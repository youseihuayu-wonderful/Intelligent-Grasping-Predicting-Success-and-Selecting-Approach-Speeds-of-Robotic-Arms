"""Observable grasp inputs, auditable evidence selection, and grouped evaluation.

Historical experiments are development evidence, never an untouched test set.
The latent body-frame grasp-point bias is retained for physical deduplication,
but is not an observable predictor or a nominal grouping dimension.
"""

from collections import Counter
import hashlib
import json
import math
from numbers import Real

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


FEATURE_NAMES = (
    "target", "scale", "mass_kg", "sliding_friction", "x_m", "y_m",
    "yaw_rad", "opening_m", "closing_force_n", "close_trigger_m",
    "descent_speed_m_s",
)
_NOMINAL_NAMES = FEATURE_NAMES[:7]
_SPLITS = ("train", "validation", "test")


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0 else result


def _vector(value, length, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def features_from_row(row: dict) -> dict:
    """Extract only pre-execution observations; labels and diagnostics are ignored.

    An absent initial_pose uses the explicitly supplied nominal settings pose.
    A present but malformed pose is an error, not a reason to silently substitute
    nominal coordinates. Quaternions follow the collector's MuJoCo wxyz order.
    """
    if not isinstance(row, dict) or not isinstance(row.get("settings"), dict):
        raise ValueError("row.settings must be a dictionary")
    settings = row["settings"]
    if settings.get("target") not in ("bowl", "bottle"):
        raise ValueError("settings.target must be bowl or bottle")
    features = {"target": settings["target"]}
    for name in FEATURE_NAMES[1:]:
        features[name] = _number(settings.get(name), f"settings.{name}")
    if "initial_pose" in row:
        pose = row["initial_pose"]
        if not isinstance(pose, dict):
            raise ValueError("initial_pose must be a dictionary when supplied")
        position = _vector(pose.get("position_m"), 3, "initial_pose.position_m")
        quaternion = _vector(pose.get("quaternion_wxyz"), 4,
                             "initial_pose.quaternion_wxyz")
        norm = math.hypot(*quaternion)
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5):
            raise ValueError("initial_pose.quaternion_wxyz must be a unit quaternion")
        w, x, y, z = (component / norm for component in quaternion)
        features["x_m"], features["y_m"] = position[:2]
        features["yaw_rad"] = math.atan2(2.0 * (w * z + x * y),
                                         1.0 - 2.0 * (y * y + z * z))
    return features


def _json_safe(value):
    """Retain invalid evidence in the audit without emitting JSON NaN/Infinity."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_number": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"invalid_value_type": type(value).__name__, "representation": repr(value)}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _metadata(row):
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary")
    if "split" in metadata or "condition_id" in metadata:
        if metadata.get("split") not in _SPLITS:
            raise ValueError("prospective metadata.split must be train, validation or test")
        condition = metadata.get("condition_id")
        if not isinstance(condition, str) or not condition:
            raise ValueError("prospective metadata.condition_id must be a nonempty string")
    if "seed" in metadata:
        seed = metadata["seed"]
        if isinstance(seed, bool) or not isinstance(seed, (int, str)) or seed == "":
            raise ValueError("metadata.seed must be an integer or nonempty string")
    return metadata


def _physical_key(row):
    # Full settings include control commands AND latent biases: repeats differing
    # only in execution error must not be mistaken for deterministic duplicates.
    settings = {
        name: value if name == "target" else _number(value, f"settings.{name}")
        for name, value in row["settings"].items()
    }
    fingerprint = row.get("initial_state_fingerprint")
    if fingerprint is not None and (not isinstance(fingerprint, str) or not fingerprint):
        raise ValueError("initial_state_fingerprint must be a nonempty string")
    state = {"fingerprint": fingerprint} if fingerprint is not None else {"pose": row.get("initial_pose")}
    return _canonical({"settings": settings, "initial_state": state})


def audit_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Quarantine unusable attempts and collapse only exact physical duplicates.

    Every source row has an audit record, including all independently detectable
    exclusion reasons. Conflicting duplicate labels or partitions abort rather
    than selecting the more convenient outcome. Clean rows are detached JSON
    copies, so callers cannot accidentally mutate the historical evidence.
    """
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")
    clean = []
    records = []
    reason_counts = Counter()
    first_by_physics = {}
    first_labelled_by_physics = {}
    first_usable_by_physics = {}
    duplicate_count = 0
    for index, row in enumerate(rows):
        record = {"source_index": index, "trial_id": None, "metadata": None,
                  "reasons": [], "decision": "excluded"}
        records.append(record)
        reasons = record["reasons"]
        if not isinstance(row, dict):
            reasons.append("invalid_row")
            record["details"] = "row must be a dictionary"
            reason_counts.update(reasons)
            continue
        record["trial_id"] = _json_safe(row.get("trial_id"))
        record["metadata"] = _json_safe(row.get("metadata", {}))
        record["trial_status"] = _json_safe(row.get("trial_status"))
        record["label"] = _json_safe(row.get("success"))
        record["warnings"] = _json_safe(row.get("warnings"))
        record["feature_pose_source"] = (
            "settled_initial_pose" if "initial_pose" in row else "nominal_settings")
        record["evidence_role"] = (
            "prospective_frozen_partition" if isinstance(row.get("metadata"), dict)
            and "split" in row["metadata"] else "historical_training_development_only")
        if row.get("trial_status") != "completed":
            reasons.append("invalid_status")
        if type(row.get("success")) is not bool:
            reasons.append("missing_or_nonbool_label")
        warnings = row.get("warnings")
        if not isinstance(warnings, list) or any(not isinstance(item, str) for item in warnings):
            reasons.append("invalid_warnings")
        elif warnings:
            reasons.append("numerical_warnings")
        try:
            metadata = _metadata(row)
        except ValueError as error:
            reasons.append("invalid_metadata")
            record["metadata_error"] = str(error)
            metadata = {}
        try:
            features_from_row(row)
            physical_key = _physical_key(row)
        except (ValueError, TypeError, OverflowError) as error:
            reasons.append("invalid_physical_inputs")
            record["physical_input_error"] = str(error)
            physical_key = None
        try:
            serialized = _canonical(row)
        except (TypeError, ValueError, OverflowError) as error:
            reasons.append("non_json_evidence")
            record["serialization_error"] = str(error)
            serialized = None
        # Track repeats even when an earlier attempt was quarantined. A bad
        # attempt must never suppress a later usable physical negative.
        if physical_key is not None:
            record["physical_fingerprint"] = hashlib.sha256(physical_key.encode()).hexdigest()
            if physical_key in first_by_physics:
                record["duplicate_of_source_index"] = first_by_physics[physical_key]
                duplicate_count += 1
            else:
                first_by_physics[physical_key] = index
            if type(row.get("success")) is bool and row.get("trial_status") == "completed":
                previous = first_labelled_by_physics.get(physical_key)
                split = metadata.get("split", "train") if "invalid_metadata" not in reasons else None
                if previous is not None:
                    if row["success"] != previous["label"]:
                        raise ValueError(
                            f"Conflicting duplicate labels at rows {previous['source_index']} and {index}")
                    if split is not None and previous["split"] is not None and split != previous["split"]:
                        raise ValueError(
                            f"Duplicate physical evidence crosses partitions at rows "
                            f"{previous['source_index']} and {index}: {previous['split']} versus {split}")
                else:
                    first_labelled_by_physics[physical_key] = {
                        "source_index": index, "label": row["success"], "split": split,
                    }
            if physical_key in first_usable_by_physics:
                reasons.append("duplicate_physical_trial")
                record["duplicate_of_clean_source_index"] = first_usable_by_physics[physical_key]
            elif not reasons:
                first_usable_by_physics[physical_key] = index
        if not reasons:
            record["decision"] = "included"
            record["clean_index"] = len(clean)
            clean.append(json.loads(serialized))
        reason_counts.update(reasons)
    audit = {
        "input_count": len(rows), "included_count": len(clean),
        "excluded_count": len(rows) - len(clean), "duplicate_count": duplicate_count,
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "row_records": records,
        "class_counts": {"0": sum(row["success"] is False for row in clean),
                         "1": sum(row["success"] is True for row in clean)},
        "feature_names": list(FEATURE_NAMES),
        "observability": {
            "pose": "settled initial position x/y and quaternion-derived yaw when supplied; otherwise nominal settings",
            "closing_force_n": "closing tendon-force command, not measured per-finger normal force",
            "excluded": ["bias_x_m", "bias_y_m", "bias_z_m", "IDs", "metadata", "seeds",
                         "success", "trial_status", "runtime", "final_pose", "contact diagnostics", "trajectory"],
            "latent_bias": "unobserved body-frame execution error; retained only in physical deduplication",
        },
        "grouping": "connected components of nominal object/state, prospective condition_id, and historical matched seed",
        "historical_partition": "train/development only; never an untouched test set",
        "duplicate_policy": "full physical settings plus initial-state fingerprint (initial pose fallback); first usable row retained",
    }
    return clean, audit


def group_ids(rows: list[dict]) -> np.ndarray:
    """Return label-independent components; nominal pose is NOT settled pose.

    A common historical seed links configurations even when their mass, friction
    or object differs. Prospective nuisance_seed values are local draws and are
    deliberately not interpreted as globally shared historical seeds.
    """
    parents = list(range(len(rows)))
    tokens_by_row = []
    owner = {}

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, row in enumerate(rows):
        features_from_row(row)
        metadata = _metadata(row)
        nominal = {"target": row["settings"]["target"]}
        nominal.update({name: _number(row["settings"][name], f"settings.{name}")
                        for name in _NOMINAL_NAMES[1:]})
        tokens = ["nominal:" + _canonical(nominal)]
        if "condition_id" in metadata:
            tokens.append("condition:" + metadata["condition_id"])
        if "split" not in metadata and "seed" in metadata:
            tokens.append("historical_seed:" + _canonical(metadata["seed"]))
        tokens_by_row.append(tokens)
        for token in tokens:
            if token in owner:
                parents[root(index)] = root(owner[token])
            else:
                owner[token] = index
    component_tokens = {}
    for index, tokens in enumerate(tokens_by_row):
        component_tokens.setdefault(root(index), set()).update(tokens)
    identifiers = {
        component: "group_" + hashlib.sha256(_canonical(sorted(tokens)).encode()).hexdigest()
        for component, tokens in component_tokens.items()
    }
    return np.asarray([identifiers[root(index)] for index in range(len(rows))], dtype=str)


def _binary_labels(y):
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be a one-dimensional sequence of binary labels")
    if labels.dtype.kind not in "biuf" or not np.isfinite(labels).all():
        raise ValueError("y must contain finite binary 0/1 labels")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain only binary 0/1 labels")
    return labels.astype(int, copy=False)


def make_cv_splits(y, groups, *, seed=5644, n_splits=3) -> list:
    """Construct one fixed grouped CV plan; fail rather than retrying folds."""
    labels = _binary_labels(y)
    group_array = np.asarray(groups)
    if group_array.ndim != 1 or len(group_array) != len(labels):
        raise ValueError("groups must be one-dimensional and match y length")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer >= 2")
    if any(value is None or (isinstance(value, Real) and not math.isfinite(value))
           or (isinstance(value, str) and not value) for value in group_array):
        raise ValueError("groups cannot contain missing or nonfinite identifiers")
    if len(np.unique(labels)) != 2:
        raise ValueError("Grouped CV requires both success and failure labels")
    if len(np.unique(group_array)) < n_splits:
        raise ValueError("Grouped CV has fewer independent groups than n_splits")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    try:
        splits = list(splitter.split(np.zeros(len(labels)), labels, group_array))
    except ValueError as error:
        raise ValueError(f"Grouped CV is not feasible: {error}") from error
    for fold, (train, validation) in enumerate(splits):
        if not len(train) or not len(validation):
            raise ValueError(f"Grouped CV fold {fold} has an empty partition")
        if np.intersect1d(train, validation).size or set(group_array[train]) & set(group_array[validation]):
            raise ValueError(f"Grouped CV fold {fold} leaks groups across partitions")
        if len(np.unique(labels[train])) != 2:
            raise ValueError(f"Grouped CV fold {fold} training partition lacks both classes")
    return splits


def evaluate_probabilities(y, probabilities) -> dict:
    """Evaluate P(success), with explicit undefined metrics and fixed bins.

    Accept either a one-dimensional probability vector or standard binary
    predict_proba columns in class order [0, 1]. The success decision threshold
    is fixed at 0.5 (equality is classified as success).
    """
    labels = _binary_labels(y)
    try:
        raw = np.asarray(probabilities)
        precision = np.finfo(float).eps
        if raw.dtype.kind == "f":
            precision = max(precision, np.finfo(raw.dtype).eps)
        probability = np.asarray(raw, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError("probabilities must be numeric") from error
    if probability.ndim == 2 and probability.shape == (len(labels), 2):
        if (not np.isfinite(probability).all() or (probability < 0).any()
                or (probability > 1).any()
                or not np.allclose(probability.sum(axis=1), 1.0, rtol=0, atol=8 * precision)):
            raise ValueError("predict_proba rows must be finite [0,1] probabilities summing to one")
        probability = probability[:, 1]
    if probability.ndim != 1 or len(probability) != len(labels):
        raise ValueError("probabilities must have shape (n,) or (n, 2) matching y")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("probabilities must be finite and within [0,1]")
    n = len(labels)
    failures = int(np.sum(labels == 0))
    successes = n - failures
    result = {
        "n_samples": n, "class_counts": {"0": failures, "1": successes},
        "decision_threshold": 0.5, "log_loss": None, "brier_score": None,
        "balanced_accuracy": None, "failure_recall": None,
        "failure_precision": None, "roc_auc": None,
        "undefined_reasons": {}, "limitations": [
            "Metrics treat supplied rows as observations; grouping and frozen partition integrity must be audited separately."
        ],
    }
    undefined = result["undefined_reasons"]
    if n:
        result["log_loss"] = float(log_loss(labels, probability, labels=[0, 1]))
        result["brier_score"] = float(brier_score_loss(labels, probability))
        predicted_success = probability >= 0.5
        predicted_failures = int(np.sum(~predicted_success))
        true_failures = int(np.sum((labels == 0) & ~predicted_success))
        if failures:
            result["failure_recall"] = true_failures / failures
        else:
            undefined["failure_recall"] = "No actual failures in this partition"
        if predicted_failures:
            result["failure_precision"] = true_failures / predicted_failures
        else:
            undefined["failure_precision"] = "No predicted failures at the fixed 0.5 threshold"
        if failures and successes:
            success_recall = float(np.sum((labels == 1) & predicted_success)) / successes
            result["balanced_accuracy"] = (result["failure_recall"] + success_recall) / 2
            result["roc_auc"] = float(roc_auc_score(labels, probability))
        else:
            reason = "Partition contains only one class; discrimination between both classes is not estimable"
            undefined["balanced_accuracy"] = reason
            undefined["roc_auc"] = reason
            result["limitations"].append(reason)
    else:
        for metric in ("log_loss", "brier_score", "balanced_accuracy", "failure_recall", "failure_precision", "roc_auc"):
            undefined[metric] = "Empty partition"
        result["limitations"].append("Empty partition")
    bins = []
    for index in range(5):
        lower, upper = index / 5, (index + 1) / 5
        mask = (probability >= lower) & ((probability <= upper) if index == 4 else (probability < upper))
        count = int(np.sum(mask))
        bins.append({
            "lower": lower, "upper": upper, "upper_inclusive": index == 4,
            "count": count,
            "mean_predicted_probability": float(np.mean(probability[mask])) if count else None,
            "observed_success_rate": float(np.mean(labels[mask])) if count else None,
            "undefined_reason": None if count else "No observations in this fixed uniform bin",
        })
    result["calibration_bins"] = bins
    return result

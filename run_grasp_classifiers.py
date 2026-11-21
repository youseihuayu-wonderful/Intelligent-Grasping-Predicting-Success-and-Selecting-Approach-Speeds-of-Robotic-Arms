"""Shared grouped-CV classifier comparison and fresh physical recommendation trials."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import importlib
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np

from collect_grasp_classifiers import (
    PHYSICS_SOURCE, ROOT, collect, execute_specs, save_json, sha256_file,
)

MODEL_NAMES = ("logistic", "lasso_logistic", "random_forest", "xgboost")


def load_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _probabilities(estimator, features):
    classes = list(estimator.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("Classifier must expose both binary classes")
    raw = np.asarray(estimator.predict_proba(features))
    precision = np.finfo(float).eps
    if raw.dtype.kind == "f":
        precision = max(precision, np.finfo(raw.dtype).eps)
    matrix = np.asarray(raw, dtype=float)
    if (matrix.shape != (len(features), 2) or not np.isfinite(matrix).all()
            or np.any(matrix < 0) or np.any(matrix > 1)
            or not np.allclose(matrix.sum(axis=1), 1, atol=8 * precision, rtol=0)):
        raise ValueError("Classifier returned invalid binary probabilities")
    return matrix[:, classes.index(1)]


def prepare_data(output):
    from grasp_classifier_data import FEATURE_NAMES, audit_rows, features_from_row, group_ids

    output = Path(output)
    protocol = load_json(output / "protocol.json")
    expected = {trial[0]: trial for trial in protocol["trials"]}
    prospective = load_json(output / "collection/trials.json")["rows"]
    if len(prospective) != len(expected) or {row["trial_id"] for row in prospective} != set(expected):
        raise ValueError("Collected trial IDs do not match the predeclared protocol")
    source = output / "collection/trials.json"
    sources = [{"path": str(source), "sha256": sha256_file(source),
                "rows": len(prospective), "historical_train_only": False}]
    rows = []
    for original in prospective:
        row = {**original, "metadata": dict(original.get("metadata", {})),
               "evidence_source": str(source)}
        spec = expected[row["trial_id"]]
        if row["settings"] != spec[1] or row["metadata"] != spec[2]:
            raise ValueError(f"Trial settings/metadata differ from frozen protocol: {row['trial_id']}")
        rows.append(row)
    clean, audit = audit_rows(rows)
    features = [features_from_row(row) for row in clean]
    y = np.asarray([int(row["success"]) for row in clean], dtype=int)
    groups = group_ids(clean)
    partitions = np.asarray([row["metadata"].get("split", "train") for row in clean])
    indices = {name: np.flatnonzero(partitions == name) for name in ("train", "validation", "test")}
    if set(partitions) != set(indices):
        raise ValueError("All three frozen partitions are required, without unknown partition labels")
    for group in np.unique(groups):
        if len(set(partitions[groups == group])) != 1:
            raise ValueError(f"Related condition group crosses partitions: {group}")
    manifest = []
    for index, row in enumerate(clean):
        manifest.append({"row_index": index, "trial_id": row["trial_id"],
                         "source": row["evidence_source"], "group_id": str(groups[index]),
                         "split": str(partitions[index]), "success": bool(row["success"]),
                         "condition_id": row["metadata"].get("condition_id")})
    summaries = {
        name: {"rows": len(selected), "groups": len(set(groups[selected])),
               "successes": int(y[selected].sum()), "failures": int((y[selected] == 0).sum())}
        for name, selected in indices.items()
    }
    save_json(output / "data_audit.json", audit)
    save_json(output / "data_sources.json", sources)
    save_json(output / "split_manifest.json", {"partitions": summaries, "rows": manifest})
    columns = ["row_index", "trial_id", "group_id", "split", *FEATURE_NAMES,
               "success", "execution_sim_s", "evidence_source"]
    with (output / "dataset.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, (row, feature) in enumerate(zip(clean, features)):
            writer.writerow({"row_index": index, "trial_id": row["trial_id"],
                             "group_id": str(groups[index]), "split": str(partitions[index]),
                             **feature, "success": int(row["success"]),
                             "execution_sim_s": row.get("execution_sim_s"),
                             "evidence_source": row["evidence_source"]})
    return clean, features, y, groups, indices, summaries


def load_csv_dataset(path):
    """Read the exported feature table without loading raw physics or model files."""
    from grasp_classifier_data import FEATURE_NAMES, features_from_row

    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"group_id", "split", "success", *FEATURE_NAMES}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"CSV is missing required columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    features, labels, groups, partitions = [], [], [], []
    group_partitions = {}
    for line, row in enumerate(rows, start=2):
        split, group = row["split"], row["group_id"]
        if split not in ("train", "validation", "test") or not group:
            raise ValueError(f"CSV line {line} requires a valid split and nonempty group_id")
        if row["success"] not in ("0", "1"):
            raise ValueError(f"CSV line {line} success must be 0 or 1")
        if group in group_partitions and group_partitions[group] != split:
            raise ValueError(f"Related condition group crosses partitions: {group}")
        group_partitions[group] = split
        settings = {"target": row["target"]}
        settings.update({name: float(row[name]) for name in FEATURE_NAMES[1:]})
        features.append(features_from_row({"settings": settings}))
        labels.append(int(row["success"]))
        groups.append(group)
        partitions.append(split)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups, dtype=str)
    partitions = np.asarray(partitions, dtype=str)
    indices = {name: np.flatnonzero(partitions == name)
               for name in ("train", "validation", "test")}
    if any(not len(selected) for selected in indices.values()):
        raise ValueError("CSV must contain nonempty train, validation and test partitions")
    summary = {
        name: {"rows": len(selected), "groups": len(set(groups[selected])),
               "successes": int(labels[selected].sum()),
               "failures": int((labels[selected] == 0).sum())}
        for name, selected in indices.items()
    }
    return features, labels, groups, indices, summary


def _fit_model(payload):
    from grasp_classifier_data import evaluate_probabilities

    name, train_x, train_y, validation_x, validation_y, cv, seed, n_iter, output = payload
    if name in ("logistic", "lasso_logistic"):
        from grasp_classifier_linear import build_search, feature_selection_state
        search = build_search(cv=cv, seed=seed, n_iter=n_iter, n_jobs=1,
                              select_features=name == "lasso_logistic")
    else:
        module = importlib.import_module("grasp_classifier_forest" if name == "random_forest"
                                         else "grasp_classifier_xgboost")
        search = module.build_search(cv=cv, seed=seed, n_iter=n_iter, n_jobs=1)
    started = time.monotonic()
    search.fit(train_x, train_y)
    estimator = search.best_estimator_
    model_path = Path(output) / "models" / f"{name}.joblib"
    joblib.dump(estimator, model_path, compress=3)
    probability = _probabilities(estimator, validation_x)
    if name in ("logistic", "lasso_logistic"):
        selection = feature_selection_state(estimator)
    else:
        selection = {"selected_feature_names": estimator.named_steps["vectorizer"].get_feature_names_out().tolist(),
                     "intercept_only": False}
    result = {
        "name": name, "model_file": str(model_path.relative_to(output)),
        "model_sha256": sha256_file(model_path), "fit_wall_s": time.monotonic() - started,
        "best_params": json_safe(search.best_params_),
        "best_cv_log_loss": -float(search.best_score_),
        "validation": evaluate_probabilities(validation_y, probability),
        "feature_selection": selection,
        "candidates": [
            {"parameters": json_safe(parameters),
             "mean_cv_log_loss": -float(search.cv_results_["mean_test_score"][index]),
             "std_cv_score": float(search.cv_results_["std_test_score"][index]),
             "rank": int(search.cv_results_["rank_test_score"][index])}
            for index, parameters in enumerate(search.cv_results_["params"])
        ],
    }
    save_json(Path(output) / "models" / f"{name}.json", result)
    return result


def fit_csv(dataset, output, *, model="logistic", seed=5644, n_iter=12):
    """Tune one route on training groups and report validation, never test scores."""
    from grasp_classifier_data import FEATURE_NAMES, evaluate_probabilities, make_cv_splits

    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model}")
    if type(n_iter) is not int or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    features, labels, groups, indices, summary = load_csv_dataset(dataset)
    train_ids, validation_ids = indices["train"], indices["validation"]
    train_x = [features[index] for index in train_ids]
    validation_x = [features[index] for index in validation_ids]
    cv = make_cv_splits(labels[train_ids], groups[train_ids], seed=seed, n_splits=3)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "models").mkdir()
    save_json(output / "cv_manifest.json", [
        {"train_rows": train_ids[left].tolist(), "validation_rows": train_ids[right].tolist(),
         "train_groups": sorted(set(groups[train_ids[left]].tolist())),
         "validation_groups": sorted(set(groups[train_ids[right]].tolist()))}
        for left, right in cv
    ])
    save_json(output / "training_protocol.json", {
        "dataset_file": Path(dataset).name, "dataset_sha256": sha256_file(dataset),
        "feature_names": list(FEATURE_NAMES), "label": "success",
        "seed": seed, "n_iter_per_search": n_iter, "cv_folds": 3,
        "cv_metric": "negative log loss", "partitions": summary,
        "test_policy": "Test rows are neither fitted nor scored by this command",
    })
    result = _fit_model((
        model, train_x, labels[train_ids], validation_x, labels[validation_ids],
        cv, seed, n_iter, str(output),
    ))
    prevalence = float(labels[train_ids].mean())
    report = {
        "model": result, "partitions": summary, "test_rows_evaluated": 0,
        "constant_predictor": {
            "training_success_prevalence": prevalence,
            "validation": evaluate_probabilities(
                labels[validation_ids], np.full(len(validation_ids), prevalence)),
        },
        "limitations": [
            "The published historical test results have already been inspected.",
            "Revised methods need new held-out conditions for a fresh final evaluation.",
            "A probability prediction does not establish better physical grasp performance.",
        ],
    }
    save_json(output / "validation_report.json", report)
    return report


def verify_physics_sources(environment):
    """Reject mixing collected labels with changed controller or object assets."""
    if sha256_file(PHYSICS_SOURCE) != environment["physical_collector_sha256"]:
        raise ValueError("Physical collector changed after protocol collection")
    for name, expected in environment["input_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Physical source or asset changed after collection: {name}")


def train(output, *, n_iter=12, model_workers=4):
    from grasp_classifier_data import evaluate_probabilities, make_cv_splits

    output = Path(output)
    if type(n_iter) is not int or n_iter < 1 or type(model_workers) is not int or model_workers < 1:
        raise ValueError("n_iter and model_workers must be positive integers")
    environment = load_json(output / "environment.json")
    verify_physics_sources(environment)
    (output / "models").mkdir(exist_ok=False)
    rows, features, y, groups, indices, partition_summary = prepare_data(output)
    seed = load_json(output / "protocol.json")["seed"]
    train_ids, validation_ids, test_ids = (indices[name] for name in ("train", "validation", "test"))
    train_x = [features[index] for index in train_ids]
    validation_x = [features[index] for index in validation_ids]
    cv = make_cv_splits(y[train_ids], groups[train_ids], seed=seed, n_splits=3)
    cv_manifest = [{"train_rows": train_ids[left].tolist(), "validation_rows": train_ids[right].tolist(),
                    "train_groups": sorted(set(groups[train_ids[left]].tolist())),
                    "validation_groups": sorted(set(groups[train_ids[right]].tolist()))}
                   for left, right in cv]
    save_json(output / "cv_manifest.json", cv_manifest)
    source_names = ("run_grasp_classifiers.py", "collect_grasp_classifiers.py", "grasp_classifier_data.py",
                    "grasp_classifier_linear.py", "grasp_classifier_forest.py", "grasp_classifier_xgboost.py",
                    "grasp_study_physics.py", "grasp_study_objects.py",
                    "assets/grasp_study/spoon.xml", "assets/grasp_study/glass.xml",
                    "assets/grasp_study/pen.xml", "pyproject.toml", "uv.lock")
    save_json(output / "training_protocol.json", {
        "seed": seed, "n_iter_per_search": n_iter, "cv_folds": 3,
        "cv_metric": "negative log loss", "model_selection": "validation log loss",
        "test_policy": "All models frozen and validation winner saved before final test evaluation",
        "probability_calibration": "Not fitted; raw probability calibration is evaluated",
        "partitions": partition_summary,
        "source_sha256": {name: sha256_file(ROOT / name) for name in source_names},
    })
    results = {}
    with ProcessPoolExecutor(max_workers=min(model_workers, len(MODEL_NAMES))) as pool:
        pending = {pool.submit(_fit_model, (
            name, train_x, y[train_ids], validation_x, y[validation_ids], cv,
            seed, n_iter, str(output),
        )): name for name in MODEL_NAMES}
        for future in as_completed(pending):
            result = future.result()
            results[result["name"]] = result
            print(json.dumps({"model": result["name"], "validation": result["validation"],
                              "best_params": result["best_params"]}), flush=True)
    winner = min(MODEL_NAMES, key=lambda name: (results[name]["validation"]["log_loss"], name))
    selection = {"selected_model": winner, "criterion": "minimum validation log loss, model-name tie break",
                 "test_used_for_selection": False}
    save_json(output / "model_selection.json", selection)
    test_x = [features[index] for index in test_ids]
    predictions = []
    for name in MODEL_NAMES:
        path = output / results[name]["model_file"]
        if sha256_file(path) != results[name]["model_sha256"]:
            raise ValueError(f"Own saved model artifact changed: {path}")
        estimator = joblib.load(path)
        probability = _probabilities(estimator, test_x)
        results[name]["test"] = evaluate_probabilities(y[test_ids], probability)
        for index, value in zip(test_ids, probability):
            predictions.append({"model": name, "trial_id": rows[index]["trial_id"],
                                "condition_id": rows[index]["metadata"]["condition_id"],
                                "predicted_success_probability": float(value),
                                "actual_success": bool(y[index])})
    prevalence = float(y[train_ids].mean())
    constant = {
        "training_success_prevalence": prevalence,
        "validation": evaluate_probabilities(y[validation_ids], np.full(len(validation_ids), prevalence)),
        "test": evaluate_probabilities(y[test_ids], np.full(len(test_ids), prevalence)),
    }
    report = {"models": results, "constant_predictor": constant,
              "selection": selection, "partitions": partition_summary,
              "limits": load_json(output / "protocol.json")["limits"]}
    save_json(output / "test_predictions.json", predictions)
    save_json(output / "comparison.json", report)
    return report


def _summarize_execution(rows):
    labelled, excluded = [], Counter()
    for row in rows:
        if row["trial_status"] != "completed":
            excluded[row["trial_status"]] += 1
        elif type(row["success"]) is not bool:
            excluded["missing_label"] += 1
        elif row.get("warnings"):
            excluded["numerical_warning"] += 1
        else:
            duration = float(row["execution_sim_s"])
            if not np.isfinite(duration) or duration < 0:
                raise ValueError("Physical execution duration must be finite and nonnegative")
            labelled.append(row)
    successful = [row for row in labelled if row["success"]]
    durations = [float(row["execution_sim_s"]) for row in successful]
    total_duration = sum(float(row["execution_sim_s"]) for row in labelled)
    return {
        "attempted": len(rows), "labelled": len(labelled),
        "successes": len(successful), "failures": len(labelled) - len(successful),
        "success_rate": len(successful) / len(labelled) if labelled else None,
        "mean_success_execution_sim_s": float(np.mean(durations)) if durations else None,
        "total_attempt_sim_s_per_success": total_duration / len(successful) if successful else None,
        "excluded_statuses": dict(excluded),
    }


def execute_recommendations(output, *, workers=3):
    from grasp_classifier_data import features_from_row

    output = Path(output)
    report = load_json(output / "comparison.json")
    protocol = load_json(output / "protocol.json")
    environment = load_json(output / "environment.json")
    verify_physics_sources(environment)
    raw = load_json(output / "collection/trials.json")["rows"]
    models = {}
    for name in MODEL_NAMES:
        info = report["models"][name]
        path = output / info["model_file"]
        if sha256_file(path) != info["model_sha256"]:
            raise ValueError(f"Own saved model artifact changed: {path}")
        models[name] = joblib.load(path)
    plan, unique_specs = [], {}
    for ordinal, group in enumerate(protocol["condition_groups"]):
        if group["split"] != "test":
            continue
        candidates = []
        for control_id, control in enumerate(group["controls"]):
            matching = [row for row in raw if row["metadata"]["condition_id"] == group["condition_id"]
                        and row["metadata"]["control_id"] == control_id
                        and row["trial_status"] == "completed" and not row.get("warnings")]
            if not matching:
                continue
            candidates.append({"control_id": control_id, "control": control,
                               "features": features_from_row(matching[0])})
        if not candidates or not any(item["control_id"] == 0 for item in candidates):
            raise ValueError(f"No physically valid baseline/candidate for {group['condition_id']}")
        candidate_x = [candidate["features"] for candidate in candidates]
        recommendations = {}
        all_probabilities = {}
        for name, estimator in models.items():
            probability = _probabilities(estimator, candidate_x)
            best = max(range(len(candidates)), key=lambda index: (
                probability[index], candidates[index]["control"]["descent_speed_m_s"],
                -candidates[index]["control_id"],
            ))
            recommendations[name] = candidates[best]
            all_probabilities[name] = [
                {"control_id": candidate["control_id"], "features": candidate["features"],
                 "predicted_success_probability": float(value)}
                for candidate, value in zip(candidates, probability)
            ]
        recommendations["fixed_baseline"] = next(item for item in candidates if item["control_id"] == 0)
        for draw in range(2):
            nuisance_seed = protocol["seed"] + 100000 + ordinal * 2 + draw
            rng = random.Random(nuisance_seed)
            error = {f"bias_{axis}_m": rng.uniform(-0.004, 0.004) for axis in "xyz"}
            for name, candidate in recommendations.items():
                control_id = candidate["control_id"]
                trial_id = f"execution.{group['condition_id']}.control_{control_id}.draw_{draw}"
                settings = {**group["nominal_settings"], **candidate["control"], **error}
                unique_specs.setdefault(trial_id, (trial_id, settings, {
                    "design": "fresh_paired_recommendation_execution",
                    "condition_id": group["condition_id"], "control_id": control_id,
                    "split": "execution", "nuisance_seed": nuisance_seed,
                }))
                plan.append({"method": name, "condition_id": group["condition_id"],
                             "draw": draw, "physical_trial_id": trial_id,
                             "selected_control_id": control_id, "selected_control": candidate["control"],
                             "candidate_predictions": all_probabilities.get(name),
                             "selection_used_observed_outcomes": False})
    save_json(output / "recommendations_before_execution.json", {
        "plan": plan, "unique_physical_attempts": len(unique_specs),
        "method_attempts": len(plan), "reused_physics": "Identical action/nuisance trials are executed once and linked across methods, not counted as independent repeats",
        "nuisance_policy": "Fresh draws, paired across methods, disjoint from collection seeds",
        "primary_model_selected_on_validation": report["selection"]["selected_model"],
    })
    rows, summary = execute_specs(list(unique_specs.values()), output / "execution", workers)
    by_id = {row["trial_id"]: row for row in rows}
    methods = {name: _summarize_execution([by_id[item["physical_trial_id"]] for item in plan
                                          if item["method"] == name])
               for name in (*MODEL_NAMES, "fixed_baseline")}
    actual = [{**item, "actual_success": by_id[item["physical_trial_id"]]["success"],
               "trial_status": by_id[item["physical_trial_id"]]["trial_status"],
               "execution_sim_s": by_id[item["physical_trial_id"]].get("execution_sim_s")}
              for item in plan]
    result = {"methods": methods, "unique_physical_execution": summary,
              "primary_model": report["selection"]["selected_model"],
              "limitations": [
                  f"Only {sum(group['split'] == 'test' for group in protocol['condition_groups'])} "
                  "held-out object/state conditions and two new draws each",
                  "Method outcomes share matched physics; rows are not independent between methods",
                  "No advantage or safety guarantee follows from a prediction score"]}
    save_json(output / "recommendations_with_outcomes.json", actual)
    save_json(output / "execution_comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "collect", "train", "execute", "fit-csv"), default="all")
    parser.add_argument("--dataset", type=Path, help="Existing exported CSV, required for fit-csv")
    parser.add_argument("--model", choices=MODEL_NAMES, default="logistic")
    parser.add_argument("--seed", type=int, default=5644)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model-workers", type=int, default=4)
    parser.add_argument("--n-iter", type=int, default=12)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "fit-csv":
        if args.dataset is None:
            parser.error("--dataset is required for --phase fit-csv")
        result = fit_csv(args.dataset, output, model=args.model,
                         seed=args.seed, n_iter=args.n_iter)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return
    if args.phase in ("all", "collect"):
        collect(output, seed=args.seed, workers=args.workers)
    if args.phase in ("all", "train"):
        result = train(output, n_iter=args.n_iter, model_workers=args.model_workers)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if args.phase in ("all", "execute"):
        result = execute_recommendations(output, workers=args.workers)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

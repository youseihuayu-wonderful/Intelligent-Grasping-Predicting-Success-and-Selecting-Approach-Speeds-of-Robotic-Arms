"""Freeze grouped classifier conditions before collecting real MuJoCo outcomes."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import importlib.util
from itertools import product
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
PHYSICS_SOURCE = ROOT / "results/grasp_feature_poc_01/run_exploration.py"
BASELINE_CONTROL = {
    "opening_m": 0.08, "closing_force_n": 4.0,
    "close_trigger_m": 0.0, "descent_speed_m_s": 0.10,
}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physics_module():
    name = "grasp_classifier_physical_adapter"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PHYSICS_SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def build_protocol(seed=5644):
    """Labels cannot influence conditions, assignments, controls or nuisance draws."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    rng = random.Random(seed)
    control_pool = [
        dict(zip(BASELINE_CONTROL, values))
        for values in product((0.04, 0.08), (2.0, 4.0, 8.0),
                              (0.0, 0.01, 0.02), (0.04, 0.10, 0.20))
    ]
    control_pool = [control for control in control_pool if control != BASELINE_CONTROL]
    groups, trials = [], []
    for target_index, target in enumerate(("bowl", "bottle")):
        indices = list(range(24))
        rng.shuffle(indices)
        assignments = {index: ("train" if rank < 16 else "validation" if rank < 20 else "test")
                       for rank, index in enumerate(indices)}
        for index in range(24):
            ordinal = target_index * 24 + index
            condition_id = f"{target}_{index:02d}"
            nominal = {
                "target": target,
                "scale": rng.choice((0.8, 1.0, 1.2)),
                "mass_kg": rng.choice((0.06, 0.18, 0.30, 0.45)),
                "sliding_friction": rng.choice((0.20, 0.40, 0.80, 1.20)),
                "x_m": rng.uniform(0.475, 0.525),
                "y_m": rng.uniform(-0.10, -0.06),
                "yaw_rad": rng.uniform(-0.15, 0.15),
            }
            controls = [dict(BASELINE_CONTROL), *rng.sample(control_pool, 3)]
            groups.append({"condition_id": condition_id, "split": assignments[index],
                           "nominal_settings": nominal, "controls": controls})
            for draw in range(2):
                nuisance_seed = seed + 1000 + ordinal * 2 + draw
                error_rng = random.Random(nuisance_seed)
                error = {f"bias_{axis}_m": error_rng.uniform(-0.004, 0.004) for axis in "xyz"}
                for control_id, control in enumerate(controls):
                    trial_id = f"{condition_id}.control_{control_id}.draw_{draw}"
                    trials.append((trial_id, {**nominal, **control, **error}, {
                        "design": "prospective_classifier_comparison",
                        "condition_id": condition_id, "split": assignments[index],
                        "control_id": control_id, "nuisance_seed": nuisance_seed,
                        "error_observability": "Latent execution uncertainty; excluded from predictive inputs",
                    }))
    return {
        "schema_version": 1, "seed": seed, "condition_groups": groups,
        "planned_attempts": len(trials), "trials": trials,
        "group_split_counts": dict(Counter(group["split"] for group in groups)),
        "baseline_control": BASELINE_CONTROL,
        "label_rule": "Existing strict bilateral contact, clearance, >=0.10 m lift, relative stability, and continuous 2 s hold",
        "nuisance_distribution": "Independent uniform [-4,+4] mm body-frame XYZ grasp error, two draws paired across controls within each condition",
        "feature_observability": "Mass/friction are simulator ground truth; settled pose available before grasp; nuisance biases, outcomes and execution diagnostics are excluded",
        "historical_evidence_policy": "Existing explored labels may supplement train only, never validation or test",
        "selection_rule": "Joint configuration maximizing predicted success; exact ties prefer faster descent then lower control_id",
        "primary_model_selection_metric": "Validation log loss; deterministic model-name tie break; test never tunes models",
        "classification_threshold": 0.5,
        "limits": [
            "Two existing geometry families with rim/neck-specific profiles; not unseen-object or real-robot generalization",
            "Strict contact labels can count contact loss without a verified object drop",
            "The trial budget establishes a runnable comparison, not guaranteed sample sufficiency or model improvement",
            "No probability calibration is fitted in this first comparison; calibration is measured and reported",
        ],
    }


def _run_spec(payload):
    spec, directory = payload
    return physics_module().run_one(spec, Path(directory))


def execute_specs(specs, directory, workers=3):
    """Keep faults as explicit nonlabels, then fail the batch instead of hiding them."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_run_spec, (spec, directory)): spec for spec in specs}
        for future in as_completed(pending):
            spec = pending[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"trial_id": spec[0], "settings": spec[1], "metadata": spec[2],
                       "trial_status": "internal_error", "success": None,
                       "error_type": type(exc).__name__, "error_message": str(exc)}
                errors.append(row)
                save_json(directory / "errors" / f"{spec[0]}.json", row)
            rows.append(row)
            print(json.dumps({"completed": len(rows), "planned": len(specs),
                              "trial_id": row["trial_id"], "status": row["trial_status"],
                              "success": row["success"]}), flush=True)
    ordering = {spec[0]: index for index, spec in enumerate(specs)}
    rows.sort(key=lambda row: ordering[row["trial_id"]])
    labelled = [row for row in rows if row["trial_status"] == "completed"]
    summary = {
        "attempted": len(rows), "labelled": len(labelled),
        "successes": sum(row["success"] is True for row in labelled),
        "failures": sum(row["success"] is False for row in labelled),
        "statuses": dict(Counter(row["trial_status"] for row in rows)),
        "numerical_warning_trials": sum(bool(row.get("warnings")) for row in rows),
        "wall_s": time.monotonic() - started,
    }
    save_json(directory / "trials.json", {"rows": rows})
    save_json(directory / "summary.json", summary)
    if errors:
        raise RuntimeError(f"{len(errors)} physical execution faults; nonlabels preserved in {directory}")
    return rows, summary


def collect(output, *, seed=5644, workers=3):
    output = Path(output)
    protocol = build_protocol(seed)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", protocol)
    environment = physics_module().provenance()
    environment.update({
        "python": sys.version, "platform": platform.platform(),
        "classifier_collector_sha256": sha256_file(Path(__file__)),
        "physical_collector_sha256": sha256_file(PHYSICS_SOURCE),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("numpy", "mujoco", "scikit-learn", "xgboost", "joblib")},
    })
    save_json(output / "environment.json", environment)
    return execute_specs(protocol["trials"], output / "collection", workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=5644)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    _, summary = collect(args.output, seed=args.seed, workers=args.workers)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

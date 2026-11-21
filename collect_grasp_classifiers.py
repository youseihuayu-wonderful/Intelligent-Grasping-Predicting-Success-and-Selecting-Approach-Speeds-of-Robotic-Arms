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

from grasp_study_objects import FRICTION_LEVELS, SCALE_LEVELS, STUDY_OBJECTS

ROOT = Path(__file__).resolve().parent
PHYSICS_SOURCE = ROOT / "grasp_study_physics.py"
BASELINE_CONTROL = {
    "opening_m": 0.08, "closing_force_n": 4.0,
    "close_trigger_m": 0.0, "descent_speed_m_s": 0.10,
}
HIGH_FORCE_CONTROL = {**BASELINE_CONTROL, "closing_force_n": 8.0}
CONTROL_LEVELS = {
    "opening_m": (0.04, 0.08),
    "closing_force_n": (2.0, 4.0, 8.0),
    "close_trigger_m": (0.0, 0.01, 0.02),
    "descent_speed_m_s": (0.04, 0.10, 0.20),
}
POSE_BOUNDS = {"x_m": (0.475, 0.525), "y_m": (-0.10, -0.06),
               "yaw_rad": (-0.15, 0.15)}
CONDITION_NAMESPACE = "five_object_v2"


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
    """Freeze a complete property grid, grouped partitions and paired controls.

    Each family's splits receive independent Latin-hypercube pose samples:
    every axis occupies each equal-width stratum once within that split.
    Neither allocation nor configuration selection consults physical outcomes.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    rng = random.Random(seed)
    control_pool = [dict(zip(CONTROL_LEVELS, values))
                    for values in product(*CONTROL_LEVELS.values())]
    joint_controls = [
        control for control in control_pool
        if sum(control[key] != BASELINE_CONTROL[key] for key in BASELINE_CONTROL) > 1
    ]
    rng.shuffle(joint_controls)
    single_contrasts = [
        {**BASELINE_CONTROL, "opening_m": 0.04},
        {**BASELINE_CONTROL, "closing_force_n": 2.0},
        {**BASELINE_CONTROL, "close_trigger_m": 0.01},
        {**BASELINE_CONTROL, "close_trigger_m": 0.02},
    ]
    groups, trials = [], []
    family_split_counts = {}
    for target_index, (target, study_object) in enumerate(STUDY_OBJECTS.items()):
        grid = list(product(SCALE_LEVELS, study_object.mass_levels_kg, FRICTION_LEVELS))
        indices = list(range(len(grid)))
        rng.shuffle(indices)
        split_indices = {"train": indices[:32], "validation": indices[32:40],
                         "test": indices[40:]}
        assignments, poses = {}, {index: {} for index in indices}
        for split, members in split_indices.items():
            for index in members:
                assignments[index] = split
            for name, (low, high) in POSE_BOUNDS.items():
                strata = list(range(len(members)))
                rng.shuffle(strata)
                for index, stratum in zip(members, strata):
                    poses[index][name] = low + (high - low) * (
                        stratum + rng.random()) / len(members)
        family_split_counts[target] = {
            split: len(members) for split, members in split_indices.items()
        }
        for index, (scale, mass, friction) in enumerate(grid):
            ordinal = target_index * len(grid) + index
            condition_id = f"{CONDITION_NAMESPACE}.s{seed}.{target}_{index:02d}"
            nominal = {
                "target": target, "scale": scale, "mass_kg": mass,
                "sliding_friction": friction, **poses[index],
            }
            controls = [
                dict(BASELINE_CONTROL), dict(HIGH_FORCE_CONTROL),
                dict(single_contrasts[index % len(single_contrasts)]),
                dict(single_contrasts[(index + 1) % len(single_contrasts)]),
                {**BASELINE_CONTROL, "descent_speed_m_s": 0.04},
                {**BASELINE_CONTROL, "descent_speed_m_s": 0.20},
                dict(joint_controls[(ordinal * 2) % len(joint_controls)]),
                dict(joint_controls[(ordinal * 2 + 1) % len(joint_controls)]),
            ]
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
                        "protocol_id": CONDITION_NAMESPACE,
                        "condition_id": condition_id, "split": assignments[index],
                        "control_id": control_id, "nuisance_seed": nuisance_seed,
                        "error_observability": "Latent execution uncertainty; excluded from predictive inputs",
                    }))
    split_counts = dict(Counter(group["split"] for group in groups))
    return {
        "schema_version": 2, "protocol_id": CONDITION_NAMESPACE, "seed": seed,
        "condition_groups": groups, "condition_group_count": len(groups),
        "planned_attempts": len(trials), "trials": trials,
        "group_split_counts": split_counts,
        "family_group_split_counts": family_split_counts,
        "planned_split_attempts": {split: count * 8 * 2 for split, count in split_counts.items()},
        "scale_levels": SCALE_LEVELS, "sliding_friction_levels": FRICTION_LEVELS,
        "study_objects": {
            name: {
                "asset_path": obj.asset_path, "mass_levels_kg": obj.mass_levels_kg,
                "geometry_description": obj.geometry_description,
                "grasp_offset_body_m_at_scale_1": obj.grasp_target.grasp_offset_m,
                "mass_interpretation": "Prescribed simulator levels, not measured real-object properties",
            }
            for name, obj in STUDY_OBJECTS.items()
        },
        "physical_source": str(PHYSICS_SOURCE.relative_to(ROOT)),
        "physical_assumptions": [
            "Panda/MuJoCo rigid-body contact simulation; no object attachment or phantom support",
            "Actual rigid collision geometries; hollow non-fracturing glass without fluids",
            "Scale applies to collision geometry and nominal grasp offset; prescribed mass scales inertia",
            "Object and finger sliding friction share the prescribed level; torsion 0.005 and rolling 0.0001",
            "MuJoCo MULTICCD enable flag 16 is enabled consistently for all five families",
            "New spoon/glass/pen collision geoms use condim 6; legacy bowl/bottle condim 3 is unchanged",
            "Glass/pen use legacy-asset contact stiffness; spoon retains MuJoCo default stiffness",
            "Pen cap remains round cylindrical geometry; no anti-roll fixture",
        ],
        "contact_model": {
            "additional_enable_flag": {"name": "MULTICCD", "value": 16, "scope": "all five families"},
            "object_collision_condim": {"bowl": 3, "bottle": 3, "spoon": 6, "glass": 6, "pen": 6},
            "torsional_friction": 0.005, "rolling_friction": 0.0001,
            "glass_pen_contact_stiffness": {
                "solimp": (0.998, 0.998, 0.001), "solref": (0.001, 1.0),
                "source": "Existing legacy asset stiffness",
            },
            "spoon_contact_stiffness": "MuJoCo defaults",
            "measurement_policy": "Raw evidence records actual object condim/friction/solref/solimp, model enableflags and observed finger-target contact dimensions",
        },
        "baseline_control": dict(BASELINE_CONTROL),
        "high_force_control": dict(HIGH_FORCE_CONTROL),
        "control_levels": CONTROL_LEVELS, "control_pool": control_pool,
        "controls_per_condition": 8, "paired_draws_per_condition": 2,
        "control_sampling": "Baseline, high-force anchor, two rotating single-parameter contrasts, both nonbaseline speeds, and two rotating multi-parameter configurations; all 54 configurations appear per family",
        "pose_bounds": POSE_BOUNDS,
        "pose_sampling": "Independent Latin-hypercube x/y/yaw strata within each family and predeclared split",
        "label_rule": "Existing strict bilateral contact, clearance, >=0.10 m lift, relative stability, and independent continuous 2 s hold",
        "nuisance_distribution": "Independent uniform [-4,+4] mm body-frame XYZ grasp error, two draws paired across controls within each condition",
        "feature_observability": "Mass/friction are simulator ground truth; settled pose available before grasp; nuisance biases, outcomes and execution diagnostics are excluded",
        "historical_evidence_policy": "No historical evidence is merged into any partition: changed MULTICCD/contact settings require a new-batch-only comparison; existing labels and the old 550-row CSV remain unchanged",
        "selection_rule": "Joint configuration maximizing predicted success; exact ties prefer faster descent then lower control_id",
        "primary_model_selection_metric": "Validation log loss; deterministic model-name tie break; test never tunes models",
        "classification_threshold": 0.5,
        "parameter_changes_from_legacy": {
            "unchanged": ["Bowl/bottle mass grids", "Scale levels 0.8/1.0/1.2",
                          "Sliding friction levels 0.2/0.4/0.8/1.2",
                          "Pose bounds", "54 joint control possibilities",
                          "Two paired uniform +/-4 mm latent XYZ draws",
                          "Strict physical label"],
            "new": ["Spoon/glass/pen rigid assets and prescribed object-specific mass grids",
                    "MULTICCD flag 16 for all families and condim 6 for new assets; legacy bowl/bottle condim 3 retained",
                    "Glass/pen adopt legacy-asset contact stiffness; spoon retains MuJoCo defaults",
                    "New-batch-only training replaces historical training supplementation because the contact model changed",
                    "Complete 48-condition property grid per family instead of 24 sampled conditions",
                    "32/8/8 train/validation/test conditions per family instead of 16/4/4",
                    "Stratified spatial/yaw poses instead of independent unstratified draws",
                    "Eight distinct controls instead of four, including required force/speed anchors",
                    "Namespaced condition IDs disjoint from legacy; 3840 planned physical trials"],
            "excluded": "Optional intermediate scale/friction levels are not part of this batch",
        },
        "limits": [
            "Five known geometry families; not unseen-object or real-robot generalization",
            "New rigid geometric approximations are not scans or validated material models; glass cannot fracture",
            "Mass/friction are prescribed simulator ground truth, not measured deployment inputs",
            "Strict contact labels can count contact loss without a verified object drop",
            "The trial budget does not guarantee sample sufficiency or model improvement",
            "No probability calibration is fitted; calibration is measured and reported",
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

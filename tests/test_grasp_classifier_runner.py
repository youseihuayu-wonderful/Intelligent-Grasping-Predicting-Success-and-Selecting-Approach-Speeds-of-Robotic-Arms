"""Synthetic evidence fixtures test protocol integrity, not grasp performance."""

from collections import defaultdict
import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from collect_grasp_classifiers import build_protocol, sha256_file
from grasp_classifier_data import FEATURE_NAMES
from run_grasp_classifiers import (
    _probabilities, _summarize_execution, fit_csv, load_csv_dataset, prepare_data,
    verify_physics_sources,
)


def fixture_row(trial_id, *, x=0.5, success=True, metadata=None, opening=0.08):
    return {
        "trial_id": trial_id,
        "settings": {
            "target": "bowl", "scale": 1.0, "mass_kg": 0.18,
            "sliding_friction": 0.8, "x_m": x, "y_m": -0.08, "yaw_rad": 0.0,
            "opening_m": opening, "closing_force_n": 4.0,
            "close_trigger_m": 0.0, "descent_speed_m_s": 0.1,
            "bias_x_m": 0.0, "bias_y_m": 0.0, "bias_z_m": 0.0,
        },
        "metadata": metadata or {}, "trial_status": "completed", "success": success,
        "warnings": [], "initial_state_fingerprint": f"fixture-{trial_id}",
        "execution_sim_s": 10.0,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class PhysicsSourceIntegrityTests(unittest.TestCase):
    def test_changing_object_geometry_after_collection_rejects_the_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "physics.py"
            asset = root / "object.xml"
            source.write_text("collector-v1", encoding="utf-8")
            asset.write_text('<geom type="box" size="0.01 0.01 0.01"/>', encoding="utf-8")
            environment = {
                "physical_collector_sha256": sha256_file(source),
                "input_sha256": {"object.xml": sha256_file(asset)},
            }
            asset.write_text('<geom type="box" size="0.02 0.01 0.01"/>', encoding="utf-8")
            with patch("run_grasp_classifiers.ROOT", root), \
                    patch("run_grasp_classifiers.PHYSICS_SOURCE", source):
                with self.assertRaisesRegex(ValueError, "changed after collection"):
                    verify_physics_sources(environment)


class CSVTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset.csv"
        self.rows = []
        for split, count in (("train", 6), ("validation", 2), ("test", 2)):
            for group in range(count):
                for label in (0, 1):
                    settings = fixture_row("csv")["settings"]
                    settings["mass_kg"] = 0.06 if label else 0.45
                    self.rows.append({
                        **{name: settings[name] for name in FEATURE_NAMES},
                        "group_id": f"{split}-{group}", "split": split,
                        "success": label, "execution_sim_s": 10.0,
                    })
        self._write()

    def _write(self):
        with self.dataset.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_related_csv_groups_cannot_cross_train_and_validation(self):
        self.rows[12]["group_id"] = self.rows[0]["group_id"]
        self._write()
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            load_csv_dataset(self.dataset)

    def test_nonfinite_csv_features_are_not_silently_imputed(self):
        self.rows[0]["mass_kg"] = "nan"
        self._write()
        with self.assertRaisesRegex(ValueError, "finite"):
            load_csv_dataset(self.dataset)

    def test_changing_test_features_and_labels_cannot_change_fitted_predictions(self):
        first = fit_csv(self.dataset, self.root / "first", n_iter=1)
        features, _, _, indices, _ = load_csv_dataset(self.dataset)
        validation = [features[index] for index in indices["validation"]]
        for row in self.rows:
            if row["split"] == "test":
                row["success"] = 1 - row["success"]
                row["mass_kg"] = 99.0
        self._write()
        second = fit_csv(self.dataset, self.root / "second", n_iter=1)
        first_model = joblib.load(self.root / "first/models/logistic.joblib")
        second_model = joblib.load(self.root / "second/models/logistic.joblib")
        np.testing.assert_array_equal(
            first_model.predict_proba(validation), second_model.predict_proba(validation))
        self.assertEqual(first["model"]["validation"], second["model"]["validation"])


class FrozenProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = build_protocol(5644)

    def test_protocol_is_deterministic_namespaced_and_frozen_before_outcomes(self):

        protocol = self.protocol
        self.assertEqual(protocol, build_protocol(5644))
        other_seed = build_protocol(5645)
        self.assertNotEqual(
            [group["nominal_settings"] for group in protocol["condition_groups"]],
            [group["nominal_settings"] for group in other_seed["condition_groups"]])
        self.assertFalse(
            {group["condition_id"] for group in protocol["condition_groups"]}
            & {group["condition_id"] for group in other_seed["condition_groups"]})
        self.assertEqual(len(protocol["condition_groups"]), 240)
        self.assertEqual(len(protocol["trials"]), 3840)
        self.assertEqual(protocol["group_split_counts"],
                         {"train": 160, "validation": 40, "test": 40})
        self.assertEqual(protocol["planned_split_attempts"],
                         {"train": 2560, "validation": 640, "test": 640})
        legacy_ids = {f"{target}_{index:02d}" for target in ("bowl", "bottle")
                      for index in range(24)}
        condition_ids = {group["condition_id"] for group in protocol["condition_groups"]}
        self.assertEqual(len(condition_ids), 240)
        self.assertFalse(condition_ids & legacy_ids)
        trial_ids = [spec[0] for spec in protocol["trials"]]
        self.assertEqual(len(set(trial_ids)), 3840)
        for _, settings, metadata in protocol["trials"]:
            self.assertNotIn("success", settings)
            self.assertNotIn("success", metadata)
        for seed in (True, -1, 2**32, 1.5, "5644"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                build_protocol(seed)

    def test_complete_object_specific_property_grids(self):
        from itertools import product

        protocol = self.protocol
        expected_masses = {
            "bowl": (0.06, 0.18, 0.30, 0.45),
            "bottle": (0.06, 0.18, 0.30, 0.45),
            "spoon": (0.015, 0.025, 0.040, 0.060),
            "glass": (0.080, 0.150, 0.250, 0.350),
            "pen": (0.005, 0.010, 0.020, 0.030),
        }
        for target, masses in expected_masses.items():
            with self.subTest(target=target):
                groups = [group for group in protocol["condition_groups"]
                          if group["nominal_settings"]["target"] == target]
                self.assertEqual(len(groups), 48)
                actual = [tuple(group["nominal_settings"][key]
                                for key in ("scale", "mass_kg", "sliding_friction"))
                          for group in groups]
                expected = set(product((0.8, 1.0, 1.2), masses, (0.2, 0.4, 0.8, 1.2)))
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(actual), len(expected))
                self.assertEqual(protocol["family_group_split_counts"][target],
                                 {"train": 32, "validation": 8, "test": 8})

    def test_spatial_yaw_strata_cover_old_bounds_in_every_family_split(self):
        from collect_grasp_classifiers import POSE_BOUNDS

        groups_by_split = defaultdict(list)
        for group in self.protocol["condition_groups"]:
            groups_by_split[group["nominal_settings"]["target"], group["split"]].append(group)
        for key, groups in groups_by_split.items():
            for name, (low, high) in POSE_BOUNDS.items():
                with self.subTest(family_split=key, axis=name):
                    values = [group["nominal_settings"][name] for group in groups]
                    self.assertTrue(all(low <= value < high for value in values))
                    strata = [int((value - low) / (high - low) * len(groups))
                              for value in values]
                    self.assertEqual(sorted(strata), list(range(len(groups))))

    def test_controls_are_distinct_have_anchors_speeds_and_all_joint_choices(self):
        from itertools import product
        from collect_grasp_classifiers import BASELINE_CONTROL, HIGH_FORCE_CONTROL, CONTROL_LEVELS

        names = tuple(BASELINE_CONTROL)
        expected_pool = set(product(*CONTROL_LEVELS.values()))
        coverage = defaultdict(set)
        for group in self.protocol["condition_groups"]:
            controls = group["controls"]
            keys = {tuple(control[name] for name in names) for control in controls}
            self.assertEqual(len(controls), 8)
            self.assertEqual(len(keys), 8)
            self.assertTrue(keys <= expected_pool)
            self.assertIn(BASELINE_CONTROL, controls)
            self.assertIn(HIGH_FORCE_CONTROL, controls)
            self.assertEqual({control["descent_speed_m_s"] for control in controls},
                             {0.04, 0.10, 0.20})
            differences = [sum(control[name] != BASELINE_CONTROL[name] for name in names)
                           for control in controls]
            self.assertIn(1, differences)
            self.assertTrue(any(count > 1 for count in differences))
            coverage[group["nominal_settings"]["target"]].update(keys)
        self.assertTrue(all(controls == expected_pool for controls in coverage.values()))

    def test_all_control_and_nuisance_variants_stay_with_their_object_condition(self):
        from grasp_classifier_data import group_ids

        protocol = self.protocol
        groups = defaultdict(list)
        declared = {group["condition_id"]: group for group in protocol["condition_groups"]}
        evidence = []
        for _, settings, metadata in protocol["trials"]:
            groups[metadata["condition_id"]].append((settings, metadata))
            evidence.append({"settings": settings, "metadata": metadata})
        partition_conditions = defaultdict(set)
        nuisance_owners = defaultdict(set)
        for condition, rows in groups.items():
            splits = {metadata["split"] for _, metadata in rows}
            self.assertEqual(len(splits), 1)
            self.assertEqual(len(rows), 16)
            self.assertEqual(splits, {declared[condition]["split"]})
            partition_conditions[next(iter(splits))].add(condition)
            per_draw = defaultdict(list)
            for settings, metadata in rows:
                per_draw[metadata["nuisance_seed"]].append(settings)
                nuisance_owners[metadata["nuisance_seed"]].add(condition)
                group = declared[condition]
                self.assertEqual({key: settings[key] for key in group["nominal_settings"]},
                                 group["nominal_settings"])
                self.assertEqual({key: settings[key] for key in group["controls"][0]},
                                 group["controls"][metadata["control_id"]])
            self.assertEqual(len(per_draw), 2)
            paired_errors = set()
            for settings in per_draw.values():
                errors = {tuple(item[f"bias_{axis}_m"] for axis in "xyz") for item in settings}
                controls = {tuple(item[key] for key in ("opening_m", "closing_force_n",
                                                        "close_trigger_m", "descent_speed_m_s"))
                            for item in settings}
                self.assertEqual(len(errors), 1)
                self.assertEqual(len(controls), 8)
                error = next(iter(errors))
                self.assertTrue(all(-0.004 <= value <= 0.004 for value in error))
                paired_errors.add(error)
            self.assertEqual(len(paired_errors), 2)
        self.assertEqual(len(nuisance_owners), 480)
        self.assertTrue(all(len(owners) == 1 for owners in nuisance_owners.values()))
        ids = group_ids(evidence)
        self.assertEqual(len(set(ids)), 240)
        partition_ids = defaultdict(set)
        condition_ids = defaultdict(set)
        for identifier, row in zip(ids, evidence):
            partition_ids[row["metadata"]["split"]].add(identifier)
            condition_ids[row["metadata"]["condition_id"]].add(identifier)
        self.assertTrue(all(len(identifiers) == 1 for identifiers in condition_ids.values()))
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(partition_conditions[left] & partition_conditions[right])
            self.assertFalse(partition_ids[left] & partition_ids[right])


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "study"
        self.historical = self.root / "results/grasp_feature_poc_01/screening/trials.json"
        self.legacy = fixture_row("legacy-negative", x=0.46, success=False, metadata={"seed": 42})
        self.prospective = [
            fixture_row(f"new-{split}", x=x, success=split != "test",
                        metadata={"condition_id": split, "split": split})
            for split, x in (("train", 0.48), ("validation", 0.50), ("test", 0.52))
        ]
        self._save_inputs()

    def _save_inputs(self):
        write_json(self.historical, {"rows": [self.legacy]})
        write_json(self.output / "collection/trials.json", {"rows": self.prospective})
        write_json(self.output / "protocol.json", {
            "trials": [[row["trial_id"], row["settings"], row["metadata"]]
                       for row in self.prospective],
        })

    def test_old_contact_model_labels_are_not_merged_into_the_new_batch(self):
        with patch("run_grasp_classifiers.ROOT", self.root):
            rows, _, labels, _, indices, summary = prepare_data(self.output)
        self.assertEqual({row["trial_id"] for row in rows},
                         {row["trial_id"] for row in self.prospective})
        self.assertEqual(labels[indices["train"]].tolist(), [1])
        self.assertEqual(summary["train"]["failures"], 0)
        self.assertEqual(json.loads(self.historical.read_text())["rows"], [self.legacy])

    def test_same_nominal_condition_cannot_cross_frozen_partitions(self):
        test_row = self.prospective[2]
        test_row["settings"]["x_m"] = self.prospective[0]["settings"]["x_m"]
        test_row["settings"]["opening_m"] = 0.04
        self._save_inputs()
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            prepare_data(self.output)

    def test_changed_control_after_protocol_freeze_is_rejected(self):
        altered = deepcopy(self.prospective)
        altered[2]["settings"]["closing_force_n"] = 8.0
        write_json(self.output / "collection/trials.json", {"rows": altered})
        with self.assertRaisesRegex(ValueError, "frozen protocol"):
            prepare_data(self.output)


class XGBoostProbabilityIntegrationTests(unittest.TestCase):
    def test_real_float32_model_predictions_are_accepted_without_renormalizing(self):
        from grasp_classifier_data import evaluate_probabilities, features_from_row
        from grasp_classifier_xgboost import build_search

        rows = [fixture_row(f"synthetic-{index}", success=index % 2 == 1)
                for index in range(64)]
        for index, row in enumerate(rows):
            row["settings"]["mass_kg"] = 0.06 if index % 2 else 0.45
        features = [features_from_row(row) for row in rows]
        labels = np.array([int(row["success"]) for row in rows])
        model = build_search(cv=2, n_iter=1).estimator
        model.set_params(classifier__n_estimators=20, classifier__max_depth=2)
        model.fit(features, labels)
        probabilities = _probabilities(model, features)
        metrics = evaluate_probabilities(labels, probabilities)
        self.assertLess(metrics["log_loss"], 0.1)
        self.assertLess(metrics["brier_score"], 0.01)


class PhysicalSummaryTests(unittest.TestCase):
    def test_physical_failures_count_but_invalid_and_numerical_faults_do_not(self):
        success = fixture_row("success")
        failure = fixture_row("failure", success=False)
        failure["execution_sim_s"] = 5.0
        invalid = fixture_row("invalid")
        invalid.update(trial_status="invalid_initial_state", success=None, execution_sim_s=None)
        warning = fixture_row("numerical", success=False)
        warning["warnings"] = ["BADQACC"]
        result = _summarize_execution([success, failure, invalid, warning])
        self.assertEqual(result["successes"], 1)
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["mean_success_execution_sim_s"], 10.0)
        self.assertEqual(result["total_attempt_sim_s_per_success"], 15.0)
        self.assertEqual(result["excluded_statuses"], {
            "invalid_initial_state": 1, "numerical_warning": 1,
        })


if __name__ == "__main__":
    unittest.main()

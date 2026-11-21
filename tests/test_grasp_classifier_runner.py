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

from collect_grasp_classifiers import build_protocol
from grasp_classifier_data import FEATURE_NAMES
from run_grasp_classifiers import (
    _probabilities, _summarize_execution, fit_csv, load_csv_dataset, prepare_data,
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
    def test_all_control_and_nuisance_variants_stay_with_their_object_condition(self):
        protocol = build_protocol(5644)
        groups = defaultdict(list)
        for _, settings, metadata in protocol["trials"]:
            groups[metadata["condition_id"]].append((settings, metadata))
        partition_conditions = defaultdict(set)
        nuisance_owners = defaultdict(set)
        for condition, rows in groups.items():
            splits = {metadata["split"] for _, metadata in rows}
            self.assertEqual(len(splits), 1)
            partition_conditions[next(iter(splits))].add(condition)
            per_draw = defaultdict(list)
            for settings, metadata in rows:
                per_draw[metadata["nuisance_seed"]].append(settings)
                nuisance_owners[metadata["nuisance_seed"]].add(condition)
            self.assertEqual(len(per_draw), 2)
            for settings in per_draw.values():
                errors = {tuple(item[f"bias_{axis}_m"] for axis in "xyz") for item in settings}
                controls = {tuple(item[key] for key in ("opening_m", "closing_force_n",
                                                        "close_trigger_m", "descent_speed_m_s"))
                            for item in settings}
                self.assertEqual(len(errors), 1)
                self.assertEqual(len(controls), 4)
        self.assertTrue(all(len(owners) == 1 for owners in nuisance_owners.values()))
        self.assertFalse(partition_conditions["train"] & partition_conditions["validation"])
        self.assertFalse(partition_conditions["train"] & partition_conditions["test"])
        self.assertFalse(partition_conditions["validation"] & partition_conditions["test"])


class PreparedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "study"
        self.historical = self.root / "historical.json"
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

    def test_historical_negative_keeps_legacy_metadata_and_enters_training_only(self):
        with patch("run_grasp_classifiers.HISTORICAL_PATHS", (self.historical,)):
            rows, _, labels, _, indices, summary = prepare_data(self.output)
        legacy_index = next(i for i, row in enumerate(rows) if row["trial_id"] == "legacy-negative")
        self.assertEqual(rows[legacy_index]["metadata"], {"seed": 42})
        self.assertEqual(labels[legacy_index], 0)
        self.assertIn(legacy_index, indices["train"])
        self.assertNotIn(legacy_index, indices["validation"])
        self.assertNotIn(legacy_index, indices["test"])
        self.assertEqual(summary["train"]["failures"], 1)
        self.assertEqual(json.loads(self.historical.read_text())["rows"], [self.legacy])

    def test_same_nominal_condition_cannot_cross_frozen_partitions(self):
        test_row = self.prospective[2]
        test_row["settings"]["x_m"] = self.legacy["settings"]["x_m"]
        test_row["settings"]["opening_m"] = 0.04
        self._save_inputs()
        with patch("run_grasp_classifiers.HISTORICAL_PATHS", (self.historical,)):
            with self.assertRaisesRegex(ValueError, "crosses partitions"):
                prepare_data(self.output)

    def test_changed_control_after_protocol_freeze_is_rejected(self):
        altered = deepcopy(self.prospective)
        altered[2]["settings"]["closing_force_n"] = 8.0
        write_json(self.output / "collection/trials.json", {"rows": altered})
        with patch("run_grasp_classifiers.HISTORICAL_PATHS", (self.historical,)):
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

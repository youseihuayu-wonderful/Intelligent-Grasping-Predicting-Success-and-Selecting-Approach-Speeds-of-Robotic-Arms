"""Small synthetic protocol regressions; no physics or model fitting."""

from copy import deepcopy
import json
import math
import unittest

import numpy as np

from grasp_classifier_data import (
    FEATURE_NAMES, audit_rows, evaluate_probabilities, features_from_row,
    group_ids, make_cv_splits,
)


def trial(identifier="unit", *, success=True, **settings):
    return {
        "trial_id": identifier,
        "settings": {
            "target": "bowl", "scale": 1.0, "mass_kg": 0.06,
            "sliding_friction": 0.8, "x_m": 0.5, "y_m": -0.08,
            "yaw_rad": 0.0, "bias_x_m": 0.0, "bias_y_m": 0.0,
            "bias_z_m": 0.0, "opening_m": 0.08, "closing_force_n": 4.0,
            "close_trigger_m": 0.0, "descent_speed_m_s": 0.1,
            **settings,
        },
        "metadata": {}, "trial_status": "completed", "success": success,
        "warnings": [], "initial_state_fingerprint": "unit-reset",
        "initial_pose": {"position_m": [0.49, -0.07, 0.72],
                         "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
    }


class ObservableFeatureTests(unittest.TestCase):
    def test_preexecution_and_completed_features_agree_without_outcome_or_latent_bias(self):
        row = trial()
        preexecution = {key: deepcopy(row[key]) for key in ("settings", "initial_pose")}
        yaw = 0.4
        quaternion = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        preexecution["initial_pose"]["quaternion_wxyz"] = quaternion
        row["initial_pose"]["quaternion_wxyz"] = quaternion
        row.update(success=False, final_pose={"position_m": [100, 100, 100]},
                   verified_hold_s=99, trajectory=["outcome"], wall_s=100)
        row["settings"].update(bias_x_m=0.006, bias_y_m=-0.006, bias_z_m=0.004)
        row["metadata"] = {"seed": 900, "prediction": 0.99}
        features = features_from_row(row)
        self.assertEqual(features, features_from_row(preexecution))
        self.assertEqual(set(features), set(FEATURE_NAMES))
        self.assertAlmostEqual(features["x_m"], 0.49)
        self.assertAlmostEqual(features["y_m"], -0.07)
        self.assertAlmostEqual(features["yaw_rad"], yaw)

    def test_missing_pose_uses_explicit_nominal_values_but_malformed_pose_never_falls_back(self):
        row = trial(x_m=0.52, yaw_rad=0.2)
        del row["initial_pose"]
        self.assertEqual(features_from_row(row)["x_m"], 0.52)
        self.assertEqual(features_from_row(row)["yaw_rad"], 0.2)
        for pose in (None, {}, {"position_m": [[0.5, 0, 0]], "quaternion_wxyz": [1, 0, 0, 0]},
                     {"position_m": [0.5, 0, 0.7], "quaternion_wxyz": [0, 0, 0, 0]}):
            with self.subTest(pose=pose):
                row["initial_pose"] = pose
                with self.assertRaises(ValueError):
                    features_from_row(row)

    def test_nonfinite_or_non_scalar_inputs_are_not_imputed(self):
        for bad in (None, True, "0.1", float("nan"), float("inf"), [0.1]):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                features_from_row(trial(mass_kg=bad))
        row = trial()
        row["initial_pose"]["position_m"][2] = float("nan")
        with self.assertRaises(ValueError):
            features_from_row(row)


class EvidenceAuditTests(unittest.TestCase):
    def test_actual_failures_survive_and_invalid_attempts_retain_all_reasons(self):
        failure = trial("negative", success=False)
        invalid = trial("invalid", success=None, mass_kg=float("nan"))
        invalid["trial_status"] = "invalid_initial_state"
        invalid["warnings"] = ["mjWARN_BADQACC"]
        invalid["metadata"] = {"latent_bad_value": float("inf")}
        integer_label = trial("integer-label", success=0, mass_kg=0.2)
        clean, audit = audit_rows([failure, invalid, integer_label])
        self.assertEqual(len(clean), 1)
        self.assertIs(clean[0]["success"], False)
        reasons = audit["row_records"][1]["reasons"]
        for reason in ("invalid_status", "missing_or_nonbool_label", "numerical_warnings", "invalid_physical_inputs"):
            self.assertIn(reason, reasons)
        self.assertIn("missing_or_nonbool_label", audit["row_records"][2]["reasons"])
        self.assertEqual(audit["class_counts"], {"0": 1, "1": 0})
        json.dumps({"clean": clean, "audit": audit}, allow_nan=False)
        clean[0]["settings"]["mass_kg"] = 2
        self.assertEqual(failure["settings"]["mass_kg"], 0.06)

    def test_latent_nuisance_is_checked_but_never_exposed_to_predictor(self):
        clean, audit = audit_rows([trial(bias_z_m=float("nan"))])
        self.assertEqual(clean, [])
        self.assertIn("invalid_physical_inputs", audit["row_records"][0]["reasons"])
        self.assertIn("bias_z_m", audit["observability"]["excluded"])

    def test_exact_repeats_deduplicate_but_different_execution_errors_remain(self):
        first = trial("first")
        duplicate = trial("repeat")
        duplicate["metadata"] = {"source_note": "different file"}
        nuisance = trial("different-draw", bias_x_m=0.006)
        clean, audit = audit_rows([first, duplicate, nuisance])
        self.assertEqual([row["trial_id"] for row in clean], ["first", "different-draw"])
        self.assertEqual(audit["duplicate_count"], 1)
        self.assertEqual(audit["row_records"][1]["duplicate_of_source_index"], 0)
        self.assertEqual(len(audit["row_records"]), 3)
        numeric_repeat = trial("numeric-repeat", scale=1, yaw_rad=-0.0)
        clean, audit = audit_rows([trial("float-representation"), numeric_repeat])
        self.assertEqual(len(clean), 1)
        self.assertEqual(audit["duplicate_count"], 1)

    def test_conflicting_duplicate_labels_and_cross_partition_repeats_raise(self):
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate labels"):
            audit_rows([trial("yes"), trial("no", success=False)])
        quarantined = trial("numerically-invalid")
        quarantined["warnings"] = ["mjWARN_BADQACC"]
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate labels"):
            audit_rows([quarantined, trial("negative", success=False)])
        held_out = trial("test")
        held_out["metadata"] = {"condition_id": "condition-test", "split": "test"}
        with self.assertRaisesRegex(ValueError, "crosses partitions"):
            audit_rows([trial("historical"), held_out])

    def test_invalid_attempt_cannot_hide_a_later_valid_physical_failure(self):
        invalid = trial("unlabelled", success=None)
        invalid["trial_status"] = "invalid_initial_state"
        clean, audit = audit_rows([invalid, trial("negative", success=False)])
        self.assertEqual([row["trial_id"] for row in clean], ["negative"])
        self.assertEqual(audit["excluded_count"], 1)
        self.assertEqual(audit["duplicate_count"], 1)
        self.assertEqual(audit["row_records"][1]["duplicate_of_source_index"], 0)
        self.assertEqual(audit["row_records"][1]["decision"], "included")

    def test_prospective_partition_metadata_cannot_be_partial(self):
        for metadata in ({"split": "test"}, {"condition_id": "new"},
                         {"split": "future", "condition_id": "new"}):
            with self.subTest(metadata=metadata):
                row = trial()
                row["metadata"] = metadata
                clean, audit = audit_rows([row])
                self.assertEqual(clean, [])
                self.assertIn("invalid_metadata", audit["row_records"][0]["reasons"])


class GroupingAndCVTests(unittest.TestCase):
    def test_nominal_groups_ignore_controls_nuisance_settling_and_outcomes(self):
        rows = [trial("one"), trial("two", success=False, opening_m=0.04, bias_x_m=0.006)]
        rows[1]["initial_pose"]["position_m"][0] += 0.01
        ids = group_ids(rows)
        self.assertEqual(ids[0], ids[1])
        rows[1]["success"] = True
        np.testing.assert_array_equal(group_ids(rows), ids)

    def test_matched_seed_transitively_connects_distinct_nominal_states(self):
        rows = [trial("a", mass_kg=0.06), trial("b", mass_kg=0.45),
                trial("c", mass_kg=0.45), trial("unrelated", x_m=0.6)]
        rows[0]["metadata"] = {"seed": 7721}
        rows[1]["metadata"] = {"seed": 7721}
        rows[2]["metadata"] = {"seed": 7722}
        ids = group_ids(rows)
        self.assertEqual(ids[0], ids[1])
        self.assertEqual(ids[1], ids[2])
        self.assertNotEqual(ids[0], ids[3])
        reverse_ids = group_ids(list(reversed(rows)))
        np.testing.assert_array_equal(ids, reverse_ids[::-1])

    def test_prospective_draw_ids_are_not_global_seed_links(self):
        rows = [trial("train", x_m=0.48), trial("test", x_m=0.52)]
        for index, row in enumerate(rows):
            row["metadata"] = {"condition_id": f"condition-{index}",
                               "split": "train" if index == 0 else "test", "nuisance_seed": 42}
        self.assertNotEqual(*group_ids(rows))
        rows[1]["metadata"]["condition_id"] = "condition-0"
        self.assertEqual(*group_ids(rows))

    def test_cv_keeps_entire_groups_together_and_rejects_single_class_training(self):
        labels = np.tile([0, 1], 6)
        groups = np.repeat([f"g-{index}" for index in range(6)], 2)
        splits = make_cv_splits(labels, groups, seed=10)
        self.assertEqual(len(splits), 3)
        validation_seen = []
        for train, validation in splits:
            self.assertFalse(set(groups[train]) & set(groups[validation]))
            self.assertEqual(set(labels[train]), {0, 1})
            validation_seen.extend(validation.tolist())
        self.assertEqual(sorted(validation_seen), list(range(len(labels))))
        with self.assertRaisesRegex(ValueError, "training partition lacks both classes"):
            make_cv_splits([0, 0, 1, 1], ["a", "a", "b", "b"], n_splits=2)
        with self.assertRaisesRegex(ValueError, "fewer independent groups"):
            make_cv_splits([0, 1], ["only", "only"])


class ProbabilityMetricTests(unittest.TestCase):
    def test_fixed_threshold_failure_metrics_and_binary_probability_columns(self):
        labels = [0, 0, 1, 1]
        probabilities = [0.2, 0.6, 0.5, 0.9]
        result = evaluate_probabilities(labels, probabilities)
        self.assertEqual(result["failure_recall"], 0.5)
        self.assertEqual(result["failure_precision"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(result["brier_score"], (0.04 + 0.36 + 0.25 + 0.01) / 4)
        matrix = [[1 - p, p] for p in probabilities]
        self.assertEqual(result, evaluate_probabilities(labels, matrix))
        self.assertEqual(sum(item["count"] for item in result["calibration_bins"]), 4)
        json.dumps(result, allow_nan=False)

    def test_float32_binary_rows_allow_only_dtype_roundoff(self):
        success = np.array([0.0347920805, 0.9652079344], dtype=np.float32)
        matrix = np.column_stack((1 - success, success))
        result = evaluate_probabilities([0, 1], matrix)
        self.assertLess(result["log_loss"], 0.1)
        self.assertLess(result["brier_score"], 0.01)
        matrix[0, 0] += 0.001
        with self.assertRaises(ValueError):
            evaluate_probabilities([0, 1], matrix)

    def test_single_class_and_empty_partitions_have_explicit_undefined_reasons(self):
        result = evaluate_probabilities([1, 1], [0.8, 1.0])
        for metric in ("roc_auc", "balanced_accuracy", "failure_recall", "failure_precision"):
            self.assertIsNone(result[metric])
            self.assertIn(metric, result["undefined_reasons"])
        self.assertIsNotNone(result["log_loss"])
        self.assertEqual(result["calibration_bins"][-1]["count"], 2)
        empty = evaluate_probabilities([], [])
        self.assertEqual(empty["n_samples"], 0)
        self.assertIsNone(empty["log_loss"])
        self.assertIn("log_loss", empty["undefined_reasons"])
        json.dumps([result, empty], allow_nan=False)

    def test_probability_shapes_nonfinite_and_out_of_range_values_raise(self):
        for probabilities in ([float("nan")], [float("inf")], [-0.01], [1.01],
                              [[0.8, 0.8]], [[0.2, 0.3, 0.5]], [], 0.5):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                evaluate_probabilities([1], probabilities)
        for labels in ([2], [float("nan")], [[1]], ["1"]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_probabilities(labels, [0.5])

    def test_exact_probability_endpoints_are_supported_without_json_infinities(self):
        result = evaluate_probabilities([0, 1, 0, 1], [0.0, 1.0, 1.0, 0.0])
        self.assertTrue(math.isfinite(result["log_loss"]))
        self.assertEqual(result["calibration_bins"][0]["count"], 2)
        self.assertEqual(result["calibration_bins"][-1]["count"], 2)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()

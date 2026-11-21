"""Invented labels exercise estimator behavior, not physical grasp evidence."""

import unittest

import numpy as np
from sklearn.model_selection import cross_validate

from grasp_classifier_forest import build_search


def feature_record(target, opening_m):
    return {
        "target": target,
        "scale": 1.0,
        "mass_kg": 0.1,
        "sliding_friction": 0.5,
        "x_m": 0.0,
        "y_m": 0.0,
        "yaw_rad": 0.0,
        "opening_m": opening_m,
        "closing_force_n": 20.0,
        "close_trigger_m": 0.02,
        "descent_speed_m_s": 0.1,
    }


class ForestSearchTests(unittest.TestCase):
    def test_search_learns_probabilities_and_refits_full_training_vocabulary(self):
        records = []
        labels = []
        groups = []
        for group in range(6):
            for target in ("bowl", "bottle"):
                for opening_m, label in ((0.02, 0), (0.08, 1)):
                    records.append(feature_record(target, opening_m))
                    labels.append(label)
                    groups.append(group)
        labels = np.asarray(labels)
        groups = np.asarray(groups)
        folds = [
            (np.flatnonzero(groups % 3 != fold), np.flatnonzero(groups % 3 == fold))
            for fold in range(3)
        ]

        search = build_search(cv=folds, seed=5644, n_iter=2)
        search.fit(records, labels)
        probabilities = search.predict_proba([
            feature_record("bowl", 0.02),
            feature_record("bowl", 0.08),
            feature_record("bottle", 0.02),
            feature_record("bottle", 0.08),
        ])

        np.testing.assert_array_equal(search.classes_, [0, 1])
        self.assertEqual(probabilities.shape, (4, 2))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        self.assertGreater(probabilities[1, 1], probabilities[0, 1])
        self.assertGreater(probabilities[3, 1], probabilities[2, 1])
        self.assertTrue(np.isfinite(search.cv_results_["mean_test_score"]).all())
        names = search.best_estimator_.named_steps["vectorizer"].get_feature_names_out()
        self.assertIn("target=bowl", names)
        self.assertIn("target=bottle", names)

    def test_fold_vocabulary_excludes_held_out_category_and_predicts_it(self):
        records = [
            feature_record(target, opening_m)
            for target in ("bowl", "bottle")
            for _ in range(4)
            for opening_m in (0.02, 0.08)
        ]
        labels = np.tile([0, 1], 8)
        bowl_indices = np.arange(8)
        bottle_indices = np.arange(8, 16)
        folds = [(bowl_indices, bottle_indices), (bottle_indices, bowl_indices)]
        # Return the actual fold estimators to observe vocabulary learning,
        # rather than inferring isolation merely from a Pipeline's structure.
        result = cross_validate(
            build_search(cv=folds, n_iter=1).estimator,
            records,
            labels,
            cv=folds,
            scoring="neg_log_loss",
            return_estimator=True,
            error_score="raise",
        )

        self.assertTrue(np.isfinite(result["test_score"]).all())
        for estimator, (training, validation) in zip(result["estimator"], folds):
            with self.subTest(training_target=records[training[0]]["target"]):
                vectorizer = estimator.named_steps["vectorizer"]
                names = vectorizer.get_feature_names_out().tolist()
                training_category = "target=" + records[training[0]]["target"]
                held_out_category = "target=" + records[validation[0]]["target"]
                self.assertIn(training_category, names)
                self.assertNotIn(held_out_category, names)
                category_index = names.index(training_category)
                training_features = vectorizer.transform([records[training[0]]])
                held_out_features = vectorizer.transform([records[validation[0]]])
                self.assertEqual(training_features[0, category_index], 1)
                self.assertEqual(held_out_features[0, category_index], 0)
                numeric_columns = [i for i, name in enumerate(names) if not name.startswith("target=")]
                np.testing.assert_array_equal(
                    training_features[:, numeric_columns],
                    held_out_features[:, numeric_columns],
                )
                probabilities = estimator.predict_proba([records[index] for index in validation])
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                self.assertGreater(probabilities[1, 1], probabilities[0, 1])


if __name__ == "__main__":
    unittest.main()

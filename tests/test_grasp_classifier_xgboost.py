"""Invented unit labels test learning behavior, not physical grasp performance."""

import unittest
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import cross_validate

from grasp_classifier_xgboost import build_search


def synthetic_features_and_labels():
    features = [
        {
            "target": "bowl" if (index // 2) % 2 == 0 else "bottle",
            "scale": 1.0,
            "mass_kg": 0.1,
            "sliding_friction": 0.6,
            "x_m": 0.0,
            "y_m": 0.0,
            "yaw_rad": 0.0,
            "opening_m": 0.04,
            "closing_force_n": 8.0 if index % 2 else 1.0,
            "close_trigger_m": 0.02,
            "descent_speed_m_s": 0.1,
        }
        for index in range(128)
    ]
    return features, np.asarray([index % 2 for index in range(128)], dtype=int)


def predefined_splits():
    first = np.arange(64)
    second = np.arange(64, 128)
    return [(first, second), (second, first)]


class XGBoostSearchTests(unittest.TestCase):
    def test_real_search_learns_probabilities_and_is_reproducible(self):
        features, labels = synthetic_features_and_labels()
        first = build_search(cv=predefined_splits(), seed=17, n_iter=2)
        second = build_search(cv=predefined_splits(), seed=17, n_iter=2)
        first.fit(features, labels)
        second.fit(features, labels)

        self.assertEqual(first.cv_results_["params"], second.cv_results_["params"])
        self.assertEqual(first.best_params_, second.best_params_)
        np.testing.assert_array_equal(first.classes_, [0, 1])
        probabilities = first.predict_proba(features)
        self.assertEqual(probabilities.shape, (128, 2))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
        self.assertGreater(probabilities[labels == 1, 1].mean(), 0.5)
        self.assertLess(probabilities[labels == 0, 1].mean(), 0.5)
        self.assertGreater(first.best_score_, -np.log(2))
        np.testing.assert_array_equal(probabilities, second.predict_proba(features))

        vectorizer = first.best_estimator_.named_steps["vectorizer"]
        names = vectorizer.get_feature_names_out().tolist()
        self.assertIn("target=bowl", names)
        self.assertIn("target=bottle", names)
        self.assertIn("closing_force_n", names)
        unseen_target = dict(features[0], target="unseen-object")
        self.assertTrue(np.all(np.isfinite(first.predict_proba([unseen_target]))))

    def test_encoding_is_fitted_only_on_each_training_fold(self):
        features, labels = synthetic_features_and_labels()
        for record in features[64:]:
            record["target"] = "held-out-category"
        estimator = build_search(cv=predefined_splits(), n_iter=1).estimator
        estimator.set_params(classifier__n_estimators=10, classifier__max_depth=2)
        result = cross_validate(
            estimator,
            features,
            labels,
            cv=predefined_splits()[:1],
            scoring="neg_log_loss",
            return_estimator=True,
            error_score="raise",
        )
        fitted = result["estimator"][0]
        names = fitted.named_steps["vectorizer"].get_feature_names_out().tolist()
        self.assertNotIn("target=held-out-category", names)
        self.assertTrue(np.isfinite(result["test_score"][0]))
        self.assertEqual(fitted.predict_proba(features[64:]).shape, (64, 2))

    def test_invalid_search_controls_are_rejected(self):
        cases = {
            "seed": [True, -1, 2**32, 1.5, None],
            "n_iter": [True, 0, -1, 1.5, None],
            "n_jobs": [True, 0, 1.5, None],
        }
        for name, invalid_values in cases.items():
            for value in invalid_values:
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        build_search(cv=predefined_splits(), **{name: value})

    def test_missing_python_or_native_dependency_never_falls_back(self):
        errors = [
            ModuleNotFoundError("No module named 'xgboost'"),
            OSError("Library not loaded: libomp.dylib"),
            ValueError("XGBoost shared library could not be loaded"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with patch("grasp_classifier_xgboost.import_module", side_effect=error):
                    with self.assertRaisesRegex(ImportError, "xgboost.*OpenMP") as raised:
                        build_search(cv=predefined_splits())
                self.assertIs(raised.exception.__cause__, error)


if __name__ == "__main__":
    unittest.main()

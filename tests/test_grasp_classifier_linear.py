"""Synthetic labels test estimator behavior, not physical grasp performance."""

import pickle
import unittest
import warnings

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler, cross_val_score, cross_validate

from grasp_classifier_linear import (
    InterceptSafeLogisticRegression,
    build_search,
    feature_selection_state,
)


def synthetic_records():
    records = []
    labels = []
    for block, target in enumerate(("bowl", "bottle")):
        for index in range(12):
            records.append({
                "target": target,
                "scale": 1.,
                "mass_kg": .1,
                "sliding_friction": .6,
                "x_m": (index - 5.5) / 100 + block * .02,
                "y_m": 0.,
                "yaw_rad": 0.,
                "opening_m": .08,
                "closing_force_n": 25.,
                "close_trigger_m": .02,
                "descent_speed_m_s": .1,
            })
            labels.append(int(index >= 6))
    return records, np.asarray(labels)


def two_folds():
    return [(np.arange(12), np.arange(12, 24)),
            (np.arange(12, 24), np.arange(12))]


class InterceptSafeLogisticTests(unittest.TestCase):
    def test_zero_columns_fit_empirical_prior_without_inventing_features(self):
        model = InterceptSafeLogisticRegression().fit(
            np.empty((4, 0)), np.array([0, 0, 0, 1]),
        )
        self.assertTrue(model.intercept_only_)
        self.assertIsNone(model.logistic_)
        self.assertEqual(model.n_features_in_, 0)
        self.assertEqual(model.coef_.shape, (1, 0))
        np.testing.assert_array_equal(model.classes_, [0, 1])
        np.testing.assert_allclose(model.predict_proba(np.empty((3, 0))),
                                   [[.75, .25]] * 3)
        np.testing.assert_allclose(model.decision_function(np.empty((3, 0))),
                                   [np.log(1 / 3)] * 3)
        np.testing.assert_array_equal(model.predict(np.empty((3, 0))), [0, 0, 0])
        with self.assertRaises(ValueError):
            model.predict_proba(np.zeros((2, 1)))

    def test_zero_column_tie_matches_sklearn_first_class_decision(self):
        model = InterceptSafeLogisticRegression().fit(
            np.empty((4, 0)), np.array([0, 1, 1, 0]),
        )
        np.testing.assert_array_equal(model.predict(np.empty((2, 0))), [0, 0])
        np.testing.assert_array_equal(model.decision_function(np.empty((2, 0))), [0., 0.])

    def test_nonempty_fit_matches_real_l2_logistic_regression(self):
        X = np.arange(20, dtype=float).reshape(-1, 1) / 10
        y = np.array([0] * 10 + [1] * 10)
        model = InterceptSafeLogisticRegression(C=.3, random_state=17).fit(X, y)
        reference = LogisticRegression(
            C=.3, solver="lbfgs", max_iter=5000, tol=1e-8,
            random_state=17,
        ).fit(X, y)
        self.assertFalse(model.intercept_only_)
        np.testing.assert_allclose(model.coef_, reference.coef_)
        np.testing.assert_allclose(model.predict_proba(X), reference.predict_proba(X))
        np.testing.assert_allclose(model.decision_function(X), reference.decision_function(X))
        np.testing.assert_array_equal(model.predict(X), reference.predict(X))

    def test_cloning_classifier_tags_and_log_loss_scoring_accept_zero_columns(self):
        model = InterceptSafeLogisticRegression(C=.03, random_state=9)
        copied = clone(model)
        self.assertTrue(is_classifier(model))
        self.assertTrue(is_classifier(copied))
        self.assertFalse(copied.__sklearn_tags__().classifier_tags.multi_class)
        self.assertEqual(copied.get_params(), model.get_params())
        self.assertFalse(hasattr(copied, "classes_"))
        y = np.array([0, 0, 1] * 4)
        scores = cross_val_score(copied, np.empty((12, 0)), y,
                                 cv=2, scoring="neg_log_loss", error_score="raise")
        expected = np.mean(np.where(y == 1, np.log(1 / 3), np.log(2 / 3)))
        np.testing.assert_allclose(scores, [expected, expected])

    def test_refitting_switches_between_intercept_and_feature_models(self):
        y = np.array([0, 0, 1, 1])
        model = InterceptSafeLogisticRegression().fit(np.empty((4, 0)), y)
        model.fit(np.arange(4, dtype=float).reshape(-1, 1), y)
        self.assertFalse(model.intercept_only_)
        self.assertEqual(model.coef_.shape, (1, 1))
        model.fit(np.empty((4, 0)), y)
        self.assertTrue(model.intercept_only_)
        np.testing.assert_allclose(model.predict_proba(np.empty((1, 0))), [[.5, .5]])

    def test_unfitted_and_single_class_models_fail_explicitly(self):
        with self.assertRaises(NotFittedError):
            InterceptSafeLogisticRegression().predict_proba(np.empty((1, 0)))
        for X in (np.empty((4, 0)), np.ones((4, 1))):
            with self.subTest(columns=X.shape[1]), self.assertRaisesRegex(
                    ValueError, "exactly two classes"):
                InterceptSafeLogisticRegression().fit(X, np.zeros(4))


class LinearSearchTests(unittest.TestCase):
    def test_all_zero_lasso_support_is_reported_as_intercept_only(self):
        records, labels = synthetic_records()
        pipeline = build_search(cv=two_folds(), select_features=True).estimator
        pipeline.set_params(selector__estimator__alpha=1000.)
        pipeline.fit(records, labels)
        probabilities = pipeline.predict_proba(records[:3])
        selector = pipeline.named_steps["selector"]
        np.testing.assert_array_equal(selector.estimator_.coef_, 0.)
        self.assertFalse(selector.get_support().any())
        self.assertEqual(selector.get_feature_names_out(
            pipeline.named_steps["vectorizer"].get_feature_names_out()).size, 0)
        state = feature_selection_state(pipeline)
        self.assertEqual(state["selected_feature_names"], [])
        self.assertEqual(state["n_features_selected"], 0)
        self.assertTrue(state["intercept_only"])
        self.assertEqual(state["selection_method"], "squared_loss_lasso")
        self.assertEqual(state["lasso_alpha"], 1000.)
        np.testing.assert_allclose(probabilities, [[.5, .5]] * 3)

    def test_retained_support_names_and_logistic_inverse_strength_are_inspectable(self):
        records, labels = synthetic_records()
        base = build_search(cv=two_folds(), select_features=True).estimator
        base.set_params(selector__estimator__alpha=.001)
        weak_C = clone(base).set_params(classifier__C=.001).fit(records, labels)
        strong_C = clone(base).set_params(classifier__C=100.).fit(records, labels)
        state = feature_selection_state(strong_C)
        self.assertIn("x_m", state["selected_feature_names"])
        self.assertFalse(state["intercept_only"])
        self.assertEqual(state["logistic_penalty"], "l2")
        self.assertEqual(state["logistic_C"], 100.)
        selector = strong_C.named_steps["selector"]
        np.testing.assert_array_equal(selector.get_support(), selector.estimator_.coef_ != 0)
        weak_distance = np.mean(np.abs(weak_C.predict_proba(records)[:, 1] - .5))
        strong_distance = np.mean(np.abs(strong_C.predict_proba(records)[:, 1] - .5))
        self.assertGreater(strong_distance, weak_distance)

    def test_vocabulary_scaling_and_lasso_are_fitted_only_on_each_training_fold(self):
        records, labels = synthetic_records()
        pipeline = build_search(cv=two_folds(), select_features=True).estimator
        pipeline.set_params(selector__estimator__alpha=.01)
        result = cross_validate(pipeline, records, labels, cv=two_folds(),
                                scoring="neg_log_loss", return_estimator=True,
                                error_score="raise")
        for fitted, (training, validation) in zip(result["estimator"], two_folds()):
            vectorizer = fitted.named_steps["vectorizer"]
            training_rows = [records[index] for index in training]
            held_out_target = records[validation[0]]["target"]
            self.assertNotIn(f"target={held_out_target}", vectorizer.get_feature_names_out())
            encoded = vectorizer.transform(training_rows)
            scaler = fitted.named_steps["scaler"]
            np.testing.assert_allclose(scaler.mean_, encoded.mean(axis=0))
            selector = fitted.named_steps["selector"]
            fold_reference = clone(selector.estimator).fit(scaler.transform(encoded),
                                                           labels[training])
            np.testing.assert_allclose(selector.estimator_.coef_, fold_reference.coef_)
        self.assertTrue(np.isfinite(result["test_score"]).all())

    def test_both_searches_train_clone_and_serialize_with_valid_probabilities(self):
        records, labels = synthetic_records()
        for selected in (False, True):
            with self.subTest(select_features=selected):
                search = build_search(cv=two_folds(), seed=29, n_iter=2,
                                      select_features=selected)
                self.assertTrue(is_classifier(clone(search)))
                # Some genuine candidates deliberately have empty support.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="No features were selected.*",
                                            category=UserWarning)
                    search.fit(records, labels)
                    probabilities = search.predict_proba(records)
                    restored = pickle.loads(pickle.dumps(search))
                    np.testing.assert_allclose(restored.predict_proba(records), probabilities)
                self.assertTrue(np.isfinite(search.best_score_))
                self.assertEqual(probabilities.shape, (24, 2))
                self.assertTrue(np.isfinite(probabilities).all())
                self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
                self.assertIn("classifier__C", search.best_params_)
                state = feature_selection_state(search.best_estimator_)
                if selected:
                    self.assertIn("selector__estimator__alpha", search.best_params_)
                else:
                    self.assertEqual(state["encoded_feature_names"], state["selected_feature_names"])
                    self.assertTrue(all(state["support_mask"]))
                    self.assertFalse(state["intercept_only"])

    def test_joint_alpha_and_C_sampling_is_reproducible(self):
        def sampled(seed):
            search = build_search(cv=two_folds(), seed=seed, n_iter=8,
                                  select_features=True)
            return list(ParameterSampler(search.param_distributions, n_iter=search.n_iter,
                                         random_state=search.random_state))

        first = sampled(31)
        self.assertEqual(first, sampled(31))
        self.assertNotEqual(first, sampled(32))
        self.assertGreater(len({p["classifier__C"] for p in first}), 1)
        self.assertGreater(len({p["selector__estimator__alpha"] for p in first}), 1)


if __name__ == "__main__":
    unittest.main()

"""Fold-local full-feature and LASSO-screened logistic probability models.

LASSO here is squared-loss regression on the binary training labels, minimizing
``||y - Xw||² / (2 n) + alpha * ||w||_1`` after standardization. It is not
L1-penalized logistic regression. The subsequent logistic model always uses L2
regularization; its ``C`` is inverse regularization strength, not LASSO alpha.

Inspect a fitted search's ``best_estimator_`` with ``feature_selection_state``.
The pipeline's ``vectorizer.get_feature_names_out()`` and, for the screened
route, ``selector.get_support()`` also expose the encoding and exact support.
"""

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, validate_data


class InterceptSafeLogisticRegression(ClassifierMixin, BaseEstimator):
    """L2 logistic regression, or an explicit empirical-prior intercept model.

    Zero-column input is legitimate after LASSO screening. In that case the
    fitted probability is the training class prevalence, the maximum-likelihood
    unpenalized intercept, with no invented feature or probability smoothing.
    Both classes must be present, just as for the ordinary logistic fit.
    """

    def __init__(self, *, C=1.0, max_iter=5000, tol=1e-8, random_state=5644):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        if (isinstance(self.C, bool) or not isinstance(self.C, Real)
                or not np.isfinite(self.C) or self.C <= 0):
            raise ValueError("C must be a finite positive inverse L2 strength")
        X, y = validate_data(self, X, y, dtype=np.float64, ensure_min_features=0)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("Logistic classification requires exactly two classes")
        self.intercept_only_ = X.shape[1] == 0
        if self.intercept_only_:
            self.logistic_ = None
            self.class_prior_ = np.array([
                np.mean(y == label) for label in self.classes_
            ])
            self.coef_ = np.empty((1, 0), dtype=float)
            self.intercept_ = np.array([
                np.log(self.class_prior_[1] / self.class_prior_[0])
            ])
            self.n_iter_ = np.array([0], dtype=np.int32)
        else:
            self.logistic_ = LogisticRegression(
                C=self.C, solver="lbfgs", max_iter=self.max_iter,
                tol=self.tol, random_state=self.random_state,
            ).fit(X, y)
            self.coef_ = self.logistic_.coef_
            self.intercept_ = self.logistic_.intercept_
            self.n_iter_ = self.logistic_.n_iter_
        return self

    def _intercept_input(self, X):
        return validate_data(
            self, X, reset=False, dtype=np.float64, ensure_min_features=0,
        )

    def predict_proba(self, X):
        check_is_fitted(self, "logistic_")
        if self.logistic_ is not None:
            return self.logistic_.predict_proba(X)
        X = self._intercept_input(X)
        return np.tile(self.class_prior_, (X.shape[0], 1))

    def decision_function(self, X):
        check_is_fitted(self, "logistic_")
        if self.logistic_ is not None:
            return self.logistic_.decision_function(X)
        X = self._intercept_input(X)
        return np.full(X.shape[0], self.intercept_[0])

    def predict(self, X):
        check_is_fitted(self, "logistic_")
        if self.logistic_ is not None:
            return self.logistic_.predict(X)
        X = self._intercept_input(X)
        # Match sklearn's first-class choice when the binary decision is zero.
        label = self.classes_[int(self.class_prior_[1] > self.class_prior_[0])]
        return np.full(X.shape[0], label)


def build_search(*, cv, seed=5644, n_iter=12, n_jobs=1, select_features=False):
    """Build a reproducible negative-log-loss search on dictionary records.

    Every CV clone learns its own vocabulary, scaling and optional LASSO
    support. ``cv`` may be a precomputed list of grouped index pairs. Search
    parallelism is controlled by ``n_jobs``; each underlying solver is serial.
    """
    steps = [
        ("vectorizer", DictVectorizer(sparse=False)),
        ("scaler", StandardScaler()),
    ]
    parameters = {
        "classifier__C": [.001, .003, .01, .03, .1, .3, 1., 3., 10., 30., 100., 300.],
    }
    if select_features:
        # SelectFromModel uses >= threshold: the smallest positive float keeps
        # every nonzero coefficient while excluding exact LASSO zeros.
        steps.append(("selector", SelectFromModel(
            Lasso(max_iter=20000, tol=1e-7, selection="cyclic", random_state=seed),
            threshold=np.nextafter(0.0, 1.0),
        )))
        steps.append(("classifier", InterceptSafeLogisticRegression(random_state=seed)))
        parameters["selector__estimator__alpha"] = [.001, .003, .01, .03, .1, .3, 1.]
    else:
        steps.append(("classifier", LogisticRegression(
            solver="lbfgs", max_iter=5000, tol=1e-8,
            random_state=seed,
        )))
    return RandomizedSearchCV(
        Pipeline(steps), param_distributions=parameters, cv=cv, n_iter=n_iter,
        scoring="neg_log_loss", refit=True, random_state=seed, n_jobs=n_jobs,
        error_score="raise", return_train_score=False,
    )


def feature_selection_state(estimator):
    """Return JSON-safe fitted support, including a declared zero-feature state.

    ``estimator`` is the fitted pipeline (``search.best_estimator_``), not the
    search itself. Unscreened models report every encoded feature as selected.
    """
    vectorizer = estimator.named_steps["vectorizer"]
    check_is_fitted(vectorizer, "vocabulary_")
    names = vectorizer.get_feature_names_out()
    selector = estimator.named_steps.get("selector")
    if selector is None:
        support = np.ones(len(names), dtype=bool)
        alpha = None
    else:
        support = selector.get_support()
        alpha = float(selector.estimator_.alpha)
    classifier = estimator.named_steps["classifier"]
    check_is_fitted(classifier, "classes_")
    return {
        "selection_method": "squared_loss_lasso" if selector is not None else "none",
        "encoded_feature_names": names.tolist(),
        "support_mask": support.tolist(),
        "selected_feature_names": names[support].tolist(),
        "n_features_selected": int(support.sum()),
        "intercept_only": bool(getattr(classifier, "intercept_only_", False)),
        "lasso_alpha": alpha,
        "logistic_penalty": "l2",
        "logistic_C": float(classifier.C),
    }

"""Fold-local dictionary encoding and CPU XGBoost probability search.

Only training records and their predeclared CV splits belong in ``fit``. The
number of boosting rounds is searched directly; no evaluation set or early
stopping consumes validation or final-test outcomes.
"""

from importlib import import_module
from numbers import Integral

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def build_search(*, cv, seed=5644, n_iter=12, n_jobs=1) -> RandomizedSearchCV:
    """Build an unfitted binary-logistic XGBoost search over feature dictionaries.

    ``cv`` may be precomputed grouped train/validation index pairs. ``n_jobs``
    controls search parallelism only: each CPU booster uses one worker. The
    fitted vectorizer and classifier are available as pipeline steps
    ``vectorizer`` and ``classifier`` on ``best_estimator_``. Missing Python or
    native XGBoost dependencies raise an explicit ImportError, never a fallback
    to a different model family.
    """
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if isinstance(n_iter, bool) or not isinstance(n_iter, Integral) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError("n_jobs must be a nonzero integer")

    try:
        xgboost = import_module("xgboost")
    except (ImportError, OSError, ValueError) as exc:
        # XGBoost's native-library loader raises XGBoostError (a ValueError)
        # when its shared library or OpenMP runtime cannot be loaded.
        raise ImportError(
            "XGBoost requires the 'xgboost' Python dependency and a working "
            "native OpenMP runtime (libomp on macOS). Install the project's "
            "pinned dependencies and required native runtime."
        ) from exc

    estimator = Pipeline([
        ("vectorizer", DictVectorizer(sparse=False)),
        ("classifier", xgboost.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            random_state=int(seed),
            n_jobs=1,
        )),
    ])
    parameters = {
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.03, 0.08, 0.15],
        "classifier__n_estimators": [60, 120, 200],
        "classifier__min_child_weight": [1, 3, 5],
        "classifier__subsample": [0.75, 1.0],
        "classifier__colsample_bytree": [0.75, 1.0],
        "classifier__reg_alpha": [0.0, 0.1, 1.0],
        "classifier__reg_lambda": [1.0, 3.0, 10.0],
    }
    return RandomizedSearchCV(
        estimator,
        param_distributions=parameters,
        n_iter=int(n_iter),
        scoring="neg_log_loss",
        cv=cv,
        refit=True,
        random_state=int(seed),
        n_jobs=int(n_jobs),
        error_score="raise",
        return_train_score=False,
    )

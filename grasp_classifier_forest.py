"""Fold-local random-forest search for pre-execution grasp feature dictionaries.

Probabilities are empirical forest estimates, not calibrated safety guarantees.
The caller supplies the shared training-only grouped folds; neither vocabulary
construction nor hyperparameter selection uses the held-out test partition.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


def build_search(*, cv, seed=5644, n_iter=12, n_jobs=1) -> RandomizedSearchCV:
    """Build an unfitted log-loss search over a modest finite forest space.

    ``cv`` accepts the precomputed grouped train/validation index pairs shared
    by all comparison models. ``n_jobs`` controls search-level parallelism;
    each forest always uses one worker to avoid nested thread multiplication.
    The fitted vocabulary is available through
    ``search.best_estimator_.named_steps['vectorizer'].get_feature_names_out()``.
    """
    estimator = Pipeline([
        ("vectorizer", DictVectorizer(sparse=False)),
        ("classifier", RandomForestClassifier(random_state=seed, n_jobs=1)),
    ])
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions={
            "classifier__n_estimators": [64, 128, 256],
            "classifier__max_depth": [3, 6, None],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", 0.75, None],
        },
        n_iter=n_iter,
        scoring="neg_log_loss",
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=False,
    )

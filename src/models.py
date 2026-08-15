"""Model zoo: baselines, bagging, boosting, stacking and voting.

Every estimator is wrapped in a Pipeline whose first step is the preprocessor,
so cross-validation refits preprocessing on each fold. Tree models skip scaling
because they are invariant to monotone transforms.

Class imbalance is handled with class weights rather than resampling. The
comparison in `imbalance_strategies` records why: SMOTE interpolates between
one-hot medical categories and invents patients who are 0.6 of the way to being
on insulin, and it distorts the calibration we rely on for cost-based
thresholding.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config as C
from src.pipeline import build_preprocessor

CV = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.SEED)


def _pipe(schema, estimator, *, scale: bool) -> Pipeline:
    return Pipeline(
        [("pre", build_preprocessor(schema, scale=scale)), ("clf", estimator)]
    )


def scale_pos_weight(y) -> float:
    """Ratio of negatives to positives, for the boosting models."""
    y = np.asarray(y)
    pos = y.sum()
    return float((len(y) - pos) / pos) if pos else 1.0


# ---------------------------------------------------------------------------
# Individual model builders
# ---------------------------------------------------------------------------
def logistic_baseline(schema) -> Pipeline:
    return _pipe(
        schema,
        LogisticRegression(
            max_iter=3000, class_weight="balanced", random_state=C.SEED, n_jobs=-1
        ),
        scale=True,
    )


def tree_baseline(schema) -> Pipeline:
    return _pipe(
        schema,
        DecisionTreeClassifier(
            class_weight="balanced", random_state=C.SEED, max_depth=6, min_samples_leaf=50
        ),
        scale=False,
    )


def random_forest(schema) -> Pipeline:
    return _pipe(
        schema,
        RandomForestClassifier(
            n_estimators=600, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEED,
        ),
        scale=False,
    )


def bagged_trees(schema) -> Pipeline:
    """Plain bagging over deep trees, to show the variance-reduction mechanism.

    Random Forest adds feature subsampling on top of bagging; keeping this
    alongside makes the difference between the two visible rather than assumed.
    """
    return _pipe(
        schema,
        BaggingClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=None, min_samples_leaf=5, class_weight="balanced",
                random_state=C.SEED,
            ),
            n_estimators=200, max_samples=0.8, n_jobs=-1, random_state=C.SEED,
        ),
        scale=False,
    )


def xgboost_model(schema, spw: float = 1.0) -> Pipeline:
    return _pipe(
        schema,
        XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=2.0, scale_pos_weight=spw, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=C.SEED, verbosity=0,
        ),
        scale=False,
    )


def lightgbm_model(schema, spw: float = 1.0) -> Pipeline:
    return _pipe(
        schema,
        LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=2.0, scale_pos_weight=spw,
            n_jobs=-1, random_state=C.SEED, verbose=-1,
        ),
        scale=False,
    )


# ---------------------------------------------------------------------------
# Heterogeneous ensembles
# ---------------------------------------------------------------------------
def _base_estimators(schema, spw: float):
    """Deliberately diverse base learners.

    A linear model, a bagged tree ensemble and two boosted ensembles make
    different errors, which is the precondition for stacking to add anything.
    The prediction-correlation heatmap in the notebook checks this held.
    """
    return [
        ("logreg", logistic_baseline(schema)),
        ("rf", random_forest(schema)),
        ("xgb", xgboost_model(schema, spw)),
        ("lgbm", lightgbm_model(schema, spw)),
    ]


def stacking_model(schema, spw: float = 1.0) -> StackingClassifier:
    """Stacked generalisation with internal cross-validation.

    `cv=CV` is the whole ballgame for meta-leakage. Each base learner produces
    out-of-fold predictions only, so the meta-learner is trained on predictions
    for rows the base learner never saw during fitting. Passing `cv=None` (or
    fitting bases on all of train and then predicting on train) would let the
    meta-learner reward whichever base model memorised hardest.
    """
    return StackingClassifier(
        estimators=_base_estimators(schema, spw),
        final_estimator=LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=C.SEED
        ),
        cv=CV,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=1,          # base pipelines already parallelise internally
    )


def voting_model(schema, spw: float = 1.0, weights=None) -> VotingClassifier:
    return VotingClassifier(
        estimators=_base_estimators(schema, spw),
        voting="soft",
        weights=weights,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Hyperparameter search spaces for RandomizedSearchCV
# ---------------------------------------------------------------------------
SEARCH_SPACES: dict[str, dict] = {
    "decision_tree": {
        "clf__max_depth": [3, 4, 5, 6, 8, 10, 12],
        "clf__min_samples_leaf": [20, 50, 100, 200],
        "clf__criterion": ["gini", "entropy"],
    },
    "logistic": {
        "clf__C": np.logspace(-3, 2, 20),
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    },
    "random_forest": {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [8, 12, 16, None],
        "clf__min_samples_leaf": [5, 10, 20, 40],
        "clf__max_features": ["sqrt", "log2", 0.3],
    },
    "xgboost": {
        "clf__n_estimators": [300, 500, 800],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 5, 10, 20],
        "clf__reg_lambda": [1.0, 2.0, 5.0, 10.0],
    },
    "lightgbm": {
        "clf__n_estimators": [300, 600, 900],
        "clf__learning_rate": [0.02, 0.05, 0.1],
        "clf__num_leaves": [15, 31, 63, 127],
        "clf__min_child_samples": [20, 40, 80],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__reg_lambda": [1.0, 2.0, 5.0],
    },
}


def build_all(schema, y_train) -> dict[str, object]:
    """The seven models compared on the untouched test set."""
    spw = scale_pos_weight(y_train)
    return {
        "Logistic Regression (baseline)": logistic_baseline(schema),
        "Decision Tree (baseline)": tree_baseline(schema),
        "Bagging (trees)": bagged_trees(schema),
        "Random Forest (bagging)": random_forest(schema),
        "XGBoost (boosting)": xgboost_model(schema, spw),
        "LightGBM (boosting)": lightgbm_model(schema, spw),
        "Stacking (heterogeneous)": stacking_model(schema, spw),
        "Soft Voting (heterogeneous)": voting_model(schema, spw),
    }


BASELINE_KEY = "Logistic Regression (baseline)"

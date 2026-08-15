"""Metrics, cost-based thresholding, and significance testing.

The scoring philosophy here: with an 9% positive class, accuracy is useless and
ROC-AUC is only half the story. The decisions that matter are made at a
threshold, and that threshold should come from what a mistake actually costs a
patient, not from the 0.5 default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config as C

rng_global = np.random.default_rng(C.SEED)


# ---------------------------------------------------------------------------
# Threshold-free and threshold-based metrics
# ---------------------------------------------------------------------------
def precision_at_k(y_true, y_prob, k: float = C.FOLLOWUP_CAPACITY) -> float:
    """Precision among the top k share of patients by predicted risk.

    This is the metric a discharge nurse actually experiences: given capacity
    to call 10% of patients, what fraction of those calls reach someone who
    would have come back?
    """
    y_true = np.asarray(y_true)
    n = max(1, int(round(k * len(y_true))))
    top = np.argsort(y_prob)[::-1][:n]
    return float(y_true[top].mean())


def lift_at_k(y_true, y_prob, k: float = C.FOLLOWUP_CAPACITY) -> float:
    """How many times better than random the top-k list is."""
    base = float(np.mean(y_true))
    return precision_at_k(y_true, y_prob, k) / base if base > 0 else np.nan


def expected_cost(
    y_true, y_pred, cost_fn: float = C.COST_FN, cost_fp: float = C.COST_FP,
    cost_tp: float = C.COST_TP, cost_tn: float = C.COST_TN,
) -> float:
    """Total expected cost per patient at a given set of hard predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    total = tn * cost_tn + fp * cost_fp + fn * cost_fn + tp * cost_tp
    return float(total / len(y_true))


def cost_optimal_threshold(
    y_true, y_prob, cost_fn: float = C.COST_FN, cost_fp: float = C.COST_FP,
    capacity: float | None = C.FOLLOWUP_CAPACITY,
) -> tuple[float, pd.DataFrame]:
    """Lowest-cost threshold the clinic can actually staff.

    Must be called on the *validation* set; choosing a threshold on test would
    be a subtle form of test-set fitting.

    The `capacity` constraint is not a technicality, it is the whole problem.
    An avoidable readmission costs roughly 200 times a follow-up call, so
    unconstrained cost minimisation always returns "call every patient" — the
    correct answer for a clinic with unlimited nurses and a useless one for the
    under-resourced clinics this project targets. Constraining the flagged rate
    to what the clinic can staff turns the threshold into a triage decision:
    given N calls, which N patients?

    Pass `capacity=None` to see the unconstrained optimum and its degeneracy.
    """
    grid = np.linspace(0.01, 0.99, 393)
    rows = []
    for t in grid:
        pred = (np.asarray(y_prob) >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(t),
                "expected_cost": expected_cost(y_true, pred, cost_fn, cost_fp),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "flagged_rate": float(pred.mean()),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            }
        )
    curve = pd.DataFrame(rows)
    curve["feasible"] = (
        curve["flagged_rate"] <= capacity if capacity is not None else True
    )

    feasible = curve[curve["feasible"]]
    if feasible.empty:                       # capacity tighter than any threshold allows
        feasible = curve
    best = float(feasible.loc[feasible["expected_cost"].idxmin(), "threshold"])
    return best, curve


def classification_metrics(y_true, y_prob, threshold: float = 0.5, label: str = "") -> dict:
    """One row of the model comparison table."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "model": label,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "brier": brier_score_loss(y_true, y_prob),
        "precision_at_10pct": precision_at_k(y_true, y_prob),
        "lift_at_10pct": lift_at_k(y_true, y_prob),
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "expected_cost_per_patient": expected_cost(y_true, pred),
    }


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------
def bootstrap_auc_difference(
    y_true, prob_a, prob_b, n_boot: int = 2000, seed: int = C.SEED
) -> dict:
    """Paired bootstrap CI for AUC(a) - AUC(b).

    Paired because both models are scored on the same resampled patients, which
    removes between-patient variance and isolates the model difference.
    """
    y_true = np.asarray(y_true)
    prob_a, prob_b = np.asarray(prob_a), np.asarray(prob_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(
            roc_auc_score(y_true[idx], prob_a[idx]) - roc_auc_score(y_true[idx], prob_b[idx])
        )

    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "auc_a": float(roc_auc_score(y_true, prob_a)),
        "auc_b": float(roc_auc_score(y_true, prob_b)),
        "mean_difference": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0),
        "n_bootstrap": int(len(diffs)),
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on the disagreements between two classifiers."""
    from scipy.stats import binomtest

    y_true = np.asarray(y_true)
    a_right = np.asarray(pred_a) == y_true
    b_right = np.asarray(pred_b) == y_true

    n01 = int(np.sum(a_right & ~b_right))   # a right, b wrong
    n10 = int(np.sum(~a_right & b_right))   # a wrong, b right

    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "significant": False}

    p = binomtest(n01, n01 + n10, 0.5).pvalue
    return {
        "n01_a_right_b_wrong": n01,
        "n10_a_wrong_b_right": n10,
        "p_value": float(p),
        "significant": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# Leakage sensitivity: what the naive split would have bought us
# ---------------------------------------------------------------------------
def leakage_sensitivity(engineered_all_encounters: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Quantify the AUC inflation from splitting encounters instead of patients.

    Trains the same model twice on the full multi-encounter data:
      A. random row split, so a patient can appear in train and test
      B. patient-grouped split, which is the honest version

    The gap is the score a submission would have claimed by accident.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.pipeline import Pipeline as SkPipeline

    from src.features import column_schema
    from src.pipeline import build_preprocessor

    df = engineered_all_encounters
    schema = column_schema(df)
    cols = schema["numeric"] + schema["binary"] + schema["ordinal"] + schema["nominal"]
    X, y, g = df[cols], df[C.TARGET], df["patient_nbr"]

    def fit_score(tr, te, name):
        model = SkPipeline(
            [
                ("pre", build_preprocessor(schema, scale=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                        random_state=seed, class_weight="balanced_subsample",
                    ),
                ),
            ]
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        prob = model.predict_proba(X.iloc[te])[:, 1]
        overlap = len(set(g.iloc[tr]) & set(g.iloc[te]))
        return {
            "split_strategy": name,
            "test_roc_auc": float(roc_auc_score(y.iloc[te], prob)),
            "patients_in_both_splits": overlap,
            "train_rows": len(tr),
            "test_rows": len(te),
        }

    idx = np.arange(len(df))
    tr_naive, te_naive = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=seed
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_group, te_group = next(gss.split(idx, y, groups=g))

    rows = [
        fit_score(tr_naive, te_naive, "Random row split (leaky)"),
        fit_score(tr_group, te_group, "Patient-grouped split (honest)"),
    ]

    # The two rows above use different test sets, so their gap confounds
    # leakage with test-set difficulty. Splitting the leaky test set into
    # patients the model *did* see in training and patients it did not
    # isolates the memorisation effect on a single fitted model.
    model = SkPipeline(
        [
            ("pre", build_preprocessor(schema, scale=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                    random_state=seed, class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    model.fit(X.iloc[tr_naive], y.iloc[tr_naive])
    prob = model.predict_proba(X.iloc[te_naive])[:, 1]

    seen = g.iloc[te_naive].isin(set(g.iloc[tr_naive])).to_numpy()
    y_te = y.iloc[te_naive].to_numpy()

    for label, mask in [
        ("  └ leaky split, patients ALSO in train", seen),
        ("  └ leaky split, patients NOT in train", ~seen),
    ]:
        if mask.sum() > 50 and len(np.unique(y_te[mask])) > 1:
            rows.append(
                {
                    "split_strategy": label,
                    "test_roc_auc": float(roc_auc_score(y_te[mask], prob[mask])),
                    "patients_in_both_splits": int(mask.sum() if mask is seen else 0),
                    "train_rows": len(tr_naive),
                    "test_rows": int(mask.sum()),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Imputer selection under simulated missingness
# ---------------------------------------------------------------------------
def imputer_ablation(
    X_train, y_train, X_val, y_val, schema, missing_rates=(0.0, 0.1, 0.2, 0.3),
    strategies=("median", "knn", "mice"), seed: int = C.SEED,
) -> pd.DataFrame:
    """Compare imputers under MCAR missingness injected into numeric columns.

    The training data has no numeric gaps, so this is a designed experiment
    rather than a repair job. It answers a deployment question: when a
    resource-limited clinic submits a record with unrecorded labs, which
    imputation strategy degrades most gracefully?
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SkPipeline

    from src.pipeline import build_preprocessor

    rng = np.random.default_rng(seed)
    num_cols = schema["numeric"]
    rows = []

    for rate in missing_rates:
        X_val_holed = X_val.copy()
        if rate > 0:
            mask = rng.random((len(X_val), len(num_cols))) < rate
            X_val_holed[num_cols] = X_val_holed[num_cols].mask(mask)

        for strategy in strategies:
            model = SkPipeline(
                [
                    ("pre", build_preprocessor(schema, numeric_imputer=strategy)),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000, class_weight="balanced", random_state=seed
                        ),
                    ),
                ]
            )
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_val_holed)[:, 1]
            rows.append(
                {
                    "missing_rate": rate,
                    "imputer": strategy,
                    "val_roc_auc": float(roc_auc_score(y_val, prob)),
                    "val_pr_auc": float(average_precision_score(y_val, prob)),
                }
            )

    return pd.DataFrame(rows)

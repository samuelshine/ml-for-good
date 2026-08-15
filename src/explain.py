"""SHAP, permutation importance and LIME.

Three methods rather than one, because agreement between independent
attribution methods is evidence and a single method's ranking is a claim.
Where they disagree, the notebook says so instead of quietly picking the
prettiest chart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


# ---------------------------------------------------------------------------
# Helpers to reach inside a fitted Pipeline
# ---------------------------------------------------------------------------
def split_pipeline(fitted_pipeline):
    """Return (preprocessor, classifier) from a fitted sklearn Pipeline."""
    return fitted_pipeline.named_steps["pre"], fitted_pipeline.named_steps["clf"]


def explainable_surrogate(fitted, prefer=("lgbm", "xgb", "rf")) -> tuple[object, str, str]:
    """Return a tree Pipeline that TreeExplainer can handle.

    A StackingClassifier is not itself explainable with TreeExplainer: it is a
    logistic meta-model sitting on four base learners, so there is no single
    tree structure to walk. Rather than fall back to a slow, approximate
    KernelExplainer, we explain the strongest tree base learner directly and
    state the substitution.

    This is defensible only because the base learners agree closely with the
    stack (Spearman ρ ≈ 0.85–0.97, see results/metrics/prediction_correlation.csv).
    `surrogate_fidelity` below measures that agreement rather than assuming it.

    Returns (pipeline, label, caveat).
    """
    from sklearn.ensemble import StackingClassifier, VotingClassifier

    if hasattr(fitted, "named_steps"):
        return fitted, "the deployed model", ""

    if isinstance(fitted, (StackingClassifier, VotingClassifier)):
        available = fitted.named_estimators_
        for key in prefer:
            if key in available:
                return (
                    available[key],
                    f"base learner '{key}' inside the ensemble",
                    "SHAP is computed on the strongest tree base learner, not on the "
                    "stack itself. The meta-learner is a logistic model over base "
                    "probabilities and has no tree structure to walk. Fidelity between "
                    "the surrogate and the deployed model is measured below.",
                )

    raise TypeError(f"no tree surrogate available for {type(fitted).__name__}")


def surrogate_fidelity(deployed, surrogate, X: pd.DataFrame) -> dict:
    """How closely does the explained surrogate track the deployed model?"""
    from scipy.stats import spearmanr

    p_dep = deployed.predict_proba(X)[:, 1]
    p_sur = surrogate.predict_proba(X)[:, 1]
    return {
        "spearman_rho": float(spearmanr(p_dep, p_sur).statistic),
        "pearson_r": float(np.corrcoef(p_dep, p_sur)[0, 1]),
        "mean_abs_diff": float(np.abs(p_dep - p_sur).mean()),
    }


def meta_learner_weights(stack) -> pd.DataFrame:
    """How much does the stack rely on each base learner?

    The meta-learner's coefficients are the honest answer to 'what did stacking
    actually do', and they are readable in a way the ensemble as a whole is not.
    """
    final = stack.final_estimator_
    names = list(stack.named_estimators_)
    coefs = np.ravel(final.coef_)
    return (
        pd.DataFrame({"base_learner": names[: len(coefs)], "meta_coefficient": coefs})
        .assign(relative_weight=lambda d: d["meta_coefficient"].abs()
                / d["meta_coefficient"].abs().sum())
        .sort_values("relative_weight", ascending=False)
        .reset_index(drop=True)
    )


def transform_frame(fitted_pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Preprocess X and return a named DataFrame SHAP can label properly."""
    pre, _ = split_pipeline(fitted_pipeline)
    arr = pre.transform(X)
    names = [str(n) for n in pre.get_feature_names_out()]
    return pd.DataFrame(arr, columns=names, index=X.index)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
def shap_values(
    fitted_pipeline, X: pd.DataFrame, max_rows: int = 4000, seed: int = C.SEED,
    must_include: list | None = None,
):
    """TreeExplainer SHAP values for the positive class.

    Sub-sampled because exact tree SHAP is O(rows); 4,000 rows is far more than
    enough for stable global rankings and keeps the notebook runnable.

    `must_include` pins specific index labels into the sample so the individual
    case studies are guaranteed to be explainable. Without it, the hand-picked
    true positive / false positive / false negative rows land in the sample only
    by luck.
    """
    import shap

    _, clf = split_pipeline(fitted_pipeline)
    Xt = transform_frame(fitted_pipeline, X)

    if len(Xt) > max_rows:
        pinned = [i for i in (must_include or []) if i in Xt.index]
        rest = Xt.drop(index=pinned)
        sampled = rest.sample(max(0, max_rows - len(pinned)), random_state=seed)
        # Keep the pinned rows first so their positions are stable and findable.
        Xt = pd.concat([Xt.loc[pinned], sampled]) if pinned else sampled

    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(Xt)

    # Binary classifiers return either (n, f) or (n, f, 2) depending on library.
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]

    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = np.asarray(base).ravel()[-1]

    return values, Xt, float(base), explainer


def global_importance(values: np.ndarray, Xt: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Mean |SHAP| per feature, with the direction of the association."""
    mean_abs = np.abs(values).mean(axis=0)

    # Correlation between a feature's value and its SHAP contribution gives the
    # direction: positive means higher values push risk up.
    direction = []
    for j in range(values.shape[1]):
        col = Xt.iloc[:, j].to_numpy()
        if np.std(col) < 1e-12:
            direction.append(0.0)
        else:
            direction.append(float(np.corrcoef(col, values[:, j])[0, 1]))

    out = pd.DataFrame(
        {
            "feature": Xt.columns,
            "mean_abs_shap": mean_abs,
            "direction": np.nan_to_num(direction),
        }
    )
    out["effect"] = np.where(
        out["direction"] > 0.05, "increases risk",
        np.where(out["direction"] < -0.05, "decreases risk", "mixed"),
    )
    return out.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)


def local_explanation(
    values: np.ndarray, Xt: pd.DataFrame, base_value: float, row_pos: int, top_n: int = 12
) -> pd.DataFrame:
    """The per-feature contributions behind one patient's prediction."""
    contrib = values[row_pos]
    out = pd.DataFrame(
        {
            "feature": Xt.columns,
            "value": Xt.iloc[row_pos].to_numpy(),
            "shap_contribution": contrib,
        }
    )
    out["abs"] = out["shap_contribution"].abs()
    out = out.sort_values("abs", ascending=False).head(top_n).drop(columns="abs")
    out.attrs["base_value"] = base_value
    out.attrs["prediction_logit"] = float(base_value + contrib.sum())
    return out.reset_index(drop=True)


def pick_case_studies(y_true, y_prob, threshold: float, seed: int = C.SEED) -> dict[str, int]:
    """Positional indices of a confident TP, FP, FN and TN for local plots."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)

    def pick(mask, by_high: bool):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return None
        order = np.argsort(y_prob[idx])
        return int(idx[order[-1]] if by_high else idx[order[0]])

    return {
        "true_positive": pick((y_true == 1) & (pred == 1), True),
        "false_positive": pick((y_true == 0) & (pred == 1), True),
        "false_negative": pick((y_true == 1) & (pred == 0), False),
        "true_negative": pick((y_true == 0) & (pred == 0), False),
    }


# ---------------------------------------------------------------------------
# Permutation importance (model-agnostic cross-check)
# ---------------------------------------------------------------------------
def permutation_scores(
    fitted_pipeline, X: pd.DataFrame, y, n_repeats: int = 5, max_rows: int = 6000,
    seed: int = C.SEED,
) -> pd.DataFrame:
    """Drop in ROC-AUC when each raw input column is shuffled.

    Computed on the *raw* columns rather than the encoded matrix, so a
    one-hot-expanded variable is judged as a whole.
    """
    from sklearn.inspection import permutation_importance

    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
        y = pd.Series(y).loc[X.index]

    result = permutation_importance(
        fitted_pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats,
        random_state=seed, n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "auc_drop": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("auc_drop", ascending=False)
        .reset_index(drop=True)
    )


def to_raw_column(encoded_name: str, raw_columns) -> str:
    """Map an encoded feature name back to the raw column it came from.

    With `verbose_feature_names_out=False`, numeric/binary/ordinal columns keep
    their own name while one-hot columns become `{column}_{category}`. Matching
    on the *longest* raw column that prefixes the encoded name avoids the trap
    of splitting on the first underscore, which would turn `number_inpatient`
    into `number`.
    """
    if encoded_name in raw_columns:
        return encoded_name
    candidates = [c for c in raw_columns if encoded_name.startswith(c + "_")]
    return max(candidates, key=len) if candidates else encoded_name


def rank_agreement(
    shap_tbl: pd.DataFrame, perm_tbl: pd.DataFrame, raw_columns, top_n: int = 15
) -> dict:
    """Do SHAP and permutation importance agree on what matters?

    SHAP is computed per encoded feature, permutation importance per raw column,
    so SHAP importances are summed back to the raw column before comparing.
    """
    from scipy.stats import spearmanr

    raw_columns = list(raw_columns)
    rolled = (
        shap_tbl.assign(raw=lambda d: d["feature"].map(
            lambda f: to_raw_column(f, raw_columns)))
        .groupby("raw", as_index=False)["mean_abs_shap"].sum()
        .sort_values("mean_abs_shap", ascending=False)
    )

    shap_top = list(rolled["raw"].head(top_n))
    perm_top = list(perm_tbl["feature"].head(top_n))
    overlap = sorted(set(shap_top) & set(perm_top))

    common = [f for f in perm_tbl["feature"] if f in set(rolled["raw"])]
    rho = np.nan
    if len(common) >= 5:
        s = rolled.set_index("raw").loc[common, "mean_abs_shap"]
        p = perm_tbl.set_index("feature").loc[common, "auc_drop"]
        rho = float(spearmanr(s, p).statistic)

    return {
        "top_n": top_n,
        "overlap_count": len(overlap),
        "overlap_pct": round(100 * len(overlap) / top_n, 1),
        "spearman_rho": rho,
        "shared_features": overlap,
        "shap_top": shap_top,
        "perm_top": perm_top,
    }


# ---------------------------------------------------------------------------
# LIME (independent local method)
# ---------------------------------------------------------------------------
def lime_explanation(fitted_pipeline, X_train: pd.DataFrame, X_row: pd.DataFrame, n_features: int = 10):
    """Local surrogate explanation for a single patient.

    LIME fits a local linear model around one point. Agreement with SHAP on the
    same patient is a genuine cross-method check, since the two make different
    assumptions.
    """
    from lime.lime_tabular import LimeTabularExplainer

    pre, clf = split_pipeline(fitted_pipeline)
    names = [str(n) for n in pre.get_feature_names_out()]

    train_enc = pre.transform(X_train)
    row_enc = pre.transform(X_row)[0]

    explainer = LimeTabularExplainer(
        train_enc, feature_names=names, class_names=["no readmit", "readmit <30d"],
        discretize_continuous=True, random_state=C.SEED,
    )
    exp = explainer.explain_instance(row_enc, clf.predict_proba, num_features=n_features)
    return pd.DataFrame(exp.as_list(), columns=["condition", "weight"])


# ---------------------------------------------------------------------------
# Domain translation
# ---------------------------------------------------------------------------
ACTIONABLE = {
    "a1c_tested", "a1c_severity_ord", "glucose_tested", "glucose_severity_ord",
    "num_med_changes", "num_meds_prescribed", "discharged_home", "time_in_hospital",
    "num_procedures", "num_medications", "insulin", "change", "diabetesMed",
    "medical_specialty", "weight_recorded",
}


def translate(shap_tbl: pd.DataFrame) -> pd.DataFrame:
    """Label each influential feature as actionable or descriptive.

    A risk model is only useful to a clinic if some of what it keys on can be
    changed. Age and race cannot be intervened on; HbA1c testing and discharge
    destination can.
    """
    from src.features import FEATURE_RATIONALE

    def classify(name: str) -> str:
        stem = name.split("_")[0]
        if name in ACTIONABLE or stem in {s.split("_")[0] for s in ACTIONABLE}:
            return "actionable"
        return "descriptive"

    out = shap_tbl.copy()
    out["lever"] = out["feature"].map(classify)
    out["clinical_meaning"] = out["feature"].map(
        lambda f: FEATURE_RATIONALE.get(f, ("", ""))[0] or "raw clinical field"
    )
    return out

"""Figures for the notebook, dashboard and deck.

Every function saves to results/figures/ and returns the Matplotlib figure, so
the notebook displays it and the deck reuses the exact same PNG. No figure is
generated twice from two different code paths.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config as C

P = C.PALETTE


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": C.FIG_DPI,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#333333",
            "grid.alpha": 0.3,
            "font.size": 10,
        }
    )


def save(fig, name: str):
    path = C.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=C.FIG_DPI, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Data audit
# ---------------------------------------------------------------------------
def plot_missingness(df: pd.DataFrame, name: str = "01_missingness"):
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]

    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(miss))))
    bars = ax.barh(miss.index[::-1], miss.values[::-1], color=P["accent"], alpha=0.85)
    for bar, val in zip(bars, miss.values[::-1]):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Missing (%)")
    ax.set_xlim(0, 105)
    ax.set_title("Missing values by column — four columns, four different policies")
    return save(fig, name)


def plot_missingness_vs_target(
    df: pd.DataFrame, cols: list[str], target: str = C.TARGET,
    name: str = "02_missingness_informative",
):
    """Is missingness related to the outcome? If yes, it is not MCAR."""
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        flag = df[col].isna()
        if flag.nunique() < 2:
            continue
        rows.append(
            {"column": col, "status": "recorded", "rate": df.loc[~flag, target].mean()}
        )
        rows.append(
            {"column": col, "status": "missing", "rate": df.loc[flag, target].mean()}
        )

    data = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=data, x="column", y="rate", hue="status", ax=ax,
        palette={"recorded": P["primary"], "missing": P["accent"]},
    )
    ax.axhline(df[target].mean(), ls="--", c="grey", lw=1,
               label=f"overall {df[target].mean():.3f}")
    ax.set_ylabel("30-day readmission rate")
    ax.set_xlabel("")
    ax.set_title("Missingness is informative, so it is encoded rather than imputed away")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return save(fig, name)


def plot_encounters_per_patient(raw: pd.DataFrame, name: str = "03_encounters_per_patient"):
    counts = raw["patient_nbr"].value_counts()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    a1.hist(counts.values, bins=np.arange(0.5, 21.5), color=P["primary"], alpha=0.85)
    a1.set_yscale("log")
    a1.set_xlabel("Encounters per patient")
    a1.set_ylabel("Patients (log)")
    a1.set_title(f"{(counts > 1).sum():,} patients appear more than once (max {counts.max()})")

    a2.bar(
        ["Encounters\n(raw rows)", "Unique\npatients"],
        [len(raw), raw["patient_nbr"].nunique()],
        color=[P["accent"], P["positive"]], alpha=0.85,
    )
    for i, v in enumerate([len(raw), raw["patient_nbr"].nunique()]):
        a2.text(i, v, f"{v:,}", ha="center", va="bottom", fontweight="bold")
    a2.set_title("Rows are not independent observations")
    a2.set_ylim(0, len(raw) * 1.15)
    fig.tight_layout()
    return save(fig, name)


def plot_class_balance(y_before: pd.Series, y_after: pd.Series, name: str = "04_class_balance"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, y, title in [
        (a1, y_before, "All encounters"),
        (a2, y_after, "One row per patient"),
    ]:
        rate = y.mean()
        ax.pie(
            [1 - rate, rate],
            labels=["Not readmitted <30d", "Readmitted <30d"],
            autopct="%1.2f%%", startangle=90,
            colors=[P["neutral"], P["accent"]],
            wedgeprops={"width": 0.45, "edgecolor": "white"},
        )
        ax.set_title(f"{title}\nn = {len(y):,}")
    fig.suptitle(
        "De-duplication lowers the positive rate: repeat encounters skew positive",
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, name)


def plot_outliers(df: pd.DataFrame, cols: list[str], name: str = "05_outliers"):
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 3.6))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        ax.boxplot(df[col].dropna(), vert=True, widths=0.5,
                   patch_artist=True,
                   boxprops={"facecolor": P["primary"], "alpha": 0.6},
                   medianprops={"color": P["accent"], "linewidth": 2})
        p995 = df[col].quantile(0.995)
        ax.axhline(p995, ls="--", c=P["accent"], lw=1)
        ax.set_title(f"{col}\np99.5 = {p995:.0f}", fontsize=9)
        ax.set_xticks([])
    fig.suptitle(
        "Heavy right tails are real high-utilising patients — winsorised, not deleted",
        fontweight="bold",
    )
    fig.tight_layout()
    return save(fig, name)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def plot_rate_by_category(
    df: pd.DataFrame, col: str, target: str = C.TARGET, top: int = 12,
    name: str | None = None, min_n: int = 100,
):
    grp = (
        df.groupby(col, observed=True)[target]
        .agg(["mean", "size"])
        .query("size >= @min_n")
        .sort_values("mean", ascending=False)
        .head(top)
    )
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.42 * len(grp))))
    bars = ax.barh(grp.index[::-1].astype(str), grp["mean"].values[::-1],
                   color=P["primary"], alpha=0.85)
    overall = df[target].mean()
    ax.axvline(overall, ls="--", c=P["accent"], lw=1.5, label=f"overall {overall:.3f}")
    for bar, (_, r) in zip(bars, grp[::-1].iterrows()):
        ax.text(r["mean"] + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{r['mean']:.3f}  (n={int(r['size']):,})", va="center", fontsize=8)
    ax.set_xlabel("30-day readmission rate")
    ax.set_title(f"Readmission rate by {col}")
    ax.legend(fontsize=8)
    ax.set_xlim(0, grp["mean"].max() * 1.35)
    return save(fig, name or f"eda_rate_by_{col}")


def plot_prior_inpatient_gradient(df: pd.DataFrame, name: str = "06_prior_inpatient"):
    d = df.copy()
    d["bucket"] = d["number_inpatient"].clip(upper=5)
    grp = d.groupby("bucket")[C.TARGET].agg(["mean", "size"])

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(grp.index.astype(int).astype(str), grp["mean"],
                  color=P["accent"], alpha=0.85)
    for bar, (_, r) in zip(bars, grp.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, r["mean"] + 0.004,
                f"{r['mean']:.1%}\nn={int(r['size']):,}",
                ha="center", fontsize=8)
    ax.axhline(df[C.TARGET].mean(), ls="--", c="grey", lw=1)
    ax.set_xlabel("Prior inpatient visits in the past year (5 = 5 or more)")
    ax.set_ylabel("30-day readmission rate")
    ax.set_ylim(0, grp["mean"].max() * 1.3)
    ax.set_title("The single strongest signal: prior admissions predict the next one")
    return save(fig, name)


def plot_a1c_replication(df: pd.DataFrame, name: str = "07_a1c_replication"):
    """Reproduce the source paper's headline result."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

    grp = df.groupby("a1c_tested")[C.TARGET].agg(["mean", "size"])
    bars = a1.bar(["Not measured", "Measured"], grp["mean"],
                  color=[P["neutral"], P["positive"]], alpha=0.85)
    for bar, (_, r) in zip(bars, grp.iterrows()):
        a1.text(bar.get_x() + bar.get_width() / 2, r["mean"] + 0.001,
                f"{r['mean']:.2%}\nn={int(r['size']):,}", ha="center", fontsize=9)
    a1.set_ylabel("30-day readmission rate")
    a1.set_title("HbA1c measurement vs readmission\n(replicates Strack et al., 2014)")
    a1.set_ylim(0, grp["mean"].max() * 1.35)

    sub = df[df["diag_1_group"].isin(
        df["diag_1_group"].value_counts().head(6).index
    )]
    piv = sub.groupby(["diag_1_group", "a1c_tested"])[C.TARGET].mean().unstack()
    piv.columns = ["Not measured", "Measured"]
    piv.plot(kind="barh", ax=a2, color=[P["neutral"], P["positive"]], alpha=0.85)
    a2.set_xlabel("30-day readmission rate")
    a2.set_ylabel("")
    a2.set_title("Stratified by primary diagnosis group")
    a2.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def plot_correlation(df: pd.DataFrame, cols: list[str], name: str = "08_correlation"):
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.4, cbar_kws={"shrink": 0.7}, ax=ax,
                annot=False)
    ax.set_title("Spearman correlation among numeric features")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    return save(fig, name)


def plot_mutual_information(mi: pd.Series, name: str = "09_mutual_information", top: int = 20):
    mi = mi.sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(mi))))
    ax.barh(mi.index[::-1], mi.values[::-1], color=P["primary"], alpha=0.85)
    ax.set_xlabel("Mutual information with the target")
    ax.set_title("Which raw signals carry information about readmission?")
    return save(fig, name)


def plot_clusters(
    coords: np.ndarray, labels: np.ndarray, rates: pd.Series,
    explained: tuple[float, float], name: str = "10_clusters",
):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))

    scatter = a1.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10",
                         s=4, alpha=0.45)
    a1.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    a1.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    a1.set_title("Patient phenotypes (KMeans on PCA projection)")
    a1.legend(*scatter.legend_elements(), title="cluster", fontsize=8, loc="best")

    bars = a2.bar(rates.index.astype(str), rates.values, color=P["accent"], alpha=0.85)
    for bar, v in zip(bars, rates.values):
        a2.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.1%}",
                ha="center", fontsize=9)
    a2.set_xlabel("Cluster")
    a2.set_ylabel("30-day readmission rate")
    a2.set_title("Readmission rate differs sharply by phenotype")
    a2.set_ylim(0, rates.max() * 1.3)
    fig.tight_layout()
    return save(fig, name)


# ---------------------------------------------------------------------------
# Model results
# ---------------------------------------------------------------------------
def plot_model_comparison(comp: pd.DataFrame, name: str = "11_model_comparison"):
    d = comp.sort_values("roc_auc")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    colours = [
        P["accent"] if "baseline" in m else
        P["positive"] if ("Stacking" in m or "Voting" in m) else P["primary"]
        for m in d["model"]
    ]
    bars = a1.barh(d["model"], d["roc_auc"], color=colours, alpha=0.88)
    for bar, v in zip(bars, d["roc_auc"]):
        a1.text(v + 0.001, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=9)
    a1.set_xlim(d["roc_auc"].min() - 0.015, d["roc_auc"].max() + 0.012)
    a1.set_xlabel("Test ROC-AUC")
    a1.set_title("Ensembles vs baselines on the untouched test set")

    bars2 = a2.barh(d["model"], d["pr_auc"], color=colours, alpha=0.88)
    for bar, v in zip(bars2, d["pr_auc"]):
        a2.text(v + 0.0006, bar.get_y() + bar.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=9)
    a2.set_xlim(d["pr_auc"].min() - 0.008, d["pr_auc"].max() + 0.008)
    a2.set_xlabel("Test PR-AUC (average precision)")
    a2.set_title("PR-AUC matters more at a 9% positive rate")
    a2.set_yticklabels([])
    fig.tight_layout()
    return save(fig, name)


def plot_roc_pr(y_true, probs: dict[str, np.ndarray], name: str = "12_roc_pr"):
    from sklearn.metrics import (
        average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
    )

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for label, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        a1.plot(fpr, tpr, lw=1.6, label=f"{label} ({roc_auc_score(y_true, p):.4f})")
        pre, rec, _ = precision_recall_curve(y_true, p)
        a2.plot(rec, pre, lw=1.6,
                label=f"{label} ({average_precision_score(y_true, p):.4f})")

    a1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
    a1.set_title("ROC curves"); a1.legend(fontsize=8, loc="lower right")

    base = float(np.mean(y_true))
    a2.axhline(base, ls="--", c="k", lw=1, alpha=0.5, label=f"random ({base:.3f})")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    a2.set_title("Precision-Recall curves"); a2.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, name)


def plot_cost_threshold(curve: pd.DataFrame, chosen: float, capacity: float,
                        name: str = "13_cost_threshold"):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(curve["threshold"], curve["expected_cost"], lw=2, color=P["primary"],
            label="Expected cost per patient")

    feasible = curve[curve["flagged_rate"] <= capacity]
    if not feasible.empty:
        ax.axvspan(feasible["threshold"].min(), curve["threshold"].max(),
                   color=P["positive"], alpha=0.08,
                   label=f"Staffable (flags ≤ {capacity:.0%})")
    ax.axvline(chosen, color=P["accent"], ls="--", lw=2,
               label=f"Chosen threshold {chosen:.3f}")
    ax.axvline(0.5, color="grey", ls=":", lw=1.5, label="Default 0.5")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost per patient (USD)")
    ax.set_title("Threshold is chosen by cost, then constrained by staffing capacity")
    ax.legend(fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(curve["threshold"], curve["recall"], color=P["warning"], lw=1.4, alpha=0.8)
    ax2.set_ylabel("Recall", color=P["warning"])
    ax2.grid(False)
    return save(fig, name)


def plot_confusion(cm_models: dict[str, dict], name: str = "14_confusion"):
    n = len(cm_models)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8))
    axes = np.atleast_1d(axes)
    for ax, (label, m) in zip(axes, cm_models.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred: no", "Pred: readmit"],
                    yticklabels=["Actual: no", "Actual: readmit"])
        ax.set_title(f"{label}\nrecall={m['recall']:.3f}  precision={m['precision']:.3f}",
                     fontsize=10)
    fig.suptitle("Confusion matrices at the deployed, cost-derived threshold",
                 fontweight="bold")
    fig.tight_layout()
    return save(fig, name)


def plot_calibration(y_true, probs: dict[str, np.ndarray], name: str = "15_calibration"):
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 0.5], [0, 0.5], "k--", lw=1, label="Perfect calibration")
    for label, p in probs.items():
        obs, pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(pred, obs, "o-", lw=1.6, ms=5, label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration: the threshold rule needs honest probabilities")
    ax.legend(fontsize=8)
    return save(fig, name)


def plot_prediction_correlation(corr: pd.DataFrame, name: str = "16_pred_correlation"):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlGn_r", vmin=0.5, vmax=1.0,
                square=True, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Base-learner agreement — stacking needs disagreement to exploit")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    return save(fig, name)


def plot_bootstrap(boot: dict, name: str = "17_bootstrap"):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.errorbar(
        boot["mean_difference"], 0,
        xerr=[[boot["mean_difference"] - boot["ci_low"]],
              [boot["ci_high"] - boot["mean_difference"]]],
        fmt="o", ms=11, capsize=8, lw=2.5, color=P["primary"],
    )
    ax.axvline(0, color=P["accent"], ls="--", lw=1.6, label="No difference")
    ax.set_yticks([])
    ax.set_xlabel("AUC difference (best ensemble − logistic baseline)")
    verdict = "significant" if boot["significant"] else "not significant"
    ax.set_title(
        f"{boot['mean_difference']:+.4f}  95% CI "
        f"[{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] — {verdict}"
    )
    ax.legend(fontsize=8)
    return save(fig, name)


# ---------------------------------------------------------------------------
# Explainability and fairness
# ---------------------------------------------------------------------------
def plot_shap_bar(tbl: pd.DataFrame, name: str = "18_shap_global", top: int = 18):
    d = tbl.head(top).iloc[::-1]
    colours = [
        P["accent"] if e == "increases risk" else
        P["positive"] if e == "decreases risk" else P["neutral"]
        for e in d["effect"]
    ]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.36 * len(d))))
    ax.barh(d["feature"], d["mean_abs_shap"], color=colours, alpha=0.88)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global feature importance (red raises risk, green lowers it)")
    return save(fig, name)


def plot_local_explanation(local: pd.DataFrame, prob: float, name: str = "19_shap_local"):
    d = local.iloc[::-1]
    colours = [P["accent"] if v > 0 else P["positive"] for v in d["shap_contribution"]]
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.4 * len(d))))
    ax.barh(d["feature"], d["shap_contribution"], color=colours, alpha=0.88)
    ax.axvline(0, color="black", lw=1)
    for y, (v, val) in enumerate(zip(d["shap_contribution"], d["value"])):
        ax.text(v + (0.008 if v > 0 else -0.008), y, f"{val:.4g}",
                va="center", ha="left" if v > 0 else "right", fontsize=8)
    ax.set_xlabel("SHAP contribution (log-odds)")
    ax.set_title(f"Why this patient scored {prob:.1%}")
    return save(fig, name)


def plot_fairness(per_group: pd.DataFrame, attribute: str, name: str | None = None):
    d = per_group[(per_group["attribute"] == attribute) & per_group["reliable"]].copy()
    if d.empty:
        return None
    d = d.sort_values("n", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in [
        (axes[0], "roc_auc", "ROC-AUC"),
        (axes[1], "tpr_recall", "Recall (true positive rate)"),
        (axes[2], "selection_rate", "Selection rate (flagged)"),
    ]:
        bars = ax.bar(d["group"].astype(str), d[col], color=P["primary"], alpha=0.85)
        for bar, v, n in zip(bars, d[col], d["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.3f}\nn={int(n):,}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, d[col].max() * 1.28)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)

    fig.suptitle(f"Subgroup performance by {attribute}", fontweight="bold")
    fig.tight_layout()
    return save(fig, name or f"20_fairness_{attribute}")

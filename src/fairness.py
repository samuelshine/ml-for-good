"""Subgroup performance audit and fairness metrics.

A model can hit a respectable overall AUC while being systematically worse for
one group. This module measures that directly rather than inferring it, and it
reports group sizes alongside every metric so a difference computed on 400
patients is not read as if it came from 40,000.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from src import config as C


def subgroup_metrics(
    y_true, y_prob, sensitive: pd.Series, threshold: float, min_size: int = 100
) -> pd.DataFrame:
    """Per-group confusion-derived metrics at the deployed threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)
    groups = pd.Series(sensitive).reset_index(drop=True)

    rows = []
    for name, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(idx)
        yt, yp, pr = y_true[idx], y_prob[idx], pred[idx]

        if len(idx) < min_size or len(np.unique(yt)) < 2:
            rows.append(
                {
                    "group": str(name), "n": len(idx),
                    "positive_rate": float(yt.mean()) if len(idx) else np.nan,
                    "reliable": False,
                }
            )
            continue

        tn, fp, fn, tp = confusion_matrix(yt, pr, labels=[0, 1]).ravel()
        rows.append(
            {
                "group": str(name),
                "n": int(len(idx)),
                "positive_rate": float(yt.mean()),
                "selection_rate": float(pr.mean()),
                "tpr_recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "fpr": float(fp / (fp + tn)) if (fp + tn) else np.nan,
                "ppv_precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                "roc_auc": float(roc_auc_score(yt, yp)),
                "reliable": True,
            }
        )

    out = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    return out


def fairness_gaps(table: pd.DataFrame) -> dict:
    """Disparity summaries across reliable groups only."""
    rel = table[table["reliable"]]
    if len(rel) < 2:
        return {}

    def gap(col):
        return float(rel[col].max() - rel[col].min())

    sel = rel["selection_rate"]
    ratio = float(sel.min() / sel.max()) if sel.max() > 0 else np.nan

    return {
        "demographic_parity_difference": gap("selection_rate"),
        "disparate_impact_ratio": ratio,
        "passes_four_fifths_rule": bool(ratio >= 0.8),
        "equal_opportunity_difference": gap("tpr_recall"),
        "equalised_odds_difference": max(gap("tpr_recall"), gap("fpr")),
        "auc_gap": gap("roc_auc"),
        "best_group": str(rel.loc[rel["roc_auc"].idxmax(), "group"]),
        "worst_group": str(rel.loc[rel["roc_auc"].idxmin(), "group"]),
    }


def full_audit(
    y_true, y_prob, sensitive_frame: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the audit across every sensitive attribute.

    Returns (per-group table, per-attribute summary).
    """
    tables, summaries = [], []
    for attr in sensitive_frame.columns:
        tbl = subgroup_metrics(y_true, y_prob, sensitive_frame[attr], threshold)
        tbl.insert(0, "attribute", attr)
        tables.append(tbl)

        gaps = fairness_gaps(tbl)
        if gaps:
            gaps["attribute"] = attr
            summaries.append(gaps)

    per_group = pd.concat(tables, ignore_index=True)
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary = summary[["attribute"] + [c for c in summary.columns if c != "attribute"]]
    return per_group, summary


def group_threshold_mitigation(
    y_true, y_prob, sensitive: pd.Series, target_tpr: float = 0.55
) -> pd.DataFrame:
    """Per-group thresholds that equalise recall, for comparison only.

    This demonstrates that the disparity is fixable by construction. It is not
    a recommendation: conditioning a clinical threshold explicitly on race
    raises legal and ethical objections that a metric improvement does not
    settle. See docs/ETHICS.md.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = pd.Series(sensitive).reset_index(drop=True)

    rows = []
    for name, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(idx)
        yt, yp = y_true[idx], y_prob[idx]
        if len(idx) < 100 or yt.sum() < 10:
            continue

        best_t, best_gap = 0.5, np.inf
        for t in np.linspace(0.02, 0.9, 177):
            tpr = ((yp >= t) & (yt == 1)).sum() / yt.sum()
            if abs(tpr - target_tpr) < best_gap:
                best_gap, best_t = abs(tpr - target_tpr), t

        pred = (yp >= best_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "group": str(name),
                "n": int(len(idx)),
                "group_threshold": round(float(best_t), 4),
                "achieved_tpr": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "selection_rate": float(pred.mean()),
                "ppv": float(tp / (tp + fp)) if (tp + fp) else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def calibration_by_group(
    y_true, y_prob, sensitive: pd.Series, n_bins: int = 5
) -> pd.DataFrame:
    """Are predicted risks equally trustworthy across groups?

    A model can rank well within every group yet be miscalibrated for one, which
    matters because the deployed decision compares a probability to a fixed
    cost-derived threshold.
    """
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(y_prob), "g": pd.Series(sensitive).values}
    )
    df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")

    out = (
        df.groupby(["g", "bin"], observed=True)
        .agg(n=("y", "size"), mean_pred=("p", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    out["calibration_error"] = (out["mean_pred"] - out["observed"]).abs()
    return out

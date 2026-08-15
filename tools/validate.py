"""Independent validation of the submission.

Cross-checks every number asserted in the README, docs, deck, dashboard and
video script against the artefacts in results/. Anything that cannot be traced
back to a generated file is reported as a failure, not a warning.

    .venv/bin/python tools/validate.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import config as C

PASS, FAIL, WARN = [], [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(f"{name}" + (f" — {detail}" if detail else ""))


def warn(name: str, detail: str = "") -> None:
    WARN.append(f"{name}" + (f" — {detail}" if detail else ""))


def close(a, b, tol=5e-4) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------
COMP = pd.read_csv(C.METRICS_DIR / "model_comparison.csv")
META = json.loads((C.METRICS_DIR / "run_metadata.json").read_text())
SIG = json.loads((C.METRICS_DIR / "significance.json").read_text())
FAIR = pd.read_csv(C.METRICS_DIR / "fairness_summary.csv")
PG = pd.read_csv(C.METRICS_DIR / "fairness_per_group.csv")
LEAK = pd.read_csv(C.METRICS_DIR / "leakage_sensitivity.csv")
AUDIT = pd.read_csv(C.METRICS_DIR / "cleaning_audit.csv")
SPLITS = pd.read_csv(C.METRICS_DIR / "split_summary.csv")
ABL = pd.read_csv(C.METRICS_DIR / "imputer_ablation.csv")
CHECKS = pd.read_csv(C.METRICS_DIR / "leakage_checklist.csv")

BEST = META["best_model"]
best = COMP[COMP["model"] == BEST].iloc[0]
base = COMP[COMP["model"] == "Logistic Regression (baseline)"].iloc[0]
BOOT = SIG["bootstrap_auc"]

README = (ROOT / "README.md").read_text()
MODELCARD = (ROOT / "docs" / "MODEL_CARD.md").read_text()
DATACARD = (ROOT / "docs" / "DATASET_CARD.md").read_text()
ETHICS = (ROOT / "docs" / "ETHICS.md").read_text()
SCRIPT = (ROOT / "docs" / "video_script.md").read_text()
DASH = (ROOT / "dashboard" / "results_dashboard.html").read_text()

print("=" * 78)
print("SECTION 1 — Do the documents match the artefacts?")
print("=" * 78)

# --- headline metrics quoted in README + model card -----------------------
check("README quotes best ROC-AUC correctly",
      f"{best['roc_auc']:.4f}" in README, f"expected {best['roc_auc']:.4f}")
check("README quotes baseline ROC-AUC correctly",
      f"{base['roc_auc']:.4f}" in README, f"expected {base['roc_auc']:.4f}")
check("Model card quotes best ROC-AUC correctly",
      f"{best['roc_auc']:.4f}" in MODELCARD)
check("Dashboard quotes best ROC-AUC correctly",
      f"{best['roc_auc']:.4f}" in DASH)

# --- bootstrap CI ---------------------------------------------------------
ci = f"[{BOOT['ci_low']:+.4f}, {BOOT['ci_high']:+.4f}]"
check("README quotes the bootstrap CI correctly", ci in README, f"expected {ci}")
check("Model card quotes the bootstrap CI correctly", ci in MODELCARD)
check("Bootstrap CI genuinely excludes zero",
      BOOT["ci_low"] > 0, f"ci_low={BOOT['ci_low']:+.5f}")
check("Bootstrap 'significant' flag agrees with the CI",
      BOOT["significant"] == (BOOT["ci_low"] > 0))
check("Claimed AUC gain equals best minus baseline",
      close(BOOT["auc_a"] - BOOT["auc_b"], best["roc_auc"] - base["roc_auc"], 1e-6))

# --- lift -----------------------------------------------------------------
check("README lift figure matches artefact",
      f"{best['lift_at_10pct']:.2f}" in README, f"expected {best['lift_at_10pct']:.2f}")
check("Lift equals precision@10% / base rate",
      close(best["lift_at_10pct"],
            best["precision_at_10pct"] / SPLITS.loc[SPLITS.split == "test", "positive_rate"].iloc[0],
            2e-3))

# --- fairness numbers quoted in prose ------------------------------------
race = PG[(PG["attribute"] == "race") & PG["reliable"]]
cauc = race[race["group"] == "Caucasian"].iloc[0]
afam = race[race["group"] == "AfricanAmerican"].iloc[0]

check("Quoted African American readmission rate (9.23%) is correct",
      close(afam["positive_rate"], 0.0923, 5e-5), f"actual {afam['positive_rate']:.4f}")
check("Quoted Caucasian readmission rate (9.07%) is correct",
      close(cauc["positive_rate"], 0.0907, 5e-5), f"actual {cauc['positive_rate']:.4f}")
check("Quoted African American selection rate (8.0%) is correct",
      close(afam["selection_rate"], 0.080, 1e-3), f"actual {afam['selection_rate']:.4f}")
check("Quoted Caucasian selection rate (10.2%) is correct",
      close(cauc["selection_rate"], 0.102, 1e-3), f"actual {cauc['selection_rate']:.4f}")
check("Quoted African American recall (19.9%) is correct",
      close(afam["tpr_recall"], 0.199, 1e-3), f"actual {afam['tpr_recall']:.4f}")
check("Quoted Caucasian recall (25.0%) is correct",
      close(cauc["tpr_recall"], 0.250, 1e-3), f"actual {cauc['tpr_recall']:.4f}")

n_failed = int((~FAIR["passes_four_fifths_rule"]).sum())
check("Claim '3 of 4 attributes fail the four-fifths rule' is true",
      n_failed == 3, f"actual {n_failed}")
check("Gender is the attribute that passes",
      FAIR.loc[FAIR["passes_four_fifths_rule"], "attribute"].tolist() == ["gender"])
check("Every disparate impact ratio marked FAIL is genuinely < 0.8",
      bool((FAIR.loc[~FAIR["passes_four_fifths_rule"], "disparate_impact_ratio"] < 0.8).all()))
check("The attribute marked pass is genuinely >= 0.8",
      bool((FAIR.loc[FAIR["passes_four_fifths_rule"], "disparate_impact_ratio"] >= 0.8).all()))

# --- leakage experiment numbers ------------------------------------------
leaky = LEAK[LEAK["split_strategy"].str.contains("Random row")].iloc[0]
honest = LEAK[LEAK["split_strategy"].str.contains("grouped")].iloc[0]
seen = LEAK[LEAK["split_strategy"].str.contains("ALSO in train")]
unseen = LEAK[LEAK["split_strategy"].str.contains("NOT in train")]

check("Leakage experiment produced all four rows", len(LEAK) == 4, f"got {len(LEAK)}")
check("Quoted leaky AUC (0.6699) is correct",
      close(leaky["test_roc_auc"], 0.6699, 5e-5), f"actual {leaky['test_roc_auc']:.4f}")
check("Quoted honest AUC (0.6759) is correct",
      close(honest["test_roc_auc"], 0.6759, 5e-5), f"actual {honest['test_roc_auc']:.4f}")
check("Claim 'leaky scored LOWER than honest' holds",
      leaky["test_roc_auc"] < honest["test_roc_auc"])
if len(seen) and len(unseen):
    check("Quoted seen-patient AUC (0.566) is correct",
          close(seen.iloc[0]["test_roc_auc"], 0.566, 1e-3),
          f"actual {seen.iloc[0]['test_roc_auc']:.4f}")
    check("Quoted unseen-patient AUC (0.687) is correct",
          close(unseen.iloc[0]["test_roc_auc"], 0.687, 1e-3),
          f"actual {unseen.iloc[0]['test_roc_auc']:.4f}")
    check("Claim 'seen patients are HARDER than unseen' holds",
          seen.iloc[0]["test_roc_auc"] < unseen.iloc[0]["test_roc_auc"])
check("Leaky split genuinely had patient overlap",
      leaky["patients_in_both_splits"] > 0, f"{int(leaky['patients_in_both_splits'])} patients")
check("Grouped split genuinely had zero overlap",
      honest["patients_in_both_splits"] == 0)

# --- cleaning audit numbers ----------------------------------------------
raw_rows = int(AUDIT.iloc[0]["rows_before"])
final_rows = int(AUDIT.iloc[-1]["rows_after"])
check("Raw row count is 101,766", raw_rows == 101_766, f"actual {raw_rows:,}")
check("Final cohort is 69,987", final_rows == 69_987, f"actual {final_rows:,}")
check("README quotes the raw row count", "101,766" in README)
check("README quotes the patient count", "71,518" in README)

expired = AUDIT[AUDIT["step"] == "drop_expired_hospice"].iloc[0]
check("Expired/hospice exclusion dropped 2,423 rows",
      int(expired["rows_dropped"]) == 2423, f"actual {int(expired['rows_dropped'])}")

# --- positive rate claim --------------------------------------------------
test_rate = float(SPLITS.loc[SPLITS.split == "test", "positive_rate"].iloc[0])
check("Post-cleaning positive rate is 8.98%", close(test_rate, 0.0898, 5e-5),
      f"actual {test_rate:.4f}")
check("README states the 11.16% -> 8.98% drop",
      "11.16" in README and "8.98" in README)

# --- imputer ablation claims ---------------------------------------------
piv = ABL.pivot(index="missing_rate", columns="imputer", values="val_roc_auc")
if 0.3 in piv.index:
    r = piv.loc[0.3]
    check("At 30% missingness KNN beats median beats MICE",
          r["knn"] > r["median"] > r["mice"],
          f"knn={r['knn']:.4f} median={r['median']:.4f} mice={r['mice']:.4f}")
    check("Quoted KNN value (0.6295) is correct", close(r["knn"], 0.6295, 5e-5))
    check("Quoted MICE value (0.5992) is correct", close(r["mice"], 0.5992, 5e-5))
if 0.0 in piv.index:
    r0 = piv.loc[0.0]
    check("At 0% missingness all imputers are identical (nothing to impute)",
          close(r0["knn"], r0["median"], 1e-9) and close(r0["median"], r0["mice"], 1e-9))

# --- threshold ------------------------------------------------------------
check("Deployed threshold in model card matches artefact",
      f"{META['threshold']:.3f}" in MODELCARD, f"expected {META['threshold']:.3f}")
check("Threshold is not the 0.5 default", not close(META["threshold"], 0.5))
check("Model comparison rows all use the recorded threshold for the best model",
      close(best["threshold"], META["threshold"], 1e-9))

# --- confusion matrix internal consistency -------------------------------
tp, fp, fn, tn = int(best["tp"]), int(best["fp"]), int(best["fn"]), int(best["tn"])
n_test = int(SPLITS.loc[SPLITS.split == "test", "rows"].iloc[0])
check("Confusion matrix sums to the test set size", tp + fp + fn + tn == n_test,
      f"{tp + fp + fn + tn} vs {n_test}")
check("Reported recall matches the confusion matrix",
      close(best["recall"], tp / (tp + fn)))
check("Reported precision matches the confusion matrix",
      close(best["precision"], tp / (tp + fp)))
check("Reported specificity matches the confusion matrix",
      close(best["specificity"], tn / (tn + fp)))
check("Reported F1 matches precision and recall",
      close(best["f1"], 2 * best["precision"] * best["recall"]
            / (best["precision"] + best["recall"])))
check("Positives in confusion matrix match the split summary",
      tp + fn == int(SPLITS.loc[SPLITS.split == "test", "positives"].iloc[0]))

# --- model card table -----------------------------------------------------
for _, r in COMP.iterrows():
    if f"{r['roc_auc']:.4f}" not in MODELCARD:
        FAIL.append(f"Model card missing ROC-AUC for {r['model']} ({r['roc_auc']:.4f})")
check("Model card lists every model's ROC-AUC",
      all(f"{r['roc_auc']:.4f}" in MODELCARD for _, r in COMP.iterrows()))

# --- video script ---------------------------------------------------------
check("Video script quotes the 83.1% demo risk", "83" in SCRIPT)
check("Video script covers all four mandated time blocks",
      all(t in SCRIPT for t in ["0:00", "0:35", "1:15", "2:10", "3:00"]))

print(f"  {len(PASS)} passed, {len(FAIL)} failed so far\n")

# ---------------------------------------------------------------------------
print("=" * 78)
print("SECTION 2 — Is the model genuinely leakage-safe?")
print("=" * 78)

import warnings

warnings.filterwarnings("ignore")

import joblib

from src.cleaning import clean
from src.features import column_schema, engineer
from src.pipeline import make_splits

cleaned, _ = clean()
feat = engineer(cleaned)
splits = make_splits(feat)
X_train, y_train = splits["train"]
X_val, y_val = splits["val"]
X_test, y_test = splits["test"]

check("Splits reconstruct to the recorded sizes",
      [len(X_train), len(X_val), len(X_test)] == SPLITS["rows"].tolist(),
      f"{[len(X_train), len(X_val), len(X_test)]} vs {SPLITS['rows'].tolist()}")

# The real test: no patient in two splits, verified on raw identifiers.
pt = feat["patient_nbr"]
g_tr, g_va, g_te = set(pt.loc[X_train.index]), set(pt.loc[X_val.index]), set(pt.loc[X_test.index])
check("Zero patient overlap train/test", len(g_tr & g_te) == 0, f"{len(g_tr & g_te)} shared")
check("Zero patient overlap train/val", len(g_tr & g_va) == 0)
check("Zero patient overlap val/test", len(g_va & g_te) == 0)
check("Every patient appears exactly once in the cohort", feat["patient_nbr"].is_unique)
check("Split index sets are disjoint",
      len(set(X_train.index) & set(X_test.index)) == 0)
check("Splits together cover the whole cohort",
      len(g_tr | g_va | g_te) == len(feat))

# Target must not be reachable from any feature.
schema = column_schema(feat)
used = set(schema["numeric"] + schema["binary"] + schema["ordinal"] + schema["nominal"])
check("Target column excluded from features", C.TARGET not in used)
check("Raw target column excluded from features", C.RAW_TARGET not in used)
check("patient_nbr excluded from features", "patient_nbr" not in used)
check("encounter_id excluded from features", "encounter_id" not in used)
check("Exploratory 'cluster' column excluded from features", "cluster" not in used)

# A feature perfectly correlated with the target would be a leak.
num_feats = schema["numeric"] + schema["binary"] + schema["ordinal"]
corrs = feat[num_feats].corrwith(feat[C.TARGET]).abs().sort_values(ascending=False)
check("No numeric feature is near-perfectly correlated with the target",
      float(corrs.max()) < 0.5, f"max |r| = {corrs.max():.4f} ({corrs.index[0]})")

# The saved model must not have been fitted on the test rows.
model = joblib.load(C.MODELS_DIR / "best_pipeline.joblib")
check("Saved model's feature list excludes identifiers and target",
      not ({C.TARGET, C.RAW_TARGET, "patient_nbr", "encounter_id"}
           & set(model.feature_names_in_)))

print(f"  running total: {len(PASS)} passed, {len(FAIL)} failed\n")

# ---------------------------------------------------------------------------
print("=" * 78)
print("SECTION 3 — Does the saved model actually reproduce the reported metrics?")
print("=" * 78)

from src import evaluate as E

prob = model.predict_proba(X_test[list(model.feature_names_in_)])[:, 1]
recomputed = E.classification_metrics(y_test, prob, META["threshold"], BEST)

for key in ["roc_auc", "pr_auc", "f1", "precision", "recall", "specificity",
            "precision_at_10pct", "lift_at_10pct"]:
    check(f"Recomputed {key} matches the reported value",
          close(recomputed[key], best[key], 1e-6),
          f"recomputed {recomputed[key]:.6f} vs reported {best[key]:.6f}")

for key in ["tp", "fp", "fn", "tn"]:
    check(f"Recomputed {key} matches the reported value",
          int(recomputed[key]) == int(best[key]),
          f"{recomputed[key]} vs {int(best[key])}")

# Independent AUC, not routed through our own helper.
from sklearn.metrics import roc_auc_score

check("Independent sklearn ROC-AUC matches the reported value",
      close(roc_auc_score(y_test, prob), best["roc_auc"], 1e-9))

# Sanity: the model must beat chance and must not be suspiciously perfect.
check("Model beats random (AUC > 0.55)", best["roc_auc"] > 0.55)
check("Model is NOT suspiciously perfect (AUC < 0.85)", best["roc_auc"] < 0.85,
      "an AUC above this on this dataset would indicate a leak")

print(f"  running total: {len(PASS)} passed, {len(FAIL)} failed\n")

# ---------------------------------------------------------------------------
print("=" * 78)
print("SECTION 4 — Notebook integrity")
print("=" * 78)

import nbformat

nb = nbformat.read(ROOT / "notebooks" / "01_ml_for_social_good_readmission.ipynb",
                   as_version=4)
code_cells = [c for c in nb.cells if c.cell_type == "code"]
errors = [o for c in code_cells for o in c.outputs if o.get("output_type") == "error"]
empty = [c for c in code_cells if not c.outputs]
counts = [c.execution_count for c in code_cells if c.execution_count is not None]

check("Notebook has zero error outputs", len(errors) == 0, f"{len(errors)} errors")
check("Every code cell produced output", len(empty) == 0, f"{len(empty)} empty")
check("Execution counts are strictly sequential (restart-and-run-all)",
      counts == sorted(counts) and counts == list(range(1, len(counts) + 1)),
      f"first={counts[:3]} last={counts[-3:]}" if counts else "none")
check("Notebook pins the project kernel",
      nb.metadata.get("kernelspec", {}).get("name") == "ml-for-good")
check("Notebook contains all 10 numbered sections",
      all(f"## {i} ·" in "".join(c.source for c in nb.cells if c.cell_type == "markdown")
          for i in range(1, 11)))

md_text = "".join(c.source for c in nb.cells if c.cell_type == "markdown")
check("Notebook uses the Observations/Inference structure",
      md_text.count("**Observations**") >= 10 and md_text.count("**Inference") >= 6,
      f"{md_text.count('**Observations**')} observations, "
      f"{md_text.count('**Inference')} inferences")

# The notebook's own reported metrics must match the artefacts.
nb_text = "".join(
    o.get("text", "") for c in code_cells for o in c.outputs if "text" in o
)
check("Notebook output contains the reported best AUC",
      f"{best['roc_auc']:.4f}" in nb_text)

print(f"  running total: {len(PASS)} passed, {len(FAIL)} failed\n")

# ---------------------------------------------------------------------------
print("=" * 78)
print("SECTION 5 — Rubric coverage and provenance")
print("=" * 78)

required_files = [
    "README.md", "requirements.txt", "Makefile",
    "notebooks/01_ml_for_social_good_readmission.ipynb",
    "docs/ETHICS.md", "docs/MODEL_CARD.md", "docs/DATASET_CARD.md",
    "docs/video_script.md",
    "presentation/CIA3_deck.pptx", "presentation/deck.html",
    "dashboard/results_dashboard.html",
    "app/streamlit_app.py", "app/synthetic.py",
    "src/train.py", "models/best_pipeline.joblib", "models/threshold.json",
]
for f in required_files:
    check(f"Deliverable present: {f}", (ROOT / f).exists())

# Rubric-mandated model families must all be present in the comparison.
families = {
    "baseline (logistic)": "Logistic Regression (baseline)",
    "baseline (tree)": "Decision Tree (baseline)",
    "bagging": "Random Forest (bagging)",
    "boosting": "XGBoost (boosting)",
    "stacking": "Stacking (heterogeneous)",
    "voting": "Soft Voting (heterogeneous)",
}
for label, model_name in families.items():
    check(f"Rubric requires {label}: present in comparison",
          model_name in COMP["model"].values)

# Citation accuracy.
check("Citation names the correct first author", "Strack" in C.CITATION)
check("Citation gives the correct year", "2014" in C.CITATION)
check("Citation gives the correct journal",
      "BioMed Research International" in C.CITATION)
check("Citation gives the correct DOI", "10.1155/2014/781670" in C.CITATION)
check("README carries the citation", "Strack" in README and "781670" in README)
check("Dataset card carries the licence", "CC BY 4.0" in DATACARD)
check("Ethics doc covers privacy", "HIPAA" in ETHICS)
check("Ethics doc covers FP/FN costs",
      "false negative" in ETHICS.lower() and "false positive" in ETHICS.lower())
check("Ethics doc covers human oversight", "override" in ETHICS.lower())
check("Ethics doc lists prohibited uses", "insurance pricing" in ETHICS.lower())

# Every leakage checklist item must pass.
check("All automated leakage checks pass",
      bool(CHECKS["passed"].all()),
      f"{int((~CHECKS['passed']).sum())} failing")

# Figures must not be stale relative to the metrics they illustrate.
newest_metric = max(p.stat().st_mtime for p in C.METRICS_DIR.glob("*"))
stale = [p.name for p in C.FIGURES_DIR.glob("*.png")
         if p.stat().st_mtime < newest_metric - 3600]
if stale:
    warn("Some figures are older than the newest metric file",
         f"{len(stale)} figures, e.g. {stale[:3]}")

# No real patient data should be shipped.
check("Raw data is gitignored",
      "data/raw/" in (ROOT / ".gitignore").read_text())
check("Demo records are synthetic",
      "synthetic" in (ROOT / "app" / "synthetic.py").read_text().lower())

# ---------------------------------------------------------------------------
print()
print("=" * 78)
print("RESULT")
print("=" * 78)
print(f"  passed : {len(PASS)}")
print(f"  failed : {len(FAIL)}")
print(f"  warnings: {len(WARN)}")

if FAIL:
    print("\nFAILURES:")
    for f in FAIL:
        print(f"  ✗ {f}")
if WARN:
    print("\nWARNINGS:")
    for w in WARN:
        print(f"  ! {w}")

if not FAIL:
    print("\n  All checks passed.")

sys.exit(1 if FAIL else 0)

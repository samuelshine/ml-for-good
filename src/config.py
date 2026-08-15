"""Central configuration: paths, seeds, column schema, cost model.

Everything that another module might want to agree on lives here, so there is
exactly one definition of "what is a count column" or "what does a false
negative cost".
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
EXPLAIN_DIR = RESULTS_DIR / "explainability"

DOCS_DIR = ROOT / "docs"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR, EXPLAIN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset provenance
# --------------------------------------------------------------------------
DATASET_NAME = "Diabetes 130-US Hospitals for Years 1999-2008"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
DATASET_UCI_PAGE = "https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008"
DATASET_LICENCE = "CC BY 4.0"

CITATION = (
    "Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., "
    "Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on "
    "hospital readmission rates: Analysis of 70,000 clinical database patient "
    "records. BioMed Research International, 2014, 781670. "
    "https://doi.org/10.1155/2014/781670"
)

RAW_CSV = RAW_DIR / "diabetic_data.csv"
IDS_CSV = RAW_DIR / "IDS_mapping.csv"

EXPECTED_RAW_SHAPE = (101766, 50)

# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------
RAW_TARGET = "readmitted"          # {'NO', '>30', '<30'}
TARGET = "readmit_lt30"            # 1 if readmitted within 30 days

# --------------------------------------------------------------------------
# Identifiers (never used as features)
# --------------------------------------------------------------------------
ID_COLS = ["encounter_id", "patient_nbr"]

# --------------------------------------------------------------------------
# Domain exclusions (Strack et al., 2014)
# --------------------------------------------------------------------------
# Discharge dispositions meaning the patient died or entered hospice. These
# encounters cannot produce a readmission, so keeping them injects label noise
# the model can never learn.
EXPIRED_OR_HOSPICE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]

# Sentinel used throughout the raw CSV for "not recorded".
MISSING_SENTINEL = "?"

# --------------------------------------------------------------------------
# The 23 medication columns
# --------------------------------------------------------------------------
MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# --------------------------------------------------------------------------
# Ordinal encodings with an explicit, defensible order
# --------------------------------------------------------------------------
AGE_ORDER = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
AGE_MIDPOINTS = {band: 5 + 10 * i for i, band in enumerate(AGE_ORDER)}

# 'None' means the test was not ordered, which is a clinical decision rather
# than a missing value. It is encoded as 0 and flagged separately.
A1C_ORDER = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
GLU_ORDER = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}

# Medication change direction, ordered from de-escalation to escalation.
MED_CHANGE_ORDER = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}

# --------------------------------------------------------------------------
# Protected / sensitive attributes used in the fairness audit
# --------------------------------------------------------------------------
SENSITIVE_COLS = ["race", "gender", "age", "payer_group"]

# --------------------------------------------------------------------------
# Split strategy
# --------------------------------------------------------------------------
TEST_SIZE = 0.20
VAL_SIZE = 0.20          # of the full dataset, taken from the non-test portion
CV_FOLDS = 5

# --------------------------------------------------------------------------
# Cost model for threshold selection (Q4: false-positive / false-negative cost)
# --------------------------------------------------------------------------
# A missed early readmission costs the health system an avoidable inpatient
# stay. A false alarm costs one nurse-led follow-up call. Sources are cited in
# docs/ETHICS.md; the ratio, not the absolute value, drives the threshold.
COST_FN = 15_000.0       # USD, avoidable 30-day readmission
COST_FP = 75.0           # USD, one nurse follow-up call + patient time
COST_TP = 75.0           # intervening on a true positive still costs the call
COST_TN = 0.0

# Follow-up capacity: share of discharges a resource-limited clinic can call.
FOLLOWUP_CAPACITY = 0.10

# Predictions inside this band are escalated to a human rather than auto-acted.
ABSTENTION_BAND = (0.40, 0.60)

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
FIG_DPI = 150
PALETTE = {
    "primary": "#2E5EAA",
    "accent": "#D1495B",
    "neutral": "#8D99AE",
    "positive": "#00916E",
    "warning": "#EDAE49",
}

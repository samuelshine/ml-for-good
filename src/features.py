"""Domain-informed feature engineering.

Each engineered feature is registered in FEATURE_RATIONALE with a clinical
reason and an expected direction of effect. Stating the expected sign *before*
looking at SHAP is what makes the explainability section a test rather than a
story: features that come out with the wrong sign get investigated, not
narrated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

# ---------------------------------------------------------------------------
# ICD-9 -> clinical chapter, following the grouping used by Strack et al. (2014)
# ---------------------------------------------------------------------------
_ICD9_RANGES: list[tuple[float, float, str]] = [
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (630, 679, "Pregnancy"),
    (680, 709, "Skin"),
    (710, 739, "Musculoskeletal"),
    (140, 239, "Neoplasms"),
    (240, 279, "Endocrine/Metabolic"),
    (280, 289, "Blood"),
    (290, 319, "Mental"),
    (320, 389, "Nervous/Sense"),
    (800, 999, "Injury"),
    (1, 139, "Infectious"),
]

# Codes that sit inside a range above but deserve their own group because the
# literature treats them separately.
_ICD9_EXACT = {785: "Circulatory", 786: "Respiratory", 787: "Digestive", 788: "Genitourinary"}

DIAG_GROUPS = sorted(
    {g for _, _, g in _ICD9_RANGES} | {"Diabetes", "Other", "Missing"}
)


def map_icd9(code) -> str:
    """Map a single ICD-9 code string to a clinical chapter.

    Diabetes (250.xx) is split out from Endocrine because it is the cohort's
    defining condition: whether diabetes is the *primary* reason for admission
    is clinically different from it being a background comorbidity.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)) or pd.isna(code):
        return "Missing"

    text = str(code).strip()
    if not text:
        return "Missing"

    # V-codes (supplementary) and E-codes (external cause) have no numeric chapter.
    if text[0].upper() in {"V", "E"}:
        return "Other"

    try:
        value = float(text)
    except ValueError:
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"

    if int(value) in _ICD9_EXACT:
        return _ICD9_EXACT[int(value)]

    for low, high, label in _ICD9_RANGES:
        if low <= value <= high:
            return label

    return "Other"


# ---------------------------------------------------------------------------
# Feature rationale registry (rendered as a table in the notebook)
# ---------------------------------------------------------------------------
FEATURE_RATIONALE: dict[str, tuple[str, str]] = {
    "total_prior_visits": (
        "Sum of outpatient, emergency and inpatient visits in the prior year.",
        "up",
    ),
    "prior_inpatient_flag": ("Any inpatient stay in the prior year.", "up"),
    "high_utilizer": ("Two or more prior inpatient stays: chronic instability.", "up"),
    "prior_emergency_ratio": (
        "Share of prior contacts that were emergencies; unplanned care signals "
        "poor disease control and weak primary-care access.",
        "up",
    ),
    "service_intensity": (
        "Labs + procedures + medications: a proxy for how sick this stay was.",
        "up",
    ),
    "meds_per_day": ("Medication count normalised by length of stay.", "up"),
    "labs_per_day": ("Lab count normalised by length of stay.", "mixed"),
    "procedures_per_day": ("Procedure count normalised by length of stay.", "mixed"),
    "num_meds_prescribed": ("How many of the 20 tracked drugs are active.", "up"),
    "num_med_changes": (
        "Count of drugs titrated up or down this stay. Regimen churn means the "
        "clinician had not yet found a stable dose at discharge.",
        "up",
    ),
    "a1c_tested": (
        "Whether HbA1c was measured. Strack et al. found measurement itself is "
        "associated with lower readmission, as it triggers regimen review.",
        "down",
    ),
    "a1c_severity_ord": ("Ordinal HbA1c result: None < Norm < >7 < >8.", "up"),
    "glucose_severity_ord": ("Ordinal serum glucose: None < Norm < >200 < >300.", "up"),
    "n_distinct_diag_groups": ("Distinct clinical chapters across diag_1..3.", "up"),
    "diabetes_is_primary": ("Diabetes is the principal diagnosis.", "up"),
    "has_circulatory": ("Any circulatory diagnosis: the top comorbidity driver.", "up"),
    "comorbidity_burden": ("Number of diagnoses coded on the record.", "up"),
    "age_midpoint": ("Numeric midpoint of the 10-year age band.", "up"),
    "emergency_admission": ("Admitted through Emergency rather than electively.", "up"),
    "discharged_home": ("Discharged home without home-health support.", "down"),
    "long_stay": ("Stay longer than one week.", "up"),
    "weight_recorded": ("Weight was documented: a proxy for care thoroughness.", "down"),
    "payer_unknown": (
        "No payer code recorded, which skews toward self-pay and uninsured "
        "patients. Kept explicitly so the fairness audit can see it.",
        "up",
    ),
    "specialty_unknown": ("Admitting specialty not recorded.", "mixed"),
}


def rationale_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"feature": k, "clinical_rationale": v[0], "expected_effect": v[1]}
            for k, v in FEATURE_RATIONALE.items()
        ]
    )


# ---------------------------------------------------------------------------
# Engineering
# ---------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add all domain features. Row-wise only, so it is split-safe.

    Nothing here touches the target or uses a statistic computed across rows,
    which is why this can run before the train/test split without leaking.
    """
    df = df.copy()
    active_meds = [c for c in C.MEDICATION_COLS if c in df.columns]

    # --- prior utilisation -------------------------------------------------
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    df["prior_inpatient_flag"] = (df["number_inpatient"] > 0).astype(int)
    df["high_utilizer"] = (df["number_inpatient"] >= 2).astype(int)
    df["prior_emergency_ratio"] = (
        df["number_emergency"] / df["total_prior_visits"].replace(0, np.nan)
    ).fillna(0.0)

    # --- intensity of this stay -------------------------------------------
    los = df["time_in_hospital"].clip(lower=1)
    df["service_intensity"] = (
        df["num_lab_procedures"] + df["num_procedures"] + df["num_medications"]
    )
    df["meds_per_day"] = df["num_medications"] / los
    df["labs_per_day"] = df["num_lab_procedures"] / los
    df["procedures_per_day"] = df["num_procedures"] / los
    df["long_stay"] = (df["time_in_hospital"] > 7).astype(int)

    # --- medication regimen ------------------------------------------------
    med = df[active_meds]
    df["num_meds_prescribed"] = (med != "No").sum(axis=1)
    df["num_med_changes"] = med.isin(["Up", "Down"]).sum(axis=1)

    # --- glycaemic testing (the source paper's research question) ----------
    df["a1c_tested"] = (df["A1Cresult"].fillna("None") != "None").astype(int)
    df["a1c_severity_ord"] = df["A1Cresult"].fillna("None").map(C.A1C_ORDER).fillna(0).astype(int)
    df["glucose_tested"] = (df["max_glu_serum"].fillna("None") != "None").astype(int)
    df["glucose_severity_ord"] = (
        df["max_glu_serum"].fillna("None").map(C.GLU_ORDER).fillna(0).astype(int)
    )

    # --- diagnoses ---------------------------------------------------------
    for i in (1, 2, 3):
        df[f"diag_{i}_group"] = df[f"diag_{i}"].map(map_icd9).astype("string")

    groups = df[["diag_1_group", "diag_2_group", "diag_3_group"]]
    df["n_distinct_diag_groups"] = groups.apply(
        lambda r: len(set(r) - {"Missing"}), axis=1
    )
    df["diabetes_is_primary"] = (df["diag_1_group"] == "Diabetes").astype(int)
    df["has_circulatory"] = (groups == "Circulatory").any(axis=1).astype(int)
    df["comorbidity_burden"] = df["number_diagnoses"]

    # --- demographics and admission context --------------------------------
    df["age_midpoint"] = df["age"].map(C.AGE_MIDPOINTS).astype("float")
    df["emergency_admission"] = (
        df.get("admission_type_id_desc", pd.Series("Unknown", index=df.index)) == "Emergency"
    ).astype(int)
    df["discharged_home"] = (
        df.get("discharge_disposition_id_desc", pd.Series("Unknown", index=df.index))
        == "Discharged to home"
    ).astype(int)

    # --- payer grouping for the fairness audit -----------------------------
    df["payer_group"] = _group_payer(df["payer_code"])

    return df


def _group_payer(series: pd.Series) -> pd.Series:
    """Collapse 18 payer codes into 4 coverage classes.

    The fairness question is about coverage type, not which specific insurer,
    and the raw codes are too sparse to audit subgroup performance reliably.
    """
    government = {"MC", "MD", "CH", "CM", "MP", "OG"}   # Medicare, Medicaid, CHAMPUS, etc.
    commercial = {"BC", "HM", "CP", "UN", "PO", "WC", "OT", "DM", "SI"}
    self_pay = {"SP", "FR"}                              # self-pay, free/charity

    def label(code) -> str:
        if pd.isna(code) or code == "Unknown":
            return "Unknown/Unrecorded"
        if code in government:
            return "Government"
        if code in self_pay:
            return "Self-pay/Charity"
        if code in commercial:
            return "Commercial"
        return "Other"

    return series.map(label).astype("string")


# ---------------------------------------------------------------------------
# Column schema for the preprocessing pipeline
# ---------------------------------------------------------------------------
def column_schema(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split the engineered frame into numeric / ordinal / nominal / drop."""
    drop = set(C.ID_COLS) | {
        C.RAW_TARGET, C.TARGET,
        "diag_1", "diag_2", "diag_3",           # replaced by their groups
        "A1Cresult", "max_glu_serum",           # replaced by ordinal + tested flags
        "admission_type_id", "discharge_disposition_id", "admission_source_id",
        "payer_code",                            # replaced by payer_group
        "age",                                   # replaced by age_midpoint
        # Exploratory-only columns. `cluster` is attached during the notebook's
        # unsupervised section; it must never silently become a model feature,
        # because the deployed model was not trained with it.
        "cluster",
    }

    numeric = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
        "total_prior_visits", "prior_emergency_ratio", "service_intensity",
        "meds_per_day", "labs_per_day", "procedures_per_day",
        "num_meds_prescribed", "num_med_changes", "n_distinct_diag_groups",
        "comorbidity_burden", "age_midpoint",
    ]

    binary = [
        "prior_inpatient_flag", "high_utilizer", "long_stay", "a1c_tested",
        "glucose_tested", "diabetes_is_primary", "has_circulatory",
        "emergency_admission", "discharged_home", "weight_recorded",
        "payer_unknown", "specialty_unknown",
    ]

    ordinal = ["a1c_severity_ord", "glucose_severity_ord"]

    known = drop | set(numeric) | set(binary) | set(ordinal)
    nominal = [c for c in df.columns if c not in known]

    return {
        "numeric": [c for c in numeric if c in df.columns],
        "binary": [c for c in binary if c in df.columns],
        "ordinal": [c for c in ordinal if c in df.columns],
        "nominal": nominal,
        "drop": sorted(drop & set(df.columns)),
    }


if __name__ == "__main__":
    from src.cleaning import clean

    cleaned, _ = clean()
    feat = engineer(cleaned)
    schema = column_schema(feat)

    print(f"Shape after engineering: {feat.shape}")
    for key, cols in schema.items():
        print(f"\n{key} ({len(cols)}):\n  {cols}")

    print("\nDiagnosis group distribution (primary):")
    print(feat["diag_1_group"].value_counts().to_string())
    print("\nPayer group distribution:")
    print(feat["payer_group"].value_counts().to_string())

"""Synthetic patient records for the live demo.

No real patient record is ever displayed. These are hand-built profiles whose
field values are drawn from clinically plausible ranges observed in the training
data, assembled so that the demo is reproducible and shows a recognisable case.
"""

from __future__ import annotations

import pandas as pd

from src import config as C
from src.features import engineer

# Baseline record. Every profile below overrides a handful of these fields.
_BASE: dict = {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 4,
    "payer_code": "MC",
    "medical_specialty": "InternalMedicine",
    "num_lab_procedures": 45,
    "num_procedures": 1,
    "num_medications": 15,
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 0,
    "number_diagnoses": 8,
    "diag_1": "428",       # heart failure
    "diag_2": "250.02",    # diabetes
    "diag_3": "401",       # hypertension
    "max_glu_serum": "None",
    "A1Cresult": "None",
    "change": "No",
    "diabetesMed": "Yes",
    "readmitted": "NO",
}

_MEDS_DEFAULT = {m: "No" for m in C.MEDICATION_COLS}
_MEDS_DEFAULT["insulin"] = "Steady"
_MEDS_DEFAULT["metformin"] = "Steady"

PROFILES: dict[str, dict] = {
    "high_risk": {
        "age": "[70-80)",
        "time_in_hospital": 11,
        "number_inpatient": 3,
        "number_emergency": 2,
        "number_outpatient": 1,
        "num_medications": 24,
        "num_lab_procedures": 62,
        "number_diagnoses": 9,
        "discharge_disposition_id": 3,      # skilled nursing facility
        "admission_type_id": 1,             # emergency
        "payer_code": "?",                  # unrecorded payer
        "A1Cresult": ">8",
        "change": "Ch",
        "_med_overrides": {"insulin": "Up", "glipizide": "Down"},
        "_label": "Elderly, three prior admissions, poor glycaemic control, unrecorded payer",
    },
    "moderate_risk": {
        "age": "[60-70)",
        "time_in_hospital": 5,
        "number_inpatient": 1,
        "number_emergency": 0,
        "number_outpatient": 2,
        "num_medications": 16,
        "number_diagnoses": 7,
        "discharge_disposition_id": 6,      # home with home health
        "A1Cresult": "Norm",
        "change": "Ch",
        "_med_overrides": {"insulin": "Steady"},
        "_label": "One prior admission, discharged with home health support",
    },
    "low_risk": {
        "age": "[40-50)",
        "time_in_hospital": 2,
        "number_inpatient": 0,
        "number_emergency": 0,
        "number_outpatient": 0,
        "num_medications": 8,
        "num_lab_procedures": 22,
        "number_diagnoses": 4,
        "discharge_disposition_id": 1,      # straight home
        "admission_type_id": 3,             # elective
        "payer_code": "BC",
        "A1Cresult": "Norm",
        "change": "No",
        "_med_overrides": {"insulin": "No", "metformin": "Steady"},
        "_label": "Middle-aged, elective short stay, no prior utilisation",
    },
}


def profile_label(profile: str) -> str:
    return PROFILES.get(profile, {}).get("_label", profile)


def make_synthetic_patient(profile: str = "high_risk", **overrides) -> pd.DataFrame:
    """Build one engineered, model-ready synthetic record.

    Returns a single-row DataFrame carrying the same engineered columns the
    trained pipeline expects.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {list(PROFILES)}")

    record = dict(_BASE)
    meds = dict(_MEDS_DEFAULT)

    spec = PROFILES[profile]
    med_overrides = spec.get("_med_overrides", {})
    record.update({k: v for k, v in spec.items() if not k.startswith("_")})
    meds.update(med_overrides)

    record.update(meds)
    record.update(overrides)

    # Two synthetic identifiers so the cleaning helpers have the columns they
    # expect. They are dropped before the model sees anything.
    record.setdefault("encounter_id", 999_999_999)
    record.setdefault("patient_nbr", 999_999_999)

    df = pd.DataFrame([record])
    return _prepare(df)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same decode + policy + engineering steps used in training."""
    from src.cleaning import (
        CleaningAudit, apply_missing_policy, binarise_target, decode_id_columns,
    )

    df = df.copy()
    df = df.replace(C.MISSING_SENTINEL, pd.NA)

    numeric = [
        "encounter_id", "patient_nbr", "admission_type_id",
        "discharge_disposition_id", "admission_source_id", "time_in_hospital",
        "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
        "number_diagnoses",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["weight"] = pd.NA

    df = binarise_target(df)
    df = decode_id_columns(df)
    df = apply_missing_policy(df, CleaningAudit())

    return engineer(df)


def profiles_frame() -> pd.DataFrame:
    """All three demo profiles, for the dashboard's comparison view."""
    return pd.concat(
        [make_synthetic_patient(p).assign(_profile=p) for p in PROFILES],
        ignore_index=True,
    )


if __name__ == "__main__":
    import joblib

    pipe = joblib.load(C.MODELS_DIR / "best_pipeline.joblib")
    for name in PROFILES:
        patient = make_synthetic_patient(name)
        cols = pipe.feature_names_in_ if hasattr(pipe, "feature_names_in_") else patient.columns
        prob = float(pipe.predict_proba(patient[list(cols)])[:, 1][0])
        print(f"{name:15s} risk={prob:6.2%}   {profile_label(name)}")

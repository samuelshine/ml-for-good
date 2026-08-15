"""Cleaning: exclusions, invalid values, de-duplication, missing-value policy.

Every function returns an audit record alongside its output so the notebook can
report exact counts without recomputing anything.

Import order matters: `clean()` runs the steps in the order they are numbered,
and that order is itself a design decision (exclusions before de-duplication,
so a patient whose first encounter ended in death is not silently represented
by a later encounter that never happened).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src import config as C
from src.data_loader import load_id_mappings, load_raw_typed


@dataclass
class CleaningAudit:
    """Row-level accounting for every record we drop, and why."""

    steps: list[dict] = field(default_factory=list)
    missing_before: pd.Series | None = None
    dropped_columns: dict[str, str] = field(default_factory=dict)

    def log(self, step: str, before: int, after: int, reason: str) -> None:
        self.steps.append(
            {
                "step": step,
                "rows_before": before,
                "rows_after": after,
                "rows_dropped": before - after,
                "pct_dropped": round(100 * (before - after) / before, 2) if before else 0.0,
                "reason": reason,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.steps)


# ---------------------------------------------------------------------------
# Step 1 — target
# ---------------------------------------------------------------------------
def binarise_target(df: pd.DataFrame) -> pd.DataFrame:
    """'<30' -> 1, everything else -> 0.

    '>30' is folded into the negative class deliberately: the clinical and
    financial question is the 30-day window that CMS penalises, and a
    readmission at day 200 is a different phenomenon with different drivers.
    """
    df = df.copy()
    df[C.TARGET] = (df[C.RAW_TARGET] == "<30").astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 2 — clinically impossible outcomes
# ---------------------------------------------------------------------------
def drop_expired_and_hospice(df: pd.DataFrame, audit: CleaningAudit) -> pd.DataFrame:
    """Remove encounters ending in death or hospice transfer.

    These patients are structurally incapable of being readmitted, so their
    label is always 0 for a reason unrelated to their risk. Keeping them
    teaches the model that terminal illness predicts *safety*. Exclusion list
    follows Strack et al. (2014).
    """
    before = len(df)
    mask = df["discharge_disposition_id"].isin(C.EXPIRED_OR_HOSPICE_DISCHARGE_IDS)
    out = df.loc[~mask].copy()
    audit.log(
        "drop_expired_hospice",
        before,
        len(out),
        f"discharge_disposition_id in {C.EXPIRED_OR_HOSPICE_DISCHARGE_IDS} "
        "= died or hospice; cannot be readmitted (Strack et al., 2014)",
    )
    return out


# ---------------------------------------------------------------------------
# Step 3 — invalid categories
# ---------------------------------------------------------------------------
def drop_invalid_gender(df: pd.DataFrame, audit: CleaningAudit) -> pd.DataFrame:
    """Remove the 3 rows coded 'Unknown/Invalid' for gender."""
    before = len(df)
    out = df.loc[df["gender"] != "Unknown/Invalid"].copy()
    audit.log(
        "drop_invalid_gender",
        before,
        len(out),
        "gender='Unknown/Invalid' is a data-entry error, not a category; "
        "too few rows to model and it would pollute the fairness audit",
    )
    return out


# ---------------------------------------------------------------------------
# Step 4 — de-duplication (the leakage-critical step)
# ---------------------------------------------------------------------------
def deduplicate_patients(df: pd.DataFrame, audit: CleaningAudit) -> pd.DataFrame:
    """Keep only each patient's first encounter.

    The dataset has 101,766 encounters from 71,518 patients. Treating rows as
    independent breaks the i.i.d. assumption twice over:

    1. A patient's later encounters leak their earlier outcome into whichever
       split they land in, inflating any random-split score.
    2. Frequent flyers (up to 40 encounters) are over-weighted in training,
       biasing the model toward a small, atypical group.

    Strack et al. resolved both by keeping the first encounter per patient. We
    follow that, and quantify what the naive alternative costs in
    `src.evaluate.leakage_sensitivity`.
    """
    before = len(df)
    out = (
        df.sort_values("encounter_id")
        .drop_duplicates(subset="patient_nbr", keep="first")
        .copy()
    )
    audit.log(
        "deduplicate_patients",
        before,
        len(out),
        "keep first encounter per patient_nbr; later encounters leak outcomes "
        "across splits and over-weight frequent readmitters",
    )
    return out


# ---------------------------------------------------------------------------
# Step 5 — decode the numeric-looking categorical IDs
# ---------------------------------------------------------------------------
def decode_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map admission/discharge/source codes to their text descriptions.

    These arrive as int64 but are nominal. Left numeric, every model would
    treat 'code 25' as five times 'code 5'.
    """
    df = df.copy()
    maps = load_id_mappings()

    unknown_labels = {"NULL", "Not Available", "Not Mapped", "Unknown", "Unknown/Invalid"}

    for col, mapping in maps.items():
        if col not in df.columns:
            continue
        decoded = df[col].map(mapping).fillna("Unknown")
        decoded = decoded.where(~decoded.isin(unknown_labels), "Unknown")
        df[col + "_desc"] = decoded.astype("string")

    return df


# ---------------------------------------------------------------------------
# Step 6 — missing-value policy
# ---------------------------------------------------------------------------
def apply_missing_policy(df: pd.DataFrame, audit: CleaningAudit) -> pd.DataFrame:
    """Handle each missing column on its own terms.

    There is no single correct imputation here because the columns are missing
    for different reasons:

    - `weight` (96.9%): not recorded as routine practice. Nothing to impute
      from. Dropped, but the *fact* of recording is kept as a feature, since a
      weighed patient probably got closer attention.
    - `medical_specialty` (49.1%) and `payer_code` (39.6%): missingness is
      informative. An absent payer code disproportionately means self-pay or
      uninsured, which is exactly the population this project cares about.
      Encoded as an explicit 'Unknown' level, never imputed away.
    - `race` (2.2%): a protected attribute. Imputing it would manufacture the
      very evidence the fairness audit is supposed to test. Kept as 'Unknown'.
    - `diag_1/2/3` (<1.5%): genuinely sparse coding. Left as NaN here and
      routed to the 'Other' diagnosis group in feature engineering.
    """
    df = df.copy()
    audit.missing_before = (df.isna().mean() * 100).round(2)

    # weight: keep the signal, drop the column
    df["weight_recorded"] = df["weight"].notna().astype(int)
    df = df.drop(columns=["weight"])
    audit.dropped_columns["weight"] = (
        "96.9% missing; replaced by weight_recorded indicator (missingness is "
        "informative, the values are not recoverable)"
    )

    # informative missingness -> explicit level
    for col in ["medical_specialty", "payer_code", "race"]:
        df[col] = df[col].fillna("Unknown").astype("string")

    df["payer_unknown"] = (df["payer_code"] == "Unknown").astype(int)
    df["specialty_unknown"] = (df["medical_specialty"] == "Unknown").astype(int)

    return df


# ---------------------------------------------------------------------------
# Step 7 — zero-variance columns
# ---------------------------------------------------------------------------
def drop_zero_variance(df: pd.DataFrame, audit: CleaningAudit) -> pd.DataFrame:
    """Drop columns with exactly one observed value.

    Only *zero* variance is dropped, not merely rare. Near-constant drugs such
    as nateglinide (99.3% 'No') still separate a real subgroup, and the
    OneHotEncoder's `min_frequency` handles them without discarding signal.
    """
    df = df.copy()
    constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    for col in constant:
        audit.dropped_columns[col] = "single observed value; carries no information"
    return df.drop(columns=constant)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, CleaningAudit]:
    """Run the full cleaning sequence and return (clean_df, audit)."""
    if df is None:
        df = load_raw_typed()

    audit = CleaningAudit()
    audit.log("raw", len(df), len(df), "as downloaded from UCI")

    df = binarise_target(df)
    df = drop_expired_and_hospice(df, audit)
    df = drop_invalid_gender(df, audit)
    df = deduplicate_patients(df, audit)
    df = decode_id_columns(df)
    df = apply_missing_policy(df, audit)
    df = drop_zero_variance(df, audit)

    return df.reset_index(drop=True), audit


if __name__ == "__main__":
    out, audit = clean()
    print(audit.to_frame().to_string(index=False))
    print(f"\nFinal shape: {out.shape}")
    print(f"Positive rate: {out[C.TARGET].mean():.4f}")
    print(f"Unique patients: {out['patient_nbr'].nunique():,} (should equal row count)")
    print("\nDropped columns:")
    for col, why in audit.dropped_columns.items():
        print(f"  {col}: {why}")

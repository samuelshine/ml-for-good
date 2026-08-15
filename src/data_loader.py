"""Download and load the UCI Diabetes 130-US Hospitals dataset.

Run directly to fetch the data:

    python -m src.data_loader
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile

import pandas as pd

from src import config as C

# Recorded on first successful download so later runs can prove they got the
# same bytes. Printed by this module if it does not match.
EXPECTED_SHA256 = "f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a"

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download(force: bool = False) -> None:
    """Fetch the zip from UCI and extract the two CSVs into data/raw/."""
    if C.RAW_CSV.exists() and C.IDS_CSV.exists() and not force:
        print(f"Raw data already present in {C.RAW_DIR}")
        return

    print(f"Downloading {C.DATASET_URL} ...")
    req = urllib.request.Request(C.DATASET_URL, headers=_UA)
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = resp.read()

    digest = _sha256(payload)
    print(f"  {len(payload):,} bytes  sha256={digest}")
    if digest != EXPECTED_SHA256:
        print(
            "  NOTE: sha256 differs from the value recorded in data_loader.py.\n"
            "        Update EXPECTED_SHA256 to the value above to pin this copy."
        )

    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        names = outer.namelist()
        # UCI wraps the CSVs in a nested folder, and sometimes a nested zip.
        inner_zips = [n for n in names if n.lower().endswith(".zip")]
        if inner_zips:
            with outer.open(inner_zips[0]) as fh, zipfile.ZipFile(io.BytesIO(fh.read())) as inner:
                _extract_csvs(inner)
        else:
            _extract_csvs(outer)

    print(f"Extracted to {C.RAW_DIR}")


def _extract_csvs(zf: zipfile.ZipFile) -> None:
    wanted = {"diabetic_data.csv": C.RAW_CSV, "IDS_mapping.csv": C.IDS_CSV}
    for name in zf.namelist():
        base = name.split("/")[-1]
        if base in wanted:
            wanted[base].write_bytes(zf.read(name))


def load_raw() -> pd.DataFrame:
    """Load diabetic_data.csv with '?' kept as-is (Phase 2 converts it)."""
    if not C.RAW_CSV.exists():
        download()
    return pd.read_csv(C.RAW_CSV, dtype=str, keep_default_na=False, na_values=[])


def load_raw_typed() -> pd.DataFrame:
    """Load with '?' mapped to NaN and numeric columns coerced.

    This is the entry point every downstream module should use.
    """
    df = load_raw()
    df = df.replace(C.MISSING_SENTINEL, pd.NA)

    numeric_cols = [
        "encounter_id", "patient_nbr", "admission_type_id",
        "discharge_disposition_id", "admission_source_id", "time_in_hospital",
        "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
        "number_diagnoses",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Everything else is genuinely categorical text.
    for col in df.columns.difference(numeric_cols):
        df[col] = df[col].astype("string")

    return df


def load_id_mappings() -> dict[str, dict[int, str]]:
    """Parse IDS_mapping.csv into {column_name: {code: description}}.

    The file is three stacked tables separated by blank lines, so it cannot be
    read with a single read_csv call.
    """
    if not C.IDS_CSV.exists():
        download()

    text = C.IDS_CSV.read_text(encoding="utf-8", errors="replace")
    mappings: dict[str, dict[int, str]] = {}
    current: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == ",":
            current = None
            continue

        left, _, right = line.partition(",")
        left, right = left.strip(), right.strip().strip('"')

        if left.endswith("_id") and right.lower() == "description":
            current = left
            mappings[current] = {}
            continue

        if current is None:
            continue

        try:
            code = int(left)
        except ValueError:
            continue

        mappings[current][code] = right if right and right != "?" else "Unknown"

    return mappings


if __name__ == "__main__":
    download(force="--force" in sys.argv)

    df = load_raw()
    print(f"\ndiabetic_data.csv shape: {df.shape}")
    assert df.shape == C.EXPECTED_RAW_SHAPE, (
        f"Expected {C.EXPECTED_RAW_SHAPE}, got {df.shape}"
    )

    maps = load_id_mappings()
    print("IDS_mapping.csv tables:")
    for key, val in maps.items():
        print(f"  {key}: {len(val)} codes")

    print(f"\nUnique patients : {df['patient_nbr'].nunique():,}")
    print(f"Encounters      : {len(df):,}")
    print("\nTarget distribution:")
    print(df[C.RAW_TARGET].value_counts(normalize=True).round(4).to_string())

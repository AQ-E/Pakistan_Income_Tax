"""
prfs_unified/data_io_dynamic.py
================================
Dynamic data loading for PRFS — accepts user-uploaded CSV/Excel data.

Rules:
  - Applies EXACTLY the same transforms as the original data_io.py (prepare_transforms).
  - Auto-detects all years in the uploaded file.
  - Sorts by year, validates uniqueness.
  - Forecast year = max(year) + 1 automatically.
  - No hardcoded year ranges anywhere in this module.
"""
from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── Column definitions (identical to data_io.py) ─────────────────────────
_LEVEL_COLS = [
    "dt", "gst", "fed", "customs", "gdp", "imports",
    "dutiable_imports", "lsm", "consumption", "exrate",
]
_RATE_COLS = ["inflation", "policy rate"]

# Minimum required columns for the pipeline to run
REQUIRED_LEVEL_COLS = ["dt", "gst", "fed", "customs", "gdp", "imports", "exrate"]
REQUIRED_RATE_COLS  = ["inflation"]

# Columns that are automatically generated (user does NOT need to supply these)
AUTO_GENERATED = [
    "log_dt", "log_gst", "log_fed", "log_customs", "log_gdp",
    "log_gdp_nonagr", "log_lsm", "log_imports", "log_dutiable_imports",
    "log_consumption", "log_exrate",
]


def _coerce_year(x) -> int:
    """Convert FY strings like '2023-24' or plain '2024' → 2024 (end-year)."""
    s = str(x).strip()
    if "-" in s:
        return int(s[:4]) + 1
    return int(s)


def prepare_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identical log-transform logic as prfs_unified/data_io.py:prepare_transforms.
    Applied to whatever years the user supplies — no hardcoding.
    """
    out = df.copy()

    # 1. Forward-fill macro rates
    for col in _RATE_COLS:
        actual = next((c for c in out.columns if c.lower() == col.lower()), None)
        if actual:
            out[actual] = pd.to_numeric(out[actual], errors="coerce").ffill()
            if actual != col:
                out[col] = out[actual]

    # 2. Log transforms for level variables
    for col in _LEVEL_COLS:
        actual = next((c for c in out.columns if c.lower() == col.lower()), None)
        if actual:
            s = pd.to_numeric(out[actual], errors="coerce").ffill()
            s[s <= 0] = np.nan
            out[actual] = s
            out[f"log_{col}"] = np.log(s)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def parse_uploaded_file(
    uploaded_file,
    sheet_name: str = 0,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parse a user-uploaded CSV or Excel file into the canonical DataFrame format.

    Returns
    -------
    (df, error_message)
    df            : cleaned+indexed DataFrame with PeriodIndex, or None on error
    error_message : human-readable error string, or None on success
    """
    try:
        raw_bytes = uploaded_file.read()
        name = uploaded_file.name.lower()

        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw_bytes))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(
                io.BytesIO(raw_bytes),
                sheet_name=sheet_name,
                engine="openpyxl",
            )
        else:
            return None, "Unsupported file type. Please upload a CSV or Excel (.xlsx) file."

        df.columns = [str(c).strip() for c in df.columns]

        # ── Find year column ───────────────────────────────────────────────
        year_col = None
        for cand in ["year_end", "Year", "year", "FY", "fy", "YEAR"]:
            if cand in df.columns:
                year_col = cand
                break

        if year_col is None:
            return None, (
                "Could not find a year column. "
                "Please include a column named 'Year', 'year_end', or 'FY'."
            )

        # Parse years
        df["year_end"] = df[year_col].apply(_coerce_year)

        # ── Validate uniqueness ────────────────────────────────────────────
        dupes = df[df.duplicated("year_end", keep=False)]["year_end"].unique()
        if len(dupes) > 0:
            return None, f"Duplicate years detected: {sorted(dupes)}. Each year must appear exactly once."

        # ── Sort + index ───────────────────────────────────────────────────
        df = df.sort_values("year_end").reset_index(drop=True)
        df.index = pd.PeriodIndex(df["year_end"], freq="Y")

        # ── Check required columns ─────────────────────────────────────────
        df_cols_lower = {c.lower(): c for c in df.columns}
        missing_required = []
        for req in REQUIRED_LEVEL_COLS:
            if req.lower() not in df_cols_lower:
                missing_required.append(req)

        if missing_required:
            return None, (
                f"Missing required columns: {missing_required}. "
                f"Please ensure your data includes at least: "
                f"{REQUIRED_LEVEL_COLS + REQUIRED_RATE_COLS}"
            )

        return df, None

    except Exception as e:
        return None, f"Failed to read file: {e}"


def validate_and_summarize(df: pd.DataFrame) -> dict:
    """
    Returns a summary dict for display in the UI:
    - year_range: (min_year, max_year)
    - n_years: number of years
    - forecast_year: max_year + 1
    - has_consumption: bool
    - has_dutiable_imports: bool
    - has_lsm: bool
    - has_policy_rate: bool
    - missing_cols: list of optional missing columns
    """
    years = df["year_end"].tolist() if "year_end" in df.columns else [int(p.year) for p in df.index]
    min_y, max_y = min(years), max(years)

    df_cols_lower = {c.lower() for c in df.columns}
    optional_cols = {
        "consumption": "Private Consumption",
        "dutiable_imports": "Dutiable Imports",
        "lsm": "Large-Scale Manufacturing (LSM)",
        "gdp_nonagr": "Non-Agricultural GDP",
        "policy rate": "SBP Policy Rate",
    }
    missing_optional = [label for col, label in optional_cols.items() if col not in df_cols_lower]

    return {
        "year_range": (min_y, max_y),
        "n_years": len(years),
        "forecast_year": max_y + 1,
        "has_consumption": "consumption" in df_cols_lower,
        "has_dutiable_imports": "dutiable_imports" in df_cols_lower,
        "has_lsm": "lsm" in df_cols_lower,
        "has_policy_rate": "policy rate" in df_cols_lower or "policy_rate" in df_cols_lower,
        "missing_optional": missing_optional,
    }


def build_template_df() -> pd.DataFrame:
    """Returns an empty template DataFrame with all expected column names."""
    cols = (
        ["year_end"]
        + _LEVEL_COLS
        + ["gdp_nonagr"]
        + _RATE_COLS
        + ["covid", "regime", "step_2024", "dummy_2024", "dummy_2025"]
    )
    return pd.DataFrame(columns=cols)


def get_template_csv() -> bytes:
    """Return a CSV template with column headers + 2 example rows."""
    template = pd.DataFrame({
        "year_end": [2023, 2024],
        "dt":       [5000000, 5500000],
        "gst":      [4000000, 4400000],
        "fed":      [500000,  550000],
        "customs":  [1000000, 1100000],
        "gdp":      [80000000, 90000000],
        "gdp_nonagr": [60000000, 67000000],
        "lsm":      [8000000, 8500000],
        "imports":  [9000000, 9500000],
        "dutiable_imports": [6000000, 6400000],
        "exrate":   [280.0, 295.0],
        "inflation": [28.0, 20.0],
        "consumption": [55000000, 62000000],
        "policy rate": [22.0, 19.0],
        "covid":    [0, 0],
        "regime":   [1, 1],
        "step_2024": [0, 1],
        "dummy_2024": [0, 1],
        "dummy_2025": [0, 0],
    })
    return template.to_csv(index=False).encode("utf-8")

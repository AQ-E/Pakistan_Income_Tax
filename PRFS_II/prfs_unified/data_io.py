"""
prfs_unified/data_io.py
=======================
Centralised data loading for the PRFS Unified application.
Handles:
  - tax_prepared_data (Excel or CSV fallback)
  - buoyancy_estimates.xlsx
  - Multi-model bundle (.pkl + .json)
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ── File-resolution helpers ──────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))

_SEARCH_DIRS = [
    _APP_ROOT,
    _THIS_DIR,
]


def _resolve(filename: str) -> Optional[str]:
    """Search common relative dirs for *filename*."""
    for d in _SEARCH_DIRS:
        p = os.path.join(d, filename)
        if os.path.isfile(p):
            return p
    return None


# ── Tax prepared data ────────────────────────────────────────────────────
def _to_year_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    try:
        years = out.index.astype(str).str.extract(r"(\d{4})")[0].astype(int)
        out.index = pd.PeriodIndex(years, freq="Y")
    except Exception:
        pass
    return out


@st.cache_data(show_spinner=False)
def load_tax_data() -> pd.DataFrame:
    """Load the canonical tax_prepared_data from xlsx (preferred) or csv."""
    # Try xlsx first
    xlsx = _resolve("tax_prepared_data.xlsx")
    if xlsx:
        df = pd.read_excel(xlsx, sheet_name="tax_prepared_data", engine="openpyxl")
        df.columns = [c.strip() for c in df.columns]
        if "Year" in df.columns:
            df["year_end"] = df["Year"].apply(
                lambda x: int(str(x)[:4]) + 1 if "-" in str(x) else int(x)
            )
        if "year_end" in df.columns:
            df = df.sort_values("year_end").reset_index(drop=True)
            df.index = pd.PeriodIndex(df["year_end"], freq="Y")
        return df

    csv = _resolve("tax_prepared_data.csv")
    if csv:
        df = pd.read_csv(csv, index_col=0)
        return _to_year_index(df)

    st.error("❌ Cannot find tax_prepared_data.xlsx or .csv")
    st.stop()


def prepare_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform levels; forward-fill rates and missing levels."""
    out = df.copy()
    levels = [
        "dt", "gst", "fed", "customs", "gdp", "imports",
        "dutiable_imports", "lsm", "consumption", "exrate",
    ]
    rates = ["inflation", "policy rate"]

    for col in rates:
        actual = next((c for c in df.columns if c.lower() == col.lower()), None)
        if actual:
            out[actual] = pd.to_numeric(out[actual], errors="coerce").ffill()
            out[col] = out[actual]

    for col in levels:
        actual = next((c for c in out.columns if c.lower() == col.lower()), None)
        if actual:
            s = pd.to_numeric(out[actual], errors="coerce")
            # Forward-fill missing levels (e.g. consumption=NaN for 2026)
            s = s.ffill()
            s[s <= 0] = np.nan
            out[actual] = s
            out[f"log_{col}"] = np.log(s)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ── Buoyancy data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_buoyancy() -> Optional[Dict]:
    path = _resolve("buoyancy_estimates.xlsx")
    if not path:
        return None
    try:
        raw = pd.read_excel(path, sheet_name="Sensitivity Analysis", header=None, engine="openpyxl")
        base_idx = proj_idx = -1
        for i, row in raw.iterrows():
            txt = str(row[1]).lower() if len(row) > 1 else ""
            if "expected base" in txt:
                base_idx = i
            if "projections" in txt and "2026-27" in txt:
                proj_idx = i
        if base_idx == -1:
            base_idx = 12
        if proj_idx == -1:
            proj_idx = 14
        base_vals = raw.iloc[base_idx, 2:9].tolist()
        proj_vals = raw.iloc[proj_idx, 2:9].tolist()
        return {
            "fy2026_base": dict(
                dt=base_vals[0], st_domestic=base_vals[1], st_imports=base_vals[2],
                st_total=base_vals[3], customs=base_vals[4], fed=base_vals[5], total=base_vals[6],
            ),
            "fy2027_buoyancy": dict(
                dt=proj_vals[0], st_domestic=proj_vals[1], st_imports=proj_vals[2],
                st_total=proj_vals[3], customs=proj_vals[4], fed=proj_vals[5], total=proj_vals[6],
            ),
        }
    except Exception:
        return None


# ── Multi-model bundle ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_multimodel_assets() -> Tuple[Optional[Dict], Optional[Dict], Optional[pd.DataFrame]]:
    """Return (bundle, meta, df_hist) or (None, None, None) if absent."""
    # Cache bust 4: Final Customs ARDL(1,0) without dummy_2022 - clean parsimonious model
    pkl_path = _resolve("tax_models_bundle.pkl")
    json_path = _resolve("tax_models_meta.json")
    csv_path = _resolve("tax_prepared_data.csv")
    xlsx_path = _resolve("tax_prepared_data.xlsx")

    if not pkl_path or not json_path:
        return None, None, None

    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load historical df – prefer xlsx (it has 2026 data), fall back to csv
    if xlsx_path:
        df = load_tax_data()
    elif csv_path:
        df = pd.read_csv(csv_path, index_col=0)
        df = _to_year_index(df)
    else:
        return bundle, meta, None

    # Compute log columns from levels (fills NaN logs for 2026 row etc.)
    df = prepare_transforms(df)

    # Pre-compute ENet residuals (same as app_multimodel_v2)
    for head, b in bundle["models"].items():
        if "enet" in b:
            model = b["enet"]["model"]
            feat_cols = b["enet"]["feature_cols"]
            y_name = b["spec"]["y"]
            train_resids = []
            valid_hist = df.dropna(subset=[y_name]).index[2:]
            for t in valid_hist:
                try:
                    row = {}
                    for c in feat_cols:
                        if c.endswith("_L0"):
                            base = c[:-3]
                            row[c] = df.loc[t, base] if base in df.columns else 0.0
                        elif "_L" in c:
                            parts = c.rsplit("_L", 1)
                            base, lag = parts[0], int(parts[1])
                            row[c] = df.shift(lag).loc[t, base] if base in df.columns else 0.0
                        else:
                            row[c] = df.loc[t, c] if c in df.columns else 0.0
                    row_df = pd.DataFrame([row], columns=feat_cols).fillna(0)
                    pred = float(model.predict(row_df)[0])
                    train_resids.append(df.loc[t, y_name] - pred)
                except Exception:
                    continue
            b["enet"]["residuals"] = train_resids if train_resids else [0.0]

    return bundle, meta, df

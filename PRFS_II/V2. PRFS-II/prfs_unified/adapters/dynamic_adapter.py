"""
prfs_unified/adapters/dynamic_adapter.py
=========================================
Wraps the Dynamic 2-Step PRFS engine (ForecastingPipeline + ScenarioEngine).
"""
from __future__ import annotations

import sys
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Ensure the PRFS engine package is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# adapters/ → prfs_unified/ → app_root/
_APP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

# These are the PRFS engine modules – imported lazily so the app
# still works even if the PRFS/ folder is absent.
_pipeline_cls = None
_scenario_cls = None


def _ensure_imports():
    global _pipeline_cls, _scenario_cls
    if _pipeline_cls is not None:
        return True
    try:
        from engine.pipeline import ForecastingPipeline
        from engine.scenario import ScenarioEngine
        _pipeline_cls = ForecastingPipeline
        _scenario_cls = ScenarioEngine
        return True
    except ImportError:
        return False


def is_available() -> bool:
    return _ensure_imports()


def run_pipeline(df_raw: pd.DataFrame):
    """Fit the full 2-stage pipeline and cache in session_state."""
    if not _ensure_imports():
        st.error("Dynamic PRFS engine not found (PRFS/engine/ not on path).")
        return
    pipeline = _pipeline_cls(df_raw)
    pipeline.run_full_pipeline()
    st.session_state["dyn_pipeline"] = pipeline
    return pipeline


def run_scenario(
    df_raw: pd.DataFrame,
    horizon: int,
    targets: Dict,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Run scenario using the cached pipeline."""
    pipeline = st.session_state.get("dyn_pipeline")
    if pipeline is None:
        st.warning("Please run the Dynamic Pipeline first.")
        st.stop()

    engine = _scenario_cls(df_raw, pipeline.best_models, pipeline.channel_models)
    results, base_df, scen_df = engine.run_scenario(horizon, targets)
    return results, base_df, scen_df


def to_standard_df(results: Dict, head: str, horizon: int) -> pd.DataFrame:
    """
    Convert Dynamic engine results for *head* into the standardised
    forecast DataFrame with columns: yhat, lo80, hi80, lo95, hi95.
    Values are in PKR **million** (consistent with multi-model output).
    Dynamic engine stores values in PKR Billion → multiply by 1000.
    """
    r = results.get(head)
    if r is None:
        return pd.DataFrame()
    return pd.DataFrame({
        "yhat": r["scenario"].values * 1000,
        "lo80": r["l80"].values * 1000,
        "hi80": r["u80"].values * 1000,
        "lo95": r["l95"].values * 1000,
        "hi95": r["u95"].values * 1000,
    })


def to_billion_df(results: Dict, head: str) -> pd.DataFrame:
    """Same but keep in Billion (for tables)."""
    r = results.get(head)
    if r is None:
        return pd.DataFrame()
    return pd.DataFrame({
        "yhat": r["scenario"].values,
        "lo80": r["l80"].values,
        "hi80": r["u80"].values,
        "lo95": r["l95"].values,
        "hi95": r["u95"].values,
    })

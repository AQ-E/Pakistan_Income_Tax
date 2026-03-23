"""
prfs_unified/buoyancy_benchmark.py
===================================
Buoyancy benchmark rendering (table + bar chart) for FY2027.
Works with any engine (multi-model or dynamic).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Key mapping between engine heads and buoyancy-file keys
B_MAP = {
    "dt": "dt",
    "gst": "st_total",
    "customs": "customs",
    "fed": "fed",
    "total": "total",
}

HEAD_LABELS = {
    "dt": "Direct Taxes (DT)",
    "gst": "Sales Tax (GST)",
    "fed": "Federal Excise Duty (FED)",
    "customs": "Customs Duty (CD)",
    "total": "TOTAL TAX REVENUE",
}


def _buoy_val(buoy: Dict, section: str, key: str) -> float:
    """Get buoyancy number, handle GST = st_domestic + st_imports."""
    if key == "gst":
        return buoy[section].get("st_domestic", 0) + buoy[section].get("st_imports", 0)
    return buoy[section].get(B_MAP.get(key, key), 0)


def render_benchmark(
    buoy_data: Dict,
    fore_head: pd.DataFrame,
    fore_total: pd.DataFrame,
    head: str,
):
    """
    Render buoyancy vs model benchmark inside the current Streamlit container.

    Parameters
    ----------
    buoy_data   : dict from load_buoyancy()
    fore_head   : forecast df for a single tax head (PKR Million, yhat/lo95/hi95)
    fore_total  : forecast df for TOTAL (PKR Million)
    head        : current tax head key
    """
    if buoy_data is None:
        st.info("Buoyancy file not found – benchmark skipped.")
        return

    st.markdown("---")
    st.subheader("📊 FY2027 Buoyancy Benchmark")

    # Prepare table rows for head + total
    rows = []
    for key, fore in [(head, fore_head), ("total", fore_total)]:
        fy26 = _buoy_val(buoy_data, "fy2026_base", key)
        fy27_buoy = _buoy_val(buoy_data, "fy2027_buoyancy", key)

        # Model FY2027 = first forecast year, in PKR Billion
        fy27_model = fore["yhat"].iloc[0] / 1000.0 if len(fore) else 0
        lo95 = fore["lo95"].iloc[0] / 1000.0 if len(fore) else 0
        hi95 = fore["hi95"].iloc[0] / 1000.0 if len(fore) else 0

        rows.append({
            "Tax Head": HEAD_LABELS.get(key, key.upper()),
            "FY26 Actual (bn)": round(fy26, 1),
            "FY27 Buoyancy (bn)": round(fy27_buoy, 1),
            "FY27 Model (bn)": round(fy27_model, 1),
            "95% CI (bn)": f"{lo95:,.0f} — {hi95:,.0f}",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Bar chart ────────────────────────────────────────────────────────
    for key, fore in [(head, fore_head)]:
        fy26 = _buoy_val(buoy_data, "fy2026_base", key)
        buoy27 = _buoy_val(buoy_data, "fy2027_buoyancy", key)
        model27 = fore["yhat"].iloc[0] / 1000.0 if len(fore) else 0
        lo95 = fore["lo95"].iloc[0] / 1000.0 if len(fore) else 0
        hi95 = fore["hi95"].iloc[0] / 1000.0 if len(fore) else 0

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["FY26 Actual", "FY27 Buoyancy", "FY27 Model"],
            y=[fy26, buoy27, model27],
            marker_color=["#636e72", "#0984e3", "#e17055"],
            text=[f"{v:.1f}" for v in [fy26, buoy27, model27]],
            textposition="auto",
            error_y=dict(
                type="data", symmetric=False,
                array=[0, 0, hi95 - model27],
                arrayminus=[0, 0, model27 - lo95],
                thickness=2, width=10, color="#2d3436",
            ),
        ))
        fig.update_layout(
            title=f"{HEAD_LABELS.get(key, key)} – Buoyancy vs Model (with 95% CI)",
            yaxis_title="PKR Billion",
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

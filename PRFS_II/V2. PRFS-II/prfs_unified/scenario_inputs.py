"""
prfs_unified/scenario_inputs.py
================================
Sidebar controls for the PRFS Unified app.
Exposes ONLY the four Dynamic PRFS macro sliders + mapping elasticities.
Single unified model selector — no "Best by MAPE", user picks directly.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# ── Labels ───────────────────────────────────────────────────────────────
TAX_LABELS = {
    "customs": "Customs Duty",
    "dt": "Income / Direct Tax (DT)",
    "fed": "Federal Excise Duty (FED)",
    "gst": "Sales Tax / GST",
}

MODEL_LABELS = {
    "ardl": "ARDL",
    "arimax": "ARIMAX (SARIMAX)",
    "enet": "ElasticNet",
    "dynamic": "Dynamic Structural Model (DSM)",
}


def render_sidebar(
    perf: pd.DataFrame | None = None,
    dynamic_available: bool = True,
    multimodel_available: bool = True,
) -> Dict:
    """Draw the sidebar and return a dict of all user choices."""
    sb = st.sidebar
    sb.title("PRFS Controls")

    # ── Tax head ─────────────────────────────────────────────────────────
    head = sb.selectbox(
        "Tax Head",
        options=list(TAX_LABELS.keys()),
        format_func=lambda k: TAX_LABELS[k],
    )

    # ── Single unified model selector (no best-by-mape) ──────────────────
    model_options = []
    if multimodel_available:
        model_options.extend(["ardl", "arimax", "enet"])
    if dynamic_available:
        model_options.append("dynamic")

    if not model_options:
        st.error("No forecasting engine available.")
        st.stop()

    model_choice = sb.selectbox(
        "Model",
        options=model_options,
        format_func=lambda m: MODEL_LABELS.get(m, m.upper()),
    )

    # ── Horizon & bootstrap ──────────────────────────────────────────────
    horizon = sb.slider("Forecast Horizon (years)", 1, 10, 3)
    n_sims = sb.select_slider("Bootstrap simulations", options=[100, 500, 1000], value=500)

    sb.markdown("---")

    # ── Scenario inputs (STRICT: only 4 dynamic PRFS sliders) ────────────
    sb.header("Macro Scenario")
    input_mode = sb.radio("Input Mode", ["Manual Sliders", "Macro Path Table"], horizontal=True)

    if input_mode == "Manual Sliders":
        gdp_growth = sb.slider("Nominal GDP Growth Target (%)", -2.0, 20.0, 10.8) / 100
        inflation = sb.slider("Inflation Target (%)", 0.0, 40.0, 6.1)
        exrate_growth = sb.slider("Exchange Rate Depreciation (%)", -5.0, 30.0, 1.0) / 100
        policy_rate = sb.slider("Policy Rate Target (%)", 5.0, 30.0, 11.2)
        targets = dict(
            gdp_growth=gdp_growth,
            inflation=inflation,
            exrate_growth=exrate_growth,
            policy_rate=policy_rate,
        )
    else:
        default = pd.DataFrame({
            "Year": range(1, horizon + 1),
            "GDP Growth (%)": [10.8] * horizon,
            "Inflation (%)": [6.1] * horizon,
            "FX Depreciation (%)": [1.0] * horizon,
            "Policy Rate (%)": [11.2] * horizon,
        })
        path = sb.data_editor(default, num_rows="fixed", hide_index=True)
        targets = dict(
            gdp_growth=(path["GDP Growth (%)"] / 100.0).tolist(),
            inflation=path["Inflation (%)"].tolist(),
            exrate_growth=(path["FX Depreciation (%)"] / 100.0).tolist(),
            policy_rate=path["Policy Rate (%)"].tolist(),
        )

    # ── Mapping elasticity overrides (advanced) ──────────────────────────
    sb.markdown("---")
    with sb.expander("⚙️ Mapping Overrides (Advanced)"):
        st.caption(
            "These elasticities control how GDP growth maps to "
            "sub-series for the multi-model engine. Default = 1.0 (proportional)."
        )
        imports_e = st.number_input("Imports ↔ GDP elasticity", 0.0, 3.0, 1.0, 0.1)
        cons_e = st.number_input("Consumption ↔ GDP elasticity", 0.0, 3.0, 1.0, 0.1)
        lsm_e = st.number_input("LSM ↔ GDP elasticity", 0.0, 3.0, 1.0, 0.1)

    elasticities = dict(imports=imports_e, consumption=cons_e, lsm=lsm_e)

    # ── Dummies ──────────────────────────────────────────────────────────
    sb.markdown("---")
    sb.subheader("Dummies")
    covid_on = sb.checkbox("COVID dummy = 1 in forecast", value=False)
    regime_on = sb.checkbox("Regime dummy = 1 in forecast", value=True)

    is_mm = model_choice in ("ardl", "arimax", "enet")

    return dict(
        head=head,
        is_multimodel=is_mm,
        model_choice=model_choice,
        horizon=horizon,
        n_sims=n_sims,
        targets=targets,
        elasticities=elasticities,
        covid_on=covid_on,
        regime_on=regime_on,
        input_mode=input_mode,
    )

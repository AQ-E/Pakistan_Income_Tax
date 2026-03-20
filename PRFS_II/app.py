"""
app_prfs_unified.py
====================
Pakistan Revenue Forecasting System (PRFS) — Unified Application.

Same layout and methodology as the original app.
The ONLY change: Tab 6 "Data Preview" has an editable table.
When the user edits/adds rows and clicks "Apply", that data becomes
the input to ALL calculations — transforms, lags, logs, training,
diagnostics, projections, and future exog generation.

Forecast year = max(year in edited table) + 1  — fully dynamic.

Run:
    streamlit run PRFS_II/app.py
"""
from __future__ import annotations

import sys, os

# Ensure package is importable regardless of CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import traceback

# ── Local imports ─────────────────────────────────────────────────────────
from prfs_unified.data_io import (
    load_tax_data,
    prepare_transforms,
    load_buoyancy,
    load_multimodel_assets,
)
from prfs_unified.scenario_inputs import render_sidebar, TAX_LABELS, MODEL_LABELS
from prfs_unified.plots import forecast_plot, forecast_table
from prfs_unified.buoyancy_benchmark import render_benchmark
from prfs_unified.utils import (
    diagnostics_ardl,
    diagnostics_arimax,
    coef_table_ardl,
    coef_table_arimax,
    coef_table_enet,
)
from prfs_unified.adapters import multimodel_adapter as mm
from prfs_unified.adapters import dynamic_adapter as dyn
from prfs_unified.mapping import build_multimodel_future_exog_from_dynamic

# ═════════════════════════════════════════════════════════════════════════
# Page config
# ═════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Pakistan Revenue Forecasting System (PRFS)",
    layout="wide",
    page_icon="📈",
)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background:linear-gradient(135deg,#0a3d62,#1e3799);padding:28px 32px;
    border-radius:12px;margin-bottom:24px">
        <h1 style="color:#fff;margin:0;font-family:'Inter',sans-serif">
          Pakistan Revenue Forecasting System (PRFS)</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════════════════
# Load / initialise data
# ─────────────────────────────────────────────────────────────────────────
# On first run, load from the default file and store in session_state.
# On subsequent runs (including after user edits in Tab 6), use whatever
# is in session_state["user_df"] — that way the editable table is the
# single source of truth for the whole pipeline.
# ═════════════════════════════════════════════════════════════════════════
try:
    if "user_df" not in st.session_state:
        _df_file = load_tax_data()
        _df_file = prepare_transforms(_df_file)
        st.session_state["user_df"] = _df_file
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

df_raw = st.session_state["user_df"]

# ── Load multi-model assets (bundle is fixed; ENet residuals re-anchored
#    to whatever data is in session_state) ─────────────────────────────────
bundle, meta, _df_hist_from_file = load_multimodel_assets()

# Rebuild df_hist from current user data (may differ from original file)
df_hist = df_raw.copy()

# Re-compute ENet residuals against the current (possibly edited) data
if bundle is not None:
    for head, b in bundle["models"].items():
        if "enet" in b:
            model      = b["enet"]["model"]
            feat_cols  = b["enet"]["feature_cols"]
            y_name     = b["spec"]["y"]
            train_resids = []
            valid_hist = df_hist.dropna(subset=[y_name]).index[2:] if y_name in df_hist.columns else []
            for t in valid_hist:
                try:
                    row = {}
                    for c in feat_cols:
                        if c.endswith("_L0"):
                            base = c[:-3]
                            row[c] = df_hist.loc[t, base] if base in df_hist.columns else 0.0
                        elif "_L" in c:
                            parts = c.rsplit("_L", 1)
                            base, lag = parts[0], int(parts[1])
                            row[c] = df_hist.shift(lag).loc[t, base] if base in df_hist.columns else 0.0
                        else:
                            row[c] = df_hist.loc[t, c] if c in df_hist.columns else 0.0
                    row_df = pd.DataFrame([row], columns=feat_cols).fillna(0)
                    pred   = float(model.predict(row_df)[0])
                    if y_name in df_hist.columns:
                        train_resids.append(df_hist.loc[t, y_name] - pred)
                except Exception:
                    continue
            b["enet"]["residuals"] = train_resids if train_resids else [0.0]

buoy_data = load_buoyancy()

multimodel_ok = bundle is not None
dynamic_ok    = dyn.is_available()
perf          = mm.perf_table(meta) if meta else None

# ── Dynamic year metadata (from current data, not hardcoded) ─────────────
_max_year = int(df_raw.index.max().year) if hasattr(df_raw.index, "year") else int(str(df_raw.index.max())[:4])
_min_year = int(df_raw.index.min().year) if hasattr(df_raw.index, "year") else int(str(df_raw.index.min())[:4])

# ── Data version: a short hash of the current data content ───────────────
# This is used as a cache-busting key — the moment the user edits any cell
# and clicks Apply, this hash changes, which forces get_cached_forecast to
# recompute instead of returning a stale cached result.
import hashlib as _hashlib
_data_version = _hashlib.md5(
    pd.util.hash_pandas_object(df_raw, index=True).values.tobytes()
).hexdigest()[:12]

# ═════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════
cfg = render_sidebar(
    perf=perf,
    dynamic_available=dynamic_ok,
    multimodel_available=multimodel_ok,
)

# Show data coverage in sidebar for confirmation
st.sidebar.info(f"📅 Data Coverage: FY{_min_year} – FY{_max_year}")
st.sidebar.caption(f"🔮 Forecast year: FY{_max_year + 1}+")

head        = cfg["head"]
horizon     = cfg["horizon"]
n_sims      = cfg["n_sims"]
targets     = cfg["targets"]
elasticities = cfg["elasticities"]
covid_on    = cfg["covid_on"]
regime_on   = cfg["regime_on"]

# ═════════════════════════════════════════════════════════════════════════
# Resolve chosen model
# ═════════════════════════════════════════════════════════════════════════
chosen = cfg["model_choice"]
is_mm  = chosen in ("ardl", "arimax", "enet")
default_model_label = MODEL_LABELS.get(chosen, chosen.upper())

# ═════════════════════════════════════════════════════════════════════════
# Generate forecasts
# ═════════════════════════════════════════════════════════════════════════
fore_head  = None
fore_total = None
exog_future = None
dyn_results = None

try:
    if is_mm:
        fore_head, exog_future = mm.forecast_head(
            bundle, meta, df_hist,
            head, chosen, horizon, n_sims,
            targets, elasticities, covid_on, regime_on,
            data_version=_data_version,
        )
        fore_total = mm.forecast_total(
            bundle, meta, df_hist,
            chosen, horizon, n_sims,
            targets, elasticities, covid_on, regime_on,
            data_version=_data_version,
        )
    else:
        # Dynamic engine
        if "dyn_pipeline" not in st.session_state:
            st.info("🔄 Dynamic pipeline not yet fitted. Click below to run.")
            if st.button("🚀 Run Dynamic Pipeline"):
                with st.spinner("Running 2-Stage Econometric Pipeline…"):
                    dyn.run_pipeline(df_raw)
                st.rerun()
            st.stop()

        dyn_results, _, _ = dyn.run_scenario(df_raw, horizon, targets)
        fore_head  = dyn.to_standard_df(dyn_results, head,    horizon)
        fore_total = dyn.to_standard_df(dyn_results, "total", horizon)

        # PeriodIndex anchored to the user's latest year (dynamic)
        years = [_max_year + i for i in range(1, horizon + 1)]
        pidx  = pd.PeriodIndex(years, freq="Y")
        if len(fore_head):
            fore_head.index  = pidx
        if len(fore_total):
            fore_total.index = pidx

except Exception as e:
    st.error(f"Forecast error: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ═════════════════════════════════════════════════════════════════════════
# Historical series (levels)
# ═════════════════════════════════════════════════════════════════════════
y_name     = None
hist_level = None

if is_mm and bundle:
    y_name     = bundle["models"][head]["spec"]["y"]
    hist_level = np.exp(df_hist[y_name])
    hist_level.name = "Historical"
    total_hist = sum(np.exp(df_hist[f"log_{h}"]) for h in ["dt", "gst", "fed", "customs"])
elif not is_mm:
    log_col = f"log_{head}"
    if log_col in df_raw.columns:
        hist_level = np.exp(df_raw[log_col].dropna())
    total_hist = sum(
        np.exp(df_raw[f"log_{h}"].dropna())
        for h in ["dt", "gst", "fed", "customs"]
        if f"log_{h}" in df_raw.columns
    )

# ═════════════════════════════════════════════════════════════════════════
# Tabs  (unchanged from original)
# ═════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Forecast Plots",
    "🎯 Forecast Accuracy",
    "📋 Model Summary",
    "🔬 Model Diagnostics",
    "📖 Model Guide",
    "📊 Data Preview",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Forecast Plots  (unchanged)
# ─────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Total Tax Revenue Projection")
    if fore_total is not None and len(fore_total):
        fig_total = forecast_plot(total_hist, fore_total, "Total Tax Revenue (Sum of Heads)", "PKR Million")
        st.plotly_chart(fig_total, use_container_width=True)
        st.dataframe(
            forecast_table(fore_total).style.format({f"Forecast (PKR Billion)": "{:,.2f}"}),
            use_container_width=True,
        )

    st.markdown("---")

    st.subheader(f"{TAX_LABELS[head]} — {default_model_label}")
    if fore_head is not None and hist_level is not None and len(fore_head):
        fig_head = forecast_plot(
            hist_level, fore_head,
            f"{TAX_LABELS[head]} Forecast ({default_model_label})",
        )
        st.plotly_chart(fig_head, use_container_width=True)
        st.dataframe(
            forecast_table(fore_head).style.format({f"Forecast (PKR Billion)": "{:,.2f}"}),
            use_container_width=True,
        )

    if fore_head is not None and fore_total is not None:
        render_benchmark(buoy_data, fore_head, fore_total, head)

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — Forecast Accuracy  (unchanged)
# ─────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Forecast Accuracy")
    st.write("Expanding-window backtest with recursive multi-step forecasting.")

    rows_all = []

    if perf is not None:
        show = perf.copy()
        sub  = show[show["tax_head"] == head].copy()
        for _, r in sub.iterrows():
            rows_all.append({
                "Model":     r["model"].upper(),
                "h1 sMAPE%": r.get("h1_smape", r.get("mae_pct", None)),
                "h3 sMAPE%": r.get("h3_smape", None),
                "h5 sMAPE%": r.get("h5_smape", None),
                "RMSE%":     r.get("rmse_pct", None),
                "Bias%":     r.get("bias_pct", None),
                "Stability": r.get("stability", None),
                "n_test":    int(r.get("n_test", 0)),
            })

    pipeline = st.session_state.get("dyn_pipeline")
    if pipeline and hasattr(pipeline, "leaderboard") and pipeline.leaderboard:
        lb = pd.DataFrame(pipeline.leaderboard)
        dsm_rows = lb[
            (lb["Tax Head"] == head.upper()) &
            (lb["Type"] == "Policy")
        ].copy()
        if not dsm_rows.empty:
            sort_col = "h1_sMAPE%" if "h1_sMAPE%" in dsm_rows.columns else "sMAPE%"
            if sort_col in dsm_rows.columns:
                dsm_rows = (
                    dsm_rows
                    .sort_values(sort_col, ascending=True)
                    .drop_duplicates(subset=["Model"], keep="first")
                )
            for _, row in dsm_rows.iterrows():
                rows_all.append({
                    "Model":     f"DSM ({row['Model']})",
                    "h1 sMAPE%": row.get("h1_sMAPE%", row.get("sMAPE%", None)),
                    "h3 sMAPE%": row.get("h3_sMAPE%", None),
                    "h5 sMAPE%": row.get("h5_sMAPE%", None),
                    "RMSE%":     row.get("RMSE%", row.get("WAPE%", None)),
                    "Bias%":     row.get("Bias%", None),
                    "Stability": row.get("Stability", None),
                    "n_test":    row.get("n_test", 8),
                })

    if rows_all:
        keep_cols = ["Model", "h1 sMAPE%", "h3 sMAPE%", "RMSE%", "n_test"]
        out_df = pd.DataFrame(rows_all).sort_values("h1 sMAPE%", na_position="last")
        out_df = out_df[[c for c in keep_cols if c in out_df.columns]]
        fmt = {
            "h1 sMAPE%": "{:.2f}%", "h3 sMAPE%": "{:.2f}%",
            "RMSE%": "{:.2f}%",     "n_test":     "{:.0f}",
        }
        st.dataframe(out_df.style.format(fmt, na_rep="—"), use_container_width=True)
        st.markdown("""
**Metric Guide:**
| Metric | Meaning |
|--------|---------|
| **h1 sMAPE%** | 1-step ahead symmetric MAPE (immediate accuracy) |
| **h3 sMAPE%** | 3-step recursive sMAPE (medium-horizon, uses predicted lags) |
| **RMSE%** | Root-mean-square error as % of mean actual |
""")
    else:
        st.info("No accuracy metrics available. Load multi-model bundle or run DSM pipeline.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — Model Summary  (unchanged)
# ─────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Model Summary")
    st.write(f"**{TAX_LABELS[head]}** · Engine: **{default_model_label}**")

    if is_mm and bundle:
        head_bundle = bundle["models"][head]
        if chosen == "ardl":
            res = head_bundle["ardl"]["res"]
            st.markdown("### Coefficients")
            st.dataframe(
                coef_table_ardl(res).style.format({"coef": "{:.4f}", "std_err": "{:.4f}", "p": "{:.4f}"}),
                use_container_width=True,
            )
            vals    = res.params
            y_n     = head_bundle["spec"]["y"]
            rho_sum = sum(vals[vals.index.str.startswith(f"{y_n}.L")])
            denom   = 1.0 - rho_sum
            lr_rows = []
            for xc in head_bundle["spec"]["x"]:
                gs = sum(vals[vals.index.str.startswith(f"{xc}.L")])
                lr_rows.append({"variable": xc, "elasticity": gs / denom if abs(denom) > 1e-4 else 0})
            st.markdown("### Implied Long-Run Elasticities (ECM)")
            st.markdown(f"**Error Correction Speed (α):** `{rho_sum - 1.0:.4f}`")
            st.dataframe(pd.DataFrame(lr_rows).style.format({"elasticity": "{:.3f}"}), use_container_width=True)
            st.markdown("### Full Output")
            st.text(res.summary().as_text())

        elif chosen == "arimax":
            res = head_bundle["arimax"]["res"]
            st.markdown("### Coefficients")
            st.dataframe(
                coef_table_arimax(res).style.format({"coef": "{:.4f}", "std_err": "{:.4f}", "z": "{:.2f}", "p": "{:.4f}"}),
                use_container_width=True,
            )
            st.markdown("### Full Output")
            st.text(res.summary().as_text())

        elif chosen == "enet":
            st.markdown("### ElasticNet Coefficients")
            st.dataframe(
                coef_table_enet(head_bundle).style.format({"coef": "{:.6f}"}),
                use_container_width=True,
            )
            st.markdown("### Settings")
            st.json(head_bundle["enet"].get("params", {}))
    else:
        pipeline = st.session_state.get("dyn_pipeline")
        if pipeline and head in pipeline.best_models:
            m = pipeline.best_models[head].get("policy_winner")

            VAR_GLOSSARY = {
                "log_imports_hat": (
                    "Predicted Imports (log)",
                    "Total predicted imports, estimated in Stage 1 from GDP, exchange rate, "
                    "policy rate and inflation. The '_hat' suffix means this is a **model-predicted** "
                    "value, not the raw observed imports."
                ),
                "log_dutiable_imports_hat": (
                    "Predicted Dutiable Imports (log)",
                    "Predicted dutiable imports (the taxable subset of total imports).",
                ),
                "log_gdp_hat": (
                    "Predicted GDP (log)",
                    "Predicted nominal GDP from Stage 1 channel equations.",
                ),
                "log_lsm_hat": (
                    "Predicted Large-Scale Manufacturing (log)",
                    "Predicted LSM index from Stage 1.",
                ),
                "log_consumption_hat": (
                    "Predicted Private Consumption (log)",
                    "Predicted private consumption expenditure from Stage 1.",
                ),
                "inflation": (
                    "Inflation Rate (%)",
                    "Consumer price inflation.",
                ),
                "log_exrate": (
                    "Exchange Rate (log, PKR/USD)",
                    "Log of the PKR/USD exchange rate.",
                ),
                "policy rate": (
                    "SBP Policy Rate (%)",
                    "State Bank of Pakistan benchmark interest rate.",
                ),
            }

            HEAD_EXPLANATIONS = {
                "customs": (
                    "Customs Duty is modelled as a function of **predicted dutiable imports** and/or "
                    "**total imports** (from Stage 1), plus the exchange rate and inflation."
                ),
                "dt": (
                    "Direct Tax (Income Tax) is modelled as a function of **predicted GDP** and/or "
                    "**LSM** (from Stage 1), plus inflation."
                ),
                "gst": (
                    "Sales Tax (GST) is modelled as a function of **predicted consumption** and "
                    "**predicted imports** (from Stage 1), plus inflation and exchange rate."
                ),
                "fed": (
                    "Federal Excise Duty is modelled as a function of **predicted LSM** "
                    "(from Stage 1) plus inflation."
                ),
            }

            if m and m.elasticities:
                st.markdown("### Policy Elasticities")
                st.markdown(f"**Winning Model:** {m.name}")
                c1, c2 = st.columns(2)

                c1.markdown("#### Short-Run")
                sr = m.elasticities.get("Short-Run", {})
                if sr:
                    sr_rows = []
                    for k, v in sr.items():
                        label, _ = VAR_GLOSSARY.get(k, (k, ""))
                        sr_rows.append({"Variable": k, "Description": label, "Elasticity": v})
                    c1.dataframe(
                        pd.DataFrame(sr_rows).style.format({"Elasticity": "{:.4f}"}),
                        use_container_width=True,
                    )

                c2.markdown("#### Long-Run")
                lr = m.elasticities.get("Long-Run", {})
                if lr:
                    lr_rows = []
                    for k, v in lr.items():
                        label, _ = VAR_GLOSSARY.get(k, (k, ""))
                        lr_rows.append({"Variable": k, "Description": label, "Elasticity": v})
                    c2.dataframe(
                        pd.DataFrame(lr_rows).style.format({"Elasticity": "{:.4f}"}),
                        use_container_width=True,
                    )
            else:
                st.info("No policy elasticities available for this tax head.")
        else:
            st.info("Run the Dynamic Pipeline first to see DSM model summaries.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — Model Diagnostics  (unchanged)
# ─────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Model Diagnostics")
    st.write(f"**{TAX_LABELS[head]}** · Engine: **{default_model_label}**")

    if is_mm and bundle:
        head_bundle = bundle["models"][head]
        if chosen == "ardl":
            res  = head_bundle["ardl"]["res"]
            diag = diagnostics_ardl(res)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Durbin–Watson",   f"{diag['durbin_watson']:.2f}")
            c2.metric("Ljung–Box p",     "NA" if diag["ljung_box_p"]    is None else f"{diag['ljung_box_p']:.3f}")
            c3.metric("Jarque–Bera p",   "NA" if diag["jb_full_p"]      is None else f"{diag['jb_full_p']:.3f}")
            c4.metric("JB trimmed p",    "NA" if diag["jb_trim_p"]      is None else f"{diag['jb_trim_p']:.3f}")
            c5.metric("Breusch–Pagan p", "NA" if diag["breusch_pagan_p"] is None else f"{diag['breusch_pagan_p']:.3f}")
            st.caption("'JB trimmed p' excludes first residual to avoid burn-in artifacts.")

            resid = pd.Series(res.resid).dropna()
            ridx  = df_hist.index[-len(resid):]
            resid.index = ridx
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=resid.index.to_timestamp(), y=resid.values, mode="lines+markers", name="Residuals"))
            fig_r.update_layout(xaxis_title="Year", yaxis_title="Residual", template="plotly_white")
            st.plotly_chart(fig_r, use_container_width=True)

        elif chosen == "arimax":
            res  = head_bundle["arimax"]["res"]
            diag = diagnostics_arimax(res)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("AIC",           f"{diag['aic']:.1f}")
            c2.metric("Durbin–Watson", f"{diag['durbin_watson']:.2f}")
            c3.metric("Ljung–Box p",   "NA" if diag["ljung_box_p"] is None else f"{diag['ljung_box_p']:.3f}")
            c4.metric("Jarque–Bera p", "NA" if diag["jb_full_p"]  is None else f"{diag['jb_full_p']:.3f}")
            c5.metric("JB trimmed p",  "NA" if diag["jb_trim_p"]  is None else f"{diag['jb_trim_p']:.3f}")

            resid = pd.Series(res.resid).dropna()
            ridx  = df_hist.index[-len(resid):]
            resid.index = ridx
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=resid.index.to_timestamp(), y=resid.values, mode="lines+markers", name="Residuals"))
            fig_r.update_layout(xaxis_title="Year", yaxis_title="Residual", template="plotly_white")
            st.plotly_chart(fig_r, use_container_width=True)

        elif chosen == "enet":
            st.write("ElasticNet diagnostics focus on stability and backtest metrics.")
            st.dataframe(
                coef_table_enet(head_bundle).head(15).style.format({"coef": "{:.6f}"}),
                use_container_width=True,
            )
            st.markdown("### Backtest")
            st.dataframe(
                perf[perf["tax_head"] == head]
                .sort_values("mae_pct")[["model", "mae_pct", "rmse_pct", "n_test"]]
                .style.format({"mae_pct": "{:.2f}%", "rmse_pct": "{:.2f}%"}),
                use_container_width=True,
            )
    else:
        pipeline = st.session_state.get("dyn_pipeline")
        if pipeline and pipeline.leaderboard:
            st.write("**DSM Tournament Diagnostics** — Expanding-window backtest with recursive forecasting")
            lb = pd.DataFrame(pipeline.leaderboard)
            diag_cols = [c for c in lb.columns if c != "obj"]
            head_lb   = lb[lb["Tax Head"] == head.upper()][diag_cols].copy()
            if not head_lb.empty:
                sort_col_diag = "h1_sMAPE%" if "h1_sMAPE%" in head_lb.columns else "sMAPE%"
                if sort_col_diag in head_lb.columns:
                    head_lb = (
                        head_lb
                        .sort_values(sort_col_diag, ascending=True)
                        .drop_duplicates(subset=["Model"], keep="first")
                    )
                show_cols = [c for c in [
                    "Model", "h1_sMAPE%", "h3_sMAPE%", "RMSE%", "n_test",
                ] if c in head_lb.columns]
                if not show_cols:
                    show_cols = [c for c in diag_cols if c in head_lb.columns]
                st.dataframe(head_lb[show_cols], use_container_width=True)

                n_values = head_lb["n_test"].unique() if "n_test" in head_lb.columns else []
                if len(n_values) == 1:
                    st.success(f"✅ Window integrity verified — all models used {int(n_values[0])} identical test origins.")
                elif len(n_values) > 1:
                    st.warning(f"⚠️ Test window mismatch detected: {sorted(n_values)}")
            else:
                st.info(f"No leaderboard entries for {TAX_LABELS[head]}.")
        else:
            st.info("Run the Dynamic Pipeline to see diagnostics.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 5 — Model Guide  (unchanged)
# ─────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Model Guide")
    st.markdown(
        "Technical methodology for each forecasting model employed in this system. "
        "All models operate on **log-transformed, annual** macro-fiscal data for Pakistan (FY1996–FY2026). "
        "The DSM engine uses a **two-stage structural identification** strategy; "
        "the Multi-Model engine runs ARDL, ARIMAX, and Elastic Net in parallel."
    )
    st.markdown("---")

    with st.expander("🏗️ DSM — Two-Stage Dynamic Structural Model (Overview)", expanded=True):
        st.markdown("""
### Motivation
Direct regression of tax revenue on macro aggregates (GDP, imports) suffers from **simultaneity bias**:
tax policy itself affects GDP and imports, creating a feedback loop that violates the OLS exogeneity assumption.
The DSM resolves this via a **two-stage instrumental-variable-style** approach analogous to 2SLS,
where Stage-1 fitted values serve as instruments for the endogenous regressors in Stage-2.

---

### Stage 1 — Channel Equations (Structural Macro Block)

Four **channel equations** are estimated by ARDL on the **full macro panel** to recover
the *exogenous* component of each intermediate variable:

| Channel | Equation | Rationale |
|---------|----------|-----------| 
| **Imports** | `log_imports = f(log_gdp, log_exrate, policy_rate, inflation)` | Demand-side import function; driven by income and price effects |
| **Dutiable Imports** | `log_dutiable_imports = f(log_imports, log_exrate, policy_rate, inflation)` | Composition of import basket subject to tariff |
| **LSM (Large-Scale Manufacturing)** | `log_lsm = f(log_gdp, policy_rate, inflation)` | Output proxy for domestic value-added tax base |
| **Consumption** | `log_consumption = f(log_gdp, inflation, policy_rate)` | Household absorption; primary GST/sales tax base |

Each channel equation is fitted using `ardl_select_order()` with AIC-optimal lag selection,
subject to the constraint `p, q ≤ n/8` to preserve degrees of freedom on the ~30-observation sample.

### Stage 2 — Tax Revenue Equations (Structural Tax Block)

| Tax Head | Dependent Variable | Stage-2 Structural Regressor |
|----------|--------------------|-------------------------------|
| **Income Tax (DT)** | `log_dt` | `log_gdp_hat`, `log_lsm_hat` |
| **GST** | `log_gst` | `log_consumption_hat`, `log_gdp_hat` |
| **FED** | `log_fed` | `log_lsm_hat`, `log_gdp_hat` |
| **Customs** | `log_customs` | `log_dutiable_imports_hat`, `log_imports_hat` |

**Three candidate Stage-2 estimators** compete for each tax head in an expanding-window backtest:
ARDL, ARIMAX, and DynamicLag. The tournament winner (lowest h1 sMAPE over 8 rolling origins)
is declared the **policy winner** and used for scenario-based projection.
""")

    with st.expander("📐 ARDL — AutoRegressive Distributed Lag (Pesaran et al., 2001)"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Reference:** Pesaran, Shin & Smith (2001), *J. of Applied Econometrics*")
            st.info(
                "Bounds-testing approach to cointegration. Applicable to regressors that are "
                "I(0), I(1), or fractionally integrated — avoids pre-testing bias inherent in "
                "Engle-Granger or Johansen procedures."
            )
            st.markdown("**Key properties**")
            st.success(
                "Consistent under mixed integration orders · "
                "Efficient with T ≈ 30–80 · "
                "Delivers SR dynamics + LR multipliers in one step"
            )
        with c2:
            st.markdown(r"""
**Specification:**

$$\log T_t = c + \sum_{i=1}^{p} \rho_i \log T_{t-i} + \sum_{j=0}^{q} \gamma_j \log \hat{X}_{t-j} + \delta D_t + \varepsilon_t$$

**Long-Run Multiplier (LRM):**
$$\theta = \frac{\sum_{j=0}^{q} \gamma_j}{1 - \sum_{i=1}^{p} \rho_i}$$

Interpretation: A permanent 1% rise in the tax base leads to a $\theta$% permanent change in revenue.

**Short-Run Coefficient:** $\gamma_0$ — the contemporaneous elasticity within the fiscal year.

**ARDL-ECM Reparameterisation (for diagnostics):**
$$\Delta \log T_t = \alpha(\log T_{t-1} - \theta \log \hat{X}_{t-1}) + \text{SR terms} + \varepsilon_t$$

$\alpha < 0$ confirms error correction — revenue converges back to its structural level after a shock.
""")

    with st.expander("📡 ARIMAX — ARIMA with Exogenous Regressors (Box & Jenkins, 1976)"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Reference:** Box, Jenkins & Reinsel (1976); Hamilton (1994) *Ch. 4*")
            st.info(
                "Extensions of the Wold decomposition theorem to include deterministic exogenous inputs. "
                "Implemented here as SARIMAX(1,1,0) via statsmodels MLE — a fixed order chosen for "
                "parsimony given the ~30-observation annual sample."
            )
            st.markdown("**Key properties**")
            st.success(
                "Handles unit-root non-stationarity via differencing · "
                "MLE estimation under Gaussian innovations · "
                "Conditional on past revenues + exogenous path"
            )
        with c2:
            st.markdown(r"""
**Specification (ARMAX(1,1,0) in differences):**

$$\Delta \log T_t = \mu + \phi \Delta \log T_{t-1} + \sum_k \beta_k X_{kt} + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)$$

**Level forecast reconstruction:**

$$\log \hat{T}_{t+h} = \log T_t + \sum_{s=1}^{h} \Delta \log \hat{T}_{t+s}$$

Recursive h-step forecasts accumulate first-difference predictions back to log-levels, then exponentiate.
""")

    with st.expander("🔁 Dynamic Lag — Partial Adjustment / Koyck Model"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Reference:** Koyck (1954); Nerlove (1958) partial adjustment model")
            st.info(
                "Reduces the infinite distributed lag to a single lagged dependent variable "
                "under geometric decay assumption."
            )
            st.markdown("**Key properties**")
            st.success(
                "Parsimonious (2–3 free parameters) · "
                "OLS closed-form solution · "
                "Directly interpretable SR and LR elasticities"
            )
        with c2:
            st.markdown(r"""
**Estimable Koyck equation:**

$$\log T_t = c + \rho \log T_{t-1} + \beta^* \log \hat{X}_t + \delta^* D_t + u_t$$

| Horizon | Formula | Interpretation |
|---------|---------|----------------|
| Short-Run | $\hat{\beta}^*$ | Elasticity of revenue to base within the fiscal year |
| Long-Run | $\hat{\beta}^* / (1 - \hat{\rho})$ | Permanent elasticity after full convergence |
""")

    with st.expander("🧮 Elastic Net — Penalised Regression (Zou & Hastie, 2005)"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Reference:** Zou & Hastie (2005), *JRSS-B 67(2)*; Tibshirani (1996) LASSO")
            st.info(
                "Regularised regression combining L1 (LASSO) and L2 (Ridge) penalties. "
                "Operates on a rich feature matrix: {Tₜ₋₁, X̂ₜ, X̂ₜ₋₁, inflation, exrate, policy_rate, regime}."
            )
            st.markdown("**Key properties**")
            st.success(
                "Consistent variable selection under multicollinearity · "
                "Groups correlated regressors (unlike LASSO) · "
                "Hyper-parameters tuned by rolling-origin CV"
            )

        with c2:
            st.markdown(r"""
**Objective function (penalised OLS):**

$$\hat{\beta} = \arg\min_{\beta} \left[ \frac{1}{2T} \|y - X\beta\|_2^2 + \alpha \left( \frac{1-\rho_{L1}}{2} \|\beta\|_2^2 + \rho_{L1} \|\beta\|_1 \right) \right]$$

**Feature matrix $X$** includes: `{log_T_{t-1}, log_imports_hat, log_imports_hat_{t-1}, log_exrate, inflation, policy_rate, regime}`
""")


    with st.expander("📊 Tax Buoyancy — Log-Log Elasticity Benchmark (Prest, 1962)"):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Reference:** Prest (1962); Choudhry (1979); IMF Fiscal Monitor")
            st.info(
                "Distinguishes *buoyancy* (observed revenue-GDP elasticity, including discretionary changes) "
                "from *elasticity* (automatic revenue response holding policy constant)."
            )
            st.markdown("**Benchmark targets**")
            st.success("Buoyancy > 1 → tax system is progressive relative to GDP · < 1 → structural revenue gap")
        with c2:
            st.markdown(r"""
**Estimating equation:**

$$\log T_t = \alpha + \beta \log Y_t + \varepsilon_t$$

where $Y_t$ is nominal GDP. OLS in log-log gives $\hat{\beta}$ = **overall tax buoyancy**.

**Pakistan context:** FBR revenue buoyancy has historically clustered around 1.0–1.3.
""")

# ─────────────────────────────────────────────────────────────────────────
# TAB 6 — Data Preview  (NOW EDITABLE)
# ─────────────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Historical Data (Model Space)")

    # ── Explanation ───────────────────────────────────────────────────────
    st.info(
        "👁️ **Historical data table is read-only.**  \n"
        "- The main table shows the data currently being used for all model calculations.  \n"
        "- To change the 2026 estimates, use the **Revised Estimates for 2026** section below.  \n"
        "- Check **'Use Revised Estimates'** and click **'Apply & Rerun'** to update the pipeline."
    )

    # ── Build the read-only frame ──────────────────────────────────────────
    _log_cols  = [c for c in df_hist.columns if c.startswith("log_")]
    _edit_cols = [c for c in df_hist.columns if c not in _log_cols]

    st.dataframe(df_hist[_edit_cols].reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.subheader("Revised Estimates for 2026")
    
    use_revised = st.checkbox(
        "☑️ Use Revised Estimates for 2026", 
        value=st.session_state.get("use_revised_2026", False),
        help="If checked, the values below will replace the 2026 historical row in all calculations."
    )

    # Load default 2026 data from file to seed the revised editor
    if "revised_2026_df" not in st.session_state:
        _base = prepare_transforms(load_tax_data())
        if 2026 in _base.index.year:
            _def_2026 = _base[_base.index.year == 2026][_edit_cols].reset_index(drop=True)
        else:
            _def_2026 = pd.DataFrame([{"year_end": 2026}])
            for c in _edit_cols:
                if c != "year_end": _def_2026[c] = 0.0
        st.session_state["revised_2026_df"] = _def_2026

    # Column config: make year_end read-only
    _col_cfg: dict = {}
    for col in st.session_state["revised_2026_df"].columns:
        if col == "year_end":
            _col_cfg[col] = st.column_config.NumberColumn("Year End", disabled=True, format="%d")
        elif col in ("covid", "regime", "step_2024", "dummy_2024", "dummy_2025", "dummy_1995", "dummy_1996", "dummy_2002", "dummy_2003"):
            _col_cfg[col] = st.column_config.CheckboxColumn(col)
        else:
            _col_cfg[col] = st.column_config.NumberColumn(col, format="%.2f")

    edited_2026_df = st.data_editor(
        st.session_state["revised_2026_df"],
        num_rows="fixed",
        use_container_width=True,
        column_config=_col_cfg,
        key="rev_2026_editor",
    )

    col_apply, col_reset = st.columns([1, 5])

    with col_apply:
        apply_clicked = st.button("✅ Apply & Rerun", type="primary")

    with col_reset:
        if st.button("↩️ Reset to Original File"):
            st.session_state.pop("user_df", None)
            st.session_state.pop("revised_2026_df", None)
            st.session_state.pop("use_revised_2026", None)
            st.session_state.pop("dyn_pipeline", None)
            st.rerun()

    # ── Apply logic ───────────────────────────────────────────────────────
    if apply_clicked:
        st.session_state["use_revised_2026"] = use_revised
        st.session_state["revised_2026_df"]  = edited_2026_df.copy()
        
        # Always start with clean data from disk
        _work = prepare_transforms(load_tax_data())
        
        if use_revised:
            # Overwrite the 2026 row with user's revised values
            _rev_row = edited_2026_df.iloc[0].to_dict()
            for col, val in _rev_row.items():
                if col in _work.columns and col != "year_end":
                    _work.loc[_work.index.year == 2026, col] = val
            
            # Re-run prepare_transforms to ensure logs match new levels
            _work = prepare_transforms(_work)

        st.session_state["user_df"] = _work
        
        # Clear caches so pipeline rebuilds with the updated data
        mm.get_cached_forecast.clear()
        st.session_state.pop("dyn_pipeline", None)

        st.success("✅ Settings applied! Re-running pipeline...")
        st.rerun()

    st.markdown("---")

    # ── Generated future exog (read-only, as before) ──────────────────────
    st.subheader("Generated Future Exog (for selected system)")
    if is_mm and exog_future is not None:
        st.dataframe(exog_future, use_container_width=True)
    elif not is_mm and dyn_results:
        st.write("Dynamic engine generates exog internally via channel equations.")
        st.json({
            k: (v if not isinstance(v, (list, pd.Series)) else list(v))
            for k, v in targets.items()
        })

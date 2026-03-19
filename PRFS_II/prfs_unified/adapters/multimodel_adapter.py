"""
prfs_unified/adapters/multimodel_adapter.py
============================================
Wraps the multi-model forecasting logic from app_multimodel_v2.py.
Uses the mapping layer to derive exogenous futures from dynamic sliders.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from prfs_unified.mapping import build_multimodel_future_exog_from_dynamic


# ── Performance helpers ──────────────────────────────────────────────────
def perf_table(meta: Dict) -> pd.DataFrame:
    return pd.DataFrame(meta["performance"])


def best_model_by_mape(perf: pd.DataFrame, head: str) -> str:
    sub = perf[perf["tax_head"] == head].sort_values("mae_pct")
    return str(sub.iloc[0]["model"]) if len(sub) else "ardl"


# ── Cached forecast ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_cached_forecast(
    model_kind: str,
    head: str,
    horizon: int,
    exog_future_json: str,
    n_sims: int = 500,
    data_version: str = "",   # ← cache-bust key: changes whenever user edits data
    _bundle=None,
    _df_hist=None,
):
    """Run a single-head forecast using the multi-model bundle."""
    bundle_head = _bundle["models"][head]
    exog_future = pd.read_json(exog_future_json).sort_index()
    exog_future.index = pd.PeriodIndex(exog_future.index, freq="Y")

    y_name = bundle_head["spec"]["y"]
    spec_x = bundle_head["spec"]["x"]   # e.g. ['log_gdp', 'inflation']

    # ── Helper: re-anchor a pre-trained result to df_hist ─────────────────
    # Keeps all coefficients exactly as trained (refit=False).
    # Only updates the "last observed values" so forecast continues from
    # wherever the user's edited data ends — not the original 2026 endpoint.
    def _apply_to_user_data(res, endog_col, exog_cols):
        """
        Returns res.apply(new_endog, new_exog, refit=False).
        Falls back to original res if anything goes wrong.
        """
        if _df_hist is None:
            return res
        try:
            if endog_col not in _df_hist.columns:
                return res
            y_new = _df_hist[endog_col].dropna()
            if len(y_new) < 5:
                return res
            # Build exog aligned to y_new index
            x_avail = [c for c in exog_cols if c in _df_hist.columns]
            if not x_avail:
                return res
            x_new = _df_hist[x_avail].reindex(y_new.index).ffill().bfill()
            # Drop any rows where either y or x is NaN
            valid = y_new.notna() & x_new.notna().all(axis=1)
            y_new = y_new[valid]
            x_new = x_new[valid]
            if len(y_new) < 5:
                return res
            return res.apply(y_new.values, exog=x_new.values, refit=False)
        except Exception:
            return res   # silent fallback — original behaviour preserved

    # ── ARDL ──────────────────────────────────────────────────────────
    if model_kind == "ardl":
        res = bundle_head["ardl"]["res"]
        # Re-anchor to user's current data so forecast starts from
        # the user's last observed value, not the original training endpoint.
        res = _apply_to_user_data(res, y_name, spec_x)
        yhat_log = res.forecast(steps=horizon, exog=exog_future)
        resid = res.resid.dropna().values
        ar_params = [v for k, v in res.params.items() if k.startswith(y_name + ".L")]

        sims = []
        for _ in range(n_sims):
            noise = np.random.choice(resid, size=horizon, replace=True)
            path = np.zeros(horizon)
            for i in range(horizon):
                ar = sum(c * path[i - (p + 1)] for p, c in enumerate(ar_params) if i - (p + 1) >= 0)
                path[i] = ar + noise[i]
            sims.append(np.exp(yhat_log.values + path))

        sims = np.array(sims)
        return pd.DataFrame(
            {
                "yhat": np.exp(yhat_log.values),
                "lo80": np.quantile(sims, 0.10, axis=0),
                "hi80": np.quantile(sims, 0.90, axis=0),
                "lo95": np.quantile(sims, 0.025, axis=0),
                "hi95": np.quantile(sims, 0.975, axis=0),
            },
            index=exog_future.index,
        )

    # ── ARIMAX ────────────────────────────────────────────────────────
    if model_kind == "arimax":
        res = bundle_head["arimax"]["res"]
        # Re-anchor to user's current data (same reason as ARDL above).
        res = _apply_to_user_data(res, y_name, spec_x)
        fc = res.get_forecast(steps=horizon, exog=exog_future)
        yhat_log = fc.predicted_mean
        ci80 = fc.conf_int(alpha=0.2)
        ci95 = fc.conf_int(alpha=0.05)
        return pd.DataFrame(
            {
                "yhat": np.exp(yhat_log.values),
                "lo80": np.exp(ci80.iloc[:, 0].values),
                "hi80": np.exp(ci80.iloc[:, 1].values),
                "lo95": np.exp(ci95.iloc[:, 0].values),
                "hi95": np.exp(ci95.iloc[:, 1].values),
            },
            index=exog_future.index,
        )

    # ── ElasticNet ────────────────────────────────────────────────────
    if model_kind == "enet":
        model = bundle_head["enet"]["model"]
        feat_cols = bundle_head["enet"]["feature_cols"]
        train_resids = bundle_head["enet"]["residuals"]

        work = pd.concat([_df_hist, exog_future], axis=0).ffill()
        preds_log = []
        for t in exog_future.index:
            row = {}
            for c in feat_cols:
                if c.endswith("_L0"):
                    row[c] = work.loc[t, c[:-3]] if c[:-3] in work.columns else 0.0
                elif "_L" in c:
                    parts = c.rsplit("_L", 1)
                    base = parts[0]
                    lag = int(parts[1])
                    row[c] = work.shift(lag).loc[t, base] if base in work.columns else 0.0
                else:
                    row[c] = work.loc[t, c] if c in work.columns else 0.0
            row_df = pd.DataFrame([row], columns=feat_cols).fillna(0)
            yhat_l = float(model.predict(row_df)[0])
            preds_log.append(yhat_l)
            work.loc[t, y_name] = yhat_l

        sim_paths = []
        for _ in range(n_sims):
            path = []
            work_sim = _df_hist.copy()
            for i, t in enumerate(exog_future.index):
                ex = exog_future.iloc[i : i + 1]
                row = {}
                for c in feat_cols:
                    if c.endswith("_L0"):
                        base = c[:-3]
                        row[c] = ex[base].iloc[0] if base in ex.columns else (work_sim[base].iloc[-1] if base in work_sim.columns else 0.0)
                    elif "_L" in c:
                        parts = c.rsplit("_L", 1)
                        base, lag = parts[0], int(parts[1])
                        try:
                            row[c] = work_sim[base].iloc[-lag] if base in work_sim.columns else 0.0
                        except (IndexError, KeyError):
                            row[c] = 0.0
                    else:
                        row[c] = work_sim[c].iloc[-1] if c in work_sim.columns else 0.0
                row_df = pd.DataFrame([row], columns=feat_cols).fillna(0)
                noisy = float(model.predict(row_df)[0]) + np.random.choice(train_resids)
                path.append(math.exp(noisy))
                new_row = pd.Series(index=_df_hist.columns, dtype=float)
                for c2 in exog_future.columns:
                    new_row[c2] = ex[c2].iloc[0]
                new_row[y_name] = noisy
                work_sim = pd.concat([work_sim, new_row.to_frame().T], ignore_index=True)
            sim_paths.append(path)

        sim_paths = np.array(sim_paths)
        return pd.DataFrame(
            {
                "yhat": [math.exp(p) for p in preds_log],
                "lo80": np.quantile(sim_paths, 0.10, axis=0),
                "hi80": np.quantile(sim_paths, 0.90, axis=0),
                "lo95": np.quantile(sim_paths, 0.025, axis=0),
                "hi95": np.quantile(sim_paths, 0.975, axis=0),
            },
            index=exog_future.index,
        )

    st.error(f"Unknown model kind: {model_kind}")
    st.stop()


# ── Convenience: run forecast for one head via dynamic targets ──────────
def forecast_head(
    bundle, meta, df_hist,
    head: str, chosen: str, horizon: int, n_sims: int,
    targets: Dict, elasticities: Dict,
    covid_on: bool, regime_on: bool,
    data_version: str = "",
) -> pd.DataFrame:
    """Build exog from dynamic targets and run the forecast."""
    spec_x = bundle["models"][head]["spec"]["x"]
    use_univariate = (head == "fed")
    exog = build_multimodel_future_exog_from_dynamic(
        df_hist, horizon, spec_x, targets,
        covid_on=covid_on, regime_on=regime_on,
        use_univariate=use_univariate,
        elasticities=elasticities,
    )
    return get_cached_forecast(
        chosen, head, horizon, exog.to_json(), n_sims,
        data_version=data_version,
        _bundle=bundle, _df_hist=df_hist,
    ), exog


def forecast_total(
    bundle, meta, df_hist,
    chosen_model: str, horizon: int, n_sims: int,
    targets: Dict, elasticities: Dict,
    covid_on: bool, regime_on: bool,
    data_version: str = "",
) -> pd.DataFrame:
    """Sum forecasts for all sub-heads using the specified model."""
    total = None
    for h in ["customs", "dt", "fed", "gst"]:
        fore, _ = forecast_head(
            bundle, meta, df_hist, h, chosen_model, horizon, n_sims,
            targets, elasticities, covid_on, regime_on,
            data_version=data_version,
        )
        total = fore if total is None else total + fore
    return total

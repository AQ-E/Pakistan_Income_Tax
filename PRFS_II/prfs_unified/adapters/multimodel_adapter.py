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
    spec_x = bundle_head["spec"]["x"]

    # ── ARDL — manual recursive forecast seeded from df_hist ──────────────
    # Why manual instead of res.forecast()?
    # res.forecast() always starts from the model's ORIGINAL training endpoint
    # (stored internally in the fitted object). If the user edits e.g. the
    # 2026 DT value, res.forecast() never sees it.
    # Instead we:
    #   1. Keep ALL pre-trained coefficients (res.params) unchanged.
    #   2. Extract the last p observed y values from df_hist as the AR seed.
    #   3. Extract historical exog (for any lag-1 exog terms) from df_hist.
    #   4. Step forward recursively using future exog from exog_future.
    # Change any historical value → AR seed changes → forecast changes. ✓
    if model_kind == "ardl":
        res    = bundle_head["ardl"]["res"]
        params = dict(res.params)

        # Parse param names into AR lags {lag: coef} and exog lags {col: {lag: coef}}
        ar_lags   = {}   # {1: 0.72, 2: 0.10, ...}
        exog_lags = {}   # {'log_gdp': {0: 0.89, 1: -0.12}, ...}
        const     = params.get("const", params.get("intercept", 0.0))

        for k, v in params.items():
            if k in ("const", "intercept"):
                continue
            if k.startswith(y_name + ".L"):
                lag = int(k[len(y_name) + 2:])
                ar_lags[lag] = v
            elif ".L" in k:
                col, lag_str = k.rsplit(".L", 1)
                lag = int(lag_str)
                exog_lags.setdefault(col, {})[lag] = v
            else:
                # No lag suffix → treat as contemporaneous exog (lag 0)
                exog_lags.setdefault(k, {})[0] = v

        max_ar_lag = max(ar_lags.keys()) if ar_lags else 1

        # ── AR Stability Enforcement ─────────────────────────────────────────
        # If sum of AR coefficients >= 1, the model is explosive (unit-root or
        # above). Scale them down to 0.95 to ensure mean-reversion at forecast
        # time, effectively bounding the runaway growth projections.
        ar_sum = sum(ar_lags.values())
        if ar_sum >= 0.99:
            scale = 0.95 / ar_sum
            ar_lags = {k: v * scale for k, v in ar_lags.items()}

        # Seed y history from df_hist (user-edited values)
        if _df_hist is not None and y_name in _df_hist.columns:
            y_hist = list(_df_hist[y_name].dropna().values)
        else:
            y_hist = list(res.fittedvalues.dropna().values)

        # Historical exog series (needed for any lag-1 exog terms at step 1)
        hist_exog = {}
        if _df_hist is not None:
            for col in exog_lags:
                if col in _df_hist.columns:
                    hist_exog[col] = list(_df_hist[col].dropna().values)

        # Recursive multi-step forecast
        yhat_log = []
        future_y = list(y_hist)   # grows with each forecasted step

        for h in range(horizon):
            exog_row = exog_future.iloc[h] if h < len(exog_future) else exog_future.iloc[-1]
            y_pred   = const

            # AR contribution — uses actual df_hist values for seeds
            for lag, coef in ar_lags.items():
                idx = -(lag)
                if abs(idx) <= len(future_y):
                    y_pred += coef * future_y[idx]

            # Exog contribution
            for col, lag_coefs in exog_lags.items():
                for lag, coef in lag_coefs.items():
                    if lag == 0:
                        # Contemporaneous — use future exog column
                        val = float(exog_row[col]) if col in exog_row.index else 0.0
                        y_pred += coef * val
                    else:
                        # Historical exog lag — go back into df_hist then exog_future
                        future_idx = h - lag   # index into exog_future (may be negative)
                        if future_idx >= 0:
                            val = float(exog_future.iloc[future_idx][col]) if col in exog_future.columns else 0.0
                        else:
                            # Still in history
                            hist_vals = hist_exog.get(col, [])
                            hist_idx  = future_idx   # negative → from end
                            val = float(hist_vals[hist_idx]) if hist_vals and abs(hist_idx) <= len(hist_vals) else 0.0
                        y_pred += coef * val

            yhat_log.append(y_pred)
            future_y.append(y_pred)

        yhat_log  = np.array(yhat_log)
        # Use parametric error variance to prevent empirical artifacts from exploding CIs
        sigma2    = float(np.var(res.resid.dropna().values[2:]))
        std_err   = float(np.sqrt(max(sigma2, 1e-12)))
        ar_coefs  = [v for _, v in sorted(ar_lags.items())]

        sims = []
        for _ in range(n_sims):
            noise = np.random.normal(0, std_err, size=horizon)
            path  = np.zeros(horizon)
            for i in range(horizon):
                ar = sum(c * path[i - (p + 1)] for p, c in enumerate(ar_coefs) if i - (p + 1) >= 0)
                path[i] = ar + noise[i]
            sims.append(np.exp(yhat_log + path))

        sims = np.array(sims)
        return pd.DataFrame(
            {
                "yhat": np.exp(yhat_log),
                "lo80": np.quantile(sims, 0.10, axis=0),
                "hi80": np.quantile(sims, 0.90, axis=0),
                "lo95": np.quantile(sims, 0.025, axis=0),
                "hi95": np.quantile(sims, 0.975, axis=0),
            },
            index=exog_future.index,
        )

    # ── ARIMAX — correct structural decomposition ──────────────────────────
    # statsmodels SARIMAX(1,1,0) with exog estimates:
    #   y_t = β·X_t + u_t          (level regression on exog)
    #   (1 - φL)(1 - L) u_t = c + ε_t    (ARIMA on the ERROR component)
    #
    # So the exog enters as a LEVEL regression — the differenced ARIMA part
    # applies to the residual u_t, NOT to y_t directly. My previous code put
    # β·X inside the difference equation, which caused values to explode.
    #
    # Correct forecast procedure:
    #   1. Compute u_T = y_T - β·X_T, u_{T-1} = y_{T-1} - β·X_{T-1}
    #   2. Δu_T = u_T - u_{T-1}
    #   3. For each step h:
    #        Δu_h = c + φ·Δu_{h-1}
    #        u_h  = u_{h-1} + Δu_h
    #        ŷ_h  = β·X_{future,h} + u_h
    #   4. Convert: exp(ŷ_h) → level PKR
    #
    # Any user edit to y_T or X_T in Data Preview changes u_T → forecast changes. ✓
    if model_kind == "arimax":
        res    = bundle_head["arimax"]["res"]
        params = dict(res.params)

        intercept = params.get("intercept", params.get("const", 0.0))
        phi       = params.get("ar.L1", 0.0)   # AR(1) on differenced errors
        sigma2    = params.get("sigma2", float(np.var(res.resid.dropna())))

        # Beta for each exog column (level regression coefficients)
        beta = {}
        for col in spec_x:
            beta[col] = params.get(col, 0.0)

        # ── Compute error series u_t = y_t - β·X_t from user data ─────────
        if _df_hist is not None and y_name in _df_hist.columns:
            y_series = _df_hist[y_name].dropna()
            # Regression component at each t
            reg_vals = np.zeros(len(y_series))
            for col, b in beta.items():
                if col in _df_hist.columns:
                    x_vals = _df_hist[col].reindex(y_series.index).fillna(0).values
                    reg_vals += b * x_vals
            u_series = y_series.values - reg_vals
            last_u   = float(u_series[-1])
            last_du  = float(u_series[-1] - u_series[-2]) if len(u_series) >= 2 else 0.0
        else:
            # Fallback: derive from model's internal data
            y_orig    = pd.Series(res.model.endog)
            reg_orig  = np.zeros(len(y_orig))
            if hasattr(res.model, 'exog') and res.model.exog is not None:
                reg_orig = res.model.exog @ np.array([beta.get(c, 0) for c in spec_x])
            u_orig    = y_orig.values - reg_orig
            last_u    = float(u_orig[-1])
            last_du   = float(u_orig[-1] - u_orig[-2]) if len(u_orig) >= 2 else 0.0

        # ── Point forecast ─────────────────────────────────────────────────
        yhat_log = []
        cur_u    = last_u
        cur_du   = last_du

        for h in range(horizon):
            exog_row = exog_future.iloc[h] if h < len(exog_future) else exog_future.iloc[-1]
            # Regression component at future time
            reg_h = sum(beta.get(col, 0.0) * float(exog_row[col])
                        for col in spec_x if col in exog_row.index)
            # ARIMA error forecast
            next_du = intercept + phi * cur_du
            next_u  = cur_u + next_du
            # Combined forecast
            yhat_h  = reg_h + next_u
            yhat_log.append(yhat_h)
            cur_du = next_du
            cur_u  = next_u

        yhat_log = np.array(yhat_log)
        # Parametric error distribution prevents drawing massive initialization artifacts from resid
        std_err  = float(np.sqrt(max(sigma2, 1e-12)))

        # ── Simulation paths for confidence intervals ──────────────────────
        sims = []
        for _ in range(n_sims):
            noise    = np.random.normal(0, std_err, size=horizon)
            sim_u    = last_u
            sim_du   = last_du
            sim_path = []
            for h in range(horizon):
                exog_row = exog_future.iloc[h] if h < len(exog_future) else exog_future.iloc[-1]
                reg_h    = sum(beta.get(col, 0.0) * float(exog_row[col])
                               for col in spec_x if col in exog_row.index)
                sim_du   = intercept + phi * sim_du + noise[h]
                sim_u    = sim_u + sim_du
                sim_path.append(np.exp(reg_h + sim_u))
            sims.append(sim_path)

        sims = np.array(sims)
        return pd.DataFrame(
            {
                "yhat": np.exp(yhat_log),
                "lo80": np.quantile(sims, 0.10, axis=0),
                "hi80": np.quantile(sims, 0.90, axis=0),
                "lo95": np.quantile(sims, 0.025, axis=0),
                "hi95": np.quantile(sims, 0.975, axis=0),
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

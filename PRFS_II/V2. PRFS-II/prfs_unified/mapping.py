"""
prfs_unified/mapping.py
=======================
CRITICAL bridge layer.

Converts the 4 dynamic PRFS macro targets into the full exogenous
future DataFrame required by the multi-model engine (app_multimodel_v2 logic).

Mapping rules (deterministic, transparent):
  - GDP growth  → drives all log-level bases by default
  - Elasticity overrides scale the effective growth for sub-series
  - Inflation is passed through as a level
  - Exchange rate uses compound log-space depreciation
"""
from __future__ import annotations

from typing import Dict, List

import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


# ── Univariate trend projection (fallback) ──────────────────────────────
def _project_univariate(series: pd.Series, horizon: int) -> np.ndarray:
    clean = series.dropna()
    y = clean.values
    x = np.arange(len(y)).reshape(-1, 1)
    m = LinearRegression().fit(x, y)
    return m.predict(np.arange(len(y), len(y) + horizon).reshape(-1, 1))


# ── Helper: resolve per-year or constant target ─────────────────────────
def _yearly(val, h: int) -> float:
    """Get the target for year index *h* (0-based)."""
    if isinstance(val, (list, tuple)):
        return val[h] if h < len(val) else val[-1]
    return float(val)


# ── Main mapping function ───────────────────────────────────────────────
def build_multimodel_future_exog_from_dynamic(
    df_hist: pd.DataFrame,
    horizon: int,
    spec_x: List[str],
    targets_dict: Dict,
    covid_on: bool = False,
    regime_on: bool = False,
    use_univariate: bool = False,
    elasticities: Dict | None = None,
) -> pd.DataFrame:
    """
    Build the exog future dataframe needed by the multi-model engine,
    using ONLY the four dynamic PRFS macro targets.

    Parameters
    ----------
    df_hist        : historical data (with log columns)
    horizon        : years to forecast
    spec_x         : column names required by the model spec
    targets_dict   : {gdp_growth, inflation, exrate_growth, policy_rate}
    covid_on       : COVID dummy flag
    regime_on      : regime dummy flag
    use_univariate : if True, use linear trend instead of growth mapping
    elasticities   : {imports, consumption, lsm} – scaling factors
    """
    if elasticities is None:
        elasticities = dict(imports=1.0, consumption=1.0, lsm=1.0)

    last = df_hist.iloc[-1]
    last_year = int(df_hist.index.max().year)
    years = [last_year + i for i in range(1, horizon + 1)]
    idx = pd.PeriodIndex(years, freq="Y")
    fut = pd.DataFrame(index=idx)

    # ── Column-growth mapping ────────────────────────────────────────────
    # Maps every log-level column to (growth_key, elasticity_multiplier)
    _map: Dict[str, tuple] = {
        "log_gdp_nonagr": ("gdp_growth", 1.0),
        "log_gdp":        ("gdp_growth", 1.0),
        "log_lsm":        ("gdp_growth", elasticities.get("lsm", 1.0)),
        "log_imports":    ("gdp_growth", elasticities.get("imports", 1.0)),
        "log_dutiable_imports": ("gdp_growth", elasticities.get("imports", 1.0)),
        "log_consumption":      ("gdp_growth", elasticities.get("consumption", 1.0)),
        "log_exrate":     ("exrate_growth", 1.0),
    }

    for col, (driver_key, elast) in _map.items():
        if col not in df_hist.columns:
            continue
        if use_univariate:
            fut[col] = _project_univariate(df_hist[col], horizon)
        else:
            # Use last non-NaN value as starting point (2026 row may have NaN logs)
            series_clean = df_hist[col].dropna()
            if len(series_clean) == 0:
                continue
            cur = float(series_clean.iloc[-1])
            vals = []
            for h in range(horizon):
                g = _yearly(targets_dict[driver_key], h) * elast
                cur = cur + np.log1p(g)
                vals.append(cur)
            fut[col] = vals

    # ── Inflation (level, not log) ───────────────────────────────────────
    if "inflation" in spec_x or True:          # always populate
        fut["inflation"] = [
            _yearly(targets_dict.get("inflation", 6.1), h)
            for h in range(horizon)
        ]

    # ── Policy rate ──────────────────────────────────────────────────────
    if "policy rate" in spec_x or "policy_rate" in spec_x:
        fut["policy rate"] = [
            _yearly(targets_dict.get("policy_rate", 11.2), h)
            for h in range(horizon)
        ]

    # ── Dummies ──────────────────────────────────────────────────────────
    fut["covid"] = 1 if covid_on else 0
    fut["regime"] = 1 if regime_on else 0
    for d in ["step_2024", "dummy_2024", "dummy_2025"]:
        if d in df_hist.columns:
            fut[d] = 1 if d == "step_2024" else 0

    # ── Fill any remaining spec_x columns from last non-NaN historical value ──
    for c in spec_x:
        if c not in fut.columns:
            if c in df_hist.columns:
                clean = df_hist[c].dropna()
                fut[c] = float(clean.iloc[-1]) if len(clean) else 0.0
            elif c in last.index and pd.notna(last[c]):
                fut[c] = float(last[c])
            else:
                fut[c] = 0.0

    return fut[spec_x].copy()

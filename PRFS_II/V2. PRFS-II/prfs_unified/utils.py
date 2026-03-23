"""
prfs_unified/utils.py
=====================
Shared diagnostic and coefficient-table helpers (ported from app_multimodel_v2).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import jarque_bera


# ── Dual Jarque-Bera ─────────────────────────────────────────────────────
def _add_dual_jb(resid: pd.Series, out: Dict):
    try:
        _, p_full, _, _ = jarque_bera(resid)
        out["jb_full_p"] = float(p_full)
        if len(resid) > 1:
            _, p_trim, _, _ = jarque_bera(resid.iloc[1:])
            out["jb_trim_p"] = float(p_trim)
        else:
            out["jb_trim_p"] = None
    except Exception:
        out["jb_full_p"] = None
        out["jb_trim_p"] = None


# ── ARDL diagnostics ─────────────────────────────────────────────────────
def diagnostics_ardl(res) -> Dict:
    resid = pd.Series(res.resid).dropna()
    out: Dict = {}
    out["durbin_watson"] = float(sm.stats.stattools.durbin_watson(resid))
    try:
        lag = min(5, max(1, len(resid) // 5))
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        out["ljung_box_p"] = None
    _add_dual_jb(resid, out)
    try:
        exog = getattr(res.model, "exog", None)
        if exog is not None:
            bp = het_breuschpagan(resid.values, exog)
            out["breusch_pagan_p"] = float(bp[1])
        else:
            out["breusch_pagan_p"] = None
    except Exception:
        out["breusch_pagan_p"] = None
    out["n_resid"] = int(len(resid))
    return out


# ── ARIMAX diagnostics ───────────────────────────────────────────────────
def diagnostics_arimax(res) -> Dict:
    resid = pd.Series(res.resid).dropna()
    out: Dict = {}
    out["aic"] = float(res.aic)
    out["bic"] = float(res.bic) if hasattr(res, "bic") else None
    out["durbin_watson"] = float(sm.stats.stattools.durbin_watson(resid))
    try:
        lag = min(5, max(1, len(resid) // 5))
        lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
        out["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0])
    except Exception:
        out["ljung_box_p"] = None
    _add_dual_jb(resid, out)
    out["n_resid"] = int(len(resid))
    return out


# ── Coefficient tables ───────────────────────────────────────────────────
def coef_table_ardl(res) -> pd.DataFrame:
    return pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "p": res.pvalues.values,
    })


def coef_table_arimax(res) -> pd.DataFrame:
    return pd.DataFrame({
        "term": res.params.index,
        "coef": res.params.values,
        "std_err": res.bse.values,
        "z": res.zvalues.values,
        "p": res.pvalues.values,
    })


def coef_table_enet(bundle_head: Dict) -> pd.DataFrame:
    model = bundle_head["enet"]["model"]
    feat_cols = bundle_head["enet"]["feature_cols"]
    enet = model.named_steps["enet"]
    out = pd.DataFrame({"term": feat_cols, "coef": enet.coef_})
    out["abs_coef"] = out["coef"].abs()
    return out.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])

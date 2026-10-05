"""
PIT Reform Optimization Engine
Production UI: supports Auto-Optimize and Policy Lab (User-Adjustable) modes.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import time
import importlib

# Force reload of core logic
import src.solver
import src.io
import src.viz
importlib.reload(src.solver)
importlib.reload(src.io)
importlib.reload(src.viz)

from src.io import load_slab_data, load_grid_data, get_data_paths
from src.solver import (optimize_schedule, compute_metrics, _schedule_to_list,
                        validate_schedule, run_manual_simulation, optimize_schedule_constrained,
                        _estimate_revenue, compute_tax)
from src.viz import (build_heatmap_dataframe, plot_etr_heatmap,
                     plot_detr_heatmap, plot_etr_curve, plot_progressivity_slope)

# ───────────────────────── Page Config ─────────────────────────
st.set_page_config(page_title="PIT Reform Optimization Engine", layout="wide",
                   page_icon="📊")

# ── IMF / World Bank Professional Theme ────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F5F7FA; }

/* ── Top institutional header bar ── */
.imf-header {
    background: linear-gradient(90deg, #003B5C 0%, #00518B 60%, #0073AC 100%);
    padding: 18px 32px 16px 32px;
    border-radius: 0 0 8px 8px;
    margin: -1rem -1rem 1.5rem -1rem;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 3px 12px rgba(0,59,92,0.25);
}
.imf-header-title {
    color: #FFFFFF;
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin: 0;
}
.imf-header-sub {
    color: #b3d4e8;
    font-size: 0.78rem;
    font-weight: 400;
    margin: 2px 0 0 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #D1DCE5;
}
section[data-testid="stSidebar"] .element-container { padding: 0 4px; }

/* ── Cards & panels ── */
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,59,92,0.08);
}
div[data-testid="stMetricValue"] { color: #003B5C !important; font-weight: 700; }
div[data-testid="stMetricLabel"] { color: #4A6B82 !important; font-size: 0.75rem !important; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Buttons ── */
button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
    background-color: #003B5C !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover { background-color: #00518B !important; }

/* ── Section headers ── */
h1, h2, h3 { color: #003B5C; }
h3 { padding-bottom: 4px; }

/* ── Data editor / tables ── */
.stDataFrame { border: 1px solid #D1DCE5; border-radius: 6px; }
thead th { background-color: #003B5C !important; color: #FFFFFF !important; font-size: 0.75rem !important; font-weight: 600 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #D1DCE5; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #4A6B82; font-weight: 500; border-radius: 4px 4px 0 0; }
.stTabs [aria-selected="true"] { color: #003B5C !important; border-bottom: 2px solid #003B5C !important; font-weight: 700; }

/* ── Expanders ── */
details summary { color: #003B5C; font-weight: 600; }

/* ── Info / Warning ── */
.stAlert { border-radius: 6px; border-left: 4px solid #003B5C; }

/* ── IMF Metric Cards ── */
.imf-metric-row {
    display: flex;
    gap: 12px;
    margin: 12px 0 20px 0;
    flex-wrap: wrap;
}
.imf-metric-card {
    flex: 1;
    min-width: 155px;
    background: #FFFFFF;
    border: 1px solid #E8EDF2;
    border-radius: 8px;
    padding: 14px 18px 12px 16px;
    box-shadow: 0 1px 4px rgba(0,59,92,0.07);
}
.imf-mc-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #4A6B82;
    margin-bottom: 5px;
}
.imf-mc-value {
    font-size: 1.22rem;
    font-weight: 700;
    color: #003B5C;
    white-space: nowrap;
}
.imf-delta-pos { color: #16a34a; font-size: 0.76rem; font-weight: 600; margin-top: 3px; }
.imf-delta-neg { color: #dc2626; font-size: 0.76rem; font-weight: 600; margin-top: 3px; }

/* ── Section label ── */
.imf-section-tag {
    display: inline-block;
    background: #003B5C;
    color: #FFFFFF;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Institutional Header ──────────────────────────────────────
st.markdown("""
<div class="imf-header">
  <div>
    <p class="imf-header-title">📊 PIT Reform Optimization Engine</p>
    <p class="imf-header-sub">Pakistan Personal Income Tax · Policy Analysis &amp; Simulation Platform</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ───────────────────────── Data Loading ─────────────────────────
_, truth_path = get_data_paths()   # only truth_path (slabs) is used from system; obs require upload

# Check if user has uploaded the observations file
_has_upload = 'uploaded_obs_bytes' in st.session_state and st.session_state.get('uploaded_obs_bytes') is not None

if not _has_upload:
    # Show friendly upload-only landing page — nothing else renders
    st.markdown("""
    <div style='text-align:center; padding:60px 20px;'>
        <h2>📊 Upload Your Observations Data</h2>
        <p style='color:#666; font-size:16px; max-width:520px; margin:auto;'>
            Upload the <b>Income Tax Liability (S / NS / AOP)</b> Excel file to begin analysis.
            The system will use internally stored slab rates and surcharge rules automatically.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _up = st.file_uploader(
        "📂 Upload Observations File (Income Tax Liability S‌/NS/AOP.xlsx)",
        type=["xlsx", "xls"],
        help="Required columns: Taxable Income Slab (Rs.), Year, Type_Tax, Number of Persons, Taxable Income (9100), Normal Income Tax (920000)"
    )
    if _up is not None:
        st.session_state.uploaded_obs_bytes = _up.getvalue()
        st.rerun()
    st.stop()

# — File is uploaded; use BytesIO directly (no temp file needed) —
import io as _io
_obs_buf = _io.BytesIO(st.session_state.uploaded_obs_bytes)
_obs_path = _obs_buf   # pass BytesIO to load_slab_data

try:
    df_slabs_agg = load_slab_data(_obs_path)
except Exception as e:
    st.error(f"❌ Could not read the uploaded file: {e}")
    st.stop()

# ───────────────────────── Session State ─────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'lab_slabs' not in st.session_state:
    st.session_state.lab_slabs = None

# ───────────────────────── Helpers ─────────────────────────

def _norm(s):
    return str(s).lower().replace('-', ' ').replace('_', ' ').strip()

# Canonical type name mapping (Excel label → app label)
_TYPE_CANON = {
    'salaried':      'Salaried',
    'non salaried':  'Non-Salaried',
    'non-salaried':  'Non-Salaried',
    'aop':           'AOP',
    'nsc':           'NSC',          # NS + AOP combined
    'non_salaried':  'Non-Salaried',
}

def _canon_type(raw):
    """Normalise an Excel Tax_Type label to the app's canonical name."""
    n = _norm(str(raw))
    return _TYPE_CANON.get(n, str(raw).strip().title())

def _get_truth_slabs(g_type, year=None):
    """Year-aware lookup into TRUTH_SLABS. Returns None if (year, type) is not in the
    slab file — no silent fallback to a different year."""
    if year is None:
        return None
    key = (int(year), _canon_type(g_type))
    v = TRUTH_SLABS.get(key)
    return v.copy() if v is not None and not v.empty else None

def _get_truth_surcharge(g_type, year=None):
    """Year-aware lookup into TRUTH_SURCHARGES, taken purely from the slab file.
    Returns zero surcharge if (year, type) is not found."""
    if year is not None:
        key = (int(year), _canon_type(g_type))
        if key in TRUTH_SURCHARGES:
            return dict(TRUTH_SURCHARGES[key])
    return {'threshold': 0.0, 'rate': 0.0}

@st.cache_data
def load_truth_slabs(file_path):
    """Parse the PIT slab file (PIT_slabs_2026.xlsx) into year-aware dicts.
    Keys are (year, canonical_type) tuples."""
    df = pd.read_excel(file_path, engine='openpyxl')
    slabs      = {}   # {(year, type): DataFrame}
    surcharges = {}   # {(year, type): {'threshold': float, 'rate': float}}

    # Inline canonical mapping — embedded here so ANY change busts the @st.cache_data
    # cache (Streamlit only hashes the function's own bytecode, not global dicts)
    _INLINE_CANON = {
        'salaried':               'Salaried',
        'non salaried':           'Non-Salaried',
        'non-salaried':           'Non-Salaried',
        'non_salaried':           'Non-Salaried',
        'aop':                    'AOP',
        'nsc':                    'NSC',          # NS + AOP combined
    }
    def _local_canon(raw):
        return _INLINE_CANON.get(str(raw).strip().lower(), str(raw).strip().title())

    for (year, raw_ttype), g in df.groupby(['Year', 'Tax_Type']):
        ttype = _local_canon(raw_ttype)   # e.g. 'Non_salaried' -> 'Non-Salaried'
        key   = (int(year), ttype)

        g_slabs  = []
        s_thresh = 0.0
        s_rate   = 0.0

        for _, r in g.iterrows():
            lower        = str(r['Lower_slab']).strip().lower()
            upper        = str(r['Upper_slab']).strip().lower()
            mtr          = r['MTR']
            tax_rate_str = str(r['TAX RATE']).lower()

            # Skip blank separator rows
            if pd.isna(mtr) and 'surcharge' not in lower and 'surcharge' not in upper:
                continue

            is_surcharge = 'surcharge' in lower or 'surcharge' in upper or 'liability' in tax_rate_str

            if is_surcharge:
                s_rate = float(mtr) if pd.notna(mtr) else 0.0
                l_num  = pd.to_numeric(r['Lower_slab'], errors='coerce')
                if pd.notna(l_num):
                    s_thresh = float(l_num)
                else:
                    nums = [float(n.replace(',', '')) for n in re.findall(r'[\d,]+', tax_rate_str)
                            if n.replace(',', '').isdigit()]
                    income_nums = [n for n in nums if n >= 100_000]
                    s_thresh = income_nums[0] if income_nums else 0.0
            else:
                l_val = pd.to_numeric(r['Lower_slab'], errors='coerce')
                u_val = np.inf if upper == '+' else pd.to_numeric(r['Upper_slab'], errors='coerce')
                if pd.isna(u_val): u_val = np.inf
                mtr_val = float(mtr) if pd.notna(mtr) else 0.0
                if pd.notna(l_val):
                    g_slabs.append({'lower_bound': float(l_val),
                                    'upper_bound': float(u_val),
                                    'marginal_rate': mtr_val})

        slabs[key]      = pd.DataFrame(g_slabs)
        surcharges[key] = {'threshold': s_thresh, 'rate': s_rate}

    return slabs, surcharges

def _fmt_table(df, filer_counts=None, avg_etrs=None):
    if df.empty: return df
    out = df.copy()
    # Filter zero-width or redundant slabs often found in raw data
    if 'upper_bound' in out.columns and 'lower_bound' in out.columns:
        out = out[out['upper_bound'] > out['lower_bound']].copy()
    out = out.reset_index(drop=True)
    
    def _rng(row):
        lo = f"{row['lower_bound']:,.0f}"
        hi = "Above" if not np.isfinite(row['upper_bound']) else f"{row['upper_bound']:,.0f}"
        return f"{lo} – {hi}"
    
    out['Income Range'] = out.apply(_rng, axis=1)
    out['MTR'] = out['marginal_rate'].map('{:.2%}'.format)
    
    # Add optional columns
    cols = ['Income Range', 'MTR']
    if filer_counts is not None:
        out['Number of Filers'] = [f"{int(n):,}" for n in filer_counts]
        cols.append('Number of Filers')
    if avg_etrs is not None:
        out['Avg ETR'] = [f"{e:.2%}" for e in avg_etrs]
        cols.append('Avg ETR')
    
    return out[cols]

def _merged_table(base_df, prop_df, base_collections=None, prop_collections=None):
    # Guard: if either frame is empty, show whatever is available
    if base_df is None or base_df.empty:
        base_df = pd.DataFrame(columns=['lower_bound', 'upper_bound', 'marginal_rate'])
    if prop_df is None or prop_df.empty:
        return pd.DataFrame(columns=['Band', 'Base MTR', 'Proposed MTR', 'Δ (pp)'])
    all_b = sorted(set(
        list(base_df['lower_bound']) + list(prop_df['lower_bound']) +
        [b for b in base_df['upper_bound'] if np.isfinite(b)] +
        [b for b in prop_df['upper_bound'] if np.isfinite(b)]
    ))
    rows = []
    for j in range(len(all_b)):
        lo = all_b[j]
        hi = all_b[j + 1] if j + 1 < len(all_b) else np.inf
        bm = base_df[(base_df['lower_bound'] <= lo) & (base_df['upper_bound'] > lo)]
        br = bm.iloc[0]['marginal_rate'] if not bm.empty else None
        pm = prop_df[(prop_df['lower_bound'] <= lo) & (prop_df['upper_bound'] > lo)]
        pr = pm.iloc[0]['marginal_rate'] if not pm.empty else None
        hi_s = "Above" if not np.isfinite(hi) else f"{hi:,.0f}"
        
        row_data = {
            'Band': f"{lo:,.0f} – {hi_s}",
            'Base MTR': f"{br:.2%}" if br is not None else "—",
            'Proposed MTR': f"{pr:.2%}" if pr is not None else "—",
            'Δ (pp)': f"{(pr-br)*100:+.2f}" if br is not None and pr is not None else "—",
        }
        
        # Add collection columns if provided
        if base_collections is not None and prop_collections is not None and j < len(base_collections):
            base_coll = base_collections[j]
            prop_coll = prop_collections[j]
            delta_coll = prop_coll - base_coll
            delta_coll_pct = (delta_coll / base_coll * 100) if base_coll > 0 else 0.0
            
            row_data['Base Collection'] = f"PKR {base_coll/1e9:,.2f}B"
            row_data['Proposed Collection'] = f"PKR {prop_coll/1e9:,.2f}B"
            row_data['Δ Collection'] = f"PKR {delta_coll/1e9:+,.2f}B"
            row_data['Δ Collection (%)'] = f"{delta_coll_pct:+.2f}%"
        
        rows.append(row_data)
    return pd.DataFrame(rows)

def _merged_table(base_df, prop_df, base_collections=None, prop_collections=None):
    # Guard: if either frame is empty, show whatever is available
    if base_df is None or base_df.empty:
        base_df = pd.DataFrame(columns=['lower_bound', 'upper_bound', 'marginal_rate'])
    if prop_df is None or prop_df.empty:
        return pd.DataFrame(columns=['Band', 'Base MTR', 'Proposed MTR', 'Δ (pp)'])
    all_b = sorted(set(
        list(base_df['lower_bound']) + list(prop_df['lower_bound']) +
        [b for b in base_df['upper_bound'] if np.isfinite(b)] +
        [b for b in prop_df['upper_bound'] if np.isfinite(b)]
    ))
    rows = []
    for j in range(len(all_b)):
        lo = all_b[j]
        hi = all_b[j + 1] if j + 1 < len(all_b) else np.inf
        bm = base_df[(base_df['lower_bound'] <= lo) & (base_df['upper_bound'] > lo)]
        br = bm.iloc[0]['marginal_rate'] if not bm.empty else None
        pm = prop_df[(prop_df['lower_bound'] <= lo) & (prop_df['upper_bound'] > lo)]
        pr = pm.iloc[0]['marginal_rate'] if not pm.empty else None
        hi_s = "Above" if not np.isfinite(hi) else f"{hi:,.0f}"
        
        row_data = {
            'Band': f"{lo:,.0f} – {hi_s}",
            'Base MTR': f"{br:.2%}" if br is not None else "—",
            'Proposed MTR': f"{pr:.2%}" if pr is not None else "—",
            'Δ (pp)': f"{(pr-br)*100:+.2f}" if br is not None and pr is not None else "—",
        }
        
        # Add collection columns if provided
        if base_collections is not None and prop_collections is not None and j < len(base_collections):
            base_coll = base_collections[j]
            prop_coll = prop_collections[j]
            delta_coll = prop_coll - base_coll
            delta_coll_pct = (delta_coll / base_coll * 100) if base_coll > 0 else 0.0
            
            row_data['Base Collection'] = f"PKR {base_coll/1e9:,.2f}B"
            row_data['Proposed Collection'] = f"PKR {prop_coll/1e9:,.2f}B"
            row_data['Δ Collection'] = f"PKR {delta_coll/1e9:+,.2f}B"
            row_data['Δ Collection (%)'] = f"{delta_coll_pct:+.2f}%"
        
        rows.append(row_data)
    return pd.DataFrame(rows)

def _merged_table_enhanced(base_df, prop_df, base_collections=None, prop_collections=None,
                          base_filers=None, prop_filers=None, base_avg_etrs=None, prop_avg_etrs=None):
    """Enhanced merged table with all base and proposed metrics."""
    if base_df is None or base_df.empty:
        base_df = pd.DataFrame(columns=['lower_bound', 'upper_bound', 'marginal_rate'])
    if prop_df is None or prop_df.empty:
        return pd.DataFrame(columns=['Band', 'Base MTR', 'Proposed MTR'])
    
    all_b = sorted(set(
        list(base_df['lower_bound']) + list(prop_df['lower_bound']) +
        [b for b in base_df['upper_bound'] if np.isfinite(b)] +
        [b for b in prop_df['upper_bound'] if np.isfinite(b)]
    ))
    
    rows = []
    for j in range(len(all_b)):
        lo = all_b[j]
        hi = all_b[j + 1] if j + 1 < len(all_b) else np.inf
        
        # Skip negligible bands (less than 2 rupees wide)
        if np.isfinite(hi) and (hi - lo) < 2:
            continue
        
        # Find MTRs
        bm = base_df[(base_df['lower_bound'] <= lo) & (base_df['upper_bound'] > lo)]
        br = bm.iloc[0]['marginal_rate'] if not bm.empty else None
        pm = prop_df[(prop_df['lower_bound'] <= lo) & (prop_df['upper_bound'] > lo)]
        pr = pm.iloc[0]['marginal_rate'] if not pm.empty else None
        
        hi_s = "Above" if not np.isfinite(hi) else f"{hi:,.0f}"
        
        row_data = {
            'Band': f"{lo:,.0f} – {hi_s}",
            'Base MTR': f"{br:.2%}" if br is not None else "—",
            'Proposed MTR': f"{pr:.2%}" if pr is not None else "—",
            'Δ MTR (pp)': f"{(pr-br)*100:+.2f}" if br is not None and pr is not None else "—",
        }
        
        # Add filer counts if provided
        if base_filers is not None and j < len(base_filers):
            row_data['Base Filers'] = f"{int(base_filers[j]):,}"
        if prop_filers is not None and j < len(prop_filers):
            row_data['Proposed Filers'] = f"{int(prop_filers[j]):,}"
        
        # Add avg ETRs if provided
        if base_avg_etrs is not None and j < len(base_avg_etrs):
            row_data['Base Avg ETR'] = f"{base_avg_etrs[j]:.2%}"
        if prop_avg_etrs is not None and j < len(prop_avg_etrs):
            row_data['Proposed Avg ETR'] = f"{prop_avg_etrs[j]:.2%}"
        
        # Add collection columns if provided
        if base_collections is not None and prop_collections is not None and j < len(base_collections):
            base_coll = base_collections[j]
            prop_coll = prop_collections[j]
            delta_coll = prop_coll - base_coll
            delta_coll_pct = (delta_coll / base_coll * 100) if base_coll > 0 else 0.0
            
            row_data['Base Collection'] = f"PKR {base_coll/1e9:,.2f}B"
            row_data['Proposed Collection'] = f"PKR {prop_coll/1e9:,.2f}B"
            row_data['Δ Collection'] = f"PKR {delta_coll/1e9:+,.2f}B"
            row_data['Δ Collection (%)'] = f"{delta_coll_pct:+.2f}%"
        
        rows.append(row_data)
    return pd.DataFrame(rows)

def _calculate_slab_filers(slabs_df, y_arr, n_arr):
    """Calculate number of filers in each slab."""
    if slabs_df.empty or len(y_arr) == 0:
        return []
    
    n_safe = np.where(n_arr > 0, n_arr, 1.0)
    y_pp = y_arr / n_safe  # Per-person income
    
    filer_counts = []
    for _, slab in slabs_df.iterrows():
        lower = slab['lower_bound']
        upper = slab['upper_bound']
        mask = (y_pp >= lower) & (y_pp < upper)
        filers = n_arr[mask].sum()
        filer_counts.append(filers)
    
    return filer_counts

def _calculate_slab_avg_etr(slabs_df, schedule_list, y_arr, n_arr, sur_slabs):
    """Calculate average ETR for each slab."""
    if slabs_df.empty or len(y_arr) == 0:
        return []
    
    n_safe = np.where(n_arr > 0, n_arr, 1.0)
    y_pp = y_arr / n_safe
    
    # Prepare schedule data
    lowers = np.array([s['lower'] for s in schedule_list])
    rates = np.array([s['rate'] for s in schedule_list])
    uppers = np.array([s['upper'] for s in schedule_list])
    
    # Build cumulative tax
    cum = np.zeros(len(schedule_list))
    for k in range(1, len(schedule_list)):
        w = uppers[k-1] - lowers[k-1]
        cum[k] = cum[k-1] + (0.0 if np.isinf(w) else w) * rates[k-1]
    
    avg_etrs = []
    for _, slab in slabs_df.iterrows():
        lower = slab['lower_bound']
        upper = slab['upper_bound']
        mask = (y_pp >= lower) & (y_pp < upper)
        
        if not np.any(mask):
            avg_etrs.append(0.0)
            continue
        
        # Calculate tax for observations in this slab
        y_pp_slab = y_pp[mask]
        n_slab = n_arr[mask]
        y_slab = y_arr[mask]
        
        idx = np.clip(np.searchsorted(lowers, y_pp_slab, side='right') - 1, 0, len(schedule_list)-1)
        base_tax_pp = np.maximum(cum[idx] + (y_pp_slab - lowers[idx]) * rates[idx], 0.0)
        
        # Apply surcharge
        sur_rt = np.zeros(len(y_pp_slab))
        if sur_slabs:
            for s in sur_slabs:
                mask_sur = (y_pp_slab >= s['lower']) & (y_pp_slab < s['upper'])
                sur_rt[mask_sur] = s['rate']
        
        nit_pp = base_tax_pp * (1.0 + sur_rt)
        total_tax = (nit_pp * n_slab).sum()
        total_income = y_slab.sum()
        
        avg_etr = total_tax / total_income if total_income > 0 else 0.0
        avg_etrs.append(avg_etr)
    
    return avg_etrs

def _calculate_band_collections(base_df, prop_df, base_schedule, prop_schedule, 
                                 y_arr, n_arr, base_sur_slabs, prop_sur_slabs, filer_scale):
    """Calculate tax collections for each band in the merged transition view."""
    if base_df is None or base_df.empty or prop_df is None or prop_df.empty:
        return [], []
    
    # Get all breakpoints
    all_b = sorted(set(
        list(base_df['lower_bound']) + list(prop_df['lower_bound']) +
        [b for b in base_df['upper_bound'] if np.isfinite(b)] +
        [b for b in prop_df['upper_bound'] if np.isfinite(b)]
    ))
    
    n_safe = np.where(n_arr > 0, n_arr, 1.0)
    y_pp = y_arr / n_safe
    
    # Prepare base schedule
    base_lowers = np.array([s['lower'] for s in base_schedule])
    base_rates = np.array([s['rate'] for s in base_schedule])
    base_uppers = np.array([s['upper'] for s in base_schedule])
    base_cum = np.zeros(len(base_schedule))
    for k in range(1, len(base_schedule)):
        w = base_uppers[k-1] - base_lowers[k-1]
        base_cum[k] = base_cum[k-1] + (0.0 if np.isinf(w) else w) * base_rates[k-1]
    
    # Prepare proposed schedule
    prop_lowers = np.array([s['lower'] for s in prop_schedule])
    prop_rates = np.array([s['rate'] for s in prop_schedule])
    prop_uppers = np.array([s['upper'] for s in prop_schedule])
    prop_cum = np.zeros(len(prop_schedule))
    for k in range(1, len(prop_schedule)):
        w = prop_uppers[k-1] - prop_lowers[k-1]
        prop_cum[k] = prop_cum[k-1] + (0.0 if np.isinf(w) else w) * prop_rates[k-1]
    
    base_collections = []
    prop_collections = []
    
    for j in range(len(all_b)):
        lo = all_b[j]
        hi = all_b[j + 1] if j + 1 < len(all_b) else np.inf
        
        # Find observations in this band
        mask = (y_pp >= lo) & (y_pp < hi)
        
        if not np.any(mask):
            base_collections.append(0.0)
            prop_collections.append(0.0)
            continue
        
        y_pp_band = y_pp[mask]
        n_band = n_arr[mask]
        
        # Base collection
        idx_base = np.clip(np.searchsorted(base_lowers, y_pp_band, side='right') - 1, 0, len(base_schedule)-1)
        base_tax_pp = np.maximum(base_cum[idx_base] + (y_pp_band - base_lowers[idx_base]) * base_rates[idx_base], 0.0)
        
        sur_rt_base = np.zeros(len(y_pp_band))
        if base_sur_slabs:
            for s in base_sur_slabs:
                mask_sur = (y_pp_band >= s['lower']) & (y_pp_band < s['upper'])
                sur_rt_base[mask_sur] = s['rate']
        
        nit_base_pp = base_tax_pp * (1.0 + sur_rt_base)
        base_coll = (nit_base_pp * n_band).sum()
        base_collections.append(base_coll)
        
        # Proposed collection (with filer scale)
        n_band_scaled = n_band * filer_scale
        idx_prop = np.clip(np.searchsorted(prop_lowers, y_pp_band, side='right') - 1, 0, len(prop_schedule)-1)
        prop_tax_pp = np.maximum(prop_cum[idx_prop] + (y_pp_band - prop_lowers[idx_prop]) * prop_rates[idx_prop], 0.0)
        
        sur_rt_prop = np.zeros(len(y_pp_band))
        if prop_sur_slabs:
            for s in prop_sur_slabs:
                mask_sur = (y_pp_band >= s['lower']) & (y_pp_band < s['upper'])
                sur_rt_prop[mask_sur] = s['rate']
        
        nit_prop_pp = prop_tax_pp * (1.0 + sur_rt_prop)
        prop_coll = (nit_prop_pp * n_band_scaled).sum()
        prop_collections.append(prop_coll)
    
    return base_collections, prop_collections

def _calculate_band_metrics(base_df, prop_df, base_schedule, prop_schedule, 
                            y_arr, n_arr, base_sur_slabs, prop_sur_slabs, filer_scale):
    """Calculate filer counts and avg ETR for each band in the merged transition view."""
    if base_df is None or base_df.empty or prop_df is None or prop_df.empty:
        return [], [], [], []
    
    # Get all breakpoints
    all_b = sorted(set(
        list(base_df['lower_bound']) + list(prop_df['lower_bound']) +
        [b for b in base_df['upper_bound'] if np.isfinite(b)] +
        [b for b in prop_df['upper_bound'] if np.isfinite(b)]
    ))
    
    n_safe = np.where(n_arr > 0, n_arr, 1.0)
    y_pp = y_arr / n_safe
    
    # Prepare schedules
    base_lowers = np.array([s['lower'] for s in base_schedule])
    base_rates = np.array([s['rate'] for s in base_schedule])
    base_uppers = np.array([s['upper'] for s in base_schedule])
    base_cum = np.zeros(len(base_schedule))
    for k in range(1, len(base_schedule)):
        w = base_uppers[k-1] - base_lowers[k-1]
        base_cum[k] = base_cum[k-1] + (0.0 if np.isinf(w) else w) * base_rates[k-1]
    
    prop_lowers = np.array([s['lower'] for s in prop_schedule])
    prop_rates = np.array([s['rate'] for s in prop_schedule])
    prop_uppers = np.array([s['upper'] for s in prop_schedule])
    prop_cum = np.zeros(len(prop_schedule))
    for k in range(1, len(prop_schedule)):
        w = prop_uppers[k-1] - prop_lowers[k-1]
        prop_cum[k] = prop_cum[k-1] + (0.0 if np.isinf(w) else w) * prop_rates[k-1]
    
    base_filers = []
    prop_filers = []
    base_avg_etrs = []
    prop_avg_etrs = []
    
    for j in range(len(all_b)):
        lo = all_b[j]
        hi = all_b[j + 1] if j + 1 < len(all_b) else np.inf
        
        # Find observations in this band
        mask = (y_pp >= lo) & (y_pp < hi)
        
        if not np.any(mask):
            base_filers.append(0.0)
            prop_filers.append(0.0)
            base_avg_etrs.append(0.0)
            prop_avg_etrs.append(0.0)
            continue
        
        y_pp_band = y_pp[mask]
        n_band = n_arr[mask]
        y_band = y_arr[mask]
        
        # Filer counts
        base_filer_count = n_band.sum()
        prop_filer_count = (n_band * filer_scale).sum()
        base_filers.append(base_filer_count)
        prop_filers.append(prop_filer_count)
        
        # Base avg ETR
        idx_base = np.clip(np.searchsorted(base_lowers, y_pp_band, side='right') - 1, 0, len(base_schedule)-1)
        base_tax_pp = np.maximum(base_cum[idx_base] + (y_pp_band - base_lowers[idx_base]) * base_rates[idx_base], 0.0)
        
        sur_rt_base = np.zeros(len(y_pp_band))
        if base_sur_slabs:
            for s in base_sur_slabs:
                mask_sur = (y_pp_band >= s['lower']) & (y_pp_band < s['upper'])
                sur_rt_base[mask_sur] = s['rate']
        
        nit_base_pp = base_tax_pp * (1.0 + sur_rt_base)
        base_total_tax = (nit_base_pp * n_band).sum()
        base_total_income = y_band.sum()
        base_avg_etr = base_total_tax / base_total_income if base_total_income > 0 else 0.0
        base_avg_etrs.append(base_avg_etr)
        
        # Proposed avg ETR
        idx_prop = np.clip(np.searchsorted(prop_lowers, y_pp_band, side='right') - 1, 0, len(prop_schedule)-1)
        prop_tax_pp = np.maximum(prop_cum[idx_prop] + (y_pp_band - prop_lowers[idx_prop]) * prop_rates[idx_prop], 0.0)
        
        sur_rt_prop = np.zeros(len(y_pp_band))
        if prop_sur_slabs:
            for s in prop_sur_slabs:
                mask_sur = (y_pp_band >= s['lower']) & (y_pp_band < s['upper'])
                sur_rt_prop[mask_sur] = s['rate']
        
        nit_prop_pp = prop_tax_pp * (1.0 + sur_rt_prop)
        prop_total_tax = (nit_prop_pp * n_band * filer_scale).sum()
        prop_total_income = (y_band * filer_scale).sum()
        prop_avg_etr = prop_total_tax / prop_total_income if prop_total_income > 0 else 0.0
        prop_avg_etrs.append(prop_avg_etr)
    
    return base_filers, prop_filers, base_avg_etrs, prop_avg_etrs

def _get_historical_data(grid_df, g_type, target_y):
    t_norm = 'Salaried' if 'salaried' in g_type.lower() and 'non' not in g_type.lower() else 'Non_Salaried'
    subset = grid_df[grid_df['Type_Tax'] == t_norm].sort_values('Annual Income')
    if subset.empty: return {}, {}
    etr_out, detr_out = {}, {}
    years = [('2025', 'ETR_FY25')]
    for label, col in years:
        if col in subset.columns:
            h_etr = np.interp(target_y, subset['Annual Income'], subset[col])
            etr_out[label] = h_etr
            detr_out[label] = np.diff(h_etr, prepend=0.0) * 100.0
    return etr_out, detr_out

# Single source of truth slab logic
try:
    TRUTH_SLABS, TRUTH_SURCHARGES = load_truth_slabs(truth_path)
except Exception as e:
    TRUTH_SLABS, TRUTH_SURCHARGES = {}, {}

REGIME_YEARS = sorted({int(yr) for (yr, _) in TRUTH_SLABS.keys()})

def _fy_label(year):
    """2027 -> '2027 (FY2026-27)'. Same year convention as the slab file."""
    y = int(year)
    return f"{y} (FY{y-1}-{str(y)[-2:]})"

def _regime_base_revenue(g_agg, g_type, data_year, regime_year, actual_tax, regime_slabs_df):
    """Base revenue under the chosen slab regime, applied to the loaded data year.

    calib_factor = actual tax (data year) / simulated tax under the data year's own slabs
    base_revenue = calib_factor x simulated tax under the chosen regime's slabs
    If regime == data year this equals actual tax exactly (same as before).
    Returns (base_revenue, calib_factor, warning_or_None)."""
    regime_list = _schedule_to_list(regime_slabs_df)
    sim_regime = _estimate_revenue(regime_list, g_agg)
    data_slabs = _get_truth_slabs(g_type, data_year)
    if data_slabs is None:
        return sim_regime, 1.0, (f"No slabs for data year {data_year} in the slab file - "
                                 f"base revenue is uncalibrated slab-formula tax.")
    sim_data = _estimate_revenue(_schedule_to_list(data_slabs), g_agg)
    if sim_data <= 0:
        return sim_regime, 1.0, "Simulated data-year tax is zero - base revenue is uncalibrated."
    factor = actual_tax / sim_data
    return factor * sim_regime, factor, None

# ─── Helper: apply surcharge slabs to simulation metrics ───────────────
def _apply_sur_to_metrics(metrics, sur_slabs):
    """
    Post-process metrics dict returned by run_manual_simulation /
    compute_metrics to fold in slab-based surcharge.
    Updates: tax, etr, delta_etr, revenue.
    sur_slabs = list of {lower, upper, rate} where rate is 0–1 decimal.
    """
    if not sur_slabs:
        return metrics
    y   = metrics['y']
    tax = metrics['tax'].copy()

    # Vectorised surcharge rate for each income level on the grid
    sur_rates = np.zeros(len(y))
    for s in sur_slabs:
        mask = (y >= s['lower']) & (y < s['upper'])
        sur_rates[mask] = s['rate']

    tax_sur       = tax * (1.0 + sur_rates)
    etr_sur       = np.where(y > 0, tax_sur / y, 0.0)
    delta_etr_sur = np.diff(etr_sur, prepend=0.0) * 100.0

    m = dict(metrics)           # shallow copy — don't mutate original
    m['tax']       = tax_sur
    m['etr']       = etr_sur
    m['delta_etr'] = delta_etr_sur
    # Scale revenue by the average surcharge uplift (weighted by tax)
    sur_uplift = (tax_sur.sum() / tax.sum()) if tax.sum() > 0 else 1.0
    m['revenue']   = metrics.get('revenue', 0.0) * sur_uplift
    return m


# ───────────────────────── Sidebar ─────────────────────────
with st.sidebar:
    # — File info & change option —
    st.header("📂 Data File")
    st.success("✅ Observations file loaded.")
    if st.button("🔄 Remove / Change File"):
        del st.session_state['uploaded_obs_bytes']
        st.rerun()

    st.markdown("---")
    st.header("⚙️ Design Mode")
    mode = st.radio("Optimization Strategy", ["Auto Optimize", "Policy Lab", "Unified Schedule"],
                    help="Auto: System finds best schedule. Lab: You design it.")

    st.markdown("---")
    st.header("📋 General Settings")
    years_avail = sorted(df_slabs_agg['year'].unique(), reverse=True)
    selected_year = st.selectbox("Data Year (observations)", years_avail,
                                 help="Year of the uploaded observations used for simulation.")

    if not REGIME_YEARS:
        st.error("❌ Slab file could not be loaded — check PIT_slabs_2026.xlsx.")
        st.stop()
    _def_regime_idx = (REGIME_YEARS.index(int(selected_year))
                       if int(selected_year) in REGIME_YEARS else len(REGIME_YEARS) - 1)
    base_regime_year = st.selectbox("Base Slab Regime", REGIME_YEARS, index=_def_regime_idx,
                                    format_func=_fy_label,
                                    help="Slab rates and surcharge used as the base. "
                                         "All changes are compared against this regime.")
    if int(base_regime_year) != int(selected_year):
        st.caption(f"ℹ️ Applying {_fy_label(base_regime_year)} slabs to {selected_year} data.")

    # Clear stale results / lab tables when the data year or base regime changes
    _ctx = (selected_year, base_regime_year)
    if st.session_state.get('_ctx') != _ctx:
        st.session_state['_ctx'] = _ctx
        st.session_state.results = {}
        st.session_state.lab_slabs = None

    uplift_target = 0.0  # Policy Lab has no revenue target; Auto Optimize sets this below


    if mode == "Auto Optimize":
        st.markdown("### Targets")
        uplift_target = st.slider("Change in Revenue (%)", 0.0, 15.0, 0.0, 0.5)
        st.markdown("### Scope")
        run_sal = st.checkbox("Optimize Salaried", value=True)
        run_nsal = st.checkbox("Optimize Non-Salaried", value=True)
        run_aop = st.checkbox("Optimize AOP", value=True)
        run_cons = st.checkbox("Optimize NSC (NS + AOP combined)", value=False)
        
        if st.button("🚀 Auto-Optimize Policy", type="primary"):
            st.session_state.results = {}
            y_grid = np.arange(0, 20_000_001, 100_000)
            groups = []
            if run_sal: groups.append('Salaried')
            if run_nsal: groups.append('Non-Salaried')
            if run_aop: groups.append('AOP')
            if run_cons: groups.append('NSC')

            for g_type in groups:
                df_slabs_agg['_norm'] = df_slabs_agg['taxpayer_type'].apply(_norm)
                g_agg = df_slabs_agg[(df_slabs_agg['year'] == selected_year) & 
                                     (df_slabs_agg['_norm'] == _norm(g_type))].copy()
                if g_agg.empty: continue
                total_tax = g_agg['normal_income_tax_920000'].sum()
                
                # Base = chosen slab regime
                base_slabs = _get_truth_slabs(g_type, base_regime_year)
                if base_slabs is None:
                    st.warning(f"⚠️ No {g_type} slabs for {_fy_label(base_regime_year)}; skipped.")
                    continue
                base_rev, _calib, _cwarn = _regime_base_revenue(g_agg, g_type, selected_year,
                                                                base_regime_year, total_tax, base_slabs)
                if _cwarn: st.warning(f"⚠️ {g_type}: {_cwarn}")
                
                with st.spinner(f"⏳ Optimizing {g_type} …"):
                    t0 = time.time()
                    res = optimize_schedule(g_agg, base_slabs, base_rev, y_grid, base_rev * (1 + uplift_target/100))
                    res.update({'g_type': g_type, 'elapsed': time.time()-t0, 'base_slabs_df': base_slabs})
                    st.session_state.results[g_type] = res
            st.success("✅ Done!")

    elif mode == "Policy Lab":
        st.markdown("### Policy Lab Setup")
        lab_type = st.selectbox("Taxpayer Type", ["Salaried", "Non-Salaried", "AOP", "NSC"],
                                help="NSC = Non-Salaried + AOP combined. Analyse it on its own; do not add NSC results to NS or AOP results (that double-counts).")

        
        # Track active lab type to handle switching
        if 'lab_type_active' not in st.session_state:
            st.session_state.lab_type_active = lab_type

        # Initialize Lab Slabs if empty OR if we switched types
        df_slabs_agg['_norm'] = df_slabs_agg['taxpayer_type'].apply(_norm)
        g_agg = df_slabs_agg[(df_slabs_agg['year'] == selected_year) & 
                             (df_slabs_agg['_norm'] == _norm(lab_type))].copy()
        total_tax = g_agg['normal_income_tax_920000'].sum()
        
        lab_nrm = _norm(lab_type)
        # Base = chosen slab regime
        base_slabs_raw = _get_truth_slabs(lab_type, base_regime_year)
        if base_slabs_raw is None:
            st.error(f"❌ No {lab_type} slabs for {_fy_label(base_regime_year)} in the slab file.")
            st.stop()
        base_list_calib = _schedule_to_list(base_slabs_raw)
        base_rev_lab, _calib_lab, _cwarn_lab = _regime_base_revenue(
            g_agg, lab_type, selected_year, base_regime_year, total_tax, base_slabs_raw)
        if _cwarn_lab:
            st.warning(f"⚠️ {_cwarn_lab}")
        else:
            st.caption(f"Base revenue ({_fy_label(base_regime_year)} on {selected_year} data): "
                       f"PKR {base_rev_lab/1e9:,.2f}B · calibration factor {_calib_lab:.3f}")

        if st.session_state.lab_slabs is None or st.session_state.lab_type_active != lab_type:
            st.session_state.lab_slabs = base_slabs_raw[base_slabs_raw['upper_bound'] > base_slabs_raw['lower_bound']].copy().reset_index(drop=True)
            st.session_state.lab_type_active = lab_type
            
        st.markdown("---")
        st.write("**Quick Actions**")
        if st.button("Reset to Base Regime"):
            st.session_state.lab_slabs = None
            if 'lab_sur_thresh' in st.session_state: del st.session_state['lab_sur_thresh']
            if 'lab_sur_rate'   in st.session_state: del st.session_state['lab_sur_rate']
            if 'lab_filer_chg'  in st.session_state: del st.session_state['lab_filer_chg']
            st.rerun()

        st.info("💡 **Policy Lab Guide**:\n- **Double-click** a cell to edit rates.\n- **Add Slabs**: Click the '+' at the bottom.\n- **Remove Slabs**: Select a row, press Delete.\n- **Final Slab**: Leave Upper Bound empty — treated as 'Above'.")

# ───────────────────────── Unified Schedule ─────────────────────────
_UNI_GROUPS = ('Salaried', 'Non-Salaried', 'AOP')
_UNI_COL = {'Salaried': '#003B5C', 'Non-Salaried': '#C8102E', 'AOP': '#E39B00'}
_UNI_PTS = (1_000_000, 3_000_000, 10_000_000, 20_000_000)   # incomes where ETR is compared
_UNI_GAP = (1_000_000, 20_000_000)                          # ETR gap = ETR(top) - ETR(bottom)

def _uni_band_label(lo, up):
    return f"{lo/1e6:.2f}M+" if np.isinf(up) else f"{lo/1e6:.2f}M–{up/1e6:.2f}M"

def _uni_bands(g_type):
    """Band rows for one taxpayer type in the selected data year (band-average approximation)."""
    d = df_slabs_agg[(df_slabs_agg['year'] == selected_year) &
                     (df_slabs_agg['taxpayer_type'].apply(_norm) == _norm(g_type))]
    d = d[pd.to_numeric(d['total_filers'], errors='coerce') > 0].sort_values('lower_bound')
    out = pd.DataFrame({
        'lower':      d['lower_bound'].astype(float).values,
        'upper':      d['upper_bound'].astype(float).values,
        'filers':     d['total_filers'].astype(float).values,
        'income':     d['taxable_income_9100'].astype(float).values,
        'actual_tax': pd.to_numeric(d['normal_income_tax_920000'], errors='coerce').fillna(0).astype(float).values,
    })
    out['band'] = [_uni_band_label(l, u) for l, u in zip(out['lower'], out['upper'])]
    out['y_pp'] = out['income'] / out['filers']
    return out

def _uni_sur_mult(y, sur, include_sur):
    y = np.atleast_1d(np.asarray(y, float))
    if include_sur and sur.get('threshold', 0) > 0 and sur.get('rate', 0) > 0:
        return np.where(y >= sur['threshold'], 1.0 + sur['rate'], 1.0)
    return np.ones(len(y))

def _uni_tax_pp(sch, sur, y, include_sur):
    y = np.atleast_1d(np.asarray(y, float))
    return compute_tax(sch, y) * _uni_sur_mult(y, sur, include_sur)

def _uni_make(bounds, rates):
    """Schedule from taxed-slab start points `bounds` (first = exemption limit) and their rates."""
    sch = [{'lower': 0.0, 'upper': float(bounds[0]), 'rate': 0.0}]
    for i, (b, r) in enumerate(zip(bounds, rates)):
        up = float(bounds[i + 1]) if i + 1 < len(bounds) else np.inf
        sch.append({'lower': float(b), 'upper': up, 'rate': float(r)})
    return sch

def _uni_from_truth(sch_list):
    """Truth schedule (first slab = 0% exemption) -> (bounds, rates) of the taxed slabs."""
    return ([sch_list[j - 1]['upper'] for j in range(1, len(sch_list))],
            [sch_list[j]['rate'] for j in range(1, len(sch_list))])

def _uni_nnls(A, b):
    """Non-negative least squares (Lawson–Hanson): minimise ||A x − b|| subject to x ≥ 0."""
    n = A.shape[1]
    x = np.zeros(n); P = np.zeros(n, dtype=bool)
    tol = 1e-10 * max(1.0, np.abs(A).max()) * max(1.0, np.abs(b).max())
    w = A.T @ (b - A @ x)
    for _ in range(5 * n + 10):
        if P.all() or (w[~P] <= tol).all():
            break
        j = np.where(~P)[0][np.argmax(w[~P])]
        P[j] = True
        for _ in range(5 * n + 10):
            z = np.zeros(n)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > 0).all():
                x = z
                break
            neg = P & (z <= 0)
            den = x[neg] - z[neg]
            alpha = np.min(np.where(den > 0, x[neg] / np.where(den > 0, den, 1), 0.0))
            x = x + alpha * (z - x)
            P = P & (x > 1e-15)
            x[~P] = 0.0
            if not P.any():
                break
        w = A.T @ (b - A @ x)
    return x

def _uni_kakwani(y, n, tax_pp):
    """Kakwani index = concentration index of tax − Gini of pre-tax income (units ranked by income)."""
    o = np.argsort(y, kind='mergesort')
    y, n, t = np.asarray(y)[o], np.asarray(n)[o], np.asarray(tax_pp)[o]
    X = np.concatenate([[0], np.cumsum(n) / n.sum()])
    Y = np.concatenate([[0], np.cumsum(n * y) / (n * y).sum()])
    T = np.concatenate([[0], np.cumsum(n * t) / (n * t).sum()]) if (n * t).sum() > 0 else Y
    gini = 1 - np.sum(np.diff(X) * (Y[1:] + Y[:-1]))
    conc = 1 - np.sum(np.diff(X) * (T[1:] + T[:-1]))
    return conc - gini

def _uni_etr(sch, sur, inc_sur, pts):
    pts = np.asarray(pts, float)
    return _uni_tax_pp(sch, sur, pts, inc_sur) / pts

if mode == "Unified Schedule":
    import plotly.graph_objects as go

    st.header("🟰 Unified Schedule — revenue-neutral, progressive options")
    st.caption(f"One schedule for Salaried, Non-Salaried and AOP. Data: {selected_year} returns · "
               f"Base: {_fy_label(base_regime_year)} schedules. Every option collects the same combined "
               "revenue as the base. Band-average approximation: each band is taxed at its average "
               "declared income per filer.")

    _bands = {g: _uni_bands(g) for g in _UNI_GROUPS}
    _missing = [g for g in _UNI_GROUPS if _bands[g].empty]
    if _missing:
        st.error(f"❌ No {', '.join(_missing)} rows for {selected_year} in the uploaded file.")
        st.stop()

    _own, _own_sur, _dat, _dat_sur = {}, {}, {}, {}
    for g in _UNI_GROUPS:
        _df = _get_truth_slabs(g, base_regime_year)
        if _df is None:
            st.error(f"❌ No {g} slabs for {_fy_label(base_regime_year)} in the slab file.")
            st.stop()
        _own[g] = _schedule_to_list(_df)
        _own_sur[g] = _get_truth_surcharge(g, base_regime_year)
        _dd = _get_truth_slabs(g, selected_year)
        _dat[g] = _schedule_to_list(_dd) if _dd is not None else None
        _dat_sur[g] = _get_truth_surcharge(g, selected_year)

    # ── Settings ──
    c1, c2, c3 = st.columns(3)
    _inc_sur = c1.checkbox("Include surcharge (s.4AB)", value=True,
                           help="When ticked, surcharge is included in the base, the calibration and every option. "
                                "When unticked, all of these are recalculated without surcharge. "
                                "Set this to match whether field 920000 includes surcharge.")
    _u_sur_rate = c2.number_input("Unified surcharge rate (%)", 0.0, 50.0, 10.0, 0.5, disabled=not _inc_sur) / 100.0
    _u_sur_thr = c3.number_input("Unified surcharge threshold (PKR)", 0.0, 1e9, 10_000_000.0, 500_000.0,
                                 disabled=not _inc_sur)
    _u_sur = {'threshold': _u_sur_thr, 'rate': _u_sur_rate}

    c4, c5, c6 = st.columns(3)
    _step = c4.number_input("Minimum rate step between slabs (percentage points)", 0.1, 20.0, 1.0, 0.5,
                            help="Each slab's rate must be at least this much higher than the slab below. "
                                 "There is no international standard for this value.") / 100.0
    _cap_on = c5.checkbox("Set a top-rate cap", value=False)
    _cap = c5.number_input("Top-rate cap (%)", 1.0, 100.0, 45.0, 0.5, disabled=not _cap_on) / 100.0
    _nslab = c6.slider("Number of slabs (incl. 0% slab)", 4, 10, (6, 8))

    # Calibration factor per group: actual tax / formula tax under the data year's own schedule
    _factor = {}
    for g in _UNI_GROUPS:
        b = _bands[g]
        if _dat[g] is None:
            _factor[g] = 1.0
            st.warning(f"⚠️ No {g} slabs for data year {selected_year}: {g} revenue is uncalibrated.")
            continue
        _form = (_uni_tax_pp(_dat[g], _dat_sur[g], b['y_pp'], _inc_sur) * b['filers']).sum()
        _factor[g] = b['actual_tax'].sum() / _form if _form > 0 else 1.0

    # Stacked arrays over all groups and bands
    _Y = np.concatenate([_bands[g]['y_pp'].values for g in _UNI_GROUPS])
    _N = np.concatenate([_bands[g]['filers'].values for g in _UNI_GROUPS])
    _F = np.concatenate([np.full(len(_bands[g]), _factor[g]) for g in _UNI_GROUPS])
    _GRP = np.concatenate([np.full(len(_bands[g]), g, dtype=object) for g in _UNI_GROUPS])
    _BASE_PP = np.concatenate([_factor[g] * _uni_tax_pp(_own[g], _own_sur[g], _bands[g]['y_pp'], _inc_sur)
                               for g in _UNI_GROUPS])
    _base_rev = {g: (_BASE_PP * _N)[_GRP == g].sum() for g in _UNI_GROUPS}
    _base_total = sum(_base_rev.values())
    _exempt = _uni_from_truth(_own['Salaried'])[0][0]
    _TX = _Y > _exempt                                   # filers above the exemption limit
    _W = np.where(_TX, _N, 0.0); _W = _W / _W.sum() if _W.sum() > 0 else _W

    # Today's progressivity benchmarks
    _K_today = _uni_kakwani(_Y, _N, _BASE_PP)
    _etr_S = _uni_etr(_own['Salaried'], _own_sur['Salaried'], _inc_sur, _UNI_PTS)
    _etr_N = _uni_etr(_own['Non-Salaried'], _own_sur['Non-Salaried'], _inc_sur, _UNI_PTS)
    _nS = _N[_TX & (_GRP == 'Salaried')].sum(); _nN = _N[_TX & (_GRP != 'Salaried')].sum()
    _wS = _nS / (_nS + _nN) if (_nS + _nN) > 0 else 0.5
    _etr_today = _wS * _etr_S + (1 - _wS) * _etr_N
    _gi = (_UNI_PTS.index(_UNI_GAP[0]), _UNI_PTS.index(_UNI_GAP[1]))
    _gap_today = _etr_today[_gi[1]] - _etr_today[_gi[0]]

    def _evaluate(name, bounds, rates):
        """All results and rule checks for one unified schedule."""
        sch = _uni_make(bounds, rates)
        new_pp = _F * _uni_tax_pp(sch, _u_sur, _Y, _inc_sur)
        d_etr = np.where(_Y > 0, (new_pp - _BASE_PP) / np.where(_Y > 0, _Y, 1), 0.0)
        grid = np.arange(_exempt + 50_000, 50_000_001, 25_000, dtype=float)
        etr_grid = _uni_etr(sch, _u_sur, _inc_sur, grid)
        etr_pts = _uni_etr(sch, _u_sur, _inc_sur, _UNI_PTS)
        rr = np.asarray(rates, float)
        steps = np.diff(np.concatenate([[0.0], rr]))
        row = {'Option': name, 'Slabs (incl. 0%)': len(sch), 'Top rate': rr.max(),
               'Avg ETR change (pts)': 100 * np.sqrt(np.sum(_W * d_etr ** 2)),
               'Kakwani': _uni_kakwani(_Y, _N, new_pp),
               'ETR gap 1M→20M (pts)': 100 * (etr_pts[_gi[1]] - etr_pts[_gi[0]])}
        row['✔ Rates rise'] = bool(np.all(steps >= _step - 1e-9))
        row['✔ ETR rises'] = bool(np.all(np.diff(etr_grid) >= -1e-12))
        row['✔ Gap ≥ today'] = bool(row['ETR gap 1M→20M (pts)'] >= 100 * _gap_today - 1e-6)
        row['✔ Kakwani ≥ today'] = bool(row['Kakwani'] >= _K_today - 1e-9)
        row['✔ Cap'] = bool(rr.max() <= _cap + 1e-9) if _cap_on else True
        row['Passes all'] = all(row[k] for k in row if k.startswith('✔'))
        tot = 0.0
        for g in _UNI_GROUPS:
            u = (new_pp * _N)[_GRP == g].sum(); tot += u
            row[f'{g} change (PKR B)'] = (u - _base_rev[g]) / 1e9
            row[f'{g} change (%)'] = (u - _base_rev[g]) / _base_rev[g] if _base_rev[g] > 0 else np.nan
        row['Total change (PKR B)'] = (tot - _base_total) / 1e9
        chg = new_pp - _BASE_PP
        row['Filers paying more'] = _N[chg > 0.5].sum()
        row['Filers paying less'] = _N[chg < -0.5].sum()
        return row, {'sch': sch, 'bounds': list(bounds), 'rates': list(rr), 'new_pp': new_pp, 'etr_pts': etr_pts}

    def _solve_layout(bounds):
        """Rates on fixed slab limits that change filers' ETR the least (filer-weighted, all groups),
        with each rate at least `_step` above the one below, and revenue exactly equal to the base."""
        b = np.asarray(bounds, float); J = len(b)
        widths = np.append(np.diff(b), np.inf)
        yt, wt, ft = _Y[_TX], _W[_TX], _F[_TX]
        U = np.clip(_Y[:, None] - b[None, :], 0, widths[None, :])           # income inside each slab
        M = U * (_uni_sur_mult(_Y, _u_sur, _inc_sur) * _F)[:, None]          # calibrated tax per unit rate
        Mt = M[_TX]
        L = np.tril(np.ones((J, J))); jv = _step * np.arange(1, J + 1)
        c = (_N[:, None] * M).sum(axis=0)                                    # revenue per unit rate
        sw = np.sqrt(wt) / yt
        A1 = sw[:, None] * (Mt @ L); h1 = sw * (_BASE_PP[_TX] - Mt @ jv)
        lam = np.sqrt(1e4)
        A2 = lam * (c @ L)[None, :] / _base_total; h2 = np.array([lam * (1 - (c @ jv) / _base_total)])
        d = _uni_nnls(np.vstack([A1, A2]), np.concatenate([h1, h2]))
        Ld = L @ d
        num, den = _base_total - c @ jv, c @ Ld
        if num < 0 or den <= 0:
            return None                                 # minimum steps alone already exceed the base revenue
        return jv + (num / den) * Ld

    _bS, _rS = _uni_from_truth(_own['Salaried'])
    _bN, _rN = _uni_from_truth(_own['Non-Salaried'])

    def _neutral_scale(bounds, rates):
        sch = _uni_make(bounds, rates)
        tot = (_F * _uni_tax_pp(sch, _u_sur, _Y, _inc_sur) * _N).sum()
        return np.asarray(rates) * (_base_total / tot) if tot > 0 else np.asarray(rates)

    def _nice(v):
        return round(v / 1e5) * 1e5 if v < 2e6 else (round(v / 5e5) * 5e5 if v < 1e7 else round(v / 1e6) * 1e6)

    # Pool of possible slab limits: 2026 limits of both schedules + percentiles of taxpayers' incomes
    _base_lims = sorted(set(_bS[1:]) | set(_bN[1:]))
    _o = np.argsort(_Y[_TX]); _cw = np.cumsum(_N[_TX][_o]) / _N[_TX].sum()
    _pct = [_nice(_Y[_TX][_o][min(np.searchsorted(_cw, q), len(_cw) - 1)]) for q in (0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99)]
    _pool = list(_base_lims)
    for v in _pct:
        if v > _exempt + 1e5 and all(abs(v - p) >= 1.5e5 for p in _pool):
            _pool.append(v)
    _pool = sorted(_pool)

    _settings_key = (selected_year, base_regime_year, _inc_sur, _u_sur_rate, _u_sur_thr, _step, _cap_on,
                     _cap, _nslab, tuple(_pool))
    st.caption(f"Possible slab limits tried (PKR M): {', '.join(f'{p/1e6:g}' for p in _pool)} — from the "
               f"{base_regime_year} salaried and non-salaried limits plus income percentiles of taxpayers in your data. "
               f"The exemption limit stays at PKR {_exempt:,.0f}.")

    if st.button("🔍 Find unified schedules", type="primary"):
        from itertools import combinations
        cands = []
        with st.spinner("Trying slab layouts…"):
            for J in range(_nslab[0] - 1, _nslab[1]):          # taxed slabs = slabs − 1
                for extra in combinations(_pool, J - 1):
                    bnd = [_exempt] + list(extra)
                    r = _solve_layout(bnd)
                    if r is None:
                        continue
                    row, det = _evaluate("", bnd, r)
                    cands.append((row, det))
            refA = _evaluate(f"A · {base_regime_year} salaried slabs, rates scaled (reference)", _bS, _neutral_scale(_bS, _rS))
            refB = _evaluate(f"B · {base_regime_year} non-salaried slabs, rates scaled (reference)", _bN, _neutral_scale(_bN, _rN))
        st.session_state['uni_res'] = {'key': _settings_key, 'cands': cands, 'A': refA, 'B': refB}

    _res = st.session_state.get('uni_res')
    if not _res:
        st.info("Set the options above and click **Find unified schedules**.")
        st.stop()
    if _res['key'] != _settings_key:
        st.warning("⚠️ Settings or data changed since the last search — click **Find unified schedules** again.")
        st.stop()

    # ── Rule summary ──
    st.subheader("Progressivity rules")
    st.markdown(
        f"- **Rates rise:** each slab's rate is at least {_step*100:.1f} points higher than the slab below.\n"
        f"- **ETR rises:** the share of income paid as tax goes up at every income above PKR {_exempt:,.0f}.\n"
        f"- **Gap ≥ today:** ETR at PKR 20M minus ETR at PKR 1M is at least today's gap of "
        f"**{100*_gap_today:.2f} points** (today = salaried and non-salaried/AOP schedules, weighted by number of taxpayers).\n"
        f"- **Kakwani ≥ today:** overall progressivity of tax across all filers is at least today's **{_K_today:.4f}**.\n"
        + (f"- **Cap:** no rate above {_cap*100:.1f}%.\n" if _cap_on else "")
        + "- **Winner:** among options passing every rule, the one that changes filers' ETR the least.")

    cands = _res['cands']
    passing = sorted([c for c in cands if c[0]['Passes all']], key=lambda c: c[0]['Avg ETR change (pts)'])
    shown = []
    if passing:
        for i, (row, det) in enumerate(passing[:5], start=1):
            row = dict(row); row['Option'] = f"{'🏆 Winner' if i == 1 else f'#{i}'} · {row['Slabs (incl. 0%)']} slabs"
            shown.append((row, det))
        st.success(f"✅ {len(passing)} of {len(cands)} slab layouts pass every rule. Showing the 5 with the least change.")
    else:
        rules = ['✔ Rates rise', '✔ ETR rises', '✔ Gap ≥ today', '✔ Kakwani ≥ today'] + (['✔ Cap'] if _cap_on else [])
        fails = {r: sum(1 for c in cands if not c[0][r]) for r in rules}
        st.error("❌ No slab layout passes every rule. Layouts failing each rule: " +
                 " · ".join(f"{r.replace('✔ ', '')}: {v} of {len(cands)}" for r, v in fails.items()) +
                 ". Showing the 5 closest options; relax the blocking rule to find a passing one.")
        for i, (row, det) in enumerate(sorted(cands, key=lambda c: c[0]['Avg ETR change (pts)'])[:5], start=1):
            row = dict(row); row['Option'] = f"#{i} (fails) · {row['Slabs (incl. 0%)']} slabs"
            shown.append((row, det))
    shown += [_res['A'], _res['B']]

    st.subheader("Options compared")
    st.caption(f"Base combined revenue: PKR {_base_total/1e9:,.1f}B (Salaried {_base_rev['Salaried']/1e9:,.1f}B · "
               f"Non-Salaried {_base_rev['Non-Salaried']/1e9:,.1f}B · AOP {_base_rev['AOP']/1e9:,.1f}B). "
               f"Calibration factors: S {_factor['Salaried']:.3f} · NS {_factor['Non-Salaried']:.3f} · "
               f"AOP {_factor['AOP']:.3f}. Today: Kakwani {_K_today:.4f}, ETR gap {100*_gap_today:.2f} points. "
               "Avg ETR change = typical change in filers' ETR (root-mean-square, filers above the exemption). "
               "Gainer/loser counts treat each band as moving together.")
    _tbl = pd.DataFrame([r for r, _ in shown])
    for col in [c for c in _tbl.columns if c.startswith('✔') or c == 'Passes all']:
        _tbl[col] = _tbl[col].map({True: '✅', False: '❌'})
    _fmt = {'Top rate': '{:.1%}', 'Avg ETR change (pts)': '{:.2f}', 'Kakwani': '{:.4f}',
            'ETR gap 1M→20M (pts)': '{:.2f}', 'Total change (PKR B)': '{:+,.2f}',
            'Filers paying more': '{:,.0f}', 'Filers paying less': '{:,.0f}'}
    for g in _UNI_GROUPS:
        _fmt[f'{g} change (PKR B)'] = '{:+,.1f}'; _fmt[f'{g} change (%)'] = '{:+.1%}'
    if not _cap_on:
        _tbl = _tbl.drop(columns=['✔ Cap'])
    st.dataframe(_tbl.style.format(_fmt, na_rep='—'), use_container_width=True, hide_index=True)

    # ETR at fixed incomes
    st.subheader("ETR at selected incomes")
    _etr_rows = [{'Schedule': f'Today: salaried ({base_regime_year})', **{f'{p/1e6:g}M': v for p, v in zip(_UNI_PTS, _etr_S)}},
                 {'Schedule': f'Today: non-salaried / AOP ({base_regime_year})', **{f'{p/1e6:g}M': v for p, v in zip(_UNI_PTS, _etr_N)}},
                 {'Schedule': 'Today: weighted by taxpayers', **{f'{p/1e6:g}M': v for p, v in zip(_UNI_PTS, _etr_today)}}]
    for row, det in shown:
        _etr_rows.append({'Schedule': row['Option'], **{f'{p/1e6:g}M': v for p, v in zip(_UNI_PTS, det['etr_pts'])}})
    _etr_df = pd.DataFrame(_etr_rows)
    _etr_df['Gap 1M→20M (pts)'] = 100 * (_etr_df['20M'] - _etr_df['1M'])
    st.dataframe(_etr_df.style.format({**{f'{p/1e6:g}M': '{:.2%}' for p in _UNI_PTS}, 'Gap 1M→20M (pts)': '{:.2f}'}),
                 use_container_width=True, hide_index=True)

    _grid = np.arange(100_000, 25_000_001, 50_000, dtype=float)
    fig_etr = go.Figure()
    fig_etr.add_scatter(x=_grid/1e6, y=_uni_etr(_own['Salaried'], _own_sur['Salaried'], _inc_sur, _grid), mode='lines',
                        name=f"Today: salaried ({base_regime_year})", line=dict(color='#003B5C', dash='dash'))
    fig_etr.add_scatter(x=_grid/1e6, y=_uni_etr(_own['Non-Salaried'], _own_sur['Non-Salaried'], _inc_sur, _grid), mode='lines',
                        name=f"Today: non-salaried / AOP ({base_regime_year})", line=dict(color='#C8102E', dash='dash'))
    for (row, det), colr in zip([shown[0], shown[-2], shown[-1]], ['#2E7D32', '#6A1B9A', '#00838F']):
        fig_etr.add_scatter(x=_grid/1e6, y=_uni_etr(det['sch'], _u_sur, _inc_sur, _grid), mode='lines',
                            name=row['Option'], line=dict(color=colr, width=3))
    for p in _UNI_GAP:
        fig_etr.add_vline(x=p/1e6, line_dash='dot', line_color='#999')
    fig_etr.update_layout(title="Statutory ETR: today vs unified options (dotted lines: 1M and 20M, where the gap is measured)",
                          height=440, plot_bgcolor='white', xaxis_title="Taxable income (PKR million)",
                          yaxis_title="Effective tax rate", yaxis_tickformat='.0%',
                          legend=dict(orientation='h', y=-0.25), margin=dict(t=50, b=30, l=30, r=10))
    fig_etr.update_xaxes(showgrid=True, gridcolor='#E8EDF2'); fig_etr.update_yaxes(showgrid=True, gridcolor='#E8EDF2')
    st.plotly_chart(fig_etr, use_container_width=True, key="uni_etr")

    # ── Detail for one option ──
    st.subheader("Option detail")
    _names = [r['Option'] for r, _ in shown]
    _pick = st.radio("Show details for", _names)
    _row, _det = shown[_names.index(_pick)]
    _sch = _det['sch']

    _cum, _rows = 0.0, []
    for s in _sch:
        _rows.append({'From (PKR)': s['lower'] + (1 if s['lower'] > 0 else 0),
                      'To (PKR)': '' if np.isinf(s['upper']) else f"{s['upper']:,.0f}",
                      'Rate': s['rate'], 'Tax at start of slab (PKR)': _cum})
        if not np.isinf(s['upper']):
            _cum += (s['upper'] - s['lower']) * s['rate']
    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown("**Unified schedule**")
        st.dataframe(pd.DataFrame(_rows).style.format({'From (PKR)': '{:,.0f}', 'Rate': '{:.2%}',
                                                       'Tax at start of slab (PKR)': '{:,.0f}'}),
                     use_container_width=True, hide_index=True)
        if _inc_sur:
            st.caption(f"Plus surcharge {_u_sur_rate:.1%} of tax where income ≥ PKR {_u_sur_thr:,.0f}.")
    with cB:
        _items = []
        for g in _UNI_GROUPS:
            pct = _row[f'{g} change (%)']
            u = _base_rev[g] + _row[f'{g} change (PKR B)'] * 1e9
            _items.append(f"<div class='imf-metric-card'><div class='imf-mc-label'>{g}</div>"
                          f"<div class='imf-mc-value'>PKR {_base_rev[g]/1e9:,.1f}B → {u/1e9:,.1f}B</div>"
                          f"<div class='{'imf-delta-pos' if pct >= 0 else 'imf-delta-neg'}'>"
                          f"{'▲' if pct >= 0 else '▼'} {pct:+.1%}</div></div>")
        st.markdown("<div class='imf-metric-row'>" + "".join(_items) + "</div>", unsafe_allow_html=True)

    _band_tbls = {}
    fig_chg = go.Figure()
    _cats = []
    for g in _UNI_GROUPS:
        m = _GRP == g
        b = _bands[g]
        new_pp = _det['new_pp'][m]; base_pp = _BASE_PP[m]
        etr_b = np.where(b['y_pp'] > 0, base_pp / b['y_pp'].where(b['y_pp'] > 0, 1), 0.0)
        etr_n = np.where(b['y_pp'] > 0, new_pp / b['y_pp'].where(b['y_pp'] > 0, 1), 0.0)
        fig_chg.add_bar(x=b['band'], y=100 * (etr_n - etr_b), name=g, marker_color=_UNI_COL[g])
        _cats += [x for x in b.sort_values('lower')['band'] if x not in _cats]
        _band_tbls[g] = pd.DataFrame({
            'Band': b['band'], 'Filers': b['filers'].round(0).astype(int),
            'Avg declared income (PKR)': b['y_pp'].round(0),
            'Base tax / filer (PKR)': np.round(base_pp, 0), 'Unified tax / filer (PKR)': np.round(new_pp, 0),
            'Change / filer (PKR)': np.round(new_pp - base_pp, 0),
            'Base ETR': etr_b, 'Unified ETR': etr_n, 'ETR change (pts)': 100 * (etr_n - etr_b),
            'Total change (PKR M)': np.round((new_pp - base_pp) * b['filers'].values / 1e6, 1)})
    _lo_map = {}
    for g in _UNI_GROUPS:
        for lb, lo in zip(_bands[g]['band'], _bands[g]['lower']):
            _lo_map.setdefault(lb, lo)
    _cats = sorted(_lo_map, key=_lo_map.get)
    fig_chg.update_layout(title="Change in ETR by income band (percentage points)", barmode='group', height=420,
                          plot_bgcolor='white', xaxis_title="Income band (declared taxable income)",
                          yaxis_title="ETR change (points)", margin=dict(t=50, b=30, l=30, r=10),
                          legend=dict(orientation='h', y=-0.3))
    fig_chg.update_xaxes(categoryorder='array', categoryarray=_cats)
    fig_chg.update_yaxes(showgrid=True, gridcolor='#E8EDF2', zeroline=True, zerolinecolor='#888')
    st.plotly_chart(fig_chg, use_container_width=True, key="uni_chg")

    for g in _UNI_GROUPS:
        with st.expander(f"📋 {g}: band-by-band detail"):
            st.dataframe(_band_tbls[g].style.format({
                'Avg declared income (PKR)': '{:,.0f}', 'Base tax / filer (PKR)': '{:,.0f}',
                'Unified tax / filer (PKR)': '{:,.0f}', 'Change / filer (PKR)': '{:+,.0f}',
                'Base ETR': '{:.2%}', 'Unified ETR': '{:.2%}', 'ETR change (pts)': '{:+.2f}',
                'Total change (PKR M)': '{:+,.1f}', 'Filers': '{:,}'}), use_container_width=True, hide_index=True)

    _buf = _io.BytesIO()
    with pd.ExcelWriter(_buf, engine='openpyxl') as _xw:
        pd.DataFrame({'Item': ['Data year', 'Base regime', 'Groups', 'Surcharge included', 'Unified surcharge',
                               'Minimum rate step (pts)', 'Top-rate cap', 'Slabs tried (incl. 0%)',
                               'Slab limits tried (PKR)', 'Kakwani today', 'ETR gap today (pts)',
                               'Calibration S', 'Calibration NS', 'Calibration AOP', 'Option shown in detail'],
                      'Value': [selected_year, _fy_label(base_regime_year), 'Salaried, Non-Salaried, AOP', _inc_sur,
                                f"{_u_sur_rate:.1%} above PKR {_u_sur_thr:,.0f}" if _inc_sur else 'n/a',
                                _step * 100, f"{_cap:.1%}" if _cap_on else 'none', f"{_nslab[0]}–{_nslab[1]}",
                                ', '.join(f'{p:,.0f}' for p in _pool), round(_K_today, 4), round(100 * _gap_today, 2),
                                round(_factor['Salaried'], 4), round(_factor['Non-Salaried'], 4),
                                round(_factor['AOP'], 4), _pick]}).to_excel(_xw, sheet_name='Settings', index=False)
        _tbl.to_excel(_xw, sheet_name='Options', index=False)
        _etr_df.to_excel(_xw, sheet_name='ETR_points', index=False)
        for i, (row, det) in enumerate(shown, start=1):
            pd.DataFrame([{'From (PKR)': s['lower'], 'To (PKR)': s['upper'], 'Rate': s['rate']} for s in det['sch']]
                         ).to_excel(_xw, sheet_name=f"Schedule_{i}", index=False)
        for g in _UNI_GROUPS:
            _band_tbls[g].to_excel(_xw, sheet_name=f"Detail_{'S' if g == 'Salaried' else ('NS' if g == 'Non-Salaried' else 'AOP')}", index=False)
    st.download_button("⬇️ Download options and tables (Excel)", _buf.getvalue(),
                       file_name=f"Unified_schedule_data{selected_year}_base{base_regime_year}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.stop()

# ───────────────────────── Main Dashboard ─────────────────────────
if mode == "Policy Lab":
    st.header(f"🧪 Policy Lab — {lab_type} Design")

    # ─── Slab Editor ───
    edited_df = st.data_editor(
        st.session_state.lab_slabs,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "lower_bound":    st.column_config.NumberColumn("Lower Bound (PKR)", format="localized", min_value=0),
            "upper_bound":    st.column_config.NumberColumn("Upper Bound (PKR)", format="localized"),
            "marginal_rate":  st.column_config.NumberColumn("MTR (decimal)",      format="%.4f", min_value=0.0, max_value=1.0)
        }
    )

    # Robustly handle the last slab’s infinity
    if not edited_df.empty:
        edited_df = edited_df.sort_values('lower_bound').reset_index(drop=True)
        last_idx = edited_df.index[-1]
        val = edited_df.loc[last_idx, 'upper_bound']
        if pd.isna(val) or val > 500_000_000 or val <= edited_df.loc[last_idx, 'lower_bound']:
            edited_df.loc[last_idx, 'upper_bound'] = np.inf
        for i in range(1, len(edited_df)):
            edited_df.loc[i-1, 'upper_bound'] = edited_df.loc[i, 'lower_bound']

    st.session_state.lab_slabs = edited_df

    # ─── Surcharge Slab Editor ───
    st.markdown("#### 📌 Surcharge Slabs")
    st.caption("Define surcharge rates by taxable income band. Once income falls in a band, that band's surcharge % is applied to the full normal tax.")

    # Build default surcharge slabs from the system truth (single threshold → one slab)
    _sur_info_default = _get_truth_surcharge(lab_type, base_regime_year)
    _def_sur_thresh   = _sur_info_default.get('threshold', 0.0)
    _def_sur_rate     = _sur_info_default.get('rate', 0.0) * 100.0  # store as %

    def _default_sur_slabs(thresh, rate_pct):
        """Build a 1-row surcharge slab DataFrame from legacy threshold+rate.
        Upper bound is stored as NaN (blank) for open-ended; inf is never put
        in the display DataFrame so the data_editor + button works correctly."""
        if thresh > 0 and rate_pct > 0:
            return pd.DataFrame([{
                'lower_bound':    float(thresh),
                'upper_bound':    float('nan'),   # blank = open-ended
                'surcharge_rate': float(rate_pct)
            }])
        # Empty table with correct dtypes
        return pd.DataFrame({
            'lower_bound':    pd.Series([], dtype='float64'),
            'upper_bound':    pd.Series([], dtype='float64'),
            'surcharge_rate': pd.Series([], dtype='float64'),
        })

    # Initialise / reset session state for surcharge slabs
    # Use a separate key per type so switching types resets cleanly
    _sur_key      = f'lab_sur_slabs_{lab_type}'
    _sur_type_key = f'lab_sur_active_type'
    if _sur_key not in st.session_state or st.session_state.get(_sur_type_key) != (lab_type, base_regime_year):
        st.session_state[_sur_key]      = _default_sur_slabs(_def_sur_thresh, _def_sur_rate)
        st.session_state[_sur_type_key] = (lab_type, base_regime_year)

    # Validate surcharge slabs
    def _validate_sur_slabs(df):
        """Returns (is_valid, error_message)."""
        if df.empty:
            return True, ""
        df2 = df.dropna(subset=['lower_bound']).copy()
        df2 = df2.sort_values('lower_bound').reset_index(drop=True)
        for i in range(len(df2) - 1):
            lo_next = df2.loc[i+1, 'lower_bound']
            hi_this = df2.loc[i, 'upper_bound']
            if pd.notna(hi_this) and not np.isinf(hi_this) and lo_next < hi_this:
                return False, f"Surcharge slabs overlap: slab {i+1} upper bound ({hi_this:,.0f}) exceeds slab {i+2} lower bound ({lo_next:,.0f})."
        return True, ""

    _sur_edited = st.data_editor(
        st.session_state[_sur_key],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "lower_bound":    st.column_config.NumberColumn("Lower Bound (PKR)",   format="localized", min_value=0),
            "upper_bound":    st.column_config.NumberColumn("Upper Bound (PKR)",   format="localized",
                                                             help="Leave blank for open-ended (last slab)."),
            "surcharge_rate": st.column_config.NumberColumn("Surcharge Rate (%)",  format="%.2f",
                                                             min_value=0.0, max_value=100.0,
                                                             help="% applied on top of normal tax.")
        },
        key=f"sur_editor_{lab_type}"
    )

    # Save edited table back to session state — keep as-is (NaN = open-ended)
    # Do NOT store np.inf here; that breaks the data_editor dynamic rows feature
    if not _sur_edited.empty:
        _sur_edited = _sur_edited.sort_values('lower_bound', na_position='last').reset_index(drop=True)

    st.session_state[_sur_key] = _sur_edited

    # Reset button
    _sr1, _sr2 = st.columns([4, 1])
    with _sr2:
        if st.button("🔄 Reset Surcharge", key="sur_reset_btn"):
            st.session_state[_sur_key] = _default_sur_slabs(_def_sur_thresh, _def_sur_rate)
            st.rerun()

    # Validate
    _sur_valid, _sur_err = _validate_sur_slabs(_sur_edited)
    if not _sur_valid:
        st.error(f"❌ Surcharge Slab Error: {_sur_err}")

    # Show active surcharge summary
    if not _sur_edited.empty and _sur_valid:
        _lines = []
        for _, _sr in _sur_edited.iterrows():
            _lo  = _sr['lower_bound']
            _hi  = _sr['upper_bound']
            _rt  = _sr['surcharge_rate']
            if pd.isna(_lo) or pd.isna(_rt): continue
            # NaN upper bound = open-ended (user left it blank)
            if pd.isna(_hi) or np.isinf(float(_hi)):
                _lines.append(f"**{_rt:.1f}%** on normal tax for income **above PKR {_lo:,.0f}**")
            else:
                _lines.append(f"**{_rt:.1f}%** on normal tax for income **PKR {_lo:,.0f} – {_hi:,.0f}**")
        if _lines:
            st.caption("⚡ Surcharge active: " + " | ".join(_lines))
        else:
            st.caption("⚡ No surcharge slabs defined.")
    else:
        st.caption("⚡ No surcharge currently applied.")

    # Convert slab table → list-of-dicts for downstream use
    def _sur_slabs_to_list(df):
        """Convert surcharge slab DataFrame to list of dicts: {lower, upper, rate}."""
        out = []
        if df.empty: return out
        df2 = df.dropna(subset=['lower_bound', 'surcharge_rate']).sort_values('lower_bound')
        for _, r in df2.iterrows():
            out.append({'lower': float(r['lower_bound']),
                        'upper': float(r['upper_bound']) if pd.notna(r['upper_bound']) else np.inf,
                        'rate':  float(r['surcharge_rate']) / 100.0})
        return out

    _lab_sur_slabs = _sur_slabs_to_list(_sur_edited)

    # ─── Filer Adjustment (same style as Surcharge Settings) ───
    st.markdown("#### 👥 Filer Count Adjustment")
    _def_filer_chg  = 0.0
    _init_filer_chg = float(st.session_state.get('lab_filer_chg', _def_filer_chg))

    fa1, fa2 = st.columns([3, 1])
    with fa1:
        new_filer_chg = st.number_input(
            "Change in Number of Filers (%)",
            min_value=-100.0, max_value=500.0,
            value=_init_filer_chg, step=1.0,
            help="Type a % change. +10 means 10% more filers (scales aggregate income Y by ×1.10). -20 means 20% fewer filers.",
            key="filer_chg_input"
        )
    with fa2:
        st.markdown("<br/>", unsafe_allow_html=True)
        if st.button("🔄 Reset Filers"):
            st.session_state['lab_filer_chg'] = _def_filer_chg
            st.rerun()

    # Persist
    st.session_state['lab_filer_chg'] = new_filer_chg
    _filer_scale = 1.0 + new_filer_chg / 100.0  # multiplier applied to Y

    if new_filer_chg != 0.0:
        st.caption(f"👥 Filer count **{new_filer_chg:+.1f}%** → Y scaled by **×{_filer_scale:.3f}**")
    else:
        st.caption("👥 No filer adjustment applied.")


    # ─── Instant Recompute ───
    if edited_df.empty:
        st.warning("⚠️ Please add at least one tax slab.")
        st.session_state.results = {}
    else:
        sch_list = _schedule_to_list(edited_df)
        is_v, err = validate_schedule(sch_list)
        if not is_v:
            st.error(f"❌ Invalid Design: {err}")
            st.session_state.results = {}
        else:
            y_grid = np.arange(0, 20_000_001, 100_000)
            res = run_manual_simulation(sch_list, g_agg, y_grid, base_rev_lab, base_list=base_list_calib)
            # ★ Apply surcharge to metrics so ALL dashboard charts reflect it
            res['metrics'] = _apply_sur_to_metrics(res['metrics'], _lab_sur_slabs)
            res.update({'g_type': lab_type, 'elapsed': 0.0, 'base_slabs_df': base_slabs_raw,
                        'lab_sur_slabs': _lab_sur_slabs, 'lab_filer_scale': _filer_scale})
            st.session_state.results = {lab_type: res}

    # Button to Refine
    if not edited_df.empty and st.button("🧙 Refine Within My Slab Structure"):
        y_grid   = np.arange(0, 20_000_001, 100_000)
        sch_list = _schedule_to_list(edited_df)
        with st.spinner("Refining..."):
            res = optimize_schedule_constrained(g_agg, sch_list, base_rev_lab * (1 + uplift_target/100), y_grid, base_list=base_list_calib)
            # ★ Apply surcharge to metrics so ALL dashboard charts reflect it
            res['metrics'] = _apply_sur_to_metrics(res['metrics'], _lab_sur_slabs)
            res.update({'g_type': lab_type, 'elapsed': 0.1, 'base_slabs_df': base_slabs_raw,
                        'lab_sur_slabs': _lab_sur_slabs, 'lab_filer_scale': _filer_scale})
            st.session_state.results = {lab_type: res}
            st.session_state.lab_slabs = res['schedule_df']
            st.rerun()

# ───────────────────────── Display Results ─────────────────────────

# Gate: require uploaded data before showing any output
_data_ready = 'uploaded_obs_bytes' in st.session_state and st.session_state.get('uploaded_obs_bytes') is not None

if not _data_ready:
    st.info("👈 Please upload your **Observations File** in the sidebar to begin analysis.")
    st.stop()

results = st.session_state.results
if not results:
    st.info("👈 Click **Auto-Optimize Policy** or adjust **Policy Lab** slabs to generate results.")
else:
    tab_labels = list(results.keys())
    tabs = st.tabs(tab_labels)
    for i, g_type in enumerate(tab_labels):
        res = results[g_type]
        m = res['metrics']
        bm = compute_metrics(_schedule_to_list(res['base_slabs_df']), res['metrics']['y'])
        
        # Apply surcharge to base metrics (bm) for accurate visualization comparison
        _s = _get_truth_surcharge(g_type, base_regime_year)
        _base_sur_for_bm = [{'lower': _s['threshold'], 'upper': np.inf, 'rate': _s['rate']}] if _s.get('threshold', 0.0) > 0 and _s.get('rate', 0.0) > 0 else []
        bm = _apply_sur_to_metrics(bm, _base_sur_for_bm)

        # ── Helpers: slab-based surcharge ────────────────────────────────────
        def _lookup_sur_rate(y_pp_val, sur_slabs):
            """Return the surcharge rate (0–1 decimal) for a given per-person income."""
            for s in sur_slabs:
                if s['lower'] <= y_pp_val < s['upper']:
                    return s['rate']
            # open-ended last slab already has upper=inf, so covers everything above
            return 0.0

        def _apply_sur_slabs_vec(y_pp_arr, sur_slabs):
            """Vectorised surcharge rate lookup → numpy array of rates (0–1)."""
            rates = np.zeros(len(y_pp_arr))
            for s in sur_slabs:
                mask = (y_pp_arr >= s['lower']) & (y_pp_arr < s['upper'])
                rates[mask] = s['rate']
            return rates

        # ── Compute slab-formula NIT Estimated for base & proposed ──────────
        def _nit_total(sch_list, y_arr, n_arr, sur_slabs):
            """NIT for aggregate-band data.
            y_arr = aggregate taxable income per band row.
            n_arr = number of persons per band row.
            Computes per-person avg income, applies slabs, scales by N.
            sur_slabs = list of {lower, upper, rate} dicts (rate in 0–1)."""
            if len(sch_list) == 0 or len(y_arr) == 0:
                return 0.0
            lws  = np.array([s['lower'] for s in sch_list])
            rs   = np.array([s['rate']  for s in sch_list])
            ups  = np.array([s['upper'] for s in sch_list])
            cum  = np.zeros(len(sch_list))
            for k in range(1, len(sch_list)):
                w = ups[k-1] - lws[k-1]
                cum[k] = cum[k-1] + (0.0 if np.isinf(w) else w) * rs[k-1]

            # Per-person average income
            n_safe = np.where(n_arr > 0, n_arr, 1.0)
            y_pp   = y_arr / n_safe

            # Apply income tax slabs
            idx    = np.clip(np.searchsorted(lws, y_pp, side='right') - 1, 0, len(sch_list)-1)
            bt_pp  = np.maximum(cum[idx] + (y_pp - lws[idx]) * rs[idx], 0.0)

            # Slab-based surcharge
            sur_rt = _apply_sur_slabs_vec(y_pp, sur_slabs) if sur_slabs else np.zeros(len(y_pp))
            nit_pp = bt_pp * (1.0 + sur_rt)

            return (nit_pp * n_arr).sum()

        # Load observation Y & N values for this g_type, filtered by year+type
        # Raw Type_Tax values in uploaded file: 'S', 'NS', 'AOP', 'NSC' (NSC = NS + AOP combined)
        _type_map   = {'Salaried': 'S', 'Non-Salaried': 'NS', 'AOP': 'AOP', 'NSC': 'NSC'}
        _tgt        = _type_map.get(g_type, g_type)
        _raw        = pd.read_excel(_io.BytesIO(st.session_state.uploaded_obs_bytes), engine='openpyxl')
        # Filter by Type_Tax
        _grp = _raw[_raw['Type_Tax'] == _tgt].copy() if 'Type_Tax' in _raw.columns else _raw.copy()
        # Filter by Year (avoids double-counting multi-year data)
        if 'Year' in _grp.columns:
            _grp = _grp[_grp['Year'] == selected_year].copy()
        _y_arr = _grp['Taxable Income (9100)'].values.astype(float) if 'Taxable Income (9100)' in _grp.columns else np.array([])

        # ── Robust filer count column detection ──────────────────────────────
        def _find_n_col(cols):
            cl = [c.lower() for c in cols]
            # Strategy 1: 'number' + 'person' or 'filer'
            for orig, lo in zip(cols, cl):
                if 'number' in lo and any(x in lo for x in ['person', 'filer']): return orig
            # Strategy 2: 'no.' or 'no ' + 'person' or 'filer'
            for orig, lo in zip(cols, cl):
                if ('no.' in lo or lo.startswith('no ')) and any(x in lo for x in ['person', 'filer']): return orig
            # Strategy 3: any column with 'persons' or 'filers' standalone
            for orig, lo in zip(cols, cl):
                if 'persons' in lo or 'filers' in lo: return orig
            # Strategy 4: numeric column code 9300
            for orig, lo in zip(cols, cl):
                if '9300' in lo: return orig
            return None

        _n_col  = _find_n_col(list(_grp.columns))
        _n_arr  = _grp[_n_col].values.astype(float) if _n_col else np.ones(len(_y_arr))

        # Debug: show column detection result
        with st.expander("🔍 Debug: Column Detection", expanded=False):
            st.write("**All columns in uploaded file:**", list(_grp.columns))
            st.write("**Filer count column detected:**", _n_col if _n_col else "❌ NOT FOUND — using N=1 (wrong!)")
            if _n_col:
                st.write("**Sample N values:**", _n_arr[:5])
                st.write("**Sample Y values:**", _y_arr[:5])
                min_len = min(5, len(_y_arr), len(_n_arr))
                if min_len > 0:
                    st.write("**Sample Y/N (per-person income):**", (_y_arr[:min_len]/_n_arr[:min_len]))
                else:
                    st.write("**Sample Y/N (per-person income):**", "Cannot compute (missing data)")

        # Surcharge slabs: lab-edited if available, else build from system truth
        def _truth_sur_to_slabs(g_type, year):
            """Convert legacy single-threshold truth surcharge → slab list."""
            s = _get_truth_surcharge(g_type, year)
            th, rt = s.get('threshold', 0.0), s.get('rate', 0.0)
            if th > 0 and rt > 0:
                return [{'lower': th, 'upper': np.inf, 'rate': rt}]
            return []

        _prop_sur_slabs = res.get('lab_sur_slabs', None)
        if _prop_sur_slabs is None:
            _prop_sur_slabs = _truth_sur_to_slabs(g_type, base_regime_year)

        # Filer scale: only applied to proposed (base always uses original Y)
        _filer_scale = res.get('lab_filer_scale', 1.0)
        _y_arr_prop  = _y_arr * _filer_scale

        # Base NIT Estimated — truth slabs + truth surcharge, original Y & N
        _base_sch      = _get_truth_slabs(g_type, base_regime_year)
        _base_sch      = _schedule_to_list(_base_sch) if _base_sch is not None else _schedule_to_list(res['base_slabs_df'])
        _base_sur_slabs = _truth_sur_to_slabs(g_type, base_regime_year)
        _nit_base      = _nit_total(_base_sch, _y_arr, _n_arr, _base_sur_slabs)

        # Proposed NIT Estimated — proposed slabs + slab surcharge + filer scale
        _nit_prop = _nit_total(res['schedule_list'], _y_arr_prop, _n_arr * _filer_scale, _prop_sur_slabs)

        _uplift_nit = (_nit_prop - _nit_base) / _nit_base if _nit_base > 0 else 0.0

        with tabs[i]:
            if _uplift_nit < -0.001:
                st.warning(f"⚠️ **Proposed NIT below baseline** (PKR {_nit_prop/1e9:,.1f}B < PKR {_nit_base/1e9:,.1f}B)")

            t_dash, t_ana, t_cmp, t_calc = st.tabs(["📈 Dashboard", "📊 ETR & CETR Heat Maps", "📋 Schedule Comparison", "🧮 Tax Calculator"])
            with t_ana:
                st.markdown(f"""
<div class="imf-section-tag">Analysis Results</div>
<h3 style="margin-top:6px;">🏆 {g_type}</h3>
""", unsafe_allow_html=True)

                # ── IMF-style metric cards ─────────────────────────────────
                total_filers   = int(_n_arr.sum()) if len(_n_arr) > 0 else m.get('total_filers', 0)
                _avg_etr_data  = _nit_base / _y_arr.sum() if _y_arr.sum() > 0 else 0.0
                max_mtr        = max([s['rate'] for s in res['schedule_list']])
                max_cetr       = m.get('band_max_jump', 0)
                _delta_arrow   = "▲" if _uplift_nit >= 0 else "▼"
                _delta_cls     = "imf-delta-pos" if _uplift_nit >= 0 else "imf-delta-neg"

                st.markdown(f"""
<div class="imf-metric-row">
  <div class="imf-metric-card">
    <div class="imf-mc-label">Base NIT Estimated</div>
    <div class="imf-mc-value">PKR {_nit_base/1e9:,.2f}B</div>
  </div>
  <div class="imf-metric-card">
    <div class="imf-mc-label">Proposed NIT Estimated</div>
    <div class="imf-mc-value">PKR {_nit_prop/1e9:,.2f}B</div>
    <div class="{_delta_cls}">{_delta_arrow} {_uplift_nit:+.2%}</div>
  </div>
  <div class="imf-metric-card">
    <div class="imf-mc-label">Number of Filers</div>
    <div class="imf-mc-value">{total_filers:,}</div>
  </div>
  <div class="imf-metric-card">
    <div class="imf-mc-label">Avg ETR (Data-Weighted)</div>
    <div class="imf-mc-value">{_avg_etr_data:.2%}</div>
  </div>
  <div class="imf-metric-card">
    <div class="imf-mc-label">MTR Max / CETR Max</div>
    <div class="imf-mc-value">{max_mtr:.1%} / {max_cetr:.2f}pp</div>
  </div>
</div>
""", unsafe_allow_html=True)

                st.markdown("---")
                y_grid = m['y']
                cmap = 'Viridis'

                fig_etr = plot_etr_heatmap(build_heatmap_dataframe(m['etr'], y_grid, bm['etr']), colorscale=cmap)
                st.plotly_chart(fig_etr, use_container_width=True, key=f"fig_etr_{g_type}")

                fig_detr = plot_detr_heatmap(build_heatmap_dataframe(m['delta_etr'], y_grid, bm['delta_etr']), colorscale=cmap)
                st.plotly_chart(fig_detr, use_container_width=True, key=f"fig_detr_{g_type}")

                # ─── Observation-level metrics using EDITED slab-based surcharge ───
                try:
                    _obs_sur_slabs = res.get('lab_sur_slabs', None)
                    if _obs_sur_slabs is None:
                        _obs_sur_slabs = _truth_sur_to_slabs(g_type, base_regime_year)

                    # Show surcharge summary
                    if _obs_sur_slabs:
                        _sur_lines = []
                        for _s in _obs_sur_slabs:
                            _lo, _hi, _rt = _s['lower'], _s['upper'], _s['rate']
                            if np.isinf(_hi):
                                _sur_lines.append(f"{_rt:.1%} on normal tax for income above PKR {_lo:,.0f}")
                            else:
                                _sur_lines.append(f"{_rt:.1%} on normal tax for income PKR {_lo:,.0f}–{_hi:,.0f}")
                        st.info("⚡ **Surcharge applied:** " + " | ".join(_sur_lines))
                    else:
                        st.info("⚡ No surcharge applied.")

                    type_mapping = {'Salaried': 'S', 'Non-Salaried': 'NS', 'AOP': 'AOP', 'NSC': 'NSC'}
                    tgt_raw  = type_mapping.get(g_type, g_type)
                    raw_obs  = pd.read_excel(_io.BytesIO(st.session_state.uploaded_obs_bytes), engine='openpyxl')
                    grp_obs  = raw_obs[raw_obs['Type_Tax'] == tgt_raw].copy() if 'Type_Tax' in raw_obs.columns else raw_obs.copy()
                    if 'Year' in grp_obs.columns:
                        grp_obs = grp_obs[grp_obs['Year'] == selected_year].copy()

                    if not grp_obs.empty and 'Taxable Income (9100)' in grp_obs.columns:
                        has_year  = 'Year'     in grp_obs.columns
                        has_ttype = 'Type_Tax' in grp_obs.columns
                        sort_cols = (["Year"] if has_year else []) + \
                                    (["Type_Tax"] if has_ttype else []) + \
                                    ["Taxable Income (9100)"]
                        grp_obs = grp_obs.sort_values(by=sort_cols).reset_index(drop=True).copy()
                        sch     = res['schedule_list']

                        y_obs    = grp_obs['Taxable Income (9100)'].values.astype(float)
                        _nc      = _find_n_col(list(grp_obs.columns))
                        n_obs    = grp_obs[_nc].values.astype(float) if _nc else np.ones(len(y_obs))
                        n_safe   = np.where(n_obs > 0, n_obs, 1.0)
                        y_pp     = y_obs / n_safe

                        lowers   = np.array([s['lower'] for s in sch])
                        rates    = np.array([s['rate']  for s in sch])
                        uppers   = np.array([s['upper'] for s in sch])

                        base_cum = np.zeros(len(sch))
                        for k in range(1, len(sch)):
                            w = uppers[k-1] - lowers[k-1]
                            base_cum[k] = base_cum[k-1] + (0.0 if np.isinf(w) else w) * rates[k-1]

                        idx      = np.clip(np.searchsorted(lowers, y_pp, side='right') - 1, 0, len(sch)-1)
                        mtr_obs  = rates[idx]
                        base_tax_pp = np.maximum(base_cum[idx] + (y_pp - lowers[idx]) * mtr_obs, 0.0)
                        base_tax = base_tax_pp * n_obs

                        # Slab-based surcharge on observation data
                        _obs_sur_rt = _apply_sur_slabs_vec(y_pp, _obs_sur_slabs) if _obs_sur_slabs else np.zeros(len(y_pp))
                        nit_est  = base_tax * (1.0 + _obs_sur_rt)
                        etr_obs  = np.where(y_obs > 0, nit_est / y_obs, 0.0)
                        detr_obs = np.zeros(len(grp_obs))
                        if has_year and has_ttype:
                            group_keys = ['Year', 'Type_Tax']
                        elif has_year:
                            group_keys = ['Year']
                        elif has_ttype:
                            group_keys = ['Type_Tax']
                        else:
                            group_keys = None

                        if group_keys:
                            for _, sub_idx in grp_obs.groupby(group_keys, sort=False).groups.items():
                                sub_idx_sorted = sorted(sub_idx)
                                sub_e = etr_obs[sub_idx_sorted]
                                sub_d = np.zeros(len(sub_e))
                                sub_d[1:] = sub_e[1:] - sub_e[:-1]
                                for j, orig_i in enumerate(sub_idx_sorted):
                                    detr_obs[orig_i] = sub_d[j]
                        else:
                            detr_obs[1:] = etr_obs[1:] - etr_obs[:-1]

                        grp_obs['MTR']          = mtr_obs
                        grp_obs['BaseTax']       = base_tax
                        grp_obs['NIT Estimated'] = nit_est
                        grp_obs['ETR']           = etr_obs
                        grp_obs['ΔETR']          = detr_obs

                        st.markdown("---")
                        st.markdown("#### 📊 Observation-Level Tax Metrics")
                        st.dataframe(grp_obs, use_container_width=True)

                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            grp_obs.to_excel(writer, index=False)
                        st.download_button(
                            label=f"📥 Download {g_type} Computed Metrics",
                            data=output.getvalue(),
                            file_name=f"{g_type}_computed_metrics.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_{g_type}"
                        )
                except Exception:
                    pass

            with t_dash:
                from src.viz import plot_revenue_contribution, plot_staircase_rates, plot_etr_curve
                import plotly.express as px

                _agg = res['agg_df'].copy()
                _agg['Slab'] = _agg.apply(
                    lambda r: f"{r['lower_bound']/1e6:.2f}M - {r['upper_bound']/1e6:.2f}M"
                    if r['upper_bound'] < np.inf else f"{r['lower_bound']/1e6:.2f}M+", axis=1)

                _chart_bg = dict(plot_bgcolor='white', paper_bgcolor='white')
                _grid_x   = dict(showgrid=True, gridcolor='#E8EDF2')
                _grid_y   = dict(showgrid=True, gridcolor='#E8EDF2')
                _margin   = dict(margin=dict(t=45, b=30, l=30, r=10))

                # Row 1 — 2 charts
                dc1, dc2 = st.columns(2)
                with dc1:
                    fig_rev = plot_revenue_contribution(res['agg_df'], res['schedule_list'])
                    fig_rev.update_layout(title="Revenue by Income Slab", height=350, **_chart_bg, **_margin)
                    fig_rev.update_xaxes(**_grid_x)
                    fig_rev.update_yaxes(**_grid_y)
                    st.plotly_chart(fig_rev, use_container_width=True, key=f"fig_rev_{g_type}")
                with dc2:
                    fig_etrc = plot_etr_curve(m, bm, historical_benchmarks={}, title="ETR Progression Curve")
                    fig_etrc.update_layout(height=350, **_margin)
                    st.plotly_chart(fig_etrc, use_container_width=True, key=f"fig_etrc_{g_type}")

                # Row 2 — 2 charts
                dc3, dc4 = st.columns(2)
                with dc3:
                    fig_dist = px.bar(_agg, x='Slab', y='total_filers', title="Distribution of Filers",
                                      color_discrete_sequence=['#003B5C'])
                    fig_dist.update_layout(height=350, yaxis_title="Number of Taxpayers",
                                           xaxis_title="Income Group", **_chart_bg, **_margin)
                    fig_dist.update_xaxes(**_grid_x)
                    fig_dist.update_yaxes(**_grid_y)
                    st.plotly_chart(fig_dist, use_container_width=True, key=f"fig_dist_{g_type}")
                with dc4:
                    if res.get('schedule_list'):
                        _y_all    = m['y']
                        _detr_all = m['delta_etr']
                        _lbs      = [s['lower'] for s in res['schedule_list']]
                        _detr_vals = []
                        for lb in _lbs:
                            idx_d = np.searchsorted(_y_all, lb)
                            _detr_vals.append(float(_detr_all[min(idx_d, len(_detr_all)-1)]))
                        _detr_df = pd.DataFrame({'Slab': [f"{lb/1e6:.1f}M" for lb in _lbs], 'ΔETR (pp)': _detr_vals})
                        fig_detr_bar = px.bar(_detr_df, x='Slab', y='ΔETR (pp)', title="ΔETR Spike Detection",
                                              color='ΔETR (pp)', color_continuous_scale='Oranges')
                        fig_detr_bar.update_layout(height=350, **_chart_bg, **_margin)
                        fig_detr_bar.update_xaxes(**_grid_x)
                        fig_detr_bar.update_yaxes(**_grid_y)
                        st.plotly_chart(fig_detr_bar, use_container_width=True, key=f"fig_detr_bar_{g_type}")
                    else:
                        st.info("No schedule data available for ΔETR chart.")

            with t_cmp:
                # Calculate additional metrics for tables
                # Get base and proposed schedules as lists
                base_schedule_list = _schedule_to_list(res['base_slabs_df']) if not res['base_slabs_df'].empty else []
                prop_schedule_list = res.get('schedule_list', [])
                
                # Calculate filer counts per slab
                base_filers = _calculate_slab_filers(res['base_slabs_df'], _y_arr, _n_arr)
                prop_filers = _calculate_slab_filers(res['schedule_df'], _y_arr * _filer_scale, _n_arr * _filer_scale)
                
                # Calculate average ETR per slab
                base_avg_etrs = _calculate_slab_avg_etr(res['base_slabs_df'], base_schedule_list, _y_arr, _n_arr, _base_sur_slabs)
                prop_avg_etrs = _calculate_slab_avg_etr(res['schedule_df'], prop_schedule_list, _y_arr * _filer_scale, _n_arr * _filer_scale, _prop_sur_slabs)
                
                # Calculate band collections for detailed transition view
                base_band_collections, prop_band_collections = _calculate_band_collections(
                    res['base_slabs_df'], res['schedule_df'],
                    base_schedule_list, prop_schedule_list,
                    _y_arr, _n_arr,
                    _base_sur_slabs, _prop_sur_slabs,
                    _filer_scale
                )
                
                # Calculate band-level filers and avg ETRs for detailed view
                base_band_filers, prop_band_filers, base_band_avg_etrs, prop_band_avg_etrs = _calculate_band_metrics(
                    res['base_slabs_df'], res['schedule_df'],
                    base_schedule_list, prop_schedule_list,
                    _y_arr, _n_arr,
                    _base_sur_slabs, _prop_sur_slabs,
                    _filer_scale
                )
                
                cb, cp = st.columns(2)
                with cb:
                    st.subheader(f"🏛️ Base: {_fy_label(base_regime_year)}")
                    _base_fmt = _fmt_table(res['base_slabs_df'], base_filers, base_avg_etrs)
                    if _base_fmt.empty:
                        st.info("No base-regime slabs available for this taxpayer type.")
                    else:
                        st.table(_base_fmt)
                with cp:
                    st.subheader("🧪 Your Lab Design")
                    _prop_fmt = _fmt_table(res['schedule_df'], prop_filers, prop_avg_etrs)
                    if _prop_fmt.empty:
                        st.info("No proposed slabs to display.")
                    else:
                        st.table(_prop_fmt)
                st.subheader("🔄 Detailed Transition View")
                try:
                    st.table(_merged_table_enhanced(res['base_slabs_df'], res['schedule_df'], 
                                                     base_band_collections, prop_band_collections,
                                                     base_band_filers, prop_band_filers,
                                                     base_band_avg_etrs, prop_band_avg_etrs))
                except Exception as _merge_err:
                    st.info(f"Transition view unavailable: {_merge_err}")

            with t_calc:
                st.markdown(f"""
<div class="imf-section-tag">Tax Calculator</div>
<h3 style="margin-top:6px;">🧮 {g_type} Tax Calculator</h3>
<p>Enter an annual income to see the tax liability under both the base regime ({_fy_label(base_regime_year)}) and your Lab Design.</p>
""", unsafe_allow_html=True)
                
                calc_income = st.number_input("Enter Annual Taxable Income (PKR)", min_value=0.0, value=1200000.0, step=50000.0, key=f"calc_in_{g_type}")
                
                # Use scalar arrays to leverage existing _nit_total logic
                _y_val = np.array([calc_income])
                _n_val = np.array([1.0])
                
                _tax_base_val = _nit_total(_base_sch, _y_val, _n_val, _base_sur_slabs)
                _tax_prop_val = _nit_total(res['schedule_list'], _y_val, _n_val, _prop_sur_slabs)
                
                _diff_val = _tax_prop_val - _tax_base_val
                _etr_base = (_tax_base_val / calc_income) if calc_income > 0 else 0.0
                _etr_prop = (_tax_prop_val / calc_income) if calc_income > 0 else 0.0

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(f"Base Tax ({base_regime_year})", f"PKR {_tax_base_val:,.0f}", help=f"Effective Tax Rate: {_etr_base:.2%}")
                with c2:
                    st.metric("Lab Design Tax", f"PKR {_tax_prop_val:,.0f}", delta=f"{_diff_val:,.0f}", delta_color="inverse", help=f"Effective Tax Rate: {_etr_prop:.2%}")
                with c3:
                    if _tax_base_val > 0:
                        _perc_chg = (_diff_val / _tax_base_val)
                        st.metric("Tax Change (%)", f"{_perc_chg:+.1%}")
                    else:
                        st.metric("Tax Change (%)", "N/A")

                # Visual comparison
                calc_df = pd.DataFrame({
                    'Scenario': ['Base Regime', 'Lab Design'],
                    'Tax Amount': [_tax_base_val, _tax_prop_val]
                })
                import plotly.express as px
                fig_calc = px.bar(calc_df, x='Scenario', y='Tax Amount', color='Scenario',
                                 color_discrete_map={'Base Regime': '#6c757d', 'Lab Design': '#003B5C'},
                                 text_auto=',.0f')
                fig_calc.update_layout(showlegend=False, height=300, margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_calc, use_container_width=True, key=f"fig_calc_{g_type}")


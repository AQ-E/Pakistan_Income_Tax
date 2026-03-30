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
                        validate_schedule, run_manual_simulation, optimize_schedule_constrained)
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
    'nsc':           'NSC',          # raw label in Excel
    'consolidated':  'NSC',          # treat consolidated as NSC
    'non_salaried':  'Non-Salaried',
    'non salaried consolidated': 'NSC',
}

def _canon_type(raw):
    """Normalise an Excel Tax_Type label to the app's canonical name."""
    n = _norm(str(raw))
    return _TYPE_CANON.get(n, str(raw).strip().title())

def _get_truth_slabs(g_type, year=None):
    """Year-aware, case-insensitive lookup into TRUTH_SLABS.
    Falls back to any year if the requested year is not found."""
    canon = _canon_type(g_type)
    # Try exact year first
    if year is not None:
        key = (int(year), canon)
        if key in TRUTH_SLABS:
            v = TRUTH_SLABS[key]
            return v.copy() if not v.empty else None
    # Fallback: any year, matching type
    for (yr, tp), v in TRUTH_SLABS.items():
        if _norm(tp) == _norm(canon):
            return v.copy() if not v.empty else None
    return None

def _get_truth_surcharge(g_type, year=None):
    """Year-aware, case-insensitive lookup into TRUTH_SURCHARGES."""
    canon = _canon_type(g_type)
    if year is not None:
        key = (int(year), canon)
        if key in TRUTH_SURCHARGES:
            return TRUTH_SURCHARGES[key]
    # Fallback
    for (yr, tp), v in TRUTH_SURCHARGES.items():
        if _norm(tp) == _norm(canon):
            return v
    return {'threshold': 0.0, 'rate': 0.0}

@st.cache_data
def load_truth_slabs(file_path):
    """Parse PIT_slabs_2025.xlsx into year-aware dicts.
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
        'nsc':                    'NSC',          # Non-Salaried Consolidated
        'consolidated':           'NSC',
        'non salaried consolidated': 'NSC',
    }
    def _local_canon(raw):
        return _INLINE_CANON.get(str(raw).strip().lower(), str(raw).strip().title())

    for (year, raw_ttype), g in df.groupby(['Year', 'Tax_Type']):
        ttype = _local_canon(raw_ttype)   # e.g. 'NSC' -> 'NSC', 'Non_salaried' -> 'Non-Salaried'
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

def _fmt_table(df):
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
    return out[['Income Range', 'MTR']]

def _merged_table(base_df, prop_df):
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
        rows.append({
            'Band': f"{lo:,.0f} – {hi_s}",
            'Base MTR': f"{br:.2%}" if br is not None else "—",
            'Proposed MTR': f"{pr:.2%}" if pr is not None else "—",
            'Δ (pp)': f"{(pr-br)*100:+.2f}" if br is not None and pr is not None else "—",
        })
    return pd.DataFrame(rows)

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
    mode = st.radio("Optimization Strategy", ["Auto Optimize", "Policy Lab"],
                    help="Auto: System finds best schedule. Lab: You design it.")

    st.markdown("---")
    st.header("📋 General Settings")
    years_avail = sorted(df_slabs_agg['year'].unique(), reverse=True)
    selected_year = st.selectbox("Baseline Year", years_avail)

    uplift_target = 0.0  # Policy Lab has no revenue target; Auto Optimize sets this below


    if mode == "Auto Optimize":
        st.markdown("### Targets")
        uplift_target = st.slider("Change in Revenue (%)", 0.0, 15.0, 0.0, 0.5)
        st.markdown("### Scope")
        run_sal = st.checkbox("Optimize Salaried", value=True)
        run_nsal = st.checkbox("Optimize Non-Salaried", value=True)
        run_aop = st.checkbox("Optimize AOP", value=True)
        run_cons = st.checkbox("Optimize NSC", value=False)
        
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
                
                # Use year-aware slab lookup from PIT_slabs_2025.xlsx
                base_slabs = _get_truth_slabs(g_type, selected_year)
                if base_slabs is None:
                    base_slabs = g_agg[['lower_bound', 'upper_bound', 'marginal_rate']].drop_duplicates().sort_values('lower_bound').copy()
                    if base_slabs['marginal_rate'].max() > 1.0: base_slabs['marginal_rate'] /= 100.0
                
                with st.spinner(f"⏳ Optimizing {g_type} …"):
                    t0 = time.time()
                    res = optimize_schedule(g_agg, base_slabs, total_tax, y_grid, total_tax * (1 + uplift_target/100))
                    res.update({'g_type': g_type, 'elapsed': time.time()-t0, 'base_slabs_df': base_slabs})
                    st.session_state.results[g_type] = res
            st.success("✅ Done!")

    else:  # Policy Lab — this runs when mode != "Auto Optimize"
        st.markdown("### Policy Lab Setup")
        lab_type = st.selectbox("Taxpayer Type", ["Salaried", "Non-Salaried", "AOP", "NSC"])
        # NSC is stored directly as 'NSC' — no remapping needed

        
        # Track active lab type to handle switching
        if 'lab_type_active' not in st.session_state:
            st.session_state.lab_type_active = lab_type

        # Initialize Lab Slabs if empty OR if we switched types
        df_slabs_agg['_norm'] = df_slabs_agg['taxpayer_type'].apply(_norm)
        g_agg = df_slabs_agg[(df_slabs_agg['year'] == selected_year) & 
                             (df_slabs_agg['_norm'] == _norm(lab_type))].copy()
        total_tax = g_agg['normal_income_tax_920000'].sum()
        
        lab_nrm = _norm(lab_type)
        # Year-aware slab lookup from PIT_slabs_2025.xlsx
        base_slabs_raw = _get_truth_slabs(lab_type, selected_year)
        if base_slabs_raw is None:
            base_slabs_raw = g_agg[['lower_bound', 'upper_bound', 'marginal_rate']].drop_duplicates().sort_values('lower_bound').copy()
            if base_slabs_raw['marginal_rate'].max() > 1.0: base_slabs_raw['marginal_rate'] /= 100.0
        base_list_calib = _schedule_to_list(base_slabs_raw)

        if st.session_state.lab_slabs is None or st.session_state.lab_type_active != lab_type:
            st.session_state.lab_slabs = base_slabs_raw[base_slabs_raw['upper_bound'] > base_slabs_raw['lower_bound']].copy().reset_index(drop=True)
            st.session_state.lab_type_active = lab_type
            
        st.markdown("---")
        st.write("**Quick Actions**")
        if st.button("Reset to Current Law"):
            st.session_state.lab_slabs = None
            if 'lab_sur_thresh' in st.session_state: del st.session_state['lab_sur_thresh']
            if 'lab_sur_rate'   in st.session_state: del st.session_state['lab_sur_rate']
            if 'lab_filer_chg'  in st.session_state: del st.session_state['lab_filer_chg']
            st.rerun()

        st.info("💡 **Policy Lab Guide**:\n- **Double-click** a cell to edit rates.\n- **Add Slabs**: Click the '+' at the bottom.\n- **Remove Slabs**: Select a row, press Delete.\n- **Final Slab**: Leave Upper Bound empty — treated as 'Above'.")

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
    _sur_info_default = _get_truth_surcharge(lab_type)
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
    if _sur_key not in st.session_state or st.session_state.get(_sur_type_key) != lab_type:
        st.session_state[_sur_key]      = _default_sur_slabs(_def_sur_thresh, _def_sur_rate)
        st.session_state[_sur_type_key] = lab_type

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
            res = run_manual_simulation(sch_list, g_agg, y_grid, total_tax, base_list=base_list_calib)
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
            res = optimize_schedule_constrained(g_agg, sch_list, total_tax * (1 + uplift_target/100), y_grid, base_list=base_list_calib)
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
        m, bm = res['metrics'], compute_metrics(_schedule_to_list(res['base_slabs_df']), res['metrics']['y'])

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
        # Raw Type_Tax values in uploaded file: 'S', 'NS', 'AOP', 'NSC'
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
            _prop_sur_slabs = _truth_sur_to_slabs(g_type, selected_year)

        # Filer scale: only applied to proposed (base always uses original Y)
        _filer_scale = res.get('lab_filer_scale', 1.0)
        _y_arr_prop  = _y_arr * _filer_scale

        # Base NIT Estimated — truth slabs + truth surcharge, original Y & N
        _base_sch      = _get_truth_slabs(g_type, selected_year)
        _base_sch      = _schedule_to_list(_base_sch) if _base_sch is not None else _schedule_to_list(res['base_slabs_df'])
        _base_sur_slabs = _truth_sur_to_slabs(g_type, selected_year)
        _nit_base      = _nit_total(_base_sch, _y_arr, _n_arr, _base_sur_slabs)

        # Proposed NIT Estimated — proposed slabs + slab surcharge + filer scale
        _nit_prop = _nit_total(res['schedule_list'], _y_arr_prop, _n_arr * _filer_scale, _prop_sur_slabs)

        _uplift_nit = (_nit_prop - _nit_base) / _nit_base if _nit_base > 0 else 0.0

        with tabs[i]:
            if _uplift_nit < -0.001:
                st.warning(f"⚠️ **Proposed NIT below baseline** (PKR {_nit_prop/1e9:,.1f}B < PKR {_nit_base/1e9:,.1f}B)")

            t_dash, t_ana, t_cmp = st.tabs(["📈 Dashboard", "📊 ETR & CETR Heat Maps", "📋 Schedule Comparison"])
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
                cmap = 'Viridis' if 'salaried' in g_type.lower() and 'non' not in g_type.lower() else 'Inferno'

                fig_etr = plot_etr_heatmap(build_heatmap_dataframe(m['etr'], y_grid, bm['etr']), colorscale=cmap)
                st.plotly_chart(fig_etr, use_container_width=True)

                fig_detr = plot_detr_heatmap(build_heatmap_dataframe(m['delta_etr'], y_grid, bm['delta_etr']), colorscale=cmap)
                st.plotly_chart(fig_detr, use_container_width=True)

                # ─── Observation-level metrics using EDITED slab-based surcharge ───
                try:
                    _obs_sur_slabs = res.get('lab_sur_slabs', None)
                    if _obs_sur_slabs is None:
                        _obs_sur_slabs = _truth_sur_to_slabs(g_type, selected_year)

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
                    st.plotly_chart(fig_rev, use_container_width=True)
                with dc2:
                    fig_etrc = plot_etr_curve(m, bm, historical_benchmarks={}, title="ETR Progression Curve")
                    fig_etrc.update_layout(height=350, **_margin)
                    st.plotly_chart(fig_etrc, use_container_width=True)

                # Row 2 — 2 charts
                dc3, dc4 = st.columns(2)
                with dc3:
                    fig_dist = px.bar(_agg, x='Slab', y='total_filers', title="Distribution of Filers",
                                      color_discrete_sequence=['#003B5C'])
                    fig_dist.update_layout(height=350, yaxis_title="Number of Taxpayers",
                                           xaxis_title="Income Group", **_chart_bg, **_margin)
                    fig_dist.update_xaxes(**_grid_x)
                    fig_dist.update_yaxes(**_grid_y)
                    st.plotly_chart(fig_dist, use_container_width=True)
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
                        st.plotly_chart(fig_detr_bar, use_container_width=True)
                    else:
                        st.info("No schedule data available for ΔETR chart.")

            with t_cmp:
                cb, cp = st.columns(2)
                with cb:
                    st.subheader("🏛️ Current Law")
                    _base_fmt = _fmt_table(res['base_slabs_df'])
                    if _base_fmt.empty:
                        st.info("No current law slabs available for this taxpayer type.")
                    else:
                        st.table(_base_fmt)
                with cp:
                    st.subheader("🧪 Your Lab Design")
                    _prop_fmt = _fmt_table(res['schedule_df'])
                    if _prop_fmt.empty:
                        st.info("No proposed slabs to display.")
                    else:
                        st.table(_prop_fmt)
                st.subheader("🔄 Detailed Transition View")
                try:
                    st.table(_merged_table(res['base_slabs_df'], res['schedule_df']))
                except Exception as _merge_err:
                    st.info(f"Transition view unavailable: {_merge_err}")


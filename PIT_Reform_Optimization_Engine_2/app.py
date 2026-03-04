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
st.set_page_config(page_title="PIT Reform Optimization Engine", layout="wide")

st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.stMetric {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("PIT Reform Optimization Engine")
st.markdown("---")

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
    'nsc':           'Consolidated',   # NSC = Non-Salaried Consolidated = Consolidated
    'consolidated':  'Consolidated',
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

    for (year, raw_ttype), g in df.groupby(['Year', 'Tax_Type']):
        ttype = _canon_type(raw_ttype)   # e.g. 'NSC' -> 'Consolidated'
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
        run_cons = st.checkbox("Optimize Consolidated", value=False)
        
        if st.button("🚀 Auto-Optimize Policy", type="primary"):
            st.session_state.results = {}
            y_grid = np.arange(0, 20_000_001, 100_000)
            groups = []
            if run_sal: groups.append('Salaried')
            if run_nsal: groups.append('Non-Salaried')
            if run_aop: groups.append('AOP')
            if run_cons: groups.append('Consolidated')

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
        lab_type = st.selectbox("Taxpayer Type", ["Salaried", "Non-Salaried", "AOP", "Consolidated (NSC)"])
        # Map display name to internal canonical name
        lab_type = 'Consolidated' if lab_type == 'Consolidated (NSC)' else lab_type

        
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
            "lower_bound":    st.column_config.NumberColumn("Lower Bound (PKR)", format="%d",    min_value=0),
            "upper_bound":    st.column_config.NumberColumn("Upper Bound (PKR)", format="%d"),
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

    # ─── Surcharge Editor (directly below slabs) ───
    st.markdown("#### 📌 Surcharge Settings")
    _sur_info_default = _get_truth_surcharge(lab_type)
    _def_thresh = _sur_info_default.get('threshold', 0.0)
    _def_rate   = _sur_info_default.get('rate', 0.0) * 100.0   # show as %

    # Load edited values from session state (persist across reruns)
    _init_thresh = st.session_state.get('lab_sur_thresh', _def_thresh)
    _init_rate   = st.session_state.get('lab_sur_rate',   _def_rate)

    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1:
        new_thresh = st.number_input(
            "Surcharge Income Threshold (PKR)",
            min_value=0.0, value=float(_init_thresh), step=500_000.0,
            help="Surcharge applies only when Taxable Income ≥ this amount.",
            key="sur_thresh_input"
        )
    with sc2:
        new_rate = st.number_input(
            "Surcharge Rate (%)",
            min_value=0.0, max_value=100.0, value=float(_init_rate), step=1.0,
            help="% added on top of normal tax for incomes above the threshold.",
            key="sur_rate_input"
        )
    with sc3:
        st.markdown("<br/>", unsafe_allow_html=True)  # vertical align
        if st.button("🔄 Reset Surcharge"):
            st.session_state['lab_sur_thresh'] = _def_thresh
            st.session_state['lab_sur_rate']   = _def_rate
            st.rerun()

    # Persist edited surcharge
    st.session_state['lab_sur_thresh'] = new_thresh
    st.session_state['lab_sur_rate']   = new_rate

    # Show a quick summary of the effective surcharge rule
    if new_thresh > 0 and new_rate > 0:
        st.caption(f"⚡ Surcharge active: **{new_rate:.1f}%** on normal tax for taxable income ≥ **PKR {new_thresh:,.0f}**")
    else:
        st.caption("⚡ No surcharge currently applied.")

    # Store effective surcharge so the results section can use it
    _lab_sur = {'threshold': new_thresh, 'rate': new_rate / 100.0}

    # ─── Filer Adjustment (directly below surcharge) ───
    st.markdown("#### 👥 Filer Count Adjustment")
    _def_filer_chg = 0.0
    _init_filer_chg = float(st.session_state.get('lab_filer_chg', _def_filer_chg))

    fa1, fa2 = st.columns([3, 1])
    with fa1:
        new_filer_chg = st.slider(
            "Change in Number of Filers (%)",
            min_value=-50.0, max_value=100.0,
            value=_init_filer_chg, step=1.0,
            help="Scales aggregate taxable income (Y) proportionally. +10% means 10% more filers at the same avg income per filer.",
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
            res = run_manual_simulation(sch_list, g_agg, y_grid, total_tax, base_list=base_list_calib)
            res.update({'g_type': lab_type, 'elapsed': 0.0, 'base_slabs_df': base_slabs_raw,
                        'lab_surcharge': _lab_sur, 'lab_filer_scale': _filer_scale})
            st.session_state.results = {lab_type: res}

    # Button to Refine
    if not edited_df.empty and st.button("🧙 Refine Within My Slab Structure"):
        y_grid   = np.arange(0, 20_000_001, 100_000)
        sch_list = _schedule_to_list(edited_df)
        with st.spinner("Refining..."):
            res = optimize_schedule_constrained(g_agg, sch_list, total_tax * (1 + uplift_target/100), y_grid, base_list=base_list_calib)
            res.update({'g_type': lab_type, 'elapsed': 0.1, 'base_slabs_df': base_slabs_raw,
                        'lab_surcharge': _lab_sur, 'lab_filer_scale': _filer_scale})
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

        # ── Compute slab-formula NIT Estimated for base & proposed ──────────
        def _nit_total(sch_list, y_arr, sur_thresh, sur_rate):
            """Apply slab schedule + surcharge to array of Y values, return total NIT."""
            lws  = np.array([s['lower'] for s in sch_list])
            rs   = np.array([s['rate']  for s in sch_list])
            ups  = np.array([s['upper'] for s in sch_list])
            cum  = np.zeros(len(sch_list))
            for k in range(1, len(sch_list)):
                w = ups[k-1] - lws[k-1]
                cum[k] = cum[k-1] + (0.0 if np.isinf(w) else w) * rs[k-1]
            idx  = np.clip(np.searchsorted(lws, y_arr, side='right') - 1, 0, len(sch_list)-1)
            bt   = np.maximum(cum[idx] + (y_arr - lws[idx]) * rs[idx], 0.0)
            nit  = np.where(y_arr >= sur_thresh, bt * (1.0 + sur_rate), bt)
            return nit.sum()

        # Load observation Y values for this g_type
        _type_map   = {'Salaried': 'S', 'Non-Salaried': 'NS', 'AOP': 'AOP', 'Consolidated': 'C'}
        _tgt        = _type_map.get(g_type, g_type)
        _raw        = pd.read_excel(_io.BytesIO(st.session_state.uploaded_obs_bytes), engine='openpyxl')
        _grp        = _raw.copy() if g_type == 'Consolidated' else (
                      _raw[_raw['Type_Tax'] == _tgt].copy() if 'Type_Tax' in _raw.columns else _raw.copy())
        _y_arr      = _grp['Taxable Income (9100)'].values.astype(float) if 'Taxable Income (9100)' in _grp.columns else np.array([])

        # Surcharge: lab-edited if available, else system truth for selected year
        _sur        = res.get('lab_surcharge', None) or _get_truth_surcharge(g_type, selected_year)
        _sur_th     = _sur.get('threshold', 0.0)
        _sur_rt     = _sur.get('rate', 0.0)

        # Filer scale: only applied to proposed (base always uses original Y)
        _filer_scale = res.get('lab_filer_scale', 1.0)
        _y_arr_prop  = _y_arr * _filer_scale   # scaled Y for proposed NIT

        # Base NIT Estimated  — truth slabs for selected year applied to original Y
        _base_sch   = _get_truth_slabs(g_type, selected_year)
        _base_sch   = _schedule_to_list(_base_sch) if _base_sch is not None else _schedule_to_list(res['base_slabs_df'])
        _base_sur   = _get_truth_surcharge(g_type, selected_year)
        _nit_base   = _nit_total(_base_sch, _y_arr, _base_sur.get('threshold', 0.0), _base_sur.get('rate', 0.0)) if len(_y_arr) else 0.0

        # Proposed NIT Estimated — proposed slabs + filer scale applied to Y
        _nit_prop   = _nit_total(res['schedule_list'], _y_arr_prop, _sur_th, _sur_rt) if len(_y_arr_prop) else 0.0

        _uplift_nit = (_nit_prop - _nit_base) / _nit_base if _nit_base > 0 else 0.0

        with tabs[i]:
            if _uplift_nit < -0.001:
                st.warning(f"⚠️ **Proposed NIT below baseline** (PKR {_nit_prop/1e9:,.1f}B < PKR {_nit_base/1e9:,.1f}B)")

            t_ana, t_cmp = st.tabs(["📊 Analysis & Heatmaps", "📋 Schedule Comparison"])
            with t_ana:
                st.markdown(f"### 🏆 {res['stage_selected']} Schedule — {g_type}")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Base NIT Estimated",     f"PKR {_nit_base/1e9:,.2f}B",
                           help="NIT computed from current-law slabs applied to uploaded Y values")
                c2.metric("Proposed NIT Estimated", f"PKR {_nit_prop/1e9:,.2f}B", f"{_uplift_nit:+.2%}",
                           help="NIT computed from proposed/lab slabs applied to same Y values")

                total_filers = m.get('total_filers', res['agg_df']['total_filers'].sum())
                c3.metric("Number of filers", f"{int(total_filers):,}")

                c4.metric("Avg ETR", f"{m.get('avg_etr', 0):.2%}")

                max_mtr = max([s['rate'] for s in res['schedule_list']])
                max_cetr = m.get('band_max_jump', 0)
                c5.metric("MTR (Max) / CETR (Max)", f"{max_mtr:.2%} / {max_cetr:.2f}pp")

                st.markdown("---")
                y_grid = m['y']
                etr_hist, detr_hist = {}, {}
                cmap = 'Viridis' if 'salaried' in g_type.lower() and 'non' not in g_type.lower() else 'Inferno'

                fig_etr = plot_etr_heatmap(build_heatmap_dataframe(m['etr'], y_grid, bm['etr']), colorscale=cmap)
                st.plotly_chart(fig_etr, use_container_width=True)

                fig_detr = plot_detr_heatmap(build_heatmap_dataframe(m['delta_etr'], y_grid, bm['delta_etr']), colorscale=cmap)
                st.plotly_chart(fig_detr, use_container_width=True)

                # ─── Observation-level metrics using EDITED surcharge ───
                try:
                    # Use lab-edited surcharge if available, else system default
                    _sur = res.get('lab_surcharge', None) or _get_truth_surcharge(g_type)
                    sur_thresh = _sur.get('threshold', 0.0)
                    sur_rate   = _sur.get('rate', 0.0)

                    # Show which surcharge is in effect (informational banner)
                    if sur_thresh > 0 and sur_rate > 0:
                        st.info(f"⚡ **Surcharge applied:** {sur_rate:.1%} on normal tax for Taxable Income ≥ PKR {sur_thresh:,.0f}")
                    else:
                        st.info("⚡ No surcharge applied.")

                    type_mapping = {'Salaried': 'S', 'Non-Salaried': 'NS', 'AOP': 'AOP', 'Consolidated': 'C'}
                    tgt_raw  = type_mapping.get(g_type, g_type)
                    raw_obs  = pd.read_excel(_io.BytesIO(st.session_state.uploaded_obs_bytes), engine='openpyxl')
                    grp_obs  = raw_obs.copy() if g_type == 'Consolidated' else (
                               raw_obs[raw_obs['Type_Tax'] == tgt_raw].copy() if 'Type_Tax' in raw_obs.columns else raw_obs.copy())

                    if not grp_obs.empty and 'Taxable Income (9100)' in grp_obs.columns:
                        # Sort by [Year, Type_Tax, Income] so within each Year×Type group
                        # rows are income-ascending — required for correct ΔETR = ETR_i - ETR_{i-1}
                        has_year  = 'Year'     in grp_obs.columns
                        has_ttype = 'Type_Tax' in grp_obs.columns
                        sort_cols = (["Year"] if has_year else []) + \
                                    (["Type_Tax"] if has_ttype else []) + \
                                    ["Taxable Income (9100)"]
                        grp_obs = grp_obs.sort_values(by=sort_cols).reset_index(drop=True).copy()
                        sch     = res['schedule_list']

                        # ── MTR & BaseTax (spec §4A) ──────────────────────────────────
                        # BaseTax = Σ_{k=1}^{i-1} (UB_k - LB_k)*r_k  +  (Y - LB_i)*r_i
                        y_obs    = grp_obs['Taxable Income (9100)'].values.astype(float)
                        lowers   = np.array([s['lower'] for s in sch])
                        rates    = np.array([s['rate']  for s in sch])
                        uppers   = np.array([s['upper'] for s in sch])

                        # Precompute cumulative tax at the bottom of each slab
                        base_cum = np.zeros(len(sch))
                        for k in range(1, len(sch)):
                            w = uppers[k-1] - lowers[k-1]
                            base_cum[k] = base_cum[k-1] + (0.0 if np.isinf(w) else w) * rates[k-1]

                        # Find slab i such that LB_i <= Y < UB_i
                        idx      = np.clip(np.searchsorted(lowers, y_obs, side='right') - 1, 0, len(sch)-1)
                        mtr_obs  = rates[idx]
                        base_tax = np.maximum(base_cum[idx] + (y_obs - lowers[idx]) * mtr_obs, 0.0)

                        # ── NIT Estimated (spec §4A surcharge) ───────────────────────
                        nit_est  = np.where(y_obs >= sur_thresh,
                                            base_tax * (1.0 + sur_rate),
                                            base_tax)

                        # ── ETR (spec §4B) ────────────────────────────────────────────
                        etr_obs  = np.where(y_obs > 0, nit_est / y_obs, 0.0)

                        # ── ΔETR (spec §4C): within each Year×Type_Tax group ─────────
                        # ΔETR_i = ETR_i - ETR_{i-1};  first row in each group = 0
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
                                sub_idx_sorted = sorted(sub_idx)  # already income-sorted after reset_index
                                sub_e = etr_obs[sub_idx_sorted]
                                sub_d = np.zeros(len(sub_e))
                                sub_d[1:] = sub_e[1:] - sub_e[:-1]  # first row stays 0
                                for j, orig_i in enumerate(sub_idx_sorted):
                                    detr_obs[orig_i] = sub_d[j]
                        else:
                            detr_obs[1:] = etr_obs[1:] - etr_obs[:-1]  # global fallback

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
                    pass  # If obs metrics fail, heatmaps above still render fine

                with st.expander("📈 Advanced Policy Charts"):
                    ca, cb = st.columns(2)
                    from src.viz import plot_staircase_rates, plot_revenue_contribution, plot_etr_curve
                    with ca: st.plotly_chart(plot_staircase_rates(m, res['schedule_list']), use_container_width=True)
                    with cb: st.plotly_chart(plot_revenue_contribution(res['agg_df'], res['schedule_list']), use_container_width=True)
                    st.plotly_chart(plot_progressivity_slope(m, bm), use_container_width=True)
                    st.plotly_chart(plot_etr_curve(m, bm, historical_benchmarks={}, title=f"{g_type} ETR Comparison"), use_container_width=True)

            with t_cmp:
                cb, cp = st.columns(2)
                with cb:
                    st.subheader("🏛️ Current Law")
                    st.table(_fmt_table(res['base_slabs_df']))
                with cp:
                    st.subheader("🧪 Your Lab Design")
                    st.table(_fmt_table(res['schedule_df']))
                st.subheader("🔄 Detailed Transition View")
                st.table(_merged_table(res['base_slabs_df'], res['schedule_df']))

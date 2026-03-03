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
slab_path, truth_path = get_data_paths()
try:
    df_slabs_agg = load_slab_data(slab_path)
    # df_grid_baseline = load_grid_data(grid_path) # obsolete dependency
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ───────────────────────── Session State ─────────────────────────
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'lab_slabs' not in st.session_state:
    st.session_state.lab_slabs = None

# ───────────────────────── Helpers ─────────────────────────

def _norm(s):
    return str(s).lower().replace('-', ' ').replace('_', ' ').strip()

@st.cache_data
def load_truth_slabs(file_path):
    df = pd.read_excel(file_path)
    slabs = {}
    surcharges = {}
    
    for (year, ttype), g in df.groupby(['Year', 'Tax_Type']):
        ttype = str(ttype).strip().lower().replace('_', '-')
        if ttype == 'non-salaried': ttype = 'Non-Salaried'
        elif ttype == 'salaried': ttype = 'Salaried'
        elif ttype == 'aop': ttype = 'AOP'
        
        g_slabs = []
        s_thresh = 0.0
        s_rate = 0.0
        
        for _, r in g.iterrows():
            lower = str(r['Lower_slab']).strip().lower()
            upper = str(r['Upper_slab']).strip().lower()
            mtr = r['MTR']
            tax_rate_str = str(r['TAX RATE']).lower()
            
            if pd.isna(mtr): continue
            
            is_surcharge = False
            if 'surcharge' in lower or 'liability' in tax_rate_str:
                is_surcharge = True
            
            if is_surcharge:
                s_rate = float(mtr)
                nums = re.findall(r'\d+', tax_rate_str.replace(',', ''))
                if nums:
                    s_thresh = float(nums[0])
                elif lower.replace('.', '').isdigit():
                    s_thresh = float(lower)
            else:
                l_val = pd.to_numeric(r['Lower_slab'], errors='coerce')
                u_val = pd.to_numeric(r['Upper_slab'], errors='coerce') if upper != '+' else np.inf
                if pd.notna(l_val) and pd.notna(mtr):
                    if pd.isna(u_val): u_val = np.inf
                    g_slabs.append({'lower_bound': float(l_val), 'upper_bound': float(u_val), 'marginal_rate': float(mtr)})
                    
        slabs[ttype] = pd.DataFrame(g_slabs)
        surcharges[ttype] = {'threshold': s_thresh, 'rate': s_rate}
        
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
    st.header("⚙️ Design Mode")
    mode = st.radio("Optimization Strategy", ["Custom Dataset Simulator", "Auto Optimize", "Policy Lab"], 
                    help="Simulator: Upload and estimate. Auto: System finds best schedule. Lab: You design it.")

    st.markdown("---")
    st.header("📋 General Settings")
    years_avail = sorted(df_slabs_agg['year'].unique(), reverse=True)
    selected_year = st.selectbox("Baseline Year", years_avail)

    st.markdown("### Targets")
    uplift_target = st.slider("Change in Revenue (%)", 0.0, 15.0, 0.0, 0.5)

    if mode == "Auto Optimize":
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
                
                t_nrm = _norm(g_type)
                if t_nrm in TRUTH_SLABS and not TRUTH_SLABS[t_nrm].empty:
                    base_slabs = TRUTH_SLABS[t_nrm].copy()
                else:
                    base_slabs = g_agg[['lower_bound', 'upper_bound', 'marginal_rate']].drop_duplicates().sort_values('lower_bound').copy()
                    if base_slabs['marginal_rate'].max() > 1.0: base_slabs['marginal_rate'] /= 100.0
                
                with st.spinner(f"⏳ Optimizing {g_type} …"):
                    t0 = time.time()
                    res = optimize_schedule(g_agg, base_slabs, total_tax, y_grid, total_tax * (1 + uplift_target/100))
                    res.update({'g_type': g_type, 'elapsed': time.time()-t0, 'base_slabs_df': base_slabs})
                    st.session_state.results[g_type] = res
            st.success("✅ Done!")

    else: # Policy Lab
        st.markdown("### Policy Lab Setup")
        lab_type = st.selectbox("Taxpayer Type", ["Salaried", "Non-Salaried", "AOP", "Consolidated"])
        
        # Track active lab type to handle switching
        if 'lab_type_active' not in st.session_state:
            st.session_state.lab_type_active = lab_type

        # Initialize Lab Slabs if empty OR if we switched types
        df_slabs_agg['_norm'] = df_slabs_agg['taxpayer_type'].apply(_norm)
        g_agg = df_slabs_agg[(df_slabs_agg['year'] == selected_year) & 
                             (df_slabs_agg['_norm'] == _norm(lab_type))].copy()
        total_tax = g_agg['normal_income_tax_920000'].sum()
        
        lab_nrm = _norm(lab_type)
        if lab_nrm in TRUTH_SLABS and not TRUTH_SLABS[lab_nrm].empty:
            base_slabs_raw = TRUTH_SLABS[lab_nrm].copy()
        else:
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
            st.rerun()

        st.info("💡 **Policy Lab Guide**:\n- **Double-click** a cell to edit.\n- **Add Slabs**: Click the '+' at the bottom.\n- **Remove Slabs**: Select a row and press Delete.\n- **Final Slab**: Leave Upper Bound empty or put a large number; the engine will treat it as 'Above'.")

# ───────────────────────── Main Dashboard ─────────────────────────
if mode == "Custom Dataset Simulator":
    st.header("📂 Custom Dataset Simulator")
    st.markdown("Upload your dataset to apply dynamic slab routing and surcharge rules.")
    
    uploaded_file = st.file_uploader("Upload Excel Dataset", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_excel(uploaded_file)
            req_cols = ["Taxable Income Slab (Rs.)", "Year", "Type_Tax", "Number of Persons", "Taxable Income (9100)", "Normal Income Tax (920000)"]
            missing = [c for c in req_cols if c not in df_upload.columns]
            if missing:
                st.error(f"❌ Missing required columns: {', '.join(missing)}")
            else:
                st.success("✅ File uploaded and validated successfully!")
                
                # Surcharge Parameters UI
                st.markdown("### ⚙️ Surcharge & Slab Settings")
                
                tax_types = ['Salaried', 'Non-Salaried', 'AOP']
                upload_types = df_upload['Type_Tax'].dropna().unique()
                for ut in upload_types:
                    if ut not in tax_types:
                        tax_types.append(ut)
                
                tabs = st.tabs(tax_types)
                
                schedules_dict = {}
                surcharges_dict = {}
                
                df_slabs_agg['_norm'] = df_slabs_agg['taxpayer_type'].apply(_norm)
                
                for i, ttype in enumerate(tax_types):
                    with tabs[i]:
                        c1, c2 = st.columns(2)
                        
                        ttype_norm = _norm(ttype)
                        default_thresh = TRUTH_SURCHARGES.get(ttype_norm, {}).get('threshold', 0.0)
                        default_rate = TRUTH_SURCHARGES.get(ttype_norm, {}).get('rate', 0.0) * 100.0
                        
                        with c1:
                            sur_thresh = st.number_input("Surcharge Threshold (PKR)", min_value=0.0, value=float(default_thresh), step=100000.0, key=f"sur_thresh_{ttype}")
                        with c2:
                            sur_rate = st.number_input("Surcharge Rate (%)", min_value=0.0, value=float(default_rate), step=0.5, key=f"sur_rate_{ttype}")
                        surcharges_dict[ttype] = {'threshold': sur_thresh, 'rate': sur_rate / 100.0}
                        
                        if ttype_norm in TRUTH_SLABS and not TRUTH_SLABS[ttype_norm].empty:
                            base_slabs_raw = TRUTH_SLABS[ttype_norm].copy()
                        else:
                            g_agg = df_slabs_agg[(df_slabs_agg['year'] == selected_year) & (df_slabs_agg['_norm'] == ttype_norm)].copy()
                            if not g_agg.empty:
                                base_slabs_raw = g_agg[['lower_bound', 'upper_bound', 'marginal_rate']].drop_duplicates().sort_values('lower_bound').copy()
                                if base_slabs_raw['marginal_rate'].max() > 1.0: base_slabs_raw['marginal_rate'] /= 100.0
                            else:
                                base_slabs_raw = pd.DataFrame({'lower_bound': [0], 'upper_bound': [np.inf], 'marginal_rate': [0.0]})
                            
                        st_key = f"sim_slabs_{ttype}"
                        if st_key not in st.session_state:
                            st.session_state[st_key] = base_slabs_raw.reset_index(drop=True)
                            
                        edited_df = st.data_editor(
                            st.session_state[st_key],
                            key=f"editor_{ttype}",
                            num_rows="dynamic",
                            use_container_width=True,
                            column_config={
                                "lower_bound": st.column_config.NumberColumn("Lower Bound (PKR)", format="%d", min_value=0),
                                "upper_bound": st.column_config.NumberColumn("Upper Bound (PKR)", format="%d"),
                                "marginal_rate": st.column_config.NumberColumn("MTR", format="%.2f", min_value=0.0, max_value=1.0)
                            }
                        )
                        
                        if not edited_df.empty:
                            edited_df = edited_df.sort_values('lower_bound').reset_index(drop=True)
                            last_idx = edited_df.index[-1]
                            val = edited_df.loc[last_idx, 'upper_bound']
                            if pd.isna(val) or val > 500_000_000 or val <= edited_df.loc[last_idx, 'lower_bound']:
                                edited_df.loc[last_idx, 'upper_bound'] = np.inf
                            for j in range(1, len(edited_df)):
                                edited_df.loc[j-1, 'upper_bound'] = edited_df.loc[j, 'lower_bound']
                                
                        st.session_state[st_key] = edited_df
                        schedules_dict[ttype] = _schedule_to_list(edited_df)
                        
                # Compute Engine
                out_dfs = []
                for (year, g_ttype), group in df_upload.groupby(['Year', 'Type_Tax']):
                    group = group.sort_values('Taxable Income (9100)').copy()
                    
                    sch = None
                    sur_thresh, sur_rate = 0.0, 0.0
                    
                    ttype_norm = _norm(g_ttype)
                    for k in tax_types:
                        if _norm(k) == ttype_norm:
                            sch = schedules_dict[k]
                            sur_thresh = surcharges_dict[k]['threshold']
                            sur_rate = surcharges_dict[k]['rate']
                            break
                            
                    if not sch:
                        out_dfs.append(group)
                        continue
                        
                    y = group['Taxable Income (9100)'].values.astype(float)
                    
                    lowers = np.array([s['lower'] for s in sch])
                    rates = np.array([s['rate'] for s in sch])
                    uppers = np.array([s['upper'] for s in sch])
                    
                    base_cum = np.zeros(len(sch))
                    for k in range(1, len(sch)):
                        width = uppers[k-1] - lowers[k-1]
                        w = np.where(np.isinf(width), 0, width)
                        base_cum[k] = base_cum[k-1] + w * rates[k-1]
                        
                    idx = np.searchsorted(lowers, y, side='right') - 1
                    idx = np.clip(idx, 0, len(sch) - 1)
                    mtr = rates[idx]
                    
                    base_tax = base_cum[idx] + (y - lowers[idx]) * mtr
                    base_tax = np.maximum(base_tax, 0.0)
                    
                    nit_est = np.where(y >= sur_thresh, base_tax * (1 + sur_rate), base_tax)
                    
                    etr = np.zeros_like(y)
                    mask = y > 0
                    etr[mask] = nit_est[mask] / y[mask]
                    
                    detr = np.zeros_like(etr)
                    detr[1:] = etr[1:] - etr[:-1]
                    
                    group['MTR'] = mtr
                    group['BaseTax'] = base_tax
                    group['NIT Estimated'] = nit_est
                    group['ETR'] = etr
                    group['ΔETR'] = detr
                    
                    out_dfs.append(group)
                    
                if out_dfs:
                    final_df = pd.concat(out_dfs, ignore_index=True)
                    st.markdown("### 📊 Processed Dataset Output")
                    st.dataframe(final_df)
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        final_df.to_excel(writer, index=False)
                    st.download_button(label="📥 Download Simulated Dataset",
                                       data=output.getvalue(),
                                       file_name="simulated_tax_output.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
        except Exception as e:
            st.error(f"Error reading dataset: {e}")

elif mode == "Policy Lab":
    st.header(f"🧪 Policy Lab — {lab_type} Design")
    
    # Render Editor
    edited_df = st.data_editor(
        st.session_state.lab_slabs,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "lower_bound": st.column_config.NumberColumn("Lower Bound (PKR)", format="%d", min_value=0),
            "upper_bound": st.column_config.NumberColumn("Upper Bound (PKR)", format="%d"),
            "marginal_rate": st.column_config.NumberColumn("MTR", format="%.2f", min_value=0.0, max_value=1.0)
        }
    )
    
    # Robustly handle the last slab's infinity
    if not edited_df.empty:
        # Sort by lower_bound to ensure we find the true visual 'last' slab
        edited_df = edited_df.sort_values('lower_bound').reset_index(drop=True)
        last_idx = edited_df.index[-1]
        
        # If the user left it blank or put a huge number, make it infinity
        val = edited_df.loc[last_idx, 'upper_bound']
        if pd.isna(val) or val > 500_000_000 or val <= edited_df.loc[last_idx, 'lower_bound']:
            edited_df.loc[last_idx, 'upper_bound'] = np.inf
            
        # Ensure contiguity: Force upper bound of previous slab to match lower bound of current
        for i in range(1, len(edited_df)):
            edited_df.loc[i-1, 'upper_bound'] = edited_df.loc[i, 'lower_bound']

    st.session_state.lab_slabs = edited_df

    # Instant Recompute
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
            res.update({'g_type': lab_type, 'elapsed': 0.0, 'base_slabs_df': base_slabs_raw})
            st.session_state.results = {lab_type: res}

    # Button to Refine
    if not edited_df.empty and st.button("🪄 Refine Within My Slab Structure"):
        y_grid = np.arange(0, 20_000_001, 100_000)
        sch_list = _schedule_to_list(edited_df)
        with st.spinner("Refining..."):
            res = optimize_schedule_constrained(g_agg, sch_list, total_tax * (1 + uplift_target/100), y_grid, base_list=base_list_calib)
            res.update({'g_type': lab_type, 'elapsed': 0.1, 'base_slabs_df': base_slabs_raw})
            st.session_state.results = {lab_type: res}
            st.session_state.lab_slabs = res['schedule_df']
            st.rerun()

# Display Results
results = st.session_state.results
if not results:
    st.info("👈 Click **Auto-Optimize Policy** or adjust **Policy Lab** to start.")
else:
    tab_labels = list(results.keys())
    tabs = st.tabs(tab_labels)
    for i, g_type in enumerate(tab_labels):
        res = results[g_type]
        m, bm = res['metrics'], compute_metrics(_schedule_to_list(res['base_slabs_df']), res['metrics']['y'])
        rev, base_rev = m['revenue'], res['base_revenue']
        uplift = (rev - base_rev) / base_rev if base_rev > 0 else 0
        
        with tabs[i]:
            if uplift < -0.001:
                st.warning(f"⚠️ **Revenue below baseline** (PKR {rev/1e9:,.1f}B < PKR {base_rev/1e9:,.1f}B) — Policy Unsafe")
            
            t_ana, t_cmp = st.tabs(["📊 Analysis & Heatmaps", "📋 Schedule Comparison"])
            with t_ana:
                st.markdown(f"### 🏆 {res['stage_selected']} Schedule — {g_type}")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Base Normal Income Tax", f"PKR {base_rev/1e9:,.1f}B")
                c2.metric("Prop Normal Income Tax", f"PKR {rev/1e9:,.1f}B", f"{uplift:+.2%}")
                
                total_filers = m.get('total_filers', res['agg_df']['total_filers'].sum())
                c3.metric("Number of filers", f"{int(total_filers):,}")
                
                c4.metric("Avg ETR", f"{m.get('avg_etr', 0):.2%}")
                
                max_mtr = max([s['rate'] for s in res['schedule_list']])
                max_cetr = m.get('band_max_jump', 0)
                c5.metric("MTR (Max) / CETR (Max)", f"{max_mtr:.2%} / {max_cetr:.2f}pp")

                st.markdown("---")
                y_grid = m['y']
                etr_hist, detr_hist = _get_historical_data(df_grid_baseline, g_type, y_grid)
                cmap = 'Viridis' if 'salaried' in g_type.lower() and 'non' not in g_type.lower() else 'Inferno'

                fig_etr = plot_etr_heatmap(build_heatmap_dataframe(m['etr'], y_grid, bm['etr']), colorscale=cmap)
                st.plotly_chart(fig_etr, use_container_width=True)

                fig_detr = plot_detr_heatmap(build_heatmap_dataframe(m['delta_etr'], y_grid, bm['delta_etr']), colorscale=cmap)
                st.plotly_chart(fig_detr, use_container_width=True)

                with st.expander("📈 Advanced Policy Charts"):
                    ca, cb = st.columns(2)
                    from src.viz import plot_staircase_rates, plot_revenue_contribution, plot_etr_curve
                    with ca: st.plotly_chart(plot_staircase_rates(m, res['schedule_list']), use_container_width=True)
                    with cb: st.plotly_chart(plot_revenue_contribution(res['agg_df'], res['schedule_list']), use_container_width=True)
                    st.plotly_chart(plot_progressivity_slope(m, bm), use_container_width=True)
                    st.plotly_chart(plot_etr_curve(m, bm, historical_benchmarks=etr_hist, title=f"{g_type} ETR Comparison"), use_container_width=True)

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

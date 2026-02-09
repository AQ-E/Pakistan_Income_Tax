
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# --- Configuration & Styling ---
st.set_page_config(layout="wide", page_title="PK Tax Analysis Tool")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        color: #7f8c8d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

FILE_PATH = "Slab wise Taxable Income Filers & Normal Tax_3012026.xlsx"

# --- Data Loading ---
@st.cache_data
def load_data(mtime):
    try:
        df = pd.read_excel(FILE_PATH)
    except FileNotFoundError:
        st.error(f"File not found: {FILE_PATH}")
        return None
    
    # Rename columns to map to canonical names
    rename_map = {
        'Year': 'year',
        'Taxpayer_type': 'taxpayer_type',
        'Slab_id': 'slab_id',
        'Slab_label': 'slab_label',
        'Lower bound': 'lower_bound',
        'upper bound': 'upper_bound',
        'marginal rate': 'marginal_rate',
        'Total Filers': 'total_filers',
        'Taxable Income\n(9100)': 'taxable_income_9100',
        'Admitted Tax\n(9203)': 'admitted_tax_9203',
        'Net Tax Chargeable \n(9200)': 'net_tax_9200',
        'Normal Income Tax\n(920000)': 'normal_tax_920000'
    }
    
    existing_cols = df.columns.tolist()
    actual_rename = {k: v for k, v in rename_map.items() if k in existing_cols}
    df = df.rename(columns=actual_rename)

    if 'slab_id' in df.columns:
        df['slab_id'] = df['slab_id'].astype(str)
        
    return df

def process_data(df_input):
    df = df_input.copy()
    
    # Fill NaNs/cleaning
    num_cols = ['taxable_income_9100', 'net_tax_9200', 'total_filers', 'normal_tax_920000', 'admitted_tax_9203']
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
            
    # Clean bounds
    if 'lower_bound' in df.columns:
        df['lower_bound'] = pd.to_numeric(df['lower_bound'], errors='coerce').fillna(0)
    if 'upper_bound' in df.columns:
        # upper_bound can be NaN for top slab
        df['upper_bound'] = pd.to_numeric(df['upper_bound'], errors='coerce')
            
    # Rule 1: Identify S_1 (reconciliation)
    # definition: lower_bound = 0 and upper_bound = 0
    df['is_reconciliation'] = (df['lower_bound'] == 0) & (df['upper_bound'] == 0)

    # Derived Metrics
    df['avg_taxable_income_per_filer'] = df['taxable_income_9100'] / df['total_filers'].replace(0, np.nan)
    df['net_tax_per_filer'] = df['net_tax_9200'] / df['total_filers'].replace(0, np.nan)
    
    df['etr_net'] = df['net_tax_9200'] / df['taxable_income_9100'].replace(0, np.nan)
    df['etr_normal'] = df['normal_tax_920000'] / df['taxable_income_9100'].replace(0, np.nan)
    
    # Revenue Shares
    group_cols = ['year', 'taxpayer_type']
    if set(group_cols).issubset(df.columns):
        totals = df.groupby(group_cols)[['taxable_income_9100', 'net_tax_9200']].transform('sum')
        df['tax_base_share'] = (df['taxable_income_9100'] / totals['taxable_income_9100']).fillna(0)
        df['revenue_share'] = (df['net_tax_9200'] / totals['net_tax_9200']).fillna(0)
    
    # Create Short Labels
    def make_short_label(row):
        lower = row.get('lower_bound', 0)
        upper = row.get('upper_bound', np.nan)
        
        def fmt(v):
            if v >= 1_000_000: return f"{v/1e6:g}M"
            if v >= 1_000: return f"{v/1e3:g}k"
            return str(v)

        if row.get('is_reconciliation', False):
            return "Reconcil."
        
        if lower == 0 and upper > 0:
            return f"< {fmt(upper)}"
        
        if pd.isna(upper) or upper == 0: 
             if lower > 0:
                 return f"> {fmt(lower)}"
             else:
                 return "0" 
        
        return f"{fmt(lower)} - {fmt(upper)}"

    df['short_label'] = df.apply(make_short_label, axis=1)
    
    return df

# --- Simulation Logic ---

def progressive_tax_calc(income, schedule):
    """
    Calculates tax based on a schedule dataframe.
    schedule: DataFrame with ['lower_bound', 'upper_bound', 'marginal_rate']
    Assumes schedule is sorted by lower_bound.
    """
    tax = 0.0
    for _, slab in schedule.iterrows():
        low = slab['lower_bound']
        up = slab['upper_bound']
        rate = slab['marginal_rate']
        
        # Skip incomplete or reconciliation rows (just in case)
        if low == 0 and up == 0:
            continue

        # Logic: rate * max(0, min(income, upper) - lower)
        # If upper is NaN/None, treat as infinity
        
        if pd.isna(up) or up == 0: # Open-ended top slab
            portion = max(0, income - low)
        else:
            portion = max(0, min(income, up) - low)
            
        tax += portion * rate
        
    return tax

def run_simulation_method1(base_data_rows, baseline_schedule, scenario_schedule):
    """
    Full schedule recomputation.
    base_data_rows: DataFrame containing filer groups (rows) to simulate.
                    Must have 'avg_taxable_income_per_filer', 'total_filers'.
    baseline_schedule: DF defining the BASELINE tax brackets.
    scenario_schedule: DF defining the SCENARIO tax brackets.
    """
    results = base_data_rows.copy()
    
    # Ensure schedules are sorted
    baseline_schedule = baseline_schedule.sort_values('lower_bound')
    scenario_schedule = scenario_schedule.sort_values('lower_bound')

    base_taxes = []
    scen_taxes = []
    
    # Compute tax for each filer group
    for _, row in results.iterrows():
        income = row['avg_taxable_income_per_filer']
        if pd.isna(income) or income <= 0:
            base_taxes.append(0)
            scen_taxes.append(0)
            continue
            
        # Baseline Tax
        t_base = progressive_tax_calc(income, baseline_schedule)
        base_taxes.append(t_base)
        
        # Scenario Tax
        t_scen = progressive_tax_calc(income, scenario_schedule)
        scen_taxes.append(t_scen)
        
    results['sim_tax_per_filer_base'] = base_taxes
    results['sim_tax_per_filer_scen'] = scen_taxes
    
    results['sim_revenue_base'] = results['sim_tax_per_filer_base'] * results['total_filers']
    results['sim_revenue_scen'] = results['sim_tax_per_filer_scen'] * results['total_filers']
    results['delta_revenue'] = results['sim_revenue_scen'] - results['sim_revenue_base']
    
    # No leakage adj
    results['delta_revenue_adj'] = results['delta_revenue']

    return results

def run_simulation_method2(base_data_rows, target_bracket, delta_rate):
    """
    Quick sensitivity: Change ONE bracket rate.
    target_bracket: dict/row with 'lower_bound', 'upper_bound'
    """
    results = base_data_rows.copy()
    
    low = target_bracket['lower_bound']
    up = target_bracket['upper_bound']
    
    deltas = []
    
    for _, row in results.iterrows():
        income = row['avg_taxable_income_per_filer']
        if pd.isna(income) or income <= 0:
            deltas.append(0)
            continue
            
        # Calculate portion of income in this bracket
        if pd.isna(up) or up == 0:
            portion = max(0, income - low)
        else:
            portion = max(0, min(income, up) - low)
            
        # Delta Tax = portion * delta_rate
        d_tax = portion * delta_rate
        d_rev = d_tax * row['total_filers']
        deltas.append(d_rev)
        
    results['delta_revenue_m2'] = deltas
    
    # No leakage adj
    results['delta_revenue_m2_adj'] = results['delta_revenue_m2']
        
    return results

# --- Main App ---

def main():
    st.markdown('<div class="main-header">PK Pakistan Income Tax Slab Analysis Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Analyze aggregates, efficiency, and policy scenarios.</div>', unsafe_allow_html=True)

    # File check
    try:
        mtime = os.path.getmtime(FILE_PATH)
    except:
        mtime = 0
    df_raw = load_data(mtime)
    
    if df_raw is None:
        return
        
    df = process_data(df_raw)

    tab_dash, tab_comp, tab_sim, tab_data = st.tabs([
        "📊 Dashboard", 
        "⚖️ Salaried vs Non-Salaried", 
        "🛠 Policy Simulator", 
        "📥 Data/Export"
    ])

    # --- TAB 1: DASHBOARD ---
    with tab_dash:
        # Filters
        c1, c2 = st.columns(2)
        with c1:
            years = sorted(df['year'].unique(), reverse=True)
            sel_year = st.selectbox("Select Tax Year", years, key="dash_year")
        with c2:
            # Defined preferred order: Salaried, then Non_Salaried (or others sorted)
            avail_types = df['taxpayer_type'].unique().tolist()
            # Sort helper: Salaried first, then others alphabetical
            avail_types.sort(key=lambda x: (0 if x == 'Salaried' else 1, x))
            
            types = ['All'] + avail_types
            sel_type = st.selectbox("Select Taxpayer Type", types, key="dash_type")
            
        mask = (df['year'] == sel_year)
        if sel_type != 'All':
            mask = mask & (df['taxpayer_type'] == sel_type)
        
        df_dash = df[mask].copy()
        
        # Metrics
        total_filers = df_dash['total_filers'].sum()
        total_income_bn = df_dash['taxable_income_9100'].sum() / 1e9
        net_tax_bn = df_dash['net_tax_9200'].sum() / 1e9
        avg_etr = (df_dash['net_tax_9200'].sum() / df_dash['taxable_income_9100'].sum()) * 100 if df_dash['taxable_income_9100'].sum() > 0 else 0
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Filers", f"{total_filers:,.0f}")
        m2.metric("Total Income (PKR bn)", f"{total_income_bn:,.1f}")
        m3.metric("Net Tax (PKR bn)", f"{net_tax_bn:,.1f}")
        m4.metric("Avg ETR (Net)", f"{avg_etr:.1f}%")
        
        st.divider()

        def render_charts(data, title_prefix):
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Revenue Share")
                total_rev = data['net_tax_9200'].sum()
                data_chart = data.copy()
                data_chart['local_share'] = (data_chart['net_tax_9200'] / total_rev) * 100 if total_rev > 0 else 0
                
                fig_bar = px.bar(
                    data_chart, x='slab_id', y='local_share',
                    labels={'local_share': 'Share (%)', 'slab_id': 'Slab ID'},
                    text_auto='.1f', hover_data=['short_label']
                )
                fig_bar.update_layout(title=None, margin=dict(t=0))
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with c2:
                st.subheader("Effective Tax Rate (%)")
                data_chart['etr_pct'] = data_chart['etr_net'] * 100
                fig_line = px.line(
                    data_chart, x='slab_id', y='etr_pct',
                    labels={'etr_pct': 'ETR %', 'slab_id': 'Slab ID'},
                    markers=True, hover_data=['short_label']
                )
                fig_line.update_layout(title=None, margin=dict(t=0))
                st.plotly_chart(fig_line, use_container_width=True)

        if sel_type == 'All':
            unique_types = sorted(df_dash['taxpayer_type'].unique(), key=lambda x: (0 if x == 'Salaried' else 1, x))
            for t in unique_types:
                st.markdown(f"### {t}")
                sub_df = df_dash[df_dash['taxpayer_type'] == t]
                if not sub_df.empty:
                    render_charts(sub_df, t)
                st.divider()
        else:
            render_charts(df_dash, sel_type)

    # --- TAB 2: COMPARISON ---
    with tab_comp:
        st.subheader("Salaried vs Non-Salaried Comparison")
        comp_year = st.selectbox("Select Year", years, key="comp_year")
        comp_df = df[df['year'] == comp_year].copy()
        agg_comp = comp_df.groupby('taxpayer_type')[['total_filers', 'taxable_income_9100', 'net_tax_9200']].sum().reset_index()
        agg_comp['etr'] = (agg_comp['net_tax_9200'] / agg_comp['taxable_income_9100']) * 100
        
        c1, c2 = st.columns(2)
        with c1:
            fig_tax = px.pie(agg_comp, values='net_tax_9200', names='taxpayer_type', title=f"Net Tax Contribution", hole=0.4)
            st.plotly_chart(fig_tax, use_container_width=True)
        with c2:
            fig_etr = px.bar(agg_comp, x='taxpayer_type', y='etr', color='taxpayer_type', title=f"Average ETR", text_auto='.1f')
            st.plotly_chart(fig_etr, use_container_width=True)

    # --- TAB 3: SIMULATOR ---
    with tab_sim:
        st.header("Policy Simulator")
        
        # 1. Selection
        # 1. Selection
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sim_year = st.selectbox("Year", years, key="sim_year")
        with col_s2:
            # Single type selection is best for schedule editing
            sim_types_avail = df['taxpayer_type'].unique().tolist()
            sim_types_avail.sort(key=lambda x: (0 if x == 'Salaried' else 1, x))
            
            sim_type = st.selectbox("Taxpayer Type", sim_types_avail, key="sim_type")

        # Prepare Data (exclude reconciliation)
        sim_mask = (df['year'] == sim_year) & (df['taxpayer_type'] == sim_type) & (~df['is_reconciliation'])
        df_sim_base = df[sim_mask].copy().sort_values('lower_bound')
        
        if df_sim_base.empty:
            st.warning("No data found.")
        else:
            # --- Method 1: Schedule Editor ---
            st.markdown("### Method 1: Full Schedule Adjustment")
            st.caption("Edit the tax brackets below. The simulator will re-calculate total tax for all filers.")
            
            # Show editor
            # We want strict types
            df_editor_input = df_sim_base[['slab_id', 'lower_bound', 'upper_bound', 'marginal_rate']].copy()
            
            edited_schedule = st.data_editor(
                df_editor_input,
                column_config={
                    "marginal_rate": st.column_config.NumberColumn(
                        "Marginal Rate (0-1)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", required=True
                    ),
                    "lower_bound": st.column_config.NumberColumn("Lower Bound", required=True),
                    "upper_bound": st.column_config.NumberColumn("Upper Bound"),
                    "slab_id": st.column_config.TextColumn("Slab ID", disabled=True)
                },
                hide_index=True,
                key="sim_editor",
                use_container_width=True
            )
            
            # --- Method 2: Quick Sensitivity ---
            st.markdown("### Method 2: Single Bracket Sensitivity")
            st.caption("Change the rate of a specific bracket to see immediate impact.")
            
            c_m2_1, c_m2_2 = st.columns(2)
            with c_m2_1:
                # Create options like "S_1: 0 - 600k"
                slab_options = df_sim_base['slab_id'].tolist()
                slab_labels = [f"{row['slab_id']} ({row['short_label']})" for _, row in df_sim_base.iterrows()]
                lbl_map = dict(zip(slab_labels, slab_options))
                
                sel_lbl = st.selectbox("Select Bracket", slab_labels)
                sel_slab_id = lbl_map[sel_lbl]
                
                # Fetch baseline info
                sel_bracket_row = df_sim_base[df_sim_base['slab_id'] == sel_slab_id].iloc[0]
                base_rate = float(sel_bracket_row['marginal_rate']) if pd.notnull(sel_bracket_row['marginal_rate']) else 0.0
                
            with c_m2_2:
                st.markdown(f"**Baseline Rate:** `{base_rate:.3f}` ({base_rate*100:.1f}%)")
                
                m2_mode = st.radio("Input Mode", ["New Scenario Rate (Absolute)", "Rate Change (pp)"], horizontal=True, label_visibility="visible")
                
                if "Absolute" in m2_mode:
                    new_rate_input = st.number_input(
                        "Scenario marginal rate (0-1)",
                        min_value=0.0, max_value=1.0,
                        value=base_rate, step=0.005, format="%.3f"
                    )
                    delta_rate = new_rate_input - base_rate
                else:
                    delta_pp_input = st.number_input(
                        "Rate change (percentage points)",
                        min_value=-100.0, max_value=100.0,
                        value=0.0, step=0.5, format="%.2f",
                        help="e.g. +1.0 means +1% tax rate (e.g. 10% -> 11%)"
                    )
                    delta_rate = delta_pp_input / 100.0

                new_rate_final = base_rate + delta_rate
                st.caption(f"Change: **{delta_rate:+.3f}** ({delta_rate*100:+.2f} pp) → Final: **{new_rate_final:.3f}**")

            # --- RUN BUTTON ---
            st.divider()
            run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

            if run_btn:
                # Run Method 1
                # Enforce numeric on edited schedule
                edited_schedule['lower_bound'] = pd.to_numeric(edited_schedule['lower_bound'], errors='coerce').fillna(0)
                edited_schedule['upper_bound'] = pd.to_numeric(edited_schedule['upper_bound'], errors='coerce')
                edited_schedule['marginal_rate'] = pd.to_numeric(edited_schedule['marginal_rate'], errors='coerce').fillna(0)
                
                # Check for changes
                # (Optional warning but we run anyway)
                
                # Baseline Schedule
                baseline_schedule = df_sim_base[['lower_bound', 'upper_bound', 'marginal_rate']].copy()
                
                res_m1 = run_simulation_method1(df_sim_base, baseline_schedule, edited_schedule)
                
                # Run Method 2
                res_m2 = run_simulation_method2(df_sim_base, sel_bracket_row, delta_rate)
                
                # --- RESULTS ---
                st.markdown("## Simulation Results")
                
                # Aggregates
                base_rev = res_m1['sim_revenue_base'].sum() / 1e9
                m1_rev = res_m1['sim_revenue_scen'].sum() / 1e9 # Pre-leakage? Or we usually care about the valid delta
                
                m1_delta_adj = res_m1['delta_revenue_adj'].sum() / 1e9
                m2_delta_adj = res_m2['delta_revenue_m2_adj'].sum() / 1e9
                
                m1_final_rev = base_rev + m1_delta_adj
                m2_final_rev = base_rev + m2_delta_adj
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                col_res1.metric("Baseline Revenue", f"{base_rev:,.2f} B")
                
                col_res2.metric(
                    "Method 1: Schedule Shift", 
                    f"{m1_final_rev:,.2f} B", 
                    delta=f"{m1_delta_adj:+.2f} B (Full)",
                    delta_color="normal"
                )
                
                col_res3.metric(
                    f"Method 2: {base_rate*100:.1f}% → {new_rate_final*100:.1f}%", 
                    f"{m2_final_rev:,.2f} B", 
                    delta=f"{m2_delta_adj:+.2f} B",
                    delta_color="normal"
                )
                
                # Charts (Waterfall / Comparisons)
                st.subheader("Comparison by Slab")
                
                # We want to show M1 Delta vs M2 Delta by Slab
                
                # Prepare comparison DF
                df_res_chart = res_m1[['slab_id', 'short_label', 'delta_revenue_adj']].copy()
                df_res_chart['Method 1 (Schedule)'] = df_res_chart['delta_revenue_adj']
                
                # Merge M2
                df_res_chart['Method 2 (Sensitivity)'] = res_m2['delta_revenue_m2_adj']
                
                # Melting for Grouped Bar
                df_melt = df_res_chart.melt(
                    id_vars=['slab_id', 'short_label'], 
                    value_vars=['Method 1 (Schedule)', 'Method 2 (Sensitivity)'],
                    var_name='Method', value_name='Delta Revenue'
                )
                
                fig_comp = px.bar(
                    df_melt, x='slab_id', y='Delta Revenue', color='Method',
                    title="Revenue Impact by Slab (PKR)",
                    barmode='group',
                    hover_data=['short_label']
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Detailed Table
                st.markdown("### Detailed Impact Table")
                
                # Merge for formatted display
                final_table = res_m1[['slab_id', 'total_filers', 'avg_taxable_income_per_filer', 'sim_revenue_base']].copy()
                final_table['M1_Delta'] = res_m1['delta_revenue_adj']
                final_table['M2_Delta'] = res_m2['delta_revenue_m2_adj']
                
                st.dataframe(final_table.style.format({
                    'total_filers': '{:,.0f}',
                    'avg_taxable_income_per_filer': '{:,.0f}',
                    'sim_revenue_base': '{:,.0f}',
                    'M1_Delta': '{:,.0f}',
                    'M2_Delta': '{:,.0f}'
                }))


    # --- TAB 4: DATA ---
    with tab_data:
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "tax_data_export.csv", "text/csv")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- CONFIGURATION & CONSTANTS ---
st.set_page_config(page_title="PIT-Sim Pakistan", layout="wide")

# IMF Style CSS
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stSidebar {
        background-color: #f0f2f6;
        border-right: 1px solid #e0e0e0;
    }
    h1 {
        color: #003366;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        border-bottom: 2px solid #003366;
        padding-bottom: 10px;
    }
    h2, h3 {
        color: #003366;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #666666;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
        border-bottom: 2px solid #003366 !important;
        color: #003366 !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Direct Column Mapping for the NEW format
NEW_COLUMN_MAPPING = {
    'Individual IDs': 'person_id',
    'Taxable income': 'taxable_income',
    'Normal income tax': 'normal_income_tax_dataset',
    'Tax reductions': 'tax_reductions',
    'Tax credits': 'tax_credits',
    'Surcharge on high earning persons u/s 4AB': 'surcharge_4AB',
    'Tax on high earning persons u/s 4C': 'tax_4C',
    'Withholding Income Tax': 'withholding_tax',
    'Advance Income Tax': 'advance_tax'
}

# Old Mapping (for fallback or reference)
TAX_CODE_MAPPING = {
    '9100': 'taxable_income',
    '920000': 'normal_income_tax_dataset',
    '9309': 'tax_reductions',
    '9329': 'tax_credits',
    '923184': 'surcharge_4AB',
    '9231822': 'tax_4C',
    '9201': 'withholding_tax',
    '9202': 'advance_tax'
}

REQUIRED_VARS = [
    'taxable_income', 'normal_income_tax_dataset', 'tax_reductions', 
    'tax_credits', 'surcharge_4AB', 'tax_4C'
]

DEFAULT_SLABS = [
    {"Lower Limit": 0, "Upper Limit": 600000, "Rate": 0.0, "Fixed Tax": 0},
    {"Lower Limit": 600000, "Upper Limit": 1200000, "Rate": 0.05, "Fixed Tax": 0},
    {"Lower Limit": 1200000, "Upper Limit": 2200000, "Rate": 0.15, "Fixed Tax": 30000},
    {"Lower Limit": 2200000, "Upper Limit": 3200000, "Rate": 0.25, "Fixed Tax": 180000},
    {"Lower Limit": 3200000, "Upper Limit": 4100000, "Rate": 0.30, "Fixed Tax": 430000},
    {"Lower Limit": 4100000, "Upper Limit": None, "Rate": 0.35, "Fixed Tax": 700000},
]

# --- CORE FUNCTIONS ---

def load_data(file):
    if hasattr(file, 'name') and file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    return df

def process_data(df):
    """Detect format and return person-level data"""
    
    # Check if it's the NEW format (person-level already)
    if 'Individual IDs' in df.columns:
        df_person = df.copy()
        df_person = df_person.rename(columns=NEW_COLUMN_MAPPING)
        
        # Ensure all numeric
        for col in df_person.columns:
            if col != 'person_id':
                df_person[col] = pd.to_numeric(df_person[col], errors='coerce').fillna(0)
        
        return df_person, {} # No labels needed for this format
    
    # Else check for the OLD format (Computation/Tax_code rows)
    elif 'Tax_code' in df.columns and any(str(c).startswith('id_') for c in df.columns):
        id_cols = [c for c in df.columns if str(c).startswith('id_')]
        label_dict = df.set_index('Tax_code')['Computation'].to_dict()
        df_long = df.melt(id_vars=['Computation', 'Tax_code'], value_vars=id_cols, 
                          var_name='person_id', value_name='value')
        
        df_long['Tax_code'] = df_long['Tax_code'].astype(str)
        df_person = df_long.pivot(index='person_id', columns='Tax_code', values='value').reset_index()
        df_person = df_person.rename(columns=TAX_CODE_MAPPING)
        
        for col in df_person.columns:
            if col != 'person_id':
                df_person[col] = pd.to_numeric(df_person[col], errors='coerce').fillna(0)
        return df_person, label_dict
    
    else:
        st.error("Unknown dataset format. Expected 'Individual IDs' column or 'Tax_code' with 'id_n' columns.")
        return None, None

def compute_slab_tax(income, slabs_df):
    if income <= 0:
        return 0
    
    tax = 0
    num_slabs = len(slabs_df)
    sorted_slabs = slabs_df.sort_values(by="Lower Limit")
    
    for idx, (_, row) in enumerate(sorted_slabs.iterrows()):
        lower = row['Lower Limit']
        upper = row['Upper Limit']
        rate = row['Rate']
        fixed = row['Fixed Tax']
        
        if idx == num_slabs - 1 or pd.isna(upper): # Last slab or infinite upper
            if income > lower:
                tax = fixed + rate * (income - lower)
                break
        else:
            if lower < income <= upper:
                tax = fixed + rate * (income - lower)
                break
    return max(0, tax)

def run_simulation(df, slabs_df, flat_shock=0.0):
    res_df = df.copy()
    res_df['adjusted_income'] = res_df['taxable_income'] * (1 + flat_shock/100)
    res_df['normal_calc'] = res_df['adjusted_income'].apply(lambda x: compute_slab_tax(x, slabs_df))
    
    # Assign Slab ID
    def get_slab_id(income, s_df):
        if income <= 0: return -1
        num = len(s_df)
        for i, (_, row) in enumerate(s_df.sort_values("Lower Limit").iterrows()):
            if i == num - 1 or pd.isna(row['Upper Limit']):
                if income > row['Lower Limit']: return i
            elif row['Lower Limit'] < income <= row['Upper Limit']:
                return i
        return -1
    
    res_df['slab_id'] = res_df['adjusted_income'].apply(lambda x: get_slab_id(x, slabs_df))
    
    res_df['after_relief'] = (res_df['normal_calc'] - res_df['tax_reductions'] - res_df['tax_credits']).clip(lower=0)
    res_df['total_liability'] = res_df['after_relief'] + res_df['surcharge_4AB'] + res_df['tax_4C']
    return res_df

def get_slab_summary(results_df, slabs_df, results_col_prefix, weights=None):
    sorted_slabs = slabs_df.sort_values(by="Lower Limit").copy()
    num_slabs = len(sorted_slabs)
    summary = []
    
    if weights is None:
        weights = pd.Series(1, index=results_df.index)
    
    for i, (_, row) in enumerate(sorted_slabs.iterrows()):
        lower = row['Lower Limit']
        upper = row['Upper Limit']
        label = f"{lower:,.0f} - {upper:,.0f}" if not (i == num_slabs - 1 or pd.isna(upper)) else f"> {lower:,.0f}"
        
        mask = results_df[results_col_prefix] == i
        
        # Proper handling of weights whether it's a Series or an integer
        if isinstance(weights, pd.Series):
            count = weights[mask].sum()
        else:
            count = mask.sum() * weights
            
        summary.append({
            "Slab Range": label,
            "Count": count
        })
    return pd.DataFrame(summary)

def get_distribution_table(df, tax_col, income_col='taxable_income'):
    df = df.sort_values(by=income_col)
    try:
        df['decile'] = pd.qcut(df[income_col], 10, labels=False, duplicates='drop') + 1
    except:
        df['decile'] = pd.qcut(df[income_col].rank(method='first'), 10, labels=False) + 1
    
    weight_col = 'weight' if 'weight' in df.columns else None
    if weight_col:
        grouped = df.groupby('decile').apply(lambda x: pd.Series({
            'Mean Income': np.average(x[income_col], weights=x[weight_col]),
            'Mean Tax': np.average(x[tax_col], weights=x[weight_col]),
            'Total Tax': (x[tax_col] * x[weight_col]).sum(),
            'Count': x[weight_col].sum()
        }))
    else:
        grouped = df.groupby('decile').agg({
            income_col: 'mean',
            tax_col: ['mean', 'sum'],
            'person_id': 'count'
        })
        grouped.columns = ['Mean Income', 'Mean Tax', 'Total Tax', 'Count']
    
    total_revenue = grouped['Total Tax'].sum()
    grouped['Tax Share %'] = (grouped['Total Tax'] / total_revenue) * 100
    grouped['Avg Tax Rate %'] = (grouped['Mean Tax'] / grouped['Mean Income'].replace(0, np.nan)) * 100
    return grouped.reset_index()

# --- UI LAYOUT ---

st.sidebar.title("PIT-Sim Controls")
uploaded_file = st.sidebar.file_uploader("Update Input File (Excel/CSV)", type=["xlsx", "csv"])

# Simple Header
st.title("PIT Sim Pakistan")

if not uploaded_file:
    # Use relative path for portability on Streamlit Cloud
    default_path = "PIT Data.xlsx"
    try:
        with open(default_path, "rb") as f:
            uploaded_file = io.BytesIO(f.read())
            uploaded_file.name = "PIT Data.xlsx"
    except:
        st.sidebar.warning("Default data not found. Please upload a file.")

if uploaded_file:
    raw_df = load_data(uploaded_file)
    df_person, labels = process_data(raw_df)
    
    if df_person is not None:
        # Silently fix missing required columns
        for var in REQUIRED_VARS:
            if var not in df_person.columns:
                df_person[var] = 0
            
        with st.sidebar:
            st.divider()
            st.write("### 🏠 Reform Parameters")
            with st.expander("📝 Baseline Slabs", expanded=False):
                st.info("💡 The last slab's 'Upper Limit' is ignored.")
                baseline_slabs = st.data_editor(pd.DataFrame(DEFAULT_SLABS), num_rows="dynamic", key="baseline_slabs")
            
            with st.expander("🚀 Reform Slabs", expanded=True):
                reform_slabs = st.data_editor(pd.DataFrame(DEFAULT_SLABS), num_rows="dynamic", key="reform_slabs")

            st.divider()
            st.write("### ⚙️ Global Sensitivity")
            flat_shock = st.number_input("Flat Income Growth (%)", -100.0, 100.0, 0.0)
            run_btn = st.button("Recalculate Impact", type="primary", use_container_width=True)

        baseline_results = run_simulation(df_person, baseline_slabs)
        reform_results = run_simulation(df_person, reform_slabs, flat_shock=flat_shock)
        
        results = df_person[['person_id', 'taxable_income', 'normal_income_tax_dataset']].copy()
        results['baseline_tax'] = baseline_results['total_liability']
        results['reform_tax'] = reform_results['total_liability']
        results['delta_tax'] = results['reform_tax'] - results['baseline_tax']
        
        tab0, tab1, tab2, tab4, tab5 = st.tabs([
            "📋 Individual PIT", "📊 Revenue Impact", "📉 Distribution", "🖼️ Visualizations", "📂 Export"
        ])
        
        with tab1:
            st.subheader("Aggregate Policy Impact")
            weights = results['weight'] if 'weight' in results.columns else 1
            total_baseline = (results['baseline_tax'] * weights).sum()
            total_reform = (results['reform_tax'] * weights).sum()
            rev_change = total_reform - total_baseline
            pct_change = (rev_change / total_baseline * 100) if total_baseline != 0 else 0
            
            results['baseline_slab_id'] = baseline_results['slab_id']
            results['reform_slab_id'] = reform_results['slab_id']

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baseline Revenue", f"{total_baseline:,.0f}")
            c2.metric("Reform Revenue", f"{total_reform:,.0f}")
            c3.metric("Revenue Change", f"{rev_change:,.0f}", delta=f"{pct_change:.1f}%")
            c4.metric("Change %", f"{pct_change:.2f}%")
            
            st.divider()
            st.subheader("Taxpayer Population by Slab")
            
            b_summary = get_slab_summary(results, baseline_slabs, 'baseline_slab_id', weights)
            r_summary = get_slab_summary(results, reform_slabs, 'reform_slab_id', weights)
            
            # Merge summaries for comparison
            comparison = b_summary.merge(r_summary, on="Slab Range", suffixes=(' (Baseline)', ' (Reform)'))
            st.dataframe(comparison, use_container_width=True)
            
        with tab0:
            st.subheader("Individual PIT Data")
            st.dataframe(df_person.head(1000), use_container_width=True)

        with tab2:
            st.subheader("Distribution Analysis")
            dist_table = get_distribution_table(results, 'reform_tax')
            st.dataframe(dist_table.style.format(precision=2), use_container_width=True)
            
            st.subheader("Mean Tax Change by Decile")
            delta_dist = results.groupby(pd.qcut(results['taxable_income'].rank(method='first'), 10, labels=False) + 1).agg({'delta_tax': 'mean'}).reset_index()
            delta_dist.columns = ['Decile', 'Mean Tax Change']
            st.dataframe(delta_dist.style.format(precision=2), use_container_width=True)


        with tab4:
            st.subheader("Visual Analytics")
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes[0, 0].hist(results['taxable_income'], bins=30, color='#003366', alpha=0.8, edgecolor='white')
            axes[0, 0].set_title("Income Distribution")
            axes[0, 1].bar(['Baseline', 'Reform'], [total_baseline, total_reform], color=['#003366', '#d4af37'])
            axes[0, 1].set_title("Total Revenue Comparison")
            axes[1, 0].plot(dist_table['decile'], dist_table['Tax Share %'], marker='o', color='#003366')
            axes[1, 0].set_title("Tax Share by Decile (%)")
            axes[1, 1].plot(dist_table['decile'], dist_table['Avg Tax Rate %'], marker='s', color='#d4af37')
            axes[1, 1].set_title("Average Tax Rate (%)")
            plt.tight_layout()
            st.pyplot(fig)

        with tab5:
            st.subheader("Export Results")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                results.to_excel(writer, sheet_name='Personnel_Impact', index=False)
                dist_table.to_excel(writer, sheet_name='Decile_Analysis', index=False)
            st.download_button("💾 Export Results (Excel)", buf.getvalue(), "PIT_Sim_Results.xlsx", use_container_width=True)

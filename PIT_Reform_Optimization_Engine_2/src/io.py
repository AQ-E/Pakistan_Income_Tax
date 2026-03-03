import pandas as pd
import numpy as np
import os

def load_slab_data(filepath):
    """
    Loads and cleans the slab-level aggregate dataset.
    """
    df = pd.read_excel(filepath)
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    
    new_cols = []
    for c in df.columns:
        c_low = c.lower()
        if 'taxable income' in c_low and '9100' in c_low:
            new_cols.append('taxable_income_9100')
        elif 'type_tax' in c_low or 'type of tax' in c_low or 'taxpayer' in c_low:
            new_cols.append('taxpayer_type')
        elif 'number' in c_low and ('persons' in c_low or 'filers' in c_low):
            new_cols.append('total_filers')
        elif 'normal income tax' in c_low and '9200' in c_low:
            new_cols.append('normal_income_tax_920000')
        elif 'mtr' in c_low:
            new_cols.append('marginal_rate')
        elif 'lower slab' in c_low or 'lower_bound' in c_low:
            new_cols.append('lower_bound')
        elif 'upper slab' in c_low or 'upper_bound' in c_low:
            new_cols.append('upper_bound')
        elif 'year' in c_low:
            new_cols.append('year')
        else:
            new_cols.append(c.lower().replace(' ', '_'))
    df.columns = new_cols

    # Robust LB / UB parser from 'Taxable Income Slab (Rs.)'
    def parse_slab(s):
        import re
        s = str(s).replace(',', '').strip()
        if '-' in s:
            p = s.split('-')
            return float(p[0]), float(p[1])
        elif 'Above' in s or '+' in s:
            p = re.findall(r'\d+', s)
            if p: return float(p[0]) + 1, np.inf
            return 0.0, np.inf
        else:
            return 0.0, np.inf

    if 'taxable_income_slab_(rs.)' in df.columns:
        df['lower_bound'], df['upper_bound'] = zip(*df['taxable_income_slab_(rs.)'].apply(parse_slab))
        
    if 'marginal_rate' in df.columns:
        df['marginal_rate'] = pd.to_numeric(df['marginal_rate'], errors='coerce').fillna(0)
    if 'upper_bound' in df.columns:
        df['upper_bound'] = pd.to_numeric(df['upper_bound'], errors='coerce').fillna(np.inf)
    if 'lower_bound' in df.columns:
        df['lower_bound'] = pd.to_numeric(df['lower_bound'], errors='coerce').fillna(0)
        
    # Map 'Type_Tax' to readable names
    if 'taxpayer_type' in df.columns:
        df['taxpayer_type'] = df['taxpayer_type'].replace({'S': 'Salaried', 'NS': 'Non-Salaried', 'AOP': 'AOP'})

    # Create a Consolidated type
    if 'year' in df.columns and 'taxpayer_type' in df.columns and 'lower_bound' in df.columns:
        agg_dict = {
            'total_filers': 'sum',
            'taxable_income_9100': 'sum',
            'normal_income_tax_920000': 'sum',
        }
        if 'marginal_rate' in df.columns: agg_dict['marginal_rate'] = 'mean'
        if 'nit_calculated' in df.columns: agg_dict['nit_calculated'] = 'sum'
        if 'etr' in df.columns: agg_dict['etr'] = 'mean'
        if 'cetr' in df.columns: agg_dict['cetr'] = 'mean'
        
        cons_df = df.groupby(['year', 'lower_bound', 'upper_bound']).agg(agg_dict).reset_index()
        cons_df['taxpayer_type'] = 'Consolidated'
        
        # We need to make sure all other columns exist
        for col in df.columns:
            if col not in cons_df.columns:
                cons_df[col] = df[col].mode()[0] if not df[col].empty else np.nan
        
        df = pd.concat([df, cons_df], ignore_index=True)
        df = df.sort_values(by=['year', 'taxpayer_type', 'lower_bound'])
    
    return df

def load_grid_data(filepath):
    """
    Loads and cleans the income-grid dataset.
    Supports both 'TOTAL TAX, ETR, CHANGE ETR' and 'Sheet1' sheet names.
    Normalizes column names for consistency.
    """
    try:
        df = pd.read_excel(filepath, sheet_name='TOTAL TAX, ETR, CHANGE ETR')
    except:
        df = pd.read_excel(filepath, sheet_name='Sheet1')
    
    # Clean column names: Collapse whitespace
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    
    # Normalize common column names
    col_map = {
        'Taxable Income': 'Annual Income',
        'ETR_23': 'ETR_FY23',
        'ETR_24': 'ETR_FY24',
        'ETR_25': 'ETR_FY25',
        'ETR_26': 'ETR_FY26'
    }
    df = df.rename(columns=col_map)
    
    # Treat ETR at Annual Income=0 as 0
    etr_cols = [c for c in df.columns if 'ETR_' in c]
    for col in etr_cols:
        if 'Annual Income' in df.columns:
            df.loc[df['Annual Income'] == 0, col] = 0
        df[col] = df[col].fillna(0)
        
    return df

def get_data_paths():
    """
    Returns the absolute paths of the data files.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    slab_file = os.path.join(base_dir, 'Income Tax Liability S_NS_AOP.xlsx')
    grid_file = os.path.join(base_dir, 'tax liability at 1 Lac.xlsx')  # Optional auxiliary
    truth_file = os.path.join(base_dir, 'PIT_slabs_2025.xlsx')
    return slab_file, truth_file

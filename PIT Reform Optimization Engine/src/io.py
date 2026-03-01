import pandas as pd
import numpy as np
import os

def load_slab_data(filepath):
    """
    Loads and cleans the slab-level aggregate dataset.
    """
    df = pd.read_excel(filepath)
    
    # Clean column names: Collapse all whitespace (including \n and extra spaces)
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    
    new_cols = []
    for c in df.columns:
        c_low = c.lower()
        if 'taxable income' in c_low or '9100' in c_low:
            new_cols.append('taxable_income_9100')
        elif 'total filers' in c_low:
            new_cols.append('total_filers')
        elif 'normal' in c_low and ('tax' in c_low or '9200' in c_low):
            new_cols.append('normal_income_tax_920000')
        elif 'taxpayer' in c_low:
            new_cols.append('taxpayer_type')
        elif 'year' in c_low:
            new_cols.append('year')
        elif 'lower' in c_low:
            new_cols.append('lower_bound')
        elif 'upper' in c_low:
            new_cols.append('upper_bound')
        elif 'marginal' in c_low:
            new_cols.append('marginal_rate')
        else:
            new_cols.append(c.lower().replace(' ', '_'))
    df.columns = new_cols
    
    # Data cleaning
    df['upper_bound'] = pd.to_numeric(df['upper_bound'], errors='coerce')
    df['upper_bound'] = df['upper_bound'].fillna(np.inf)
    
    df['marginal_rate'] = df['marginal_rate'].fillna(0)
    
    # Ensure lower bounds are numeric
    df['lower_bound'] = pd.to_numeric(df['lower_bound'], errors='coerce').fillna(0)
    
    # Validate lower bounds ascending within each (Year, Taxpayer_type)
    df = df.sort_values(by=['year', 'taxpayer_type', 'lower_bound'])
    
    return df

def load_grid_data(filepath):
    """
    Loads and cleans the income-grid dataset.
    """
    df = pd.read_excel(filepath, sheet_name='TOTAL TAX, ETR, CHANGE ETR')
    
    # Clean column names
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    
    # Treat ETR at Annual Income=0 as 0
    etr_cols = [c for c in df.columns if c.startswith('ETR_')]
    for col in etr_cols:
        df.loc[df['Annual Income'] == 0, col] = 0
        df[col] = df[col].fillna(0)
        
    return df

def get_data_paths():
    """
    Returns the absolute paths of the data files.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    slab_file = os.path.join(base_dir, 'Slab wise Taxable Income Filers & Normal Tax_3012026.xlsx')
    grid_file = os.path.join(base_dir, 'tax liability at various income levels.xlsx')
    return slab_file, grid_file

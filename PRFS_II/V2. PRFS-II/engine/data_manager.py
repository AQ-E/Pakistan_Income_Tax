import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

def load_data(path, sheet_name="Sheet1"):
    """
    Loads raw Excel data and performs initial cleaning.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Handle Pakistan FY formatting or use existing year_end
    if 'Year' in df.columns:
        df['year_end'] = df['Year'].apply(lambda x: int(str(x)[:4]) + 1 if '-' in str(x) else int(x))
    
    if 'year_end' in df.columns:
        df = df.sort_values('year_end').reset_index(drop=True)
        df.index = pd.PeriodIndex(df['year_end'], freq='Y')
    
    return df

def prepare_transforms(df):
    """
    Applies log transforms strictly to level variables.
    Rule: if x <= 0 set to NaN before log.
    Includes forward-fill for macro variables to ensure starting points are valid.
    """
    transformed_df = df.copy()
    
    levels = ['dt', 'gst', 'fed', 'customs', 'gdp', 'imports', 'dutiable_imports', 'lsm', 'consumption', 'exrate']
    rates = ['inflation', 'policy rate']
    
    # 1. Forward fill macro rates to handle missing end-of-series targets
    for col in rates:
        actual_col = next((c for c in df.columns if c.lower() == col.lower()), None)
        if actual_col:
            transformed_df[actual_col] = pd.to_numeric(df[actual_col], errors='coerce').ffill()
            transformed_df[col] = transformed_df[actual_col]

    # 2. Log transforms for levels
    for col in levels:
        actual_col = next((c for c in df.columns if c.lower() == col.lower()), None)
        if actual_col:
            # Enforce rule: if x <= 0 set to NaN
            s = pd.to_numeric(df[actual_col], errors='coerce')
            s[s <= 0] = np.nan
            transformed_df[f'log_{col}'] = np.log(s)
    
    # 3. Final cleanup: ensures we have log_dt and log_gdp for the tournament
    # but we don't drop the final row if it's useful for forecasting
    transformed_df = transformed_df.replace([np.inf, -np.inf], np.nan)
    
    return transformed_df

def load_buoyancy_data(file_path):
    """
    Reads Buoyancy Excel: Sensitivity Analysis B12:I20
    Returns a dict of FY2026 Actuals and FY2027 Buoyancy Projections.
    """
    try:
        # Load without headers to find markers
        df = pd.read_excel(file_path, sheet_name='Sensitivity Analysis', header=None, engine='openpyxl')
        
        # We look for "Expected Base Figures 2025-26" (which is FY26)
        # Based on screenshot, it's typically row 12 or 13 (0-indexed)
        # Let's search for the string to be robust
        base_row_idx = -1
        proj_row_idx = -1
        
        for i, row in df.iterrows():
            row_str = str(row[1]).lower() if len(row) > 1 else ""
            if "expected base" in row_str:
                base_row_idx = i
            if "projections" in row_str and "2026-27" in row_str:
                proj_row_idx = i
        
        if base_row_idx == -1 or proj_row_idx == -1:
            # Fallback to fixed indices from visual inspection
            base_row_idx = 12 if base_row_idx == -1 else base_row_idx
            proj_row_idx = 14 if proj_row_idx == -1 else proj_row_idx

        # Columns mapping: B=1 (Desc), C=2 (DT), D=3 (STD), E=4 (STM), F=5 (Total), G=6 (CD), H=7 (FED), I=8 (Total)
        # Note: image shows CD/FED order might vary. We follow the header names if possible.
        # Header row is usually at base_row_idx - 1
        headers = df.iloc[base_row_idx - 1, 2:9].tolist()
        base_vals = df.iloc[base_row_idx, 2:9].tolist()
        proj_vals = df.iloc[proj_row_idx, 2:9].tolist()
        
        col_map = {
            'dt': 0, 'st_d': 1, 'st_m': 2, 'st_total': 3, 'cd': 4, 'fed': 5, 'total': 6
        }
        
        # Clean mapping based on labels
        result = {}
        # Units in Billion PKR
        result['fy2026_base'] = {
            'dt': base_vals[0], 'st_domestic': base_vals[1], 'st_imports': base_vals[2],
            'st_total': base_vals[3], 'customs': base_vals[4], 'fed': base_vals[5], 'total': base_vals[6]
        }
        result['fy2027_buoyancy'] = {
            'dt': proj_vals[0], 'st_domestic': proj_vals[1], 'st_imports': proj_vals[2],
            'st_total': proj_vals[3], 'customs': proj_vals[4], 'fed': proj_vals[5], 'total': proj_vals[6]
        }
        return result
    except Exception as e:
        return None

def standardize_to_billion(val, source_unit='million'):
    """Standardizes reporting to PKR Billion"""
    if source_unit.lower() == 'million':
        return val / 1000.0
    return val

def run_diagnostics_step1(df, vars_to_test):
    """
    Step 1 Pre-Estimation: ADF + KPSS
    """
    results = []
    for var in vars_to_test:
        if var not in df.columns: continue
        series = df[var].dropna()
        if len(series) < 10: continue
        
        # ADF Test (Trend + Constant as macro series usually have both)
        adf_res = adfuller(series, regression='ct') 
        # KPSS Test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_res = kpss(series, regression='ct', nlags="auto")
        
        results.append({
            'Variable': var,
            'ADF Stat': round(adf_res[0], 3),
            'ADF p-val': round(adf_res[1], 3),
            'ADF Result': 'Stationary' if adf_res[1] < 0.05 else 'Non-Stationary',
            'KPSS Stat': round(kpss_res[0], 3),
            'KPSS p-val': round(kpss_res[1], 3),
            'KPSS Result': 'Stationary' if kpss_res[1] > 0.05 else 'Non-Stationary/Trend'
        })
    return pd.DataFrame(results)

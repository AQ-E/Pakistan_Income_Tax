import pandas as pd
import numpy as np

def update_load_slab_data(filepath):
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
        elif 'nit' in c_low:
            new_cols.append('nit_calculated')
        elif 'cetr' in c_low:
            new_cols.append('cetr')
        elif 'etr' in c_low:
            new_cols.append('etr')
        else:
            new_cols.append(c.lower().replace(' ', '_'))
    df.columns = new_cols

    if 'marginal_rate' in df.columns:
        df['marginal_rate'] = pd.to_numeric(df['marginal_rate'], errors='coerce').fillna(0)
    if 'upper_bound' in df.columns:
        df['upper_bound'] = pd.to_numeric(df['upper_bound'], errors='coerce').fillna(np.inf)
    if 'lower_bound' in df.columns:
        df['lower_bound'] = pd.to_numeric(df['lower_bound'], errors='coerce').fillna(0)

    # validate
    if 'year' in df.columns and 'taxpayer_type' in df.columns and 'lower_bound' in df.columns:
        df = df.sort_values(by=['taxpayer_type', 'lower_bound'])
    print("New columns:", list(df.columns))

update_load_slab_data('C:/Users/LENOVO/Downloads/Pakistan_Income_Tax_Slabs_app/PIT Reform Optimization Engine/Income Tax Liability S_NS_AOP.xlsx')

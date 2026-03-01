import numpy as np
import pandas as pd

def build_base_taxes(slabs_df):
    """
    Computes the cumulative base tax for each slab.
    """
    df = slabs_df.sort_values('lower_bound').copy()
    
    # Check if 'base_tax' column already exists (maybe from optimizer candidates)
    # If so, return it directly to avoid recalculation errors or double-counts
    # Wait, the column might be stale. Always best to recalculate if not sure.
    # But candidates from optimizer might be raw (no 'base_tax').
    
    df['base_tax'] = 0.0 # Placeholder
    
    # Calculate cumulative tax up to lower bound of each slab
    # Start with first slab (base tax 0 if min_exempt is covered correctly)
    
    current_cumulative = 0.0
    base_taxes = [0.0]
    
    # Iterate through slabs
    # We need to know previous upper bound to calculate step
    # Slabs are ordered.
    
    for i in range(len(df)):
        if i == 0:
            current_cumulative = 0.0
        else:
            prev_slab = df.iloc[i-1]
            width = prev_slab['upper_bound'] - prev_slab['lower_bound']
            tax_chunk = width * prev_slab['marginal_rate']
            current_cumulative += tax_chunk
            
        base_taxes.append(current_cumulative)
        
    # The last base_tax is for the (N+1)th theoretical slab, but we need N values.
    # df has N rows.
    # base_taxes has N+1 entries. entry 0 corresponds to slab 0.
    
    df['base_tax'] = base_taxes[:-1]
    
    return df

def calculate_surtax(income, surtax_params):
    """
    Calculates nonlinear surtax:
    extra_tax(y) = surtax_rate * max(0, y - surtax_threshold)^p
    """
    if not surtax_params or income <= 0:
        return 0.0
        
    threshold = surtax_params.get('threshold', 0)
    rate = surtax_params.get('rate', 0.0)
    power = surtax_params.get('power', 1.0)
    
    if income <= threshold:
        return 0.0
        
    taxable_excess = income - threshold
    extra_tax = rate * np.power(taxable_excess, power)
    
    return extra_tax

def calculate_total_tax(income, slabs_with_base, surtax_params=None):
    """
    Calculates total tax (Slab + Surtax).
    """
    if income <= 0:
        return 0.0
    
    # 1. Slab Tax
    # Optimized lookup
    # Find slab where lower <= income < upper
    # Note: last slab has upper = inf
    
    # Manual filter is slow for large grids in loops, but okay for single scalar calls
    # For scalars:
    slab_tax = 0.0
    
    # Optim: Assume slabs sorted. Last matching lower_bound wins.
    # Or strict interval check.
    
    relevant_slab = slabs_with_base[
        (income >= slabs_with_base['lower_bound']) & 
        (income < slabs_with_base['upper_bound'])
    ]
    
    if not relevant_slab.empty:
        row = relevant_slab.iloc[0]
        slab_tax = row['base_tax'] + (income - row['lower_bound']) * row['marginal_rate']
    else:
        # Fallback (e.g. income exactly equals upper bound of last finite slab? No, < is strict)
        # Check if income >= last lower bound (infinite slab) which should be caught above if upper is inf.
        # If no match, check max lower bound
        last_row = slabs_with_base.iloc[-1]
        if income >= last_row['lower_bound']:
             slab_tax = last_row['base_tax'] + (income - last_row['lower_bound']) * last_row['marginal_rate']
    
    slab_tax = max(0.0, slab_tax)
    
    # 2. Surtax
    extra_tax = calculate_surtax(income, surtax_params)
    
    return slab_tax + extra_tax

def calculate_tax(income, slabs_df): # Legacy wrapper
    if 'base_tax' not in slabs_df.columns:
        slabs_df = build_base_taxes(slabs_df)
    return calculate_total_tax(income, slabs_df, surtax_params=None)

def calculate_etr(income, tax):
    if income <= 0: return 0.0
    return tax / income

def get_tax_grid(income_grid, slabs_df, surtax_params=None):
    """
    Computes tax and ETR for a grid of income levels (Vectorized).
    """
    slabs_with_base = build_base_taxes(slabs_df)
    
    # Vectorized Slab Tax Calculation
    # Use np.searchsorted to find slab indices for all income points at once
    # bins = lower_bounds
    # But searchsorted finds insertion points.
    
    lower_bounds = slabs_with_base['lower_bound'].values
    upper_bounds = slabs_with_base['upper_bound'].values
    rates = slabs_with_base['marginal_rate'].values
    base_taxes = slabs_with_base['base_tax'].values
    
    # For slab i: lower[i] <= y < upper[i]
    # np.searchsorted(lower_bounds, y, side='right') - 1 gives index i such that lower[i] <= y
    
    slab_indices = np.searchsorted(lower_bounds, income_grid, side='right') - 1
    slab_indices = np.clip(slab_indices, 0, len(slabs_with_base) - 1)
    
    # Calculate Slab Tax
    # tax = base[i] + (y - lower[i]) * rate[i]
    
    y = income_grid
    idx = slab_indices
    
    slab_tax = base_taxes[idx] + (y - lower_bounds[idx]) * rates[idx]
    slab_tax = np.maximum(0.0, slab_tax)
    
    # Calculate Surtax (Vectorized)
    extra_tax = np.zeros_like(y, dtype=float)
    if surtax_params:
        thresh = surtax_params.get('threshold', 0)
        rate = surtax_params.get('rate', 0.0)
        power = surtax_params.get('power', 1.0)
        
        excess = np.maximum(0.0, y - thresh)
        extra_tax = rate * np.power(excess, power)
        
    total_tax = slab_tax + extra_tax
    
    # ETR
    etr = np.zeros_like(total_tax)
    mask = y > 0
    etr[mask] = total_tax[mask] / y[mask]
    
    return pd.DataFrame({
        'income': y,
        'tax': total_tax,
        'slab_tax': slab_tax, # For diagnostics
        'surtax': extra_tax,
        'etr': etr
    })

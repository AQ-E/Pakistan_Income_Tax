import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Core Metric Computation
# ---------------------------------------------------------------------------

def compute_metrics(grid_df, compliance_band=None):
    """
    Computes delta ETR and checks for progressivity / strict convexity.
    grid_df must have columns: 'income', 'etr'.
    Returns the same dataframe augmented with diagnostic columns.
    """
    df = grid_df.sort_values('income').copy()

    # ΔETR in percentage-points (pp)
    df['delta_etr'] = df['etr'].diff().fillna(0) * 100

    # Δ²ETR  (second difference – for strict convexity)
    df['d2_etr'] = df['delta_etr'].diff().fillna(0)

    # Progressivity:  ETR non-decreasing  →  ΔETR >= 0
    df['is_progressive'] = df['delta_etr'] >= -1e-7

    # Weak convexity:  ΔETR non-decreasing  →  Δ²ETR >= 0
    df['is_convex'] = df['d2_etr'] >= -1e-7

    # Strict convexity:  ΔETR_i >= ΔETR_{i-1} + eps  (eps = 1e-6 pp)
    eps = 1e-6
    delta_vals = df['delta_etr'].values
    strict = np.ones(len(delta_vals), dtype=bool)
    strict[0] = True
    strict[1] = True  # no prior pair at index 1
    for k in range(2, len(delta_vals)):
        strict[k] = delta_vals[k] >= delta_vals[k - 1] + eps
    df['is_strict_convex'] = strict

    # Compliance-band flag
    df['in_compliance_band'] = False
    if compliance_band:
        c_min, c_max = compliance_band
        df['in_compliance_band'] = (df['income'] >= c_min) & (df['income'] <= c_max)

    return df


# ---------------------------------------------------------------------------
# Strict-Convexity Check  (used by optimizer as hard constraint)
# ---------------------------------------------------------------------------

def check_strict_convexity(grid_df, eps=1e-6):
    """
    Returns (pass: bool, n_violations: int, violation_details: list[dict]).
    Each violation dict has: income, delta_etr_i, delta_etr_prev.
    """
    df = grid_df.sort_values('income').copy()
    delta = (df['etr'].diff().fillna(0) * 100).values
    incomes = df['income'].values

    violations = []
    for k in range(2, len(delta)):
        if delta[k] < delta[k - 1] + eps:
            violations.append({
                'income': incomes[k],
                'delta_etr_i': round(delta[k], 6),
                'delta_etr_prev': round(delta[k - 1], 6),
            })
    return len(violations) == 0, len(violations), violations[:10]


# ---------------------------------------------------------------------------
# Compliance-Band Scalar Metrics
# ---------------------------------------------------------------------------

def get_compliance_metrics(metrics_df):
    """
    Scalar summaries for the World-Bank compliance band.
    """
    band_df = metrics_df[metrics_df['in_compliance_band']]

    if band_df.empty:
        return {
            'max_jump_pp': 0.0,
            'volatility': 0.0,
            'avg_delta': 0.0,
            'points_count': 0,
        }

    return {
        'max_jump_pp': float(band_df['delta_etr'].max()),
        'volatility': float(band_df['delta_etr'].std()),
        'avg_delta': float(band_df['delta_etr'].mean()),
        'points_count': len(band_df),
    }


# ---------------------------------------------------------------------------
# Violation Summary (for UI diagnostic banner)
# ---------------------------------------------------------------------------

def get_violation_summary(metrics_df):
    """
    Returns a summary of progressivity and convexity violations.
    """
    prog_violations = metrics_df[~metrics_df['is_progressive']]
    convex_violations = metrics_df[~metrics_df['is_strict_convex']]

    summary = {
        'total_rows': len(metrics_df),
        'prog_pass': prog_violations.empty,
        'convex_pass': convex_violations.empty,
        'prog_violations_count': len(prog_violations),
        'convex_violations_count': len(convex_violations),
        'prog_violation_ranges': [],
        'convex_violation_ranges': [],
    }

    for _, row in prog_violations.head(10).iterrows():
        summary['prog_violation_ranges'].append(
            f"Income {row['income']:,.0f}: ΔETR={row['delta_etr']:.4f}")

    for _, row in convex_violations.head(10).iterrows():
        summary['convex_violation_ranges'].append(
            f"Income {row['income']:,.0f}: ΔETR={row['delta_etr']:.4f}, d²ETR={row['d2_etr']:.4f}")

    return summary

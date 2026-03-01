"""
Production-grade Pakistan PIT slab optimizer.
Guarantees: ALWAYS returns a best schedule — never fails.

Feasibility ladder:
  Stage 0 — fallback: base schedule (or repaired).  Always available.
  Stage 1 — tight convexity; pick best among zero-violation if any.
  Stage 2 — minimise convexity-violation count + spike + volatility.
  Stage 3 — minimise convexity-violation magnitude + spike + volatility.
  Stage 4 — maximise revenue; still report convexity metrics.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

_RNG = np.random.RandomState(42)

# ═══════════════════════════════════════════════════════════════════════
# 1.  CORE TAX ENGINE  (vectorised)
# ═══════════════════════════════════════════════════════════════════════

def compute_tax(schedule, y_grid):
    """
    Vectorised slab-tax computation.
    schedule : list[dict] with keys 'lower', 'upper', 'rate'.
    """
    y = np.asarray(y_grid, dtype=np.float64)
    lowers = np.array([s['lower'] for s in schedule])
    uppers = np.array([s['upper'] for s in schedule])
    rates  = np.array([s['rate']  for s in schedule])

    base_cum = np.zeros(len(schedule))
    for k in range(1, len(schedule)):
        width = uppers[k-1] - lowers[k-1]
        base_cum[k] = base_cum[k-1] + width * rates[k-1]

    idx = np.searchsorted(lowers, y, side='right') - 1
    idx = np.clip(idx, 0, len(schedule) - 1)
    tax = base_cum[idx] + (y - lowers[idx]) * rates[idx]
    return np.maximum(tax, 0.0)


def _schedule_to_list(slabs_df):
    out = []
    # Drop rows that are completely empty or have critical None values from the editor
    clean_df = slabs_df.dropna(subset=['lower_bound', 'marginal_rate']).copy()
    for _, r in clean_df.sort_values('lower_bound').iterrows():
        # Ensure values are numeric, default to 0 if something is weird
        try:
            lo = float(r['lower_bound'])
            up = float(r['upper_bound']) if pd.notna(r['upper_bound']) else np.inf
            rt = float(r['marginal_rate'])
            out.append({'lower': lo, 'upper': up, 'rate': rt})
        except (TypeError, ValueError):
            continue
    return out


def _list_to_df(schedule_list):
    return pd.DataFrame([
        {'lower_bound': s['lower'], 'upper_bound': s['upper'], 'marginal_rate': s['rate']}
        for s in schedule_list
    ])


# ═══════════════════════════════════════════════════════════════════════
# 2.  METRIC COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(schedule_list, y_grid,
                    compliance_band=(600_000, 3_000_000)):
    """
    Returns dict with arrays (y, tax, etr, delta_etr, d2_etr)
    and scalar summaries.
    """
    y   = np.asarray(y_grid, dtype=np.float64)
    tax = compute_tax(schedule_list, y)

    etr = np.zeros_like(y)
    mask = y > 0
    etr[mask] = tax[mask] / y[mask]

    delta_etr = np.diff(etr, prepend=0.0) * 100.0   # pp
    d2_etr    = np.diff(delta_etr, prepend=0.0)

    # Convexity (soft)
    neg = d2_etr[2:] < -1e-8
    convex_viol_count = int(neg.sum())
    convex_viol_mag   = float(-d2_etr[2:][neg].sum()) if convex_viol_count else 0.0

    # Band metrics
    blo, bhi = compliance_band
    bmask = (y >= blo) & (y <= bhi)
    bd = delta_etr[bmask]
    band_max_jump   = float(bd.max())        if len(bd) else 0.0
    band_volatility = float(np.sum(bd**2))   if len(bd) else 0.0
    band_p95        = float(np.percentile(bd, 95)) if len(bd) > 3 else 0.5

    # Violation details (top 10)
    viol_details = []
    vidx = np.where(d2_etr[2:] < -1e-8)[0] + 2
    for vi in vidx[:10]:
        viol_details.append({
            'income': float(y[vi]),
            'delta_etr_i': float(delta_etr[vi]),
            'delta_etr_prev': float(delta_etr[vi-1]),
            'd2_etr': float(d2_etr[vi]),
        })

    return {
        'y': y, 'tax': tax, 'etr': etr,
        'delta_etr': delta_etr, 'd2_etr': d2_etr,
        'convex_viol_count': convex_viol_count,
        'convex_viol_mag':   convex_viol_mag,
        'band_max_jump':     band_max_jump,
        'band_volatility':   band_volatility,
        'band_p95_delta':    band_p95,
        'viol_details':      viol_details,
    }


# ═══════════════════════════════════════════════════════════════════════
# 3.  REVENUE FROM AGGREGATES
# ═══════════════════════════════════════════════════════════════════════

def _estimate_revenue(schedule_list, agg_df, base_list=None, base_revenue_reported=None):
    """
    Simulation of revenue. 
    If base_list and base_revenue_reported are provided, it performs 'Base Calibration'.
    This ensures that if schedule_list == base_list, the result is exactly base_revenue_reported.
    """
    # Raw simulation
    total = 0.0
    for _, row in agg_df.iterrows():
        n = row['total_filers']
        if n > 0:
            avg_y = row['taxable_income_9100'] / n
            t = compute_tax(schedule_list, np.array([avg_y]))[0]
            total += t * n
            
    if base_list is not None and base_revenue_reported is not None:
        # Calculate calibration factor
        baseline_sim = 0.0
        for _, row in agg_df.iterrows():
            n = row['total_filers']
            if n > 0:
                avg_y = row['taxable_income_9100'] / n
                t_base = compute_tax(base_list, np.array([avg_y]))[0]
                baseline_sim += t_base * n
        
        if baseline_sim > 0:
            calib_factor = base_revenue_reported / baseline_sim
            total *= calib_factor
            
    return total


# ═══════════════════════════════════════════════════════════════════════
# 4.  THRESHOLD POOL
# ═══════════════════════════════════════════════════════════════════════

_ANCHORS = np.array([
    300_000, 600_000, 900_000, 1_200_000, 1_800_000,
    2_400_000, 3_000_000, 3_600_000, 4_800_000,
    6_000_000, 8_000_000, 10_000_000, 15_000_000, 20_000_000,
])

def _build_pool(agg_df, step=50_000):
    incomes = []
    for _, row in agg_df.iterrows():
        n = max(1, int(row['total_filers']))
        if n > 0 and row['taxable_income_9100'] > 0:
            avg = row['taxable_income_9100'] / n
            incomes.extend([avg] * min(n, 200))
    incomes = np.array(incomes) if incomes else np.array([600_000])
    qs = np.percentile(incomes, [10,20,30,40,50,60,70,80,90,95,99])
    qs = np.round(qs / step) * step
    pool = np.unique(np.concatenate([qs, _ANCHORS]))
    return np.sort(pool[pool > 0])


def _pick_thresholds(pool, n, method):
    if len(pool) <= n:
        return pool[:n]
    if method == 'q':
        idx = np.linspace(0, len(pool)-1, n+2).astype(int)[1:-1]
    else:
        idx = np.geomspace(1, len(pool), n+2).astype(int)[1:-1]
        idx = np.clip(idx-1, 0, len(pool)-1)
    return pool[np.unique(idx)][:n]


def _gen_thresh_sets(pool, K):
    """Generate threshold sets of size n=K-1, forcing 600k as first boundary."""
    n = K - 1
    if n < 1: return []
    out, seen = [], set()
    
    # Force 600k into the pool if not there
    active_pool = np.unique(np.concatenate([pool, [600_000]]))
    active_pool = active_pool[active_pool >= 600_000] # Ensure exemption
    
    for m in ['q', 'g']:
        t = _pick_thresholds(active_pool, n, m)
        t = np.sort(np.unique(t))
        # Force 600k to be the start
        if 600_000 not in t:
            t = np.sort(np.concatenate([[600_000], t[:-1]]))
        
        if len(t) == n:
            k = tuple(t)
            if k not in seen:
                seen.add(k); out.append(t)
    return out


# ═══════════════════════════════════════════════════════════════════════
# 5.  INNER RATE OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════

def _make_sch(thresholds, rates):
    n = len(rates)
    sch, lb = [], 0
    for i in range(n):
        ub = float(thresholds[i]) if i < n-1 else np.inf
        sch.append({'lower': lb, 'upper': ub, 'rate': float(rates[i])})
        lb = ub
    return sch


def _inner_opt(thresholds, target, agg, max_rate, starter, max_jump, iters=60, 
               base_list=None, base_rev_reported=None):
    n = len(thresholds) + 1
    rates = np.linspace(0, max_rate*0.6, n)
    rates[0] = 0.0
    if n > 1: rates[1] = min(rates[1], starter)

    best_r, best_gap = rates.copy(), float('inf')

    for it in range(iters):
        sch = _make_sch(thresholds, rates)
        rev = _estimate_revenue(sch, agg, base_list=base_list, base_revenue_reported=base_rev_reported)
        ratio = target / rev if rev > 0 else 2.0
        gap = abs(1 - ratio)
        if gap < best_gap:
            best_gap, best_r = gap, rates.copy()
        if gap < 0.003: break
        lr = 0.5 * (0.96 ** it)
        rates *= (1 + (ratio - 1) * lr)
        rates = np.clip(rates, 0, max_rate)
        rates[0] = 0.0
        if n > 1: rates[1] = min(rates[1], starter)
        for j in range(1, n):
            rates[j] = max(rates[j], rates[j-1])
            rates[j] = min(rates[j], rates[j-1] + max_jump)
        rates = np.clip(rates, 0, max_rate)

    sch = _make_sch(thresholds, best_r)
    final_rev = _estimate_revenue(sch, agg, base_list=base_list, base_revenue_reported=base_rev_reported)
    return sch, final_rev


# ═══════════════════════════════════════════════════════════════════════
# 6.  REPAIR OPERATOR
# ═══════════════════════════════════════════════════════════════════════

def _repair(sch, target, agg, max_rate=0.65, base_list=None, base_rev_reported=None):
    sch = [dict(s) for s in sch]
    rev = _estimate_revenue(sch, agg, base_list=base_list, base_revenue_reported=base_rev_reported)
    for _ in range(80):
        if rev >= target * 0.999: break
        sch[-1]['rate'] = min(sch[-1]['rate'] + 0.01, max_rate)
        if len(sch) > 2:
            sch[-2]['rate'] = min(sch[-2]['rate'] + 0.005, sch[-1]['rate'])
        rev = _estimate_revenue(sch, agg, base_list=base_list, base_revenue_reported=base_rev_reported)
    return sch, rev


# ═══════════════════════════════════════════════════════════════════════
# 7.  UNIFIED SCORER  (same weights for ALL candidates)
# ═══════════════════════════════════════════════════════════════════════

def _score(m, target_rev):
    """
    ACCELERATED PROGRESSIVITY score. Higher = better.
    Goal: High revenue + 'Rate of change of ETR' high for top earners.
    """
    rev = m.get('revenue', 0)
    uplift = (rev - target_rev) / target_rev if target_rev > 0 else 0

    # Hard constraint: negative uplift → big penalty
    if uplift < -0.001:
        return -1_000_000 * abs(uplift)

    s = 0.0
    s += 8000 * uplift                         # Priority 1: High Revenue

    # Objective: Multiplier for 'Progressivity Acceleration'
    # We want dETR(high) >> dETR(low)
    detr = m.get('delta_etr', [])
    if len(detr) > 10:
        mid = len(detr) // 2
        d_low = np.mean(detr[:mid])
        d_high = np.mean(detr[mid:])
        
        if d_high > d_low:
            # Reward high rate of change at the top
            acceleration_factor = (d_high / d_low) if d_low > 0 else 1.0
            s += 2000 * acceleration_factor
        else:
            # Heavy penalty for flat or regressive top-end
            s -= 5000 * (d_low - d_high + 0.1)

    # Maintain convexity and spikes
    s -= 300 * m['convex_viol_count']
    s -= 500 * m['convex_viol_mag']
    
    s -= 100 * m['band_max_jump']               # Moderate penalty for jumps
    s -= 10 * m['band_volatility']              # Low penalty for volatility to allow 'steeper' curves
    
    return s


# ═══════════════════════════════════════════════════════════════════════
# 8.  MAIN — optimize_schedule()
# ═══════════════════════════════════════════════════════════════════════

def optimize_schedule(agg_df, base_schedule_df, base_revenue, y_grid,
                      target_revenue=None, compliance_band=(600_000, 3_000_000)):
    """
    Production entry point.  ALWAYS returns a result dict — never fails.
    target_revenue: if provided, the solver aims for this. If None, uses base_revenue.
    """
    if target_revenue is None:
        target_revenue = base_revenue
    log = []
    np.random.seed(42)

    # ── Base schedule ──
    base_list = _schedule_to_list(base_schedule_df)
    base_m    = compute_metrics(base_list, y_grid, compliance_band=compliance_band)
    log.append("Base: %d slabs, p95 ΔETR=%.4f pp" % (len(base_list), base_m['band_p95_delta']))

    # ── Pool ──
    pool = _build_pool(agg_df)
    log.append("Threshold pool: %d values" % len(pool))

    # ── Stage 0 fallback (always available) ──
    base_rev_sim = _estimate_revenue(base_list, agg_df)
    if base_rev_sim >= target_revenue * 0.999:
        fb, fb_rev = base_list, base_rev_sim
    else:
        fb, fb_rev = _repair(base_list, target_revenue, agg_df)
    fb_m = compute_metrics(fb, y_grid, compliance_band=compliance_band)
    fb_m['revenue'] = fb_rev
    log.append("Stage 0 fallback: rev %.0f (%.2f%% of original)" %
               (fb_rev, fb_rev/base_revenue*100))

    best = {'sch': fb, 'metrics': fb_m, 'score': _score(fb_m, target_revenue), 'stage': 0}

    # ── Stages 1-4 ──
    STAGES = [
        {'id': 1, 'K': range(10, 16), 'mr': 0.40, 'sc': 0.03, 'mj': 0.10,
         'desc': 'Stage 1: 10-15 slabs, max 40%'},
        {'id': 2, 'K': range(14, 21), 'mr': 0.40, 'sc': 0.04, 'mj': 0.12,
         'desc': 'Stage 2: 14-20 slabs, max 40%'},
        {'id': 3, 'K': range(18, 26), 'mr': 0.40, 'sc': 0.05, 'mj': 0.15,
         'desc': 'Stage 3: 18-25 slabs, max 40%'},
        {'id': 4, 'K': range(12, 18), 'mr': 0.42, 'sc': 0.05, 'mj': 0.18,
         'desc': 'Stage 4: max-revenue search, 40% target (relaxed)'},
    ]

    for stg in STAGES:
        log.append("--- %s ---" % stg['desc'])
        n_cand = 0
        best_before = best['score']

        for K in stg['K']:
            for thr in _gen_thresh_sets(pool, K):
                if len(thr) != K - 1: continue
                sch, rev = _inner_opt(thr, target_revenue, agg_df,
                                      stg['mr'], stg['sc'], stg['mj'],
                                      base_list=base_list, base_rev_reported=base_revenue)
                if rev < target_revenue * 0.999:
                    sch, rev = _repair(sch, target_revenue, agg_df,
                                       min(stg['mr'] + 0.10, 0.65),
                                       base_list=base_list, base_rev_reported=base_revenue)
                    if rev < target_revenue * 0.999:
                        continue

                m = compute_metrics(sch, y_grid, compliance_band=compliance_band)
                m['revenue'] = rev
                sc = _score(m, target_revenue)
                n_cand += 1

                if sc > best['score']:
                    best = {'sch': sch, 'metrics': m, 'score': sc, 'stage': stg['id']}

        log.append("  %d candidates; best score %.1f → %.1f" %
                   (n_cand, best_before, best['score']))

        if stg['id'] == 1 and best['metrics']['convex_viol_count'] == 0 and best['stage'] == 1:
            log.append("  → Zero-violation found, stopping early.")
            break

    # ── Package result ──
    final_m = best['metrics']
    log.append("=== SELECTED Stage %d | Rev %.0f | Viol %d | MaxJump %.4f pp ===" %
               (best['stage'], final_m['revenue'],
                final_m['convex_viol_count'], final_m['band_max_jump']))

    extra = _calculate_extra_metrics(best['sch'], agg_df, base_list=base_list, base_rev_reported=base_revenue)
    final_m.update(extra)

    return {
        'schedule_list':  best['sch'],
        'schedule_df':    _list_to_df(best['sch']),
        'metrics':        final_m,
        'base_metrics':   base_m,
        'base_revenue':   base_revenue,
        'stage_selected': best['stage'],
        'log':            log,
        'agg_df':         agg_df,
    }


def _calculate_extra_metrics(sch, agg_df, base_list=None, base_rev_reported=None):
    """
    Computes Kakwani Index and Top 1% Tax Share based on aggregates.
    """
    df = agg_df.copy()
    df['avg_y'] = df['taxable_income_9100'] / df['total_filers'].replace(0, 1)
    df['t_per_filer'] = compute_tax(sch, df['avg_y'].values)
    df['total_tax_raw'] = df['t_per_filer'] * df['total_filers']
    
    # Calibration
    raw_total = df['total_tax_raw'].sum()
    if base_list is not None and base_rev_reported is not None:
        base_sim = 0.0
        for _, row in agg_df.iterrows():
            n = row['total_filers']
            if n > 0:
                ay = row['taxable_income_9100'] / n
                base_sim += compute_tax(base_list, np.array([ay]))[0] * n
        if base_sim > 0:
            cal = base_rev_reported / base_sim
            df['total_tax'] = df['total_tax_raw'] * cal
        else:
            df['total_tax'] = df['total_tax_raw']
    else:
        df['total_tax'] = df['total_tax_raw']

    total_rev = df['total_tax'].sum()
    total_inc = df['taxable_income_9100'].sum()
    total_fil = df['total_filers'].sum()
    
    if total_rev == 0 or total_inc == 0 or total_fil == 0:
        return {'top_1pct_share': 0, 'kakwani': 0}

    df = df.sort_values('avg_y')
    df['cum_filers'] = df['total_filers'].cumsum() / total_fil
    df['cum_income'] = df['taxable_income_9100'].cumsum() / total_inc
    df['cum_tax']    = df['total_tax'].cumsum() / total_rev
    
    f = df['cum_filers'].values
    y = df['cum_income'].values
    gini_inc = 1 - np.sum((y[1:] + y[:-1]) * (f[1:] - f[:-1]))
    
    t = df['cum_tax'].values
    conc_tax = 1 - np.sum((t[1:] + t[:-1]) * (f[1:] - f[:-1]))
    
    kakwani = conc_tax - gini_inc
    
    top_1_filers = total_fil * 0.01
    df_desc = df.sort_values('avg_y', ascending=False)
    df_desc['cum_filers_desc'] = df_desc['total_filers'].cumsum()
    top_slabs = df_desc[df_desc['cum_filers_desc'] <= top_1_filers + df_desc['total_filers'].iloc[0]]
    top_1pct_tax_share = top_slabs['total_tax'].sum() / total_rev
    
    return {
        'top_1pct_share': float(top_1pct_tax_share),
        'kakwani': float(kakwani),
        'total_filers': total_fil,
        'avg_etr': total_rev / total_inc
    }


# ═══════════════════════════════════════════════════════════════════════
# 10. POLICY LAB HELPERS
# ═══════════════════════════════════════════════════════════════════════

def validate_schedule(sch_list):
    """
    Checks for: strictly increasing thresholds, no overlap, rates in [0, 1].
    Returns (is_valid, error_msg).
    """
    if not sch_list:
        return False, "Schedule is empty."
    
    sch = sorted(sch_list, key=lambda x: x['lower'])
    
    for i, s in enumerate(sch):
        if s['rate'] < 0 or s['rate'] > 1.0:
            return False, f"Slab {i+1}: Rate must be between 0% and 100%."
        if s['upper'] <= s['lower']:
            return False, f"Slab {i+1}: Upper bound must be strictly greater than lower bound."
        if i > 0:
            if abs(s['lower'] - sch[i-1]['upper']) > 1.0:
                return False, f"Slab {i+1}: Non-contiguous boundary ({s['lower']} != {sch[i-1]['upper']})."
    
    if not np.isinf(sch[-1]['upper']):
        return False, "Last slab must have an infinite (np.inf) upper bound."
    
    return True, ""


def run_manual_simulation(sch_list, agg_df, y_grid, base_revenue, base_list=None):
    """
    Direct simulation for user-provided schedule.
    """
    m = compute_metrics(sch_list, y_grid)
    rev = _estimate_revenue(sch_list, agg_df, base_list=base_list, base_revenue_reported=base_revenue)
    m['revenue'] = rev
    
    extra = _calculate_extra_metrics(sch_list, agg_df, base_list=base_list, base_rev_reported=base_revenue)
    m.update(extra)
    
    return {
        'schedule_list': sch_list,
        'schedule_df': pd.DataFrame([
            {'lower_bound': s['lower'], 'upper_bound': s['upper'], 'marginal_rate': s['rate']}
            for s in sch_list
        ]),
        'metrics': m,
        'base_revenue': base_revenue,
        'stage_selected': 'Manual',
        'log': ["Policy Lab Manual Simulation"],
        'agg_df': agg_df,
    }


def optimize_schedule_constrained(agg_df, user_sch_list, target_revenue, y_grid,
                                  step_size=500_000, base_list=None):
    """
    Refines user schedule: 
    - Keep number of slabs fixed.
    - Thresholds movable within +/- 1 step.
    """
    def _get_candidates():
        usr_thr = [s['upper'] for s in user_sch_list[:-1]]
        cands = [np.array(usr_thr)]
        for i in range(len(usr_thr)):
            for shift in [-step_size, step_size]:
                new_t = np.array(usr_thr)
                new_t[i] += shift
                if (new_t > 0).all() and (np.diff(new_t) > 0).all():
                    cands.append(new_t)
        return cands

    best_sch = user_sch_list
    best_m = compute_metrics(user_sch_list, y_grid)
    best_rev = _estimate_revenue(user_sch_list, agg_df, base_list=base_list, base_revenue_reported=target_revenue/1.05) # approximation
    best_m['revenue'] = best_rev
    best_score = _score(best_m, target_revenue)
    
    pool = _get_candidates()
    
    for thr in pool:
        hints = [s['rate'] for s in user_sch_list]
        for it in range(40):
            sch_cand = _make_sch(thr, hints)
            rev_cand = _estimate_revenue(sch_cand, agg_df, base_list=base_list, base_revenue_reported=target_revenue/1.05)
            
            ratio = target_revenue / rev_cand if rev_cand > 0 else 1.2
            hints[1:] = [np.clip(r * ratio, 0, 0.45) for r in hints[1:]]
            for j in range(2, len(hints)):
                hints[j] = max(hints[j], hints[j-1])
            
            sch_res = _make_sch(thr, hints)
            m_res = compute_metrics(sch_res, y_grid)
            m_res['revenue'] = _estimate_revenue(sch_res, agg_df, base_list=base_list, base_revenue_reported=target_revenue/1.05)
            sc = _score(m_res, target_revenue)
            
            if sc > best_score:
                best_score = sc
                best_sch = sch_res
                best_m = m_res
            
            if abs(ratio - 1) < 0.001: break

    final_m = best_m
    extra = _calculate_extra_metrics(best_sch, agg_df, base_list=base_list, base_rev_reported=target_revenue/1.05)
    final_m.update(extra)
    
    return {
        'schedule_list':  best_sch,
        'schedule_df':    pd.DataFrame([
            {'lower_bound': s['lower'], 'upper_bound': s['upper'], 'marginal_rate': s['rate']}
            for s in best_sch
        ]),
        'metrics':        final_m,
        'base_revenue':   _estimate_revenue(user_sch_list, agg_df, base_list=base_list, base_revenue_reported=target_revenue/1.05),
        'stage_selected': 'Refined',
        'log':            ["Constrained optimization complete"],
        'agg_df':         agg_df,
    }

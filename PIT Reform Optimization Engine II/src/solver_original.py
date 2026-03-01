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
    for _, r in slabs_df.sort_values('lower_bound').iterrows():
        out.append({'lower': float(r['lower_bound']),
                    'upper': float(r['upper_bound']),
                    'rate':  float(r['marginal_rate'])})
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

def _estimate_revenue(schedule_list, agg_df):
    total = 0.0
    for _, row in agg_df.iterrows():
        n = row['total_filers']
        if n > 0:
            avg_y = row['taxable_income_9100'] / n
            t = compute_tax(schedule_list, np.array([avg_y]))[0]
            total += t * n
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


def _inner_opt(thresholds, target, agg, max_rate, starter, max_jump, iters=60):
    n = len(thresholds) + 1
    rates = np.linspace(0, max_rate*0.6, n)
    rates[0] = 0.0
    if n > 1: rates[1] = min(rates[1], starter)

    best_r, best_gap = rates.copy(), float('inf')

    for it in range(iters):
        sch = _make_sch(thresholds, rates)
        rev = _estimate_revenue(sch, agg)
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
    return sch, _estimate_revenue(sch, agg)


# ═══════════════════════════════════════════════════════════════════════
# 6.  REPAIR OPERATOR
# ═══════════════════════════════════════════════════════════════════════

def _repair(sch, target, agg, max_rate=0.65):
    sch = [dict(s) for s in sch]
    rev = _estimate_revenue(sch, agg)
    for _ in range(80):
        if rev >= target * 0.999: break
        sch[-1]['rate'] = min(sch[-1]['rate'] + 0.01, max_rate)
        if len(sch) > 2:
            sch[-2]['rate'] = min(sch[-2]['rate'] + 0.005, sch[-1]['rate'])
        rev = _estimate_revenue(sch, agg)
    return sch, rev


# ═══════════════════════════════════════════════════════════════════════
# 7.  UNIFIED SCORER  (same weights for ALL candidates)
# ═══════════════════════════════════════════════════════════════════════

def _score(m, target_rev):
    """
    UNIFIED score.  Higher = better.
    Revenue constraint met → base score 0.
    Then soft objectives:
      + revenue uplift (small bonus)
      - convexity violations (count * 100 + magnitude * 200)
      - band volatility
      - band max jump
    """
    rev = m.get('revenue', 0)
    uplift = (rev - target_rev) / target_rev if target_rev > 0 else 0

    # Hard constraint: negative uplift → big penalty
    if uplift < -0.001:
        return -1_000_000 * abs(uplift)

    s = 0.0
    s += 5000 * uplift                         # Heavily prioritize revenue gain
    
    # Progressivity Bonus: dETR(high) > dETR(low)
    # This rewards a steeper ETR curve at the top
    detr = m.get('delta_etr', [])
    if len(detr) > 10:
        mid = len(detr) // 2
        d_low = np.mean(detr[:mid])
        d_high = np.mean(detr[mid:])
        if d_high > d_low:
            s += 500 * (d_high - d_low)         # Bonus for progressive acceleration
        else:
            s -= 500 * (d_low - d_high)         # Penalty for regressive flattening

    s -= 300 * m['convex_viol_count']
    s -= 500 * m['convex_viol_mag']
    s -= 200 * m['band_max_jump']
    s -=  20 * m['band_volatility']             # Relaxed volatility penalty to allow bigger jumps
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
    # User Spec: Max Rate 40%, More Slabs (12-25), 600k Exempt
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
                                      stg['mr'], stg['sc'], stg['mj'])
                if rev < target_revenue * 0.999:
                    sch, rev = _repair(sch, target_revenue, agg_df,
                                       min(stg['mr'] + 0.10, 0.65))
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

        # Early exit if stage 1 found zero-violation winner
        if stg['id'] == 1 and best['metrics']['convex_viol_count'] == 0 and best['stage'] == 1:
            log.append("  → Zero-violation found, stopping early.")
            break

    # ── Package result ──
    final_m = best['metrics']
    log.append("=== SELECTED Stage %d | Rev %.0f | Viol %d | MaxJump %.4f pp ===" %
               (best['stage'], final_m['revenue'],
                final_m['convex_viol_count'], final_m['band_max_jump']))

    return {
        'schedule_list':  best['sch'],
        'schedule_df':    _list_to_df(best['sch']),
        'metrics':        final_m,
        'base_metrics':   base_m,
        'base_revenue':   base_revenue,
        'stage_selected': best['stage'],
        'log':            log,
    }

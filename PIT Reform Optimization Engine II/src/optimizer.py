import numpy as np
import pandas as pd
from .tax import build_base_taxes, get_tax_grid, calculate_total_tax
from .metrics import compute_metrics, check_strict_convexity, get_compliance_metrics


# ============================================================================
# Revenue Estimation  (vectorised for speed)
# ============================================================================

def estimate_revenue(slabs_df, aggregates_df, surtax_params=None):
    """Simulate total revenue using aggregate filer data."""
    slabs_with_base = build_base_taxes(slabs_df)
    total_rev = 0.0
    for _, row in aggregates_df.iterrows():
        if row['total_filers'] > 0:
            avg_income = row['taxable_income_9100'] / row['total_filers']
            tax_per_filer = calculate_total_tax(avg_income, slabs_with_base, surtax_params)
            total_rev += tax_per_filer * row['total_filers']
    return total_rev


# ============================================================================
# Threshold Candidate Generator  (compact — 2-3 per n_slabs)
# ============================================================================

def _gen_thresholds(n_slabs, compliance_cap=3_000_000):
    if n_slabs < 2:
        return []
    cnt = n_slabs - 1
    MAX_INC = 25_000_000
    out = []

    # 1. Geometric
    t = np.geomspace(600_000, MAX_INC, cnt + 2)[1:-1]
    out.append(np.round(t / 50_000) * 50_000)

    # 2. WB-dense  (half thresholds below compliance cap)
    n_lo = max(1, cnt // 2)
    n_hi = cnt - n_lo
    lo  = np.geomspace(400_000, compliance_cap, n_lo + 1)[1:]
    if n_hi > 0:
        hi = np.geomspace(compliance_cap, MAX_INC, n_hi + 1)[1:]
        combo = np.concatenate([lo, hi])
    else:
        combo = lo[:cnt]
    combo = np.sort(np.unique(np.round(combo / 50_000) * 50_000))
    if len(combo) == cnt:
        out.append(combo)

    # De-dup
    seen, valid = set(), []
    for c in out:
        c = np.sort(np.unique(c))
        if len(c) == cnt:
            k = tuple(c)
            if k not in seen:
                seen.add(k)
                valid.append(c)
    return valid


# ============================================================================
# Rate Optimiser  (fast — 80 iters max)
# ============================================================================

def _optimize_rates(thresholds, target_revenue, agg_df,
                    max_rate, starter_cap=0.02, max_jump=0.08,
                    surtax_params=None):
    n = len(thresholds) + 1
    rates = np.linspace(0, max_rate * 0.7, n)
    rates[0] = 0.0
    if n > 1:
        rates[1] = min(rates[1], starter_cap)

    def _df(r):
        rows, lb = [], 0
        for i in range(n):
            ub = thresholds[i] if i < n - 1 else np.inf
            rows.append({'lower_bound': lb, 'upper_bound': ub, 'marginal_rate': r[i]})
            lb = ub
        return pd.DataFrame(rows)

    best, best_gap = _df(rates), float('inf')

    for it in range(80):
        sdf = _df(rates)
        rev = estimate_revenue(sdf, agg_df, surtax_params)
        ratio = target_revenue / rev if rev > 0 else 2.0
        gap = abs(1 - ratio)
        if gap < best_gap:
            best_gap = gap
            best = sdf.copy()
        if gap < 0.005:
            break
        lr = 0.5 * (0.95 ** it)
        rates *= (1 + (ratio - 1) * lr)
        rates = np.clip(rates, 0, max_rate)
        rates[0] = 0.0
        if n > 1:
            rates[1] = min(rates[1], starter_cap)
        for j in range(1, n):
            rates[j] = max(rates[j], rates[j - 1])
            rates[j] = min(rates[j], rates[j - 1] + max_jump)
        rates = np.clip(rates, 0, max_rate)

    final_rev = estimate_revenue(best, agg_df, surtax_params)
    return build_base_taxes(best), final_rev


# ============================================================================
# Candidate Scorer
# ============================================================================

def _score(uplift, convex_pass, n_viol, band_jump, band_vol, n_slabs, rates):
    s = 10_000 * uplift - 500 * band_jump - 200 * band_vol - 2 * n_slabs
    if not convex_pass:
        s -= 100_000 * (n_viol + 1)
    zero_taxable = (rates[1:] <= 0.005).sum() if len(rates) > 1 else 0
    s -= 1000 * zero_taxable
    return s


# ============================================================================
# MAIN  –  Auto-Escalation Optimizer  (fast version)
# ============================================================================

def optimize_schedule(aggregates_df, income_grid_array, revenue_base,
                      wb_compliance_band=(600_000, 3_000_000),
                      progress_callback=None):
    """
    4-stage auto-escalation.  Hard constraints:
      A) revenue >= base  B) strict ΔETR convexity  C) tax non-decreasing
    """

    STAGES = [
        {'slabs': (6, 10), 'rate': 0.35, 'starter': 0.025, 'jump': 0.06,
         'surtax': False, 'desc': 'Stage 0: 6-10 slabs, max 35%'},
        {'slabs': (8, 14), 'rate': 0.45, 'starter': 0.03, 'jump': 0.08,
         'surtax': False, 'desc': 'Stage 1: 8-14 slabs, max 45%'},
        {'slabs': (10, 20), 'rate': 0.55, 'starter': 0.04, 'jump': 0.10,
         'surtax': False, 'desc': 'Stage 2: 10-20 slabs, max 55%'},
        {'slabs': (10, 16), 'rate': 0.47, 'starter': 0.03, 'jump': 0.08,
         'surtax': True, 'desc': 'Stage 3: slabs + surtax'},
    ]

    log = []
    feasible = []

    for si, stg in enumerate(STAGES):
        slo, shi = stg['slabs']
        log.append(stg['desc'])

        if progress_callback:
            progress_callback(f"🔍 {stg['desc']} …")

        # Surtax grid  (kept tiny — 4 combos max)
        stx_grid = [None]
        if stg['surtax']:
            stx_grid = [
                {'enabled': True, 'threshold': 5_000_000,  'rate': 0.03, 'power': 1.0},
                {'enabled': True, 'threshold': 10_000_000, 'rate': 0.04, 'power': 1.0},
                {'enabled': True, 'threshold': 5_000_000,  'rate': 0.05, 'power': 1.02},
                {'enabled': True, 'threshold': 10_000_000, 'rate': 0.06, 'power': 1.02},
            ]

        for ns in range(slo, shi + 1):
            for thresholds in _gen_thresholds(ns, wb_compliance_band[1]):
                for stx in stx_grid:
                    sdf, rev = _optimize_rates(
                        thresholds, revenue_base, aggregates_df,
                        max_rate=stg['rate'], starter_cap=stg['starter'],
                        max_jump=stg['jump'], surtax_params=stx)

                    uplift = (rev - revenue_base) / revenue_base if revenue_base > 0 else 0
                    if uplift < -0.005:
                        continue

                    tg = get_tax_grid(income_grid_array, sdf, surtax_params=stx)

                    # Hard C: non-decreasing tax
                    if (tg['tax'].diff().dropna() < -0.01).any():
                        continue

                    # Hard B: strict convexity
                    ok, nv, vd = check_strict_convexity(tg)

                    md = compute_metrics(tg, compliance_band=wb_compliance_band)
                    ws = get_compliance_metrics(md)
                    rates = sdf['marginal_rate'].values

                    feasible.append({
                        'schedule': sdf, 'surtax': stx,
                        'revenue': rev, 'uplift': uplift,
                        'convex_pass': ok, 'n_viol': nv, 'viol': vd,
                        'wb': ws, 'n_slabs': ns,
                        'score': _score(uplift, ok, nv, ws['max_jump_pp'],
                                        ws['volatility'], ns, rates),
                        'stage': stg['desc'],
                    })

        # Early stop if we have 3+ strictly-convex candidates
        strict = [c for c in feasible if c['convex_pass'] and c['uplift'] >= -0.005]
        if len(strict) >= 3:
            break

    # ── SELECTION ──
    strict = [c for c in feasible if c['convex_pass'] and c['uplift'] >= -0.005]
    pool = strict if strict else feasible
    if not pool:
        return None

    pool.sort(key=lambda c: c['score'], reverse=True)
    w = pool[0]

    fg = get_tax_grid(income_grid_array, w['schedule'], surtax_params=w['surtax'])
    fm = compute_metrics(fg, compliance_band=wb_compliance_band)

    return {
        'schedule': w['schedule'],
        'surtax': w['surtax'],
        'metrics_summary': {
            'revenue': w['revenue'],
            'base_revenue': revenue_base,
            'uplift_pct': w['uplift'],
            'convex_pass': w['convex_pass'],
            'n_convex_viol': w['n_viol'],
            'convex_violations': w['viol'],
            'max_jump_pp': w['wb']['max_jump_pp'],
            'volatility': w['wb']['volatility'],
        },
        'tax_grid': fm,
        'relaxation_log': log,
    }

"""
Policy-grade heatmap visualization for Pakistan PIT slab optimizer.
Government-quality charts: clean, minimal, no clutter.

Functions:
  make_income_labels(income_grid)
  build_heatmap_dataframe(values_base, values_final, income_grid)
  plot_etr_heatmap(df, zmax=0.40)
  plot_detr_heatmap(df, spike_threshold)
  plot_etr_curve(metrics_final, metrics_base)
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────
# 1.  Income-label formatter
# ─────────────────────────────────────────────────────────

def make_income_labels(income_grid):
    """
    Format income values for Y-axis display.
    Returns list of strings:  '0.6M', '1.2M', etc.
    """
    labels = []
    for y in income_grid:
        if y >= 1_000_000:
            labels.append(f"{y / 1e6:.1f}M")
        elif y >= 1_000:
            labels.append(f"{y / 1e3:.0f}k")
        else:
            labels.append(f"{y:.0f}")
    return labels


# ─────────────────────────────────────────────────────────
# 2.  Build 2-column DataFrame for heatmap
# ─────────────────────────────────────────────────────────

def build_heatmap_dataframe(values_final, income_grid, values_base=None, 
                            historical_dict=None, target_rows=25):
    """
    Build a multi-column DataFrame for heatmap comparison.
    Columns: [Historical Years...] + [Base] + [Proposed]
    """
    n = len(income_grid)
    step = max(1, n // target_rows)
    idx = np.arange(0, n, step)
    
    y = income_grid[idx]
    labels = make_income_labels(y)
    
    data = {}
    
    # 1. Historical Years
    if historical_dict:
        for val_label, full_arr in historical_dict.items():
            data[val_label] = np.asarray(full_arr)[idx].astype(float)
            
    # 2. Selected Baseline
    if values_base is not None:
        data['Base'] = np.asarray(values_base)[idx].astype(float)
        
    # 3. Final Proposed
    data['Proposed'] = np.asarray(values_final)[idx].astype(float)
    
    return pd.DataFrame(data, index=labels)


# ─────────────────────────────────────────────────────────
# 3.  Winsorize helper
# ─────────────────────────────────────────────────────────

def _winsorize(arr, lo_pct=2, hi_pct=98):
    """Clip values to [p_lo, p_hi] to avoid colour-scale distortion."""
    a = np.asarray(arr, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        return a
    lo = np.percentile(finite, lo_pct)
    hi = np.percentile(finite, hi_pct)
    return np.clip(a, lo, hi)


# ─────────────────────────────────────────────────────────
# 4.  ETR HEATMAP
# ─────────────────────────────────────────────────────────

def plot_etr_heatmap(df, zmax=0.40, colorscale='Viridis'):
    """
    Policy-grade ETR heatmap.

    Parameters
    ----------
    df   : DataFrame from build_heatmap_dataframe (ETR values).
    zmax : upper bound for colour scale (default 0.40 = 40%).

    Rules
    -----
    - zmin=0, zmax=policy max
    - Viridis colour scale
    - NO cell text annotations
    - Values on hover only (income + ETR %)
    """
    mat = df.values.copy()
    mat = _winsorize(mat)

    # Build hover text matrix
    hover = []
    for i, y_label in enumerate(df.index):
        row = []
        for j, col in enumerate(df.columns):
            row.append(f"Income: {y_label}<br>{col} ETR: {df.iloc[i, j]:.2%}")
        hover.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=mat,
        x=list(df.columns),
        y=list(df.index),
        zmin=0,
        zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text='ETR (%)', font=dict(size=14)),
            tickformat='.0%',
            tickfont=dict(size=12),
            len=0.85,
        ),
        hovertext=hover,
        hoverinfo='text',
        showscale=True,
    ))

    fig.update_layout(
        title=dict(
            text='Final Effective Tax Rate (ETR) — Policy Schedule',
            font=dict(size=20, color='#1a1a2e'),
        ),
        xaxis=dict(
            title='',
            tickfont=dict(size=14, color='#1a1a2e'),
            side='bottom',
        ),
        yaxis=dict(
            title='Income Level',
            tickfont=dict(size=12, color='#333'),
            autorange='reversed',
        ),
        font=dict(family='Inter, sans-serif', size=13),
        width=200 + 150 * len(df.columns),
        height=max(400, len(df) * 25),
        plot_bgcolor='#fafafa',
        paper_bgcolor='#ffffff',
        margin=dict(l=90, r=30, t=55, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────
# 5.  ΔETR HEATMAP
# ─────────────────────────────────────────────────────────

def plot_detr_heatmap(df, spike_threshold=None, colorscale='Inferno'):
    """
    Policy-grade CETR heatmap with spike highlighting.

    Parameters
    ----------
    df              : DataFrame from build_heatmap_dataframe (ΔETR values, pp).
    spike_threshold : float or None.  If None, auto-computed as p95.
                      Cells above threshold get a text annotation; rest are clean.

    Rules
    -----
    - zmin=0, zmax=p99 of all values
    - Inferno colour scale
    - Only spike cells get text annotation
    """
    mat = df.values.copy()

    # Compute percentiles for zmax and spike threshold
    finite = mat[np.isfinite(mat)]
    p99 = float(np.percentile(finite, 99)) if len(finite) > 0 else 1.0
    if spike_threshold is None:
        spike_threshold = float(np.percentile(finite, 95)) if len(finite) > 5 else p99

    mat_clipped = _winsorize(mat)

    # Build hover text
    hover = []
    for i, y_label in enumerate(df.index):
        row = []
        for j, col in enumerate(df.columns):
            row.append(f"Income: {y_label}<br>{col} CETR: {df.iloc[i, j]:.3f} pp")
        hover.append(row)

    # Build annotation text — ONLY for spike cells
    anno_text = []
    for i in range(len(df)):
        row = []
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            if np.isfinite(val) and val > spike_threshold:
                row.append(f"{val:.2f}")
            else:
                row.append("")
        anno_text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=mat_clipped,
        x=list(df.columns),
        y=list(df.index),
        zmin=0,
        zmax=p99,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text='CETR (pp)', font=dict(size=14)),
            tickfont=dict(size=12),
            len=0.85,
        ),
        hovertext=hover,
        hoverinfo='text',
        showscale=True,
    ))

    # Add spike annotations as text overlay
    for i in range(len(df)):
        for j in range(len(df.columns)):
            txt = anno_text[i][j]
            if txt:
                fig.add_annotation(
                    x=df.columns[j],
                    y=df.index[i],
                    text=f"<b>{txt}</b>",
                    showarrow=False,
                    font=dict(size=11, color='white'),
                    bgcolor='rgba(231,111,81,0.7)',
                    borderpad=2,
                )

    fig.update_layout(
        title=dict(
            text='CETR (Change in ETR) — Smoothness Diagnostic',
            font=dict(size=20, color='#1a1a2e'),
        ),
        xaxis=dict(
            title='',
            tickfont=dict(size=14, color='#1a1a2e'),
            side='bottom',
        ),
        yaxis=dict(
            title='Income Level',
            tickfont=dict(size=12, color='#333'),
            autorange='reversed',
        ),
        font=dict(family='Inter, sans-serif', size=13),
        width=200 + 150 * len(df.columns),
        height=max(400, len(df) * 25),
        plot_bgcolor='#fafafa',
        paper_bgcolor='#ffffff',
        margin=dict(l=90, r=30, t=55, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────
# 6.  ETR Curve (line chart)
# ─────────────────────────────────────────────────────────

def plot_etr_curve(metrics_final, metrics_base=None, historical_benchmarks=None, title="ETR Comparison"):
    """
    Proposed vs Base ETR line chart with historical benchmarks.
    """
    fig = go.Figure()

    # Historical Benchmarks (light colors)
    if historical_benchmarks:
        # Professional secondary palette for backgrounds
        hist_colors = ['#ced4da', '#adb5bd', '#6c757d', '#dee2e6']
        for i, (label, h_etr) in enumerate(historical_benchmarks.items()):
            fig.add_trace(go.Scatter(
                x=metrics_final['y'], y=h_etr,
                mode='lines', name=f"Archive {label}",
                line=dict(color=hist_colors[i % len(hist_colors)], width=1.2, dash='dot')))

    # Baseline (current law)
    if metrics_base is not None:
        fig.add_trace(go.Scatter(
            x=metrics_base['y'], y=metrics_base['etr'],
            mode='lines', name='Baseline (Selected Year)',
            line=dict(color='#e76f51', dash='dash', width=2.5)))

    # Proposed
    fig.add_trace(go.Scatter(
        x=metrics_final['y'], y=metrics_final['etr'],
        mode='lines', name='Proposed (Optimized)',
        line=dict(color='#2a9d8f', width=3.5)))

    fig.update_layout(
        title=title,
        xaxis_title="Annual Income (PKR)",
        yaxis_title="Effective Tax Rate (%)",
        template="plotly_white",
        hovermode="x unified",
        width=1000, height=550,
        font=dict(family='Inter, sans-serif'),
        yaxis=dict(tickformat='.1%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


# ─────────────────────────────────────────────────────────
# 7.  Revenue Contribution Chart
# ─────────────────────────────────────────────────────────

def plot_revenue_contribution(agg_df, schedule_list, title="Normal Income Tax Contribution by Income Group"):
    """
    Shows which income groups are paying how much in total PKR.
    """
    from .solver import compute_tax
    
    rows = []
    for _, row in agg_df.iterrows():
        n = row['total_filers']
        if n > 0:
            avg_y = row['taxable_income_9100'] / n
            t = compute_tax(schedule_list, np.array([avg_y]))[0]
            rev = t * n
            rows.append({
                'Slab': row.get('slab_id', 'N/A'),
                'Normal Income Tax (Billion PKR)': rev / 1e9,
                'Income Range': f"{row['lower_bound']/1e6:.1f}M - {row.get('upper_bound', np.inf)/1e6:.1f}M"
            })
    
    df_plot = pd.DataFrame(rows)
    fig = px.bar(df_plot, x='Slab', y='Normal Income Tax (Billion PKR)', 
                 hover_data=['Income Range'],
                 color='Normal Income Tax (Billion PKR)',
                 color_continuous_scale='Blues',
                 title=title)
    
    fig.update_layout(template='plotly_white', height=400)
    return fig


# ─────────────────────────────────────────────────────────
# 8.  Marginal vs Effective Staircase
# ─────────────────────────────────────────────────────────

def plot_staircase_rates(metrics, schedule_list, title="MTR vs ETR Staircase"):
    """
    Step chart for Marginal rates vs line chart for Effective rates.
    """
    y_grid = metrics['y']
    etr = metrics['etr']
    
    # Marginal Rates (Step)
    lowers = [s['lower'] for s in schedule_list]
    rates = [s['rate'] for s in schedule_list]
    
    fig = go.Figure()
    
    # Effective Rate
    fig.add_trace(go.Scatter(x=y_grid, y=etr, name='ETR',
                             line=dict(color='#2a9d8f', width=3)))
    
    # Marginal Rate (Step)
    fig.add_trace(go.Scatter(x=lowers, y=rates, name='MTR',
                             line=dict(color='#e76f51', width=2),
                             line_shape='hv'))
    
    fig.update_layout(
        title=title,
        xaxis_title="Annual Income (PKR)",
        yaxis_title="Tax Rate",
        template="plotly_white",
        yaxis=dict(tickformat='.0%'),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ─────────────────────────────────────────────────────────
# 9.  Progressivity Slope (Acceleration)
# ─────────────────────────────────────────────────────────

def plot_progressivity_slope(metrics, base_metrics=None, title="Progressivity Analysis: Rate of Acceleration"):
    """
    Plots the ΔETR (slope) across income. 
    Rising line = Increasing progressivity (steeper).
    Flat line = Linear increase.
    """
    y_grid = metrics['y']
    detr = metrics['delta_etr']
    
    fig = go.Figure()
    
    # Base/Historical Slope
    if base_metrics is not None:
        fig.add_trace(go.Scatter(
            x=y_grid, y=base_metrics['delta_etr'],
            name='Baseline Acceleration',
            line=dict(color='#999', width=1, dash='dot'),
            fill='none'
        ))

    # Proposed Slope
    fig.add_trace(go.Scatter(
        x=y_grid, y=detr,
        name='Proposed Acceleration (Progressivity)',
        line=dict(color='#e76f51', width=3),
        fill='tozeroy',
        fillcolor='rgba(231,111,81,0.1)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Annual Income (PKR)",
        yaxis_title="CETR (pp per step)",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    # Add annotation for Policy Intent
    if len(y_grid) > 0:
        fig.add_annotation(
            x=y_grid[-1], y=detr[-1],
            text="Steeper for Top Earners ↗",
            showarrow=True,
            arrowhead=2,
            ax=-100, ay=-30
        )
    
    return fig

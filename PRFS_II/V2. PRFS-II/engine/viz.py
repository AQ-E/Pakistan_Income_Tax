import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

def plot_forecast_comparison(head, baseline, scenario, start_year):
    years = list(range(start_year + 1, start_year + 1 + len(baseline)))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=baseline, name="Baseline", line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=years, y=scenario, name="Scenario", line=dict(color='red')))
    
    fig.update_layout(
        title=f"Forecast: {head}",
        xaxis_title="Year",
        yaxis_title="Revenue (PKR Billion)",
        template="plotly_white"
    )
    return fig

def create_impact_table(results, horizon):
    rows = []
    for head, res in results.items():
        # Get final year impact
        base_val = res['baseline'].iloc[-1]
        scen_val = res['scenario'].iloc[-1]
        
        rows.append({
            'Tax Head': head.upper(),
            'Baseline (bn)': f"{base_val:,.1f}",
            'Scenario (bn)': f"{scen_val:,.1f}",
            'Abs Change (bn)': f"{scen_val - base_val:,.1f}",
            '% Change': f"{(scen_val/base_val - 1)*100:.1f}%" if base_val != 0 else "0.0%"
        })
    return pd.DataFrame(rows)

def plot_contribution_waterfall(head, base_fc, scen_fc):
    total_change = scen_fc.iloc[-1] - base_fc.iloc[-1]
    
    fig = go.Figure(go.Waterfall(
        name = "Contribution", orientation = "v",
        measure = ["relative", "total"],
        x = ["Policy Impact", "Total Scenario"],
        y = [total_change, scen_fc.iloc[-1]],
        base = base_fc.iloc[-1],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    fig.update_layout(title=f"Revenue Contribution: {head}")
    return fig

def plot_sensitivity_heatmap(engine, horizon, head_name):
    grid = []
    gdp_range = [0.01, 0.03, 0.05, 0.07]
    inf_range = [5, 10, 15, 20]
    
    for g in gdp_range:
        row = []
        for i in inf_range:
            res, _, _ = engine.run_scenario(horizon, {'gdp_growth': g, 'inflation': i})
            pct_change = (res[head_name]['scenario'].iloc[-1] / res[head_name]['baseline'].iloc[-1] - 1) * 100
            row.append(pct_change)
        grid.append(row)
        
    fig = px.imshow(grid, x=inf_range, y=gdp_range, 
                    labels=dict(x="Inflation (%)", y="GDP Growth (%)"),
                    title=f"Sensitivity Heatmap: {head_name} (% Change vs Baseline)")
    return fig

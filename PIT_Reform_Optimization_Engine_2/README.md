# 🇵🇰 Pakistan PIT Slab Policy Dashboard

Progressive income tax microsimulation and aggregate analysis dashboard.

## 🚀 How to Run

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 🛠 Features
- **Aggregate Slab Analysis**: Explore filers, taxable income, and tax collection by slab.
- **Custom Slab Editor**: Define and test custom tax slabs and rates.
- **ETR Heatmap**: Analyze Effective Tax Rates across an income grid.
- **Progressivity Check**: Detect violations in ETR and ΔETR (convexity).
- **Optimizer**: Automatically adjust rates to meet revenue targets while maintaining progressivity.

- `app.py`: Main Streamlit application.
- `src/`:
    - `io.py`: Data loading and cleaning.
    - `solver.py`: Core tax engine, metrics, and policy optimizer.
    - `viz.py`: Plotly visualization modules.
    - `__init__.py`: Package marker.

## 📊 Data Sources
- `Slab wise Taxable Income Filers & Normal Tax_3012026.xlsx`: Aggregate tax data (2023-2025).
- `tax liability at various income levels.xlsx`: Historical ETR baseline data.

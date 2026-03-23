import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ardl import ARDL, ardl_select_order
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
import pmdarima as pm

class BaseTaxModel:
    def __init__(self, name):
        self.name = name
        self.model_res = None
        self.y_col = None
        self.x_cols = []
        self.diagnostics = {}
        self.performance = {}
        self.elasticities = {}

    def fit(self, df, y, x):
        pass

    def forecast(self, exog_df, steps=1):
        pass

class ARDLModel(BaseTaxModel):
    def __init__(self, max_p=1, max_q=1): # Reduced default lags for annual data stability
        super().__init__("ARDL")
        self.max_p = max_p
        self.max_q = max_q

    def fit(self, df, y, x):
        self.y_col = y
        self.x_cols = x
        data = df[[y]+x].dropna()
        n_obs = len(data)
        # Hard constraint: lags + variables < n_obs / 2
        p = min(self.max_p, max(1, n_obs // 8))
        q = min(self.max_q, max(1, n_obs // 8))
        
        try:
            self.sel_order = ardl_select_order(data[y], p, data[x], q, ic="aic", trend="c")
            self.model_res = self.sel_order.model.fit()
            self.rmse = np.std(self.model_res.resid)
            self._extract_elasticities()
            return self.model_res
        except Exception as e:
            # Fallback to simple ARDL(1,0)
            try:
                self.model_res = ARDL(data[y], 1, data[x], 0, trend="c").fit()
                self.rmse = np.std(self.model_res.resid)
                self._extract_elasticities()
                return self.model_res
            except Exception as e2:
                raise ValueError(f"ARDL Final Fit Error: {e2}")

    def _extract_elasticities(self):
        """
        Extract SR (coefficients) and LR (b / (1 - sum(rho))) elasticities.
        """
        params = self.model_res.params
        # LR Elasticities
        # For ARDL(p, q1, q2...): y_t = c + sum(rho_i y_{t-i}) + sum(gamma_j x_{t-j})
        # LR = sum(gamma) / (1 - sum(rho))
        ar_sum = sum([params[k] for k in params.index if k.startswith(f'{self.y_col}.L')])
        denom = 1 - ar_sum
        
        lr_elasts = {}
        for col in self.x_cols:
            gamma_sum = sum([params[k] for k in params.index if col in k])
            lr_elasts[col] = gamma_sum / denom if abs(denom) > 1e-5 else np.nan
            
        self.elasticities = {
            'Short-Run': {c: params.get(c, params.get(f'{c}.L0', np.nan)) for c in self.x_cols},
            'Long-Run': lr_elasts
        }

    def forecast(self, exog_df, steps=1):
        # Validation
        for col in self.x_cols:
            if col not in exog_df.columns:
                raise ValueError(f"Model {self.name} requires '{col}' for forecasting, but it is missing.")
            if exog_df[col].isna().any():
                raise ValueError(f"Model {self.name} received NaN for exogenous variable '{col}'.")
                
        # Enforce column selection and order to avoid 'exog_oos' errors
        return self.model_res.forecast(steps=steps, exog=exog_df[self.x_cols])

class ARIMAXModel(BaseTaxModel):
    def __init__(self):
        super().__init__("ARIMAX")

    def fit(self, df, y, x):
        self.y_col = y
        self.x_cols = x
        auto_mod = pm.auto_arima(df[y], X=df[x], seasonal=False, stepwise=True, ic='aic', error_action='ignore', suppress_warnings=True)
        self.model_res = auto_mod
        self.rmse = np.std(auto_mod.resid())
        return self.model_res

    def forecast(self, exog_df, steps=1):
        for col in self.x_cols:
            if col not in exog_df.columns: raise ValueError(f"ARIMAX missing {col}")
            if exog_df[col].isna().any(): raise ValueError(f"ARIMAX {col} is NaN")
            
        return pd.Series(self.model_res.predict(n_periods=steps, X=exog_df[self.x_cols]))

class VECMModel(BaseTaxModel):
    """
    Small VECM/VAR for 3-4 variables.
    """
    def __init__(self, k_ar_diff=1):
        super().__init__("VECM")
        self.k_ar_diff = k_ar_diff

    def fit(self, df, y, x):
        self.y_col = y
        self.x_cols = x
        data = df[[y] + x].dropna()
        
        # Cointegration Rank (Johansen)
        rank_res = select_coint_rank(data, det_order=0, k_ar_diff=self.k_ar_diff)
        self.rank = rank_res.rank
        
        self.model = VECM(data, k_ar_diff=self.k_ar_diff, coint_rank=self.rank, deterministic="co")
        self.model_res = self.model.fit()
        return self.model_res

    def forecast(self, exog_df, steps=1):
        # VECM forecasts all variables together
        fc = self.model_res.predict(steps=steps)
        # Return first column (y)
        return pd.Series(fc[:, 0], index=exog_df.index[:steps])

class ARIMABaseline(BaseTaxModel):
    def __init__(self):
        super().__init__("ARIMA-Base")

    def fit(self, df, y, x=None):
        self.y_col = y
        self.model_res = ARIMA(df[y], order=(1,1,1)).fit()
        return self.model_res

    def forecast(self, exog_df, steps=1):
        return self.model_res.forecast(steps=steps)

class DynamicLagModel(BaseTaxModel):
    """
    Standard dynamic regression: y ~ L(y) + x
    """
    def __init__(self, lags=1):
        super().__init__("DynamicLag")
        self.lags = lags

    def fit(self, df, y, x):
        self.y_col = y
        self.x_cols = x
        data = df[[y] + x].copy()
        
        rhs_cols = list(x)
        for i in range(1, self.lags + 1):
            data[f'lag{i}_{y}'] = data[y].shift(i)
            rhs_cols.append(f'lag{i}_{y}')
        
        data = data.dropna()
        X = sm.add_constant(data[rhs_cols])
        Y = data[y]
        self.model_res = sm.OLS(Y, X).fit()
        self.rmse = np.std(self.model_res.resid)
        self.rhs_cols = rhs_cols
        self.params = self.model_res.params
        self._extract_elasticities()
        return self.model_res

    def _extract_elasticities(self):
        # Roughly: beta / (1 - sum(rho))
        denom = 1 - sum([self.params.get(f'lag{i}_{self.y_col}', 0) for i in range(1, self.lags + 1)])
        self.elasticities = {
            'Short-Run': {c: self.params.get(c, 0) for c in self.x_cols},
            'Long-Run': {c: self.params.get(c, 0) / denom if abs(denom) > 1e-5 else 0 for c in self.x_cols}
        }

    def forecast(self, exog_df, steps=1):
        # Validation
        for col in self.x_cols:
            if col not in exog_df.columns: raise ValueError(f"DynamicLag missing {col}")
            if exog_df[col].isna().any(): raise ValueError(f"DynamicLag {col} is NaN")

        X = sm.add_constant(exog_df[self.x_cols], has_constant='add')
        # We need to add the lags manually if they aren't in exog_df
        for i in range(1, self.lags + 1):
            lag_col = f'lag{i}_{self.y_col}'
            if lag_col not in X.columns:
                X[lag_col] = exog_df[self.y_col].values if self.y_col in exog_df.columns else 0
        
        # Ensure column order matches training
        X = X[['const'] + self.rhs_cols]
        return self.model_res.predict(X)

class OLSModel(BaseTaxModel):
    def __init__(self):
        super().__init__("OLS")

    def fit(self, df, y, x):
        self.y_col = y
        self.x_cols = x
        data = df[[y] + x].dropna()
        self.model_res = sm.OLS(data[y], sm.add_constant(data[x])).fit()
        self.rmse = np.std(self.model_res.resid)
        self.params = self.model_res.params
        self.elasticities = {'Short-Run': self.params.to_dict(), 'Long-Run': self.params.to_dict()}
        return self.model_res

    def forecast(self, exog_df, steps=1):
        X = sm.add_constant(exog_df[self.x_cols], has_constant='add')
        # Ensure exog_df has all columns in same order as model
        X = X[self.model_res.model.exog_names]
        return self.model_res.predict(X)

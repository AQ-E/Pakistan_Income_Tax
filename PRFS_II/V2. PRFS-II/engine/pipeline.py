import pandas as pd
import numpy as np
import statsmodels.api as sm
from engine.models import ARDLModel, ARIMAXModel, VECMModel, ARIMABaseline, DynamicLagModel, OLSModel
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan, recursive_olsresiduals
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import jarque_bera
import math

class ForecastingPipeline:
    def __init__(self, df):
        self.df = df
        self.best_models = {}
        self.leaderboard = []
        self.channel_models = {}
        self.channel_predictions = pd.DataFrame(index=df.index)
        self.channel_predictions['log_gdp'] = df['log_gdp'] if 'log_gdp' in df.columns else np.nan
        self.df_stage2 = df.copy() # Will hold predicted channels
        self.dummies = self.identify_dummies()

    def identify_dummies(self):
        """Automatically identify and categorize binary columns."""
        res = {'all': [], 'policy': [], 'shock': []}
        policy_keywords = ['regime', 'policy', 'reform', 'tax_law', 'measure']
        
        for col in self.df.columns:
            unique_vals = self.df[col].dropna().unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                res['all'].append(col)
                if any(k in col.lower() for k in policy_keywords):
                    res['policy'].append(col)
                else:
                    res['shock'].append(col)
        return res

    def run_stage1_channels(self):
        """
        Stage 1 - Channel/Base Equations
        Imports, LSM, Consumption, Dutiable Imports (Proxy if needed)
        """
        # Robustness: Proxy consumption if NaN found (often missing in FBR/PBS tables)
        if 'log_consumption' not in self.df.columns or self.df['log_consumption'].isna().sum() > 0:
             # Proxy: Consumption ~ 0.8 * GDP (Natural fallback for buoyancy comparison)
             self.df['log_consumption'] = self.df['log_gdp'] + np.log(0.8)
        # A) Imports base
        self._fit_channel('log_imports', ['log_gdp', 'log_exrate', 'policy rate', 'inflation'])
        
        # B) Dutiable Imports
        if 'log_dutiable_imports' in self.df.columns:
            self._fit_channel('log_dutiable_imports', ['log_imports', 'log_exrate', 'policy rate', 'inflation'])
        else:
            # Proxy
            self.channel_predictions['log_dutiable_imports'] = self.channel_predictions['log_imports']
            
        # C) LSM base
        self._fit_channel('log_lsm', ['log_gdp', 'policy rate', 'inflation'])
        
        # D) Consumption base
        self._fit_channel('log_consumption', ['log_gdp', 'inflation', 'policy rate'])

    def _fit_channel(self, y, x_pool):
        # 1. Enforce VIF < 10 and Parsimony
        x_final = [v for v in x_pool if v in self.df.columns]
        while len(x_final) > 1 and self.check_vif(self.df, x_final) >= 10:
            x_final.pop() 
            
        # 2. Fit and Generate predicted series
        try:
            model = ARDLModel()
            model.fit(self.df, y, x_final)
            self.channel_models[y] = model
            # Use in-sample prediction to cover the full historical series (fills gaps)
            if hasattr(model.model_res, 'predict'):
                # For ARDL, predict() without arguments usually returns in-sample
                pred = model.model_res.predict()
            else:
                pred = model.model_res.fittedvalues
            
            # Align and ensure no NaNs in the transmission series
            s_pred = pd.Series(pred, index=self.df.index).ffill().bfill()
            self.channel_predictions[y] = s_pred
        except:
            model = OLSModel()
            model.fit(self.df, y, x_final)
            self.channel_models[y] = model
            pred = model.model_res.predict(sm.add_constant(self.df[x_final], has_constant='add'))
            s_pred = pd.Series(pred, index=self.df.index).ffill().bfill()
            self.channel_predictions[y] = s_pred

    def check_vif(self, df, x_cols):
        # Filtering to existing columns only to prevent KeyError
        valid_cols = [c for c in x_cols if c in df.columns]
        if not valid_cols: return 0
        
        dataset = df[valid_cols].dropna()
        if len(dataset) < 5: return 0
        X = sm.add_constant(dataset)
        vifs = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
        return max(vifs) if vifs else 0

    def run_full_pipeline(self):
        if 'gst' in self.df.columns and 'log_gst' not in self.df.columns:
            self.df['log_gst'] = np.log(self.df['gst'].replace(0, np.nan))

        self.run_stage1_channels()
        
        # Stage 2 - Tax Head Equations
        combined_df = self.df.copy()
        for col in self.channel_predictions.columns:
            combined_df[f'{col}_hat'] = self.channel_predictions[col]
            
        # Ensure log_gdp_hat exists even if not model-fitted
        if 'log_gdp_hat' not in combined_df.columns and 'log_gdp' in combined_df.columns:
            combined_df['log_gdp_hat'] = combined_df['log_gdp']
            
        self.df_stage2 = combined_df # Store for UI access
            
        tax_specs = {
            'dt': {'y': 'log_dt', 'bases': ['log_gdp_hat', 'log_lsm_hat'], 'others': ['inflation']},
            'gst': {'y': 'log_gst', 'bases': ['log_consumption_hat', 'log_imports_hat'], 'others': ['inflation', 'log_exrate']},
            'fed': {'y': 'log_fed', 'bases': ['log_lsm_hat'], 'others': ['inflation']},
            'customs': {'y': 'log_customs', 'bases': ['log_dutiable_imports_hat', 'log_imports_hat'], 'others': ['log_exrate', 'inflation']}
        }
        
        for head, spec in tax_specs.items():
            y_col = spec['y']
            
            # Temporary storage for this head's candidates
            matches = []
            
            for base in spec['bases']:
                if base not in combined_df.columns: continue
                
                x_cols = [base] + [o for o in spec['others'] if o in combined_df.columns]
                
                # Dynamic Dummy Selection (Top 3 by correlation with y)
                if self.dummies['all']:
                    y_ser = combined_df[y_col].dropna()
                    dummy_corrs = []
                    for d in self.dummies['all']:
                        d_ser = combined_df.loc[y_ser.index, d]
                        if d_ser.std() > 0:
                            corr = abs(y_ser.corr(d_ser))
                            dummy_corrs.append((d, corr))
                    
                    top_dummies = [d for d, c in sorted(dummy_corrs, key=lambda x: x[1], reverse=True)[:3]]
                    x_cols += top_dummies

                while len(x_cols) > 1 and self.check_vif(combined_df, x_cols) >= 10:
                    x_cols.pop()
                
                # Tournament: Structural vs Baseline
                candidates = [
                    (ARDLModel(), "Policy"),
                    (ARIMAXModel(), "Policy"),
                    (DynamicLagModel(), "Policy"),
                    (ARIMABaseline(), "Forecast-Only")
                ]
                
                for m, m_type in candidates:
                    try:
                        m.fit(combined_df, y_col, x_cols if m_type == "Policy" else None)
                        diag = self.get_diagnostics(m)
                        perf = self.backtest_model(m, combined_df, y_col, x_cols if m_type == "Policy" else [])
                        
                        vif = self.check_vif(combined_df, x_cols) if m_type == "Policy" else 0
                        policy_score = self.calculate_policy_score(m, diag, vif, head) if m_type == "Policy" else 0

                        match_info = {
                            'Tax Head': head.upper(),
                            'Base': base if m_type == "Policy" else "N/A",
                            'Model': m.name,
                            'Type': m_type,
                            'sMAPE%': round(perf['smape'], 2),
                            'WAPE%': round(perf['wape'], 2),
                            'RMSLE': round(perf['rmsle'], 4),
                            'PolicyScore': round(policy_score, 2),
                            'VIF': round(vif, 2),
                            'Autocorr-p': round(diag['autocorr_p'], 3),
                            'n_test': perf.get('n_test', 8),
                            'obj': m
                        }
                        matches.append(match_info)
                        self.leaderboard.append(match_info)
                    except Exception as e:
                        # st.error(f"Fit failed for {m.name}: {e}")
                        continue
            
            # Selection Logic (Bifurcated: Accuracy vs Policy)
            if matches:
                m_df = pd.DataFrame(matches)
                # 1. Baseline Winner (Accuracy-focused: minimized sMAPE)
                f_winner = m_df.sort_values('sMAPE%').iloc[0]
                
                # 2. Policy Winner (Scenario-focused: prioritized PolicyScore, then sMAPE)
                p_candidates = m_df[m_df['Type'] == "Policy"]
                if not p_candidates.empty:
                    # Sort by PolicyScore (desc) then sMAPE (asc)
                    p_winner = p_candidates.sort_values(by=['PolicyScore', 'sMAPE%'], ascending=[False, True]).iloc[0]
                    
                    self.best_models[head] = {
                        'forecast_winner': f_winner['obj'],
                        'policy_winner': p_winner['obj']
                    }
                else:
                    self.best_models[head] = {
                        'forecast_winner': f_winner['obj'],
                        'policy_winner': None
                    }

    def calculate_policy_score(self, model, diag, vif, head):
        """
        Policy Validity Score (0-100)
        Weighs diagnostics, VIF, Theory-Signs, and Elasticity Plausibility.
        """
        score = 100
        
        # 1. Diagnostic Penalties
        if diag['autocorr_p'] < 0.05: score -= 30 # Serial Correlation error
        if vif > 10: score -= 20 # Multicollinearity
        
        # 2. Theory Check: Most tax bases should have positive coefficients
        # (LSM, GDP, Imports etc should naturally grow revenue)
        if hasattr(model, 'elasticities') and 'Short-Run' in model.elasticities:
            sr = model.elasticities['Short-Run']
            # Find the primary base (usually the first x_col)
            base_col = model.x_cols[0]
            if base_col in sr and sr[base_col] < 0:
                score -= 40 # THEORY BREACH: Negative elasticity with base
                
        # 3. Elasticity Plausibility (Ideally between 0.5 and 2.5)
        if hasattr(model, 'elasticities') and 'Long-Run' in model.elasticities:
            lr = model.elasticities['Long-Run']
            base_col = model.x_cols[0]
            if base_col in lr:
                val = lr[base_col]
                if val > 3.0 or val < 0.2:
                    score -= 15 # EXPLODING or WEAK elasticity
        
        # 4. Forecast Stability (Unit Root check)
        ar_sum = 0
        if hasattr(model, 'params'):
            p = model.params
            ar_sum = sum([p[k] for k in p.index if head.lower() in k.lower() and ('lag' in k or 'L' in k)])
        
        if ar_sum >= 1.0:
            score -= 30 # EXPLOSIVE: Model will trend to infinity
        elif ar_sum > 0.95:
             score -= 10 # NEAR UNIT ROOT: Unstable for long-run forecasting

        # 5. Overfitting Penalty (# of predictors)
        score -= (len(model.x_cols) * 2)
        
        return max(0, score)

    def get_diagnostics(self, model):
        # ... logic ...
        res = model.model_res
        resid = res.resid if hasattr(res, 'resid') else res.resid()
        try:
            if hasattr(res, 'test_serial_correlation'): # ARDL
                bg_p = res.test_serial_correlation(lags=1)[0, 1]
            else:
                from statsmodels.stats.diagnostic import acorr_ljungbox
                bg_p = acorr_ljungbox(resid, lags=[1])['lb_pvalue'].iloc[0]
        except: bg_p = 0.5
        return {'autocorr_p': bg_p}

    def backtest_model(self, model, df, y_col, x_cols, n_test=8):
        """Standardized Backtest using sMAPE, WAPE and RMSLE on level scale."""
        actuals = []
        preds = []
        
        n = len(df)
        for i in range(n - n_test, n):
            train = df.iloc[:i]
            test = df.iloc[i:i+1]
            try:
                # Skip if no actual value to compare against
                actual_log_val = test[y_col].iloc[0]
                if pd.isna(actual_log_val): continue
                
                model.fit(train, y_col, x_cols)
                fc_val = model.forecast(test, steps=1).iloc[0]
                
                preds.append(np.exp(fc_val))
                actuals.append(np.exp(actual_log_val))
            except: continue
        
        preds = np.array(preds)
        actuals = np.array(actuals)
        
        if len(actuals) == 0: return {'smape': 999, 'wape': 999, 'rmsle': 999}
        
        # Metrics
        # Avoid division by zero in sMAPE
        denom = (np.abs(actuals) + np.abs(preds))
        smape = 100/len(actuals) * np.sum(2 * np.abs(preds - actuals) / np.where(denom == 0, 1, denom))
        wape = 100 * np.sum(np.abs(actuals - preds)) / max(np.sum(actuals), 1e-9)
        
        # RMSLE is RMSE on the log scale
        log_preds = np.log(np.maximum(preds, 1e-9))
        log_actuals = np.log(np.maximum(actuals, 1e-9))
        rmsle = np.sqrt(np.mean((log_actuals - log_preds)**2))
        
        return {'smape': smape, 'wape': wape, 'rmsle': rmsle, 'n_test': len(actuals)}

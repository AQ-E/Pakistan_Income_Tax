import pandas as pd
import numpy as np
import math

class ScenarioEngine:
    def __init__(self, df, best_models, channel_models):
        self.df = df
        self.best_models = best_models
        self.channel_models = channel_models # Taken from pipeline

    def run_scenario(self, horizon, targets):
        """
        Runs recursive forecasting for all macro channels and tax heads.
        Returns results standardized in PKR Billion.
        """
        last_row = self.df.iloc[-1].copy()
        
        # Prepare last row: ensure logs AND macro rates are computed if levels exist
        levels = ['dt', 'gst', 'fed', 'customs', 'gdp', 'imports', 'dutiable_imports', 'lsm', 'consumption', 'exrate']
        for col in levels:
            if col in last_row and pd.isna(last_row.get(f'log_{col}')):
                val = last_row[col]
                # If level is missing, check if previous row had it (very rare but possible in bad data)
                if pd.isna(val) and len(self.df) > 1:
                    val = self.df[col].iloc[-2]
                
                if not pd.isna(val) and val > 0:
                    last_row[f'log_{col}'] = math.log(val)
        
        # Ensure hats exist for exogenous variables
        if 'log_gdp' in last_row: last_row['log_gdp_hat'] = last_row['log_gdp']
        if 'log_exrate' in last_row: last_row['log_exrate_hat'] = last_row['log_exrate']
        
        # Ensure rates are not NaN
        for col in ['inflation', 'policy rate']:
            if col in last_row and pd.isna(last_row[col]):
                if len(self.df) > 1:
                    last_row[col] = self.df[col].iloc[-2]
                else:
                    last_row[col] = 10.0 # Emergency fallback

        base_path = []
        scen_path = []
        
        cur_base = last_row.copy()
        cur_scen = last_row.copy()
        
        results = {head: {
            'baseline': [], 'scenario': [],
            'l80': [], 'u80': [], 'l95': [], 'u95': []
        } for head in self.best_models.keys()}

        for h in range(1, horizon + 1):
            # Step 1: Propagate Macro Channels
            cur_base = self.step_macro(cur_base, targets, is_scenario=False)
            cur_scen = self.step_macro(cur_scen, targets, is_scenario=True)
            
            # Step 2: Propagate Tax heads recursively
            for head, winners in self.best_models.items():
                model = winners.get('policy_winner')
                if not model: continue
                
                # Single step predict
                df_b = pd.DataFrame([cur_base])
                df_s = pd.DataFrame([cur_scen])
                
                # VALIDATION: Check for NaNs and missing columns before predict
                for d_ver, state in [('Base', cur_base), ('Scen', cur_scen)]:
                    missing = [c for c in model.x_cols if c not in state.index]
                    if missing:
                        raise ValueError(f"CRITICAL: Model for {head} requires {missing}, but they are missing from forecast context.")
                    
                    nans = [c for c in model.x_cols if pd.isna(state[c])]
                    if nans:
                         raise ValueError(f"CRITICAL: Forecast for {head} halted. Variable(s) {nans} computed as NaN in {d_ver} path.")

                # Forecast
                b_fc = model.forecast(df_b, steps=1).iloc[0]
                s_fc = model.forecast(df_s, steps=1).iloc[0]
                
                # Update for next step lags
                cur_base[model.y_col] = b_fc
                cur_scen[model.y_col] = s_fc
                
                # Convert to PKR Billion (storage is million)
                results[head]['baseline'].append(np.exp(b_fc) / 1000.0)
                results[head]['scenario'].append(np.exp(s_fc) / 1000.0)

                # Confidence Intervals: SE_h = RMSE * sqrt(h)
                rmse = getattr(model, 'rmse', 0.15) # Default 15% error if model fails to report
                se_h = rmse * math.sqrt(h)
                
                results[head]['l80'].append(np.exp(s_fc - 1.282 * se_h) / 1000.0)
                results[head]['u80'].append(np.exp(s_fc + 1.282 * se_h) / 1000.0)
                results[head]['l95'].append(np.exp(s_fc - 1.960 * se_h) / 1000.0)
                results[head]['u95'].append(np.exp(s_fc + 1.960 * se_h) / 1000.0)
            
            base_path.append(cur_base.copy())
            scen_path.append(cur_scen.copy())

        # Cleanup results and compute total
        total_baseline = None
        total_scenario = None
        total_l80 = None
        total_u80 = None
        total_l95 = None
        total_u95 = None
        
        for head in list(results.keys()):
            for k in ['baseline', 'scenario', 'l80', 'u80', 'l95', 'u95']:
                results[head][k] = pd.Series(results[head][k])
            
            if total_baseline is None:
                total_baseline = results[head]['baseline'].copy()
                total_scenario = results[head]['scenario'].copy()
                total_l80 = results[head]['l80'].copy()
                total_u80 = results[head]['u80'].copy()
                total_l95 = results[head]['l95'].copy()
                total_u95 = results[head]['u95'].copy()
            else:
                total_baseline += results[head]['baseline']
                total_scenario += results[head]['scenario']
                total_l80 += results[head]['l80']
                total_u80 += results[head]['u80']
                total_l95 += results[head]['l95']
                total_u95 += results[head]['u95']
        
        results['total'] = {
            'baseline': total_baseline if total_baseline is not None else pd.Series([0]*horizon),
            'scenario': total_scenario if total_scenario is not None else pd.Series([0]*horizon),
            'l80': total_l80 if total_l80 is not None else pd.Series([0]*horizon),
            'u80': total_u80 if total_u80 is not None else pd.Series([0]*horizon),
            'l95': total_l95 if total_l95 is not None else pd.Series([0]*horizon),
            'u95': total_u95 if total_u95 is not None else pd.Series([0]*horizon)
        }
            
        return results, pd.DataFrame(base_path), pd.DataFrame(scen_path)

    def step_macro(self, row, targets, is_scenario=True):
        new_row = row.copy()
        
        # 1. Dummy Management: Reset all shock dummies to 0, toggle policy dummies
        active_policy_dummies = targets.get('active_dummies', [])
        all_dummies = targets.get('all_dummies', [])
        for d in all_dummies:
            if d in new_row.index:
                new_row[d] = 1 if d in active_policy_dummies else 0

        # 2. Consistent Level Paths
        if is_scenario:
            # Shift log_gdp
            g_rate = targets.get('gdp_growth', 0.108)
            new_row['log_gdp'] += math.log(max(1 + g_rate, 0.001))
            new_row['log_gdp_hat'] = new_row['log_gdp']
            
            new_row['inflation'] = targets.get('inflation', row['inflation'])
            new_row['policy rate'] = targets.get('policy_rate', row['policy rate'])
            
            # exrate propagation
            new_row['exrate'] *= (1 + targets.get('exrate_growth', 0.0))
            if new_row['exrate'] <= 0: new_row['exrate'] = 1e-6
            new_row['log_exrate'] = math.log(new_row['exrate'])
        else:
            # Baseline uses nominal growth (10.8%)
            new_row['log_gdp'] += math.log(1.108)
            new_row['log_gdp_hat'] = new_row['log_gdp']
            
        # 3. Channel Equations Propagation (Stage 1)
        order = ['log_imports', 'log_dutiable_imports', 'log_lsm', 'log_consumption']
        for ch_name in order:
            if ch_name in self.channel_models:
                ch_model = self.channel_models[ch_name]
                # Forecast 1 step
                pred = ch_model.forecast(pd.DataFrame([new_row]), steps=1).iloc[0]
                new_row[ch_name] = pred
                new_row[f"{ch_name}_hat"] = pred
            
        return new_row

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm

class HARSimulator:
    def __init__(self, df, Y, shift, mode='rolling', use_lasso=True, fixed_windows=None):
        self.df = df.copy()
        self.Y = Y
        self.shift = shift
        self.mode = mode  # 'rolling' or 'incremental'
        self.use_lasso = use_lasso
        self.fixed_windows = fixed_windows
        self.train_size = 1045
        self.test_size = 260
        self.results = []
        self.pred = np.array([])
        self.true = np.array([])

    def add_rolling_features(self):
        for wl in range(1, 31):
            self.df[f'roll_{wl}'] = self.df['vol'].rolling(wl, min_periods=1).mean()

    def run_simulation(self):
        for outer in range(6):
            if self.mode == 'rolling':
                train_start = self.test_size * outer
            else:  # incremental
                train_start = 0

            train_end = self.train_size + outer * self.test_size
            test_start = train_end
            test_end = train_end + self.test_size

            Y_train_all = self.Y[self.shift:][train_start:train_end]
            Y_test = self.Y[self.shift:][test_start:test_end]

            feature_cols = [f'roll_{wl}' for wl in range(1, 31)]
            X_all = self.df[feature_cols].values[:-self.shift][train_start:train_end]
            X_test_all = self.df[feature_cols].values[:-self.shift][test_start:test_end]

            if self.fixed_windows:
                best_windows = self.fixed_windows
            elif self.use_lasso:
                scaler = StandardScaler()
                X_all_scaled = scaler.fit_transform(X_all)
                tscv = TimeSeriesSplit(n_splits=5)
                lasso = LassoCV(cv=tscv, max_iter=100000)
                lasso.fit(X_all_scaled, Y_train_all)

                coef = lasso.coef_
                selected_indices = np.where(coef != 0)[0]
                best_windows = [i + 1 for i in selected_indices]

                if len(best_windows) == 0:
                    print(f"[Window {outer + 1}] No windows selected. Skipping.")
                    continue
            else:
                raise ValueError("Either fixed_windows must be provided or use_lasso must be True.")

            print(f"[Window {outer + 1}] Selected windows: {best_windows}")

            selected_features = [f'roll_{wl}' for wl in best_windows]
            X_train = self.df[selected_features].values[:-self.shift][train_start:train_end]
            X_test = self.df[selected_features].values[:-self.shift][test_start:test_end]

            scaler_final = StandardScaler()
            X_train_scaled = scaler_final.fit_transform(X_train)
            X_test_scaled = scaler_final.transform(X_test)

            X_train_sm = sm.add_constant(X_train_scaled)
            X_test_sm = sm.add_constant(X_test_scaled)

            ols = sm.OLS(Y_train_all, X_train_sm).fit()
            #print(f"\n[Window {outer + 1}] OLS Regression Summary")

            preds = np.maximum(0, ols.predict(X_test_sm))

            self.pred = np.concatenate((self.pred, preds)) if self.pred.size else preds
            self.true = np.concatenate((self.true, Y_test)) if self.true.size else Y_test

            mae = mean_absolute_error(Y_test, preds)/100
            rmse = np.sqrt(mean_squared_error(Y_test, preds)) / 100
            self.results.append({
                "window": outer + 1,
                "MAE": mae,
                "RMSE": rmse,
                "Used Windows": best_windows
            })
        for i in range(len(self.results)):
            print(f"Window {self.results[i]['window']}: MAE = {self.results[i]['MAE']:.4f}, RMSE = {self.results[i]['RMSE']:.4f}")

    def plot_predictions(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.true, label='true',color='blue',linestyle='--')
        plt.plot(self.pred, label='predictions',color='red',linewidth=2)
        plt.legend()
        model_name = "Flexible HAR" if self.use_lasso else f"HAR({', '.join(map(str, self.fixed_windows))})"
        plt.title(f"Walk-forward predictions - {model_name} with {self.mode} testing")
        plt.show()

    def summary(self):
        df_results = pd.DataFrame(self.results)
        #print(df_results[["window", "MAE", "RMSE", "Used Windows"]])
        #print("Global RMSE:", np.sqrt(np.mean((self.true - self.pred) ** 2)))
        return df_results

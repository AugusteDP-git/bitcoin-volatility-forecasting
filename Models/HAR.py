import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm

class HARSimulator:
    """Walk-forward simulation d’un HAR flexible (fenêtres 1-30 jours)."""

    def __init__(self, df, Y, shift, mode='rolling', use_lasso=True, fixed_windows=None):
        self.df = df.copy()
        self.Y = Y
        self.shift = shift
        self.mode = mode
        self.use_lasso = use_lasso
        self.fixed_windows = fixed_windows
        self.train_size = 1045
        self.test_size = 260
        self.results = []
        self.pred = np.array([])
        self.true = np.array([])
        self.add_rolling_features()

    def add_rolling_features(self):
        for wl in range(1, 31):
            self.df[f'roll_{wl}'] = self.df['vol'].rolling(wl, min_periods=1).mean()

    def run_simulation(self):
        for outer in range(6):
            train_start = 0 if self.mode == 'incremental' else self.test_size * outer
            train_end = self.train_size + outer * self.test_size
            test_start = train_end
            test_end = train_end + self.test_size

            y_train = self.Y[self.shift:][train_start:train_end]
            y_test = self.Y[self.shift:][test_start:test_end]

            feature_cols = [f'roll_{wl}' for wl in range(1, 31)]
            X_train_full = self.df[feature_cols].values[:-self.shift][train_start:train_end]
            X_test_full = self.df[feature_cols].values[:-self.shift][test_start:test_end]

            # ------------------------------------------------------------------ #
            if self.fixed_windows is not None:
                best_windows = self.fixed_windows
                sel_cols = [f'roll_{w}' for w in best_windows]
                X_train = self.df[sel_cols].values[:-self.shift][train_start:train_end]
                X_test = self.df[sel_cols].values[:-self.shift][test_start:test_end]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                X_train_sm = sm.add_constant(X_train_scaled)
                X_test_sm = sm.add_constant(X_test_scaled)

                ols = sm.OLS(y_train, X_train_sm).fit()
                preds = np.maximum(0, ols.predict(X_test_sm))

            else:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_full)
                X_test = scaler.transform(X_test_full)

                tscv = TimeSeriesSplit(n_splits=5)
                lasso = LassoCV(cv=tscv, max_iter=100_000).fit(X_train, y_train)
                preds = np.maximum(0, lasso.predict(X_test))
                best_windows = [i + 1 for i, c in enumerate(lasso.coef_) if c != 0]

            if len(best_windows) == 0:
                print(f"[Bloc {outer + 1}] Aucune fenêtre retenue. Skipped.")
                continue

            print(f"[Bloc {outer + 1}] Windows taken into account : {best_windows}")

            self.pred = np.concatenate((self.pred, preds)) if self.pred.size else preds
            self.true = np.concatenate((self.true, y_test)) if self.true.size else y_test

            mae = mean_absolute_error(y_test, preds) / 100
            rmse = np.sqrt(mean_squared_error(y_test, preds)) / 100
            self.results.append(
                {"bloc": outer + 1, "MAE": mae, "RMSE": rmse, "Fenêtres utilisées": best_windows}
            )

    def plot_predictions(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.true, label='Observé',color='blue', linestyle='--')
        plt.plot(self.pred, label='Prévu',color='red', linewidth=2)
        plt.legend()
        titre = "Flexible HAR (Lasso)" if self.fixed_windows is None else f"HAR fixe {self.fixed_windows}"
        plt.title(f"Walk-forward predictions – {titre} – mode {self.mode}")
        plt.show()

    def summary(self):
        print(pd.DataFrame(self.results))


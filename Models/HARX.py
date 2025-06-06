import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm

class HARXSimulator:
    def __init__(self, df, Y, shift, har_windows=False, mode='rolling', lag=1):
        self.df = df.copy()
        self.Y = Y
        self.shift = shift
        self.har_windows = har_windows
        self.mode = mode
        self.lag = lag
        self.train_size = 1045
        self.test_size = 260
        self.results = []
        self.pred = np.array([])
        self.true = np.array([])

        self.base_exog_features = [
            'spread', 'bid_depth', 'ask_depth', 'difference',
            'bid_volume', 'ask_volume', 'volume_difference',
            'weighted_spread', 'bid_slope', 'ask_slope'
        ]

        self.add_rolling_features()
        self.add_lagged_orderbook_features(self.lag)

        # Drop rows with NaN caused by rolling/lags
        self.df.dropna(inplace=True)

        # Align Y with cleaned df index to avoid misalignment
        self.Y = self.Y[self.df.index]

    def add_rolling_features(self):
        for wl in range(1, 31):
            self.df[f'roll_{wl}'] = self.df['vol'].rolling(wl, min_periods=1).mean()

    def add_lagged_orderbook_features(self, max_lag):
      num_rows = len(self.df)
      lagged_columns = []
      lagged_values = []

      for lag in range(0, max_lag + 1):
          for feature in self.base_exog_features:
              new_col = f'{feature}_lag{lag}'
              shifted_array = self.df[feature].shift(lag).values  
              lagged_columns.append(new_col)
              lagged_values.append(shifted_array)

      lagged_matrix = np.column_stack(lagged_values)

      lagged_df = pd.DataFrame(lagged_matrix, columns=lagged_columns, index=self.df.index)

      self.df = pd.concat([self.df, lagged_df], axis=1)

      self.exog_features = lagged_columns


    def run_simulation(self):
        # Choose HAR features (rolling vol) manually or dynamically
        if self.har_windows:
          har_features = [f'roll_{wl}' for wl in (1,6,24)]
        else:
          har_features = [f'roll_{wl}' for wl in range(1, 31)]
        candidate_features = har_features + self.exog_features

        for outer in range(6):
            if self.mode == 'rolling':
                train_start = self.test_size * outer
            elif self.mode == 'incremental':
                train_start = 0
            else:
                raise ValueError("Invalid mode. Use 'rolling' or 'incremental'.")

            train_end = self.train_size + outer * self.test_size
            test_start = train_end
            test_end = train_end + self.test_size

            # Ensure target y_t+1 is predicted using x_t
            Y_train_all = self.Y[self.shift:][train_start:train_end]
            Y_test = self.Y[self.shift:][test_start:test_end]

            X_full = self.df[candidate_features].values[:-self.shift]
            X_train = X_full[train_start:train_end]
            X_test = X_full[test_start:test_end]

            if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                print(f"[Window {outer + 1}] Empty train/test set. Skipping.")
                continue

            # Fit scaler only on training data to avoid leakage
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Lasso with time-aware cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            lasso = LassoCV(cv=tscv, max_iter=100_000)
            lasso.fit(X_train_scaled, Y_train_all)

            selected_indices = np.where(lasso.coef_ != 0)[0]
            if self.har_windows:

                forced_har_features = [f'roll_{w}' for w in (1,6,24)]
                forced_indices = [i for i, feat in enumerate(candidate_features) if feat in forced_har_features]
                
                selected_indices = np.unique(np.concatenate([selected_indices, forced_indices]))
            if len(selected_indices) == 0:
                print(f"[Window {outer + 1}] No features selected by Lasso. Skipping.")
                continue

            selected_features = [candidate_features[i] for i in selected_indices]
            print(f"[Window {outer + 1}] Selected features: {selected_features}")

            X_train_selected = X_train_scaled[:, selected_indices]
            X_test_selected = X_test_scaled[:, selected_indices]

            X_train_sm = sm.add_constant(X_train_selected)
            X_test_sm = sm.add_constant(X_test_selected)

            ols = sm.OLS(Y_train_all, X_train_sm).fit()
            preds = np.maximum(0, ols.predict(X_test_sm))

            self.pred = np.concatenate((self.pred, preds)) if self.pred.size else preds
            self.true = np.concatenate((self.true, Y_test)) if self.true.size else Y_test

            mae = mean_absolute_error(Y_test, preds) /100
            rmse = np.sqrt(mean_squared_error(Y_test, preds)) / 100

            self.results.append({
                "window": outer + 1,
                "MAE": mae,
                "RMSE": rmse,
                "Selected Features": selected_features
            })

    def plot_predictions(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.true, label='true',color='blue',linestyle='--')
        plt.plot(self.pred, label='predictions',color='red',linewidth=2)
        plt.legend()
        plt.title(f"Walk-forward predictions - HARX with post-Lasso OLS ({self.mode} mode)")
        plt.show()

    def summary(self):
        print(pd.DataFrame(self.results))

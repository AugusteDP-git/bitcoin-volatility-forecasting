import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

class HARXSimulator:
    def __init__(self, df, Y, shift, har_windows=[1, 6, 24], mode='rolling', GCT = False):
        self.GCT = GCT
        self.df = df.copy()
        self.Y = Y
        self.shift = shift
        self.har_windows = har_windows
        self.mode = mode  # 'rolling' or 'incremental'
        self.train_size = 1040
        self.test_size = 260
        self.results = []
        self.pred = np.array([])
        self.true = np.array([])

        self.exog_features = [
            'spread', 'bid_depth', 'ask_depth', 'difference',
            'bid_volume', 'ask_volume', 'volume_difference',
            'weighted_spread', 'bid_slope', 'ask_slope'
        ]
        
        if GCT:
            self.exog_features = ['ask_depth', 'bid_depth', 'bid_volume', 'spread', 'volume_difference']

    def add_rolling_features(self):
        for wl in range(1, 31):
            self.df[f'roll_{wl}'] = self.df['vol'].rolling(wl, min_periods=1).mean()

    def run_simulation(self):
        har_features = [f'roll_{wl}' for wl in self.har_windows]
        selected_features = har_features + self.exog_features

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

            Y_train_all = self.Y[self.shift:][train_start:train_end]
            Y_test = self.Y[self.shift:][test_start:test_end]

            X_train = self.df[selected_features].values[:-self.shift][train_start:train_end]
            X_test = self.df[selected_features].values[:-self.shift][test_start:test_end]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            X_train_sm = sm.add_constant(X_train_scaled)
            X_test_sm = sm.add_constant(X_test_scaled)

            ols = sm.OLS(Y_train_all, X_train_sm).fit()

            preds = np.maximum(0, ols.predict(X_test_sm))

            self.pred = np.concatenate((self.pred, preds)) if self.pred.size else preds
            self.true = np.concatenate((self.true, Y_test)) if self.true.size else Y_test

            mae = mean_absolute_error(Y_test, preds)/100
            rmse = np.sqrt(mean_squared_error(Y_test, preds)) / 100

            self.results.append({
                "window": outer + 1,
                "MAE": mae,
                "RMSE": rmse,
                "Used Windows": self.har_windows
            })

    def plot_predictions(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.true, label='true',color='blue',linestyle='--')
        plt.plot(self.pred, label='predictions',color='red',linewidth=2)
        plt.legend()
        if self.GCT:
            plt.title(f"Walk-forward predictions - HARX({', '.join(map(str, self.har_windows))}) with GCT features")
        else:
            plt.title(f"Walk-forward predictions - HARX({', '.join(map(str, self.har_windows))}) with {self.mode} testing")
        plt.show()

    def summary(self):
        df_results = pd.DataFrame(self.results)
        print(df_results)
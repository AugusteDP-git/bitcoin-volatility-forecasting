from arch import arch_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def garch_forecast(alpha0, alpha1, beta1, ut, ht, steps_ahead=1):
    indicator = 1 if ut < 0 else 0
    ut_carre = ut**2
    base = alpha1 + beta1
    core_term = alpha1 * ut_carre + beta1 * ht

    if base != 1:
        sum_term = alpha0 * (1 - base**steps_ahead) / (1 - base)
    else:
        sum_term = alpha0 * steps_ahead

    decay_term = (base ** (steps_ahead - 1)) * core_term
    variance_forecast = sum_term + decay_term
    return np.sqrt(variance_forecast)


def split_mean_rmse(rmse_series, n_intervals=10):
    n = len(rmse_series)
    interval_size = n // n_intervals
    
    mean_rmses = []
    for i in range(n_intervals):
        start_idx = i * interval_size
        if i == n_intervals - 1:
            end_idx = n
        else:
            end_idx = (i + 1) * interval_size
        
        interval_rmse = rmse_series[start_idx:end_idx]
        mean_rmse = np.mean(interval_rmse)
        mean_rmses.append(mean_rmse)
    
    return mean_rmses


def test_garch(df_returns: pd.DataFrame, df_vol: pd.DataFrame, plot=True):
    df_returns = df_returns.copy()
    df_vol = df_vol.copy()
    df_returns['timestamp'] = pd.to_datetime(df_returns['timestamp'])
    df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'])

    df_returns_aligned = df_returns[df_returns['timestamp'] >= df_vol['timestamp'].min()]
    df = pd.merge(df_returns_aligned, df_vol[['timestamp', 'vol']], on='timestamp', how='left')
    df.rename(columns={'vol_y': 'vol'}, inplace=True)
    if 'vol_x' in df.columns:
        df.drop(columns=['vol_x'], inplace=True)

    df = df.dropna().reset_index(drop=True)

    best_lb = None
    best_rmse = float('inf')
    candidate_windows = list(range(70, 88))

    N = len(df)
    print("Starting lookback window optimization...")
    for lb in candidate_windows:
        window_rmses = []

        for i in range(lb, N - 1):
            train_set = df.iloc[i - lb:i]
            model = arch_model(train_set['returns'], mean='Zero', vol='Garch', p=1, q=1, dist='skewt', rescale=False)
            model_fit = model.fit(disp='off', show_warning=False)

            alpha0 = model_fit.params['omega']
            alpha1 = model_fit.params['alpha[1]']
            beta1 = model_fit.params['beta[1]']
            ut = train_set['returns'].iloc[-1]
            ht = model_fit.conditional_volatility.iloc[-1] ** 2

            pred_vol = garch_forecast(alpha0, alpha1, beta1, ut, ht, steps_ahead=1)
            pred_scaled = pred_vol * 100 * np.sqrt(365)

            rmse = np.sqrt((pred_scaled - df['vol'].iloc[i + 1])**2)
            window_rmses.append(rmse / 100)

        mean_rmse = np.mean(window_rmses)

        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_lb = lb

    lb = best_lb
    print(f"Starting volatility prediction...")

    all_preds = []
    all_trues = []
    all_timestamps = []

    for i in range(lb, N - 1):
        train_set = df.iloc[i - lb:i]

        model = arch_model(train_set['returns'], mean='Zero', vol='Garch', p=1, q=1, dist='skewt', rescale=False)
        model_fit = model.fit(disp='off', show_warning=False)

        alpha0 = model_fit.params['omega']
        alpha1 = model_fit.params['alpha[1]']
        beta1 = model_fit.params['beta[1]']
        ut = train_set['returns'].iloc[-1]
        ht = model_fit.conditional_volatility.iloc[-1] ** 2

        pred_vol = garch_forecast(alpha0, alpha1, beta1, ut, ht, steps_ahead=1)
        pred_scaled = pred_vol * 100 * np.sqrt(365)

        true_vol = df['vol'].iloc[i + 1]
        timestamp = df['timestamp'].iloc[i + 1]

        all_preds.append(pred_scaled)
        all_trues.append(true_vol)
        all_timestamps.append(timestamp)

    # Construct prediction DataFrame with timestamps
    preds_df = pd.DataFrame({
        'pred': all_preds,
        'timestamp': all_timestamps
    })

    if plot:
        print("\nPlotting GARCH predictions...")
        plt.figure(figsize=(14, 6))
        plt.plot(pd.Series(all_trues).reset_index(drop=True), label="True Volatility", color='blue', linestyle='--')
        plt.plot(pd.Series(all_preds).reset_index(drop=True), label="Predicted Volatility", color='red')
        plt.title(f"GARCH - Combined Volatility Prediction (lookback={lb})")
        plt.xlabel("Time Step")
        plt.ylabel("Volatility")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        splits = np.array_split(all_preds, 10)
        all_trues = np.array_split(all_trues,10)
        for i in range(4,len(splits)):
            print(f'Interval {i+1}: Mean RMSE = {np.sqrt(mean_squared_error(splits[i], all_trues[i]))/100:.4f}, mean MAE = {mean_absolute_error(splits[i], all_trues[i])/100:.4f}')

    return preds_df, best_lb
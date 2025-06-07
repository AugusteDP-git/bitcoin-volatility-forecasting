import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from Models.LSTMxGARCH import DualLSTMModel
import torch

def test_lstm(df, param_grid,
              orderbook_timesteps=30, future_horizon=1,
              epochs=5, batch_size=32):
    
    df = df.dropna().reset_index(drop=True)
    dfs = np.array_split(df, 10)

    predictions = []
    timestamps_all = []
    rmse_list = []
    mae_list = []
    all_true = []

    for i in range(3, 9):
        df_train    = pd.concat([dfs[j] for j in range(i-3, i-1)], ignore_index=True).reset_index(drop=True)
        df_validate = dfs[i].iloc[5:].reset_index(drop=True)
        df_test     = dfs[i+1].iloc[5:].reset_index(drop=True)

       
        X_train_vol, X_train_ord, y_train, _ = DualLSTMModel.prepare_data(
            df_train, orderbook_timesteps, future_horizon
        )
        X_val_vol, X_val_ord, y_val, _ = DualLSTMModel.prepare_data(
            df_validate, orderbook_timesteps, future_horizon
        )
        X_test_vol, X_test_ord, y_test, ts_test = DualLSTMModel.prepare_data(
            df_test, orderbook_timesteps, future_horizon
        )

        if X_train_vol.numel() == 0 or X_val_vol.numel() == 0 or X_test_vol.numel() == 0:
            continue

       
        best_model, best_params, best_rmse = None, None, float('inf')
        for params in ParameterGrid(param_grid):
            model = DualLSTMModel(
                hidden_size_1=params['hidden_size_1'],
                hidden_size_2=params['hidden_size_2'],
                dropout_rate=params['dropout_rate']
            )
            model.train_model(
                X_train_vol, X_train_ord, y_train,
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                epochs=epochs,
                batch_size=batch_size
            )
            mse_val = evaluate_model_mse(model, X_val_vol, X_val_ord, y_val)[0]
            if mse_val < best_rmse:
                best_rmse = mse_val
                best_params = params
                best_model = model


       
        new_df_train = pd.concat([df_train, df_validate], ignore_index=True).reset_index(drop=True)
        scaler = StandardScaler()
        feature_cols = ['spread', 'bid_depth', 'ask_depth', 
                        'bid_volume',  'volume_difference',
                  'vol']
        scaled_features = [col for col in feature_cols if col != 'vol']
        new_df_train[scaled_features] = scaler.fit_transform(new_df_train[scaled_features])

        X_vol_all, X_ord_all, y_all, _ = DualLSTMModel.prepare_data(
            new_df_train, orderbook_timesteps, future_horizon
        )
        best_model.train_model(X_vol_all, X_ord_all, y_all)

        
        df_test[scaled_features] = scaler.transform(df_test[scaled_features])

       
        rmse, mae, y_true_np, y_pred_np = evaluate_model_mse(
            best_model, X_test_vol, X_test_ord, y_test
        )
      

        predictions.append(y_pred_np)
        timestamps_all.append(ts_test)
        rmse_list.append(rmse)
        mae_list.append(mae)
        all_true.append(y_true_np)

    
    flat_preds = np.concatenate(predictions, axis=0)
    flat_ts    = np.concatenate(timestamps_all, axis=0)
    flat_true  = np.concatenate(all_true, axis=0)

   
    df_lstm = pd.DataFrame({'timestamp': flat_ts, 'pred': flat_preds})

   
    for idx, (r, m) in enumerate(zip(rmse_list, mae_list), start=1):
        print(f"Interval {idx} — Test RMSE: {r/100:.4f}, MAE: {m/100:.4f}")

   
    dfs_trimmed = [dfs[j].iloc[orderbook_timesteps + 5:].reset_index(drop=True) for j in range(10)]
    merged_true = pd.concat(dfs_trimmed[4:10], ignore_index=True)['vol'].values

    plt.figure(figsize=(10, 6))
    plt.plot(merged_true, label='Real Volatility', linestyle='--', color='blue')
    plt.plot(flat_preds,  label='Predictions LSTM', color='red')
    plt.title('Volatility Predictions — LSTM (6 intervals)')
    plt.xlabel('Timestamp')
    plt.ylabel('Annualized volatility (%)')
    plt.legend()
    plt.grid(True)
    plt.show()



    start_idx, end_idx = 1000, 1200
    zoom_true = flat_true[start_idx:end_idx]
    zoom_pred = flat_preds[start_idx:end_idx]
    zoom_x    = np.arange(start_idx, end_idx)

    plt.figure(figsize=(10, 6))
    plt.plot(zoom_x, zoom_true, '--', label='Real Volatility', color='blue')
    plt.plot(zoom_x, zoom_pred,  '-' , label='LSTM Predictions', color='red')
    plt.title(f'Volatility Predictions — Zoom [{start_idx}:{end_idx}]')
    plt.xlabel('Time Step Index')
    plt.ylabel('Annualized volatility (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return df_lstm, rmse_list, mae_list


def evaluate_model_mse(model, X_vol, X_orderbook, y_true):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_vol       = X_vol.to(device)
    X_orderbook = X_orderbook.to(device)
    y_true      = y_true.to(device).view(-1, 1)

    with torch.no_grad():
        y_pred = model(X_vol, X_orderbook)

    y_pred_np = y_pred.cpu().numpy().flatten()
    y_true_np = y_true.cpu().numpy().flatten()
    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae  = mean_absolute_error(y_true_np, y_pred_np)
    return rmse, mae, y_true_np, y_pred_np

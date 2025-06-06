from Models.GARCH import test_garch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import copy
import matplotlib.pyplot as plt
def test_lstm_garch(df_returns, df_vol, model_class, feature_cols, param_grid, epochs=5, batch_size=32):
    results_df, best_lb = test_garch(df_returns, df_vol,plot = False)

    df_vol = df_vol.copy()
    df_vol['garch_vol'] = np.nan
    df_vol.loc[df_vol.index[:len(results_df)], 'garch_vol'] = results_df["pred"].values
    dfs = np.array_split(df_vol, 10)

    predictions, rmse_list, mae_list = [], [], []

    for i in range(3, 8):
        df_train = pd.concat([copy.deepcopy(dfs[j]) for j in range(i-3, i-1)], ignore_index=True)
        df_validate = copy.deepcopy(dfs[i][5:])  # Skip 5 to avoid overlap
        df_test = copy.deepcopy(dfs[i+1][5:])


        X_train_vol, X_train_orderbook, X_train_garch, y_train = model_class.prepare_data(df_train, lb_garch=best_lb)
        X_val_vol, X_val_orderbook, X_val_garch, y_val = model_class.prepare_data(df_validate, lb_garch=best_lb)
        X_test_vol, X_test_orderbook, X_test_garch, y_test = model_class.prepare_data(df_test, lb_garch=best_lb)


        model, best_params, best_rmse = model_class.grid_search(
            param_grid,
            X_train_vol, X_train_orderbook, X_train_garch, y_train,
            X_val_vol, X_val_orderbook, X_val_garch, y_val,
            epochs=epochs,
            batch_size=batch_size
        )

        print(f'Cross-validated on interval {i}')


        new_df_train = pd.concat([df_train, df_validate], ignore_index=True)
        scaler = StandardScaler()
        features_to_scale = [col for col in feature_cols if col != 'vol']
        new_df_train[features_to_scale] = scaler.fit_transform(new_df_train[features_to_scale])

        X_vol, X_orderbook, X_garch, y = model_class.prepare_data(new_df_train, lb_garch=best_lb)
        model.train_model(X_vol, X_orderbook, X_garch, y)


        df_test[features_to_scale] = scaler.transform(df_test[features_to_scale])
        print(f'Predicting interval {i+1}...')

        rmse, mae, y_true, y_pred, alphas = model.evaluate(
            model, X_test_vol, X_test_orderbook, X_test_garch, y_test
        )

        plt.figure(figsize=(10, 6))
        plt.plot(alphas, label="Alpha (weight on GARCH)", color='green')
        plt.title("Dynamic Alpha Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Alpha Value")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        plt.show()

        predictions.append(y_pred)
        rmse_list.append(rmse)
        mae_list.append(mae)


    final_pred = [p for interval in predictions for p in interval]
    print('RMSE per interval:', rmse_list)
    print('MAE per interval:', mae_list)

    dfs_trimmed = [df.iloc[best_lb + 5:].reset_index(drop=True) for df in dfs]
    merged = pd.concat(dfs_trimmed[4:9], ignore_index=True)

    plt.figure(figsize=(10, 6))
    plt.plot(merged['vol'], label='True Volatility', color='blue', linestyle='--')
    plt.plot(final_pred, label='LSTM-GARCH Predictions', color='red')
    plt.title('Volatility Predictions - LSTM with GARCH')
    plt.xlabel('Time Step')
    plt.ylabel('Annualized Volatility (%)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return final_pred, rmse_list, mae_list
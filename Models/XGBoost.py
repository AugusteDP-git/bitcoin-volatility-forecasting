import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Full function with manual grid search + custom exponential weighted MSE loss

def exp_weighted_mse_obj(alpha=5.0):
    def loss(y_pred, dtrain):
        y_true = dtrain.get_label()
        max_y = np.max(np.abs(y_true)) + 1e-6
        weight = np.exp(alpha * np.abs(y_true) / max_y)
        grad = 2 * weight * (y_pred - y_true)
        hess = 2 * weight
        return grad, hess
    return loss

def test_xgboost(df: pd.DataFrame, feat_lags, n_lags_vol=5, n_splits=10):
    if 'timestamp' not in df.columns:
        raise ValueError("'timestamp' column must be present in input DataFrame.")

    timestamps = df['timestamp'].copy()  # Save timestamps before dropping
    df = df.drop(columns=['timestamp'])

    lagged_features = []
    Unlagged_features = df.columns.difference(["vol"])

    for lag in range(1, n_lags_vol + 1):
        lagged_features.append(df["vol"].shift(lag).rename(f"vol_lag{lag}"))
        
    for wl in [1,6,24]:
            df[f'roll_{wl}'] = df['vol'].shift(1).rolling(wl, min_periods=1).mean()

    for col, max_lag in feat_lags.items():
        if col not in df.columns:
            continue
        for lag in range(1, max_lag + 1):
            lagged_features.append(df[col].shift(lag).rename(f"{col}_lag{lag}"))

    df = pd.concat([df] + lagged_features, axis=1).dropna().reset_index(drop=True)
    timestamps = timestamps.iloc[-len(df):].reset_index(drop=True)

    for col in Unlagged_features:
        df.drop(columns=[col], inplace=True)

    interval_size = len(df) // n_splits
    intervals = [df.iloc[i * interval_size: (i + 1) * interval_size] for i in range(n_splits)]
    interval_times = [timestamps.iloc[i * interval_size: (i + 1) * interval_size].reset_index(drop=True)
                      for i in range(n_splits)]

    param_grid = [
        {'max_depth': d, 'n_estimators': n, 'reg_lambda': l, 'gamma': g}
        for d in [3, 5, 7, 9]
        for n in [50, 100, 200]
        for l in [0.001, 0.01, 0.1, 1]
        for g in [0, 0.1, 0.5, 1]
    ]

    results = []
    all_preds = pd.DataFrame(columns=['pred', 'timestamp'])
    all_trues = []

    for i in range(n_splits - 4):
        print(f"\n[Window {i + 1}] Training on intervals {i}-{i + 2}, validating on {i + 3}, testing on {i + 4}")

        df_train = pd.concat(intervals[i:i + 3], axis=0).reset_index(drop=True)
        df_cv = intervals[i + 3]
        df_test = intervals[i + 4]
        ts_test = interval_times[i + 4]

        feature_cols = df_train.columns.difference(["vol"])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[feature_cols])
        X_cv = scaler.transform(df_cv[feature_cols])
        X_test = scaler.transform(df_test[feature_cols])

        y_train = df_train["vol"].values
        y_cv = df_cv["vol"].values
        y_test = df_test["vol"].values

        best_rmse = float("inf")
        best_model = None
        

        for params in param_grid:
            for alpha in [0,3,5,7,10]:
                full_params = {
                    'max_depth': params['max_depth'],
                    'eta': 0.1,
                    'reg_lambda': params['reg_lambda'],
                    'gamma': params['gamma'],
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'verbosity': 0
                }

                dtrain = xgb.DMatrix(X_train, label=y_train)
                dval = xgb.DMatrix(X_cv, label=y_cv)

                model = xgb.train(
                    full_params,
                    dtrain,
                    num_boost_round=params['n_estimators'],
                    obj=exp_weighted_mse_obj(alpha),
                    evals=[(dval, 'eval')],
                    verbose_eval=False
                )

                preds_val = model.predict(dval)
                rmse = np.sqrt(mean_squared_error(y_cv, preds_val))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model

        dtest = xgb.DMatrix(X_test, label=y_test)
        preds_test = best_model.predict(dtest)
        preds_df = pd.DataFrame({'pred': preds_test, 'timestamp': ts_test})
        all_preds = pd.concat([all_preds, preds_df], ignore_index=True)
        all_trues.extend(y_test)

        results.append({
            "Window": i + 1,
            "RMSE": np.sqrt(mean_squared_error(y_test, preds_test)) / 100,
            "MAE": mean_absolute_error(y_test, preds_test) / 100
        })

    results_df = pd.DataFrame(results)
    print("\n[Summary]")
    print(results_df)

    plt.figure(figsize=(14, 6))
    plt.plot(all_trues, label="True Volatility", color='blue', linestyle='--')
    plt.plot(all_preds['pred'], label="Predicted Volatility", color='red', linewid
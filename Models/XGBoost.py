import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


def test_xgboost(df: pd.DataFrame, n_lags: int = 1, n_splits: int = 10):
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])

    for lag in range(1, n_lags + 1):
        df[f"vol_lag{lag}"] = df["vol"].shift(lag)
    

    initial_features = df.columns.difference([ 'vol'])
    order_book_cols = [col for col in initial_features if col not in ['vol']]
    
    for col in order_book_cols:
        for lag in range(1, n_lags + 1):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df = df.dropna().reset_index(drop=True)

    interval_size = len(df) // n_splits
    intervals = [df.iloc[i * interval_size : (i + 1) * interval_size] for i in range(n_splits)]

    results = []
    all_preds = []
    all_trues = []

    for i in range(n_splits - 4):
        print(f"\n[Window {i+1}] Training on intervals {i}-{i+2}, validating on {i+3}, testing on {i+4}")

        df_train = pd.concat(intervals[i:i+3], axis=0).reset_index(drop = True)
        df_test = intervals[i+4]


        feature_cols = df_train.columns.difference(["vol"])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[feature_cols])
        X_test = scaler.transform(df_test[feature_cols])

        y_train = df_train["vol"]
        y_test = df_test["vol"]

        cv_train_df = intervals[i + 3]
        X_cv = scaler.transform(cv_train_df[feature_cols])
        y_cv = cv_train_df["vol"]

        param_grid = {
        'max_depth': [3, 5, 7, 9],                 
        'n_estimators': [50, 100, 200],            
        'gamma': [0, 0.01, 0.1, 0.5, 1],           
        'reg_lambda': [0.001, 0.01, 0.1, 1, 10],    
        }

        base_model = XGBRegressor(
            objective='reg:squarederror',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=3,
            verbose=0,
            n_jobs=-1
        )
        grid_search.fit(X_cv, y_cv)
        best_params = grid_search.best_params_

        model = XGBRegressor(
            **best_params,
            objective='reg:squarederror',
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        X = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_cv)], axis=0).reset_index(drop=True)
        y = pd.concat([y_train.reset_index(drop=True), y_cv.reset_index(drop=True)], axis=0).reset_index(drop=True)
        model.fit(X, y)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)


        all_preds.extend(preds)
        all_trues.extend(y_test.reset_index(drop=True))

        results.append({
            "Window": i + 1,
            "RMSE": rmse / 100,
            "MAE": mae / 100
        })


    results_df = pd.DataFrame(results)
    print("\n[Summary]")
    print(results_df)

    plt.figure(figsize=(14, 6))
    plt.plot(all_trues, label="True Volatility", color='blue', linestyle = '--')
    plt.plot(all_preds, label="Predicted Volatility", color='red', linewidth=2)
    plt.title("XGBoost - Combined Volatility Prediction")
    plt.xlabel("Time Step")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results_df
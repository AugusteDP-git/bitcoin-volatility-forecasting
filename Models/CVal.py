import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from Models.TM_G import TM_G
from Models.TM_SG import TM_SG
def cross_validation(df_train, df_validate):
  feature_cols = df_train.columns.difference(["vol"])

  scaler = StandardScaler()
  df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
  df_validate[feature_cols] = scaler.transform(df_validate[feature_cols])


  best_rmse = np.inf
  best_params = None

  lambdas = [0.001, 0.01,0.1,1,5,10]
  lbs = [5, 10, 15, 25, 30]
  for lb in lbs:
    for lambda_reg in lambdas:
      rmse = 0
      model = TM_G(n_features=len(feature_cols), lb = lb, lambda_reg=lambda_reg)
      model.train_model(df_train, epochs=2000)
      y_pred, y_true, gating = model.predict(df_validate)


      rmse = np.sqrt(mean_squared_error(y_pred, y_true))
      if rmse < best_rmse:
        best_rmse = rmse
        best_params = {
            'lb' : lb,
            'lambda_reg': lambda_reg
        }
  return best_params

def cross_validation_S(df_train, df_validate):
  feature_cols = df_train.columns.difference(["vol", "sentiment"])

  scaler = StandardScaler()
  df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
  df_validate[feature_cols] = scaler.transform(df_validate[feature_cols])


  best_rmse = np.inf
  best_params = None

  lambdas = [0.001, 0.01,0.1,1,5,10]
  lbs = [5, 10, 15, 25, 30]
  for lb in lbs:
    for lambda_reg in lambdas:
        rmse = 0
        model = TM_SG(n_features=len(feature_cols), lb = lb, lambda_reg=lambda_reg)
        model.train_model(df_train, epochs=2000)
        y_pred, y_true, g_hist, g_order, g_sentiment = model.predict(df_validate)


        rmse = np.sqrt(mean_squared_error(y_pred, y_true))
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {
                'lb' : lb,
                'lambda_reg': lambda_reg

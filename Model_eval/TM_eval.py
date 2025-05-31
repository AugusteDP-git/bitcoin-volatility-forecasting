import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import torch

def evaluate_TM(model, df_test, plot = True):
    y_pred, y_true, gating = model.predict(df_test)


    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    mae = mean_absolute_error(y_pred, y_true)
    # Plot
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="Real volatility", color = 'blue', linestyle = '--')
        plt.plot(y_pred, label="Predicted volatility", color = 'red', linewidth=2)
        plt.legend()
        plt.title("Volatility Prediction")
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.stackplot(range(len(gating)), gating, 1-gating,
                    labels=['Historical volatility', 'Orderbook'],
                    colors=['blue', 'red'])

        plt.title("Contribution of Gate Components Over Time")
        plt.xlabel("Time Steps")
        plt.ylabel("Proportion of Contribution")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

  

    return rmse, mae, y_pred, y_true

def evaluate_STM(model, df_test, plot = True):
    y_pred, y_true, g_hist, g_order, g_sentiment = model.predict(df_test)

    rmse = np.sqrt(mean_squared_error(y_pred, y_true))
    mae = mean_absolute_error(y_pred, y_true)
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(y_true, label="True volatility", color = 'blue', linestyle = '--')
        plt.plot(y_pred, label="Predicted volatility", color = 'red', linewidth=2)
        plt.legend()
        plt.title("Volatility Prediction")
        plt.show()

        plt.figure(figsize=(10, 
import numpy as np
import torch


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from sklearn.metrics import mean_squared_error, mean_absolute_error



def evaluate_model_mse(model, X_vol, X_orderbook, y_true):


    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)



    X_vol = X_vol.to(device)
    X_orderbook = X_orderbook.to(device)
    y_true = y_true.to(device).view(-1, 1)
   

    with torch.no_grad():
        y_pred = model.forward(X_vol, X_orderbook)

    y_pred_np = y_pred.cpu().numpy().flatten()
    y_true_np = y_true.cpu().numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae = mean_absolute_error(y_true_np, y_pred_np)

    return rmse, mae, y_true_np, y_pred_np

def evaluate_model_mse_garch_with_alpha(model, x_vol, x_ord, x_garch, y_true):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_vol, x_ord, x_garch, y_true = x_vol.to(device), x_ord.to(device), x_garch.to(device), y_true.to(device)
    with torch.no_grad():
        y_pred, alphas = model.forward(x_vol, x_ord, x_garch,return_alpha = True)

    y_pred_np = y_pred.cpu().numpy().flatten()
    y_true_np = y_true.cpu().numpy().flatten()
    alpha_np = alphas.cpu().numpy().flatten()

    rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
    mae = mean_absolute_error(y_true_np, y_pred_np)

    return rmse, mae, y_true_np, y_pred_np,alpha_np


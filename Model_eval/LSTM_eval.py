import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
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



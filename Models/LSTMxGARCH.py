import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim





class DualLSTMModel_garch(nn.Module):


    def __init__(self, hidden1=256, hidden2=256, dropout_rate=0.8):
        super().__init__()
        self.lstm_vol = nn.LSTM(1, hidden1, batch_first=True)
        self.lstm_orderbook = nn.LSTM(5, hidden1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden1 * 2, hidden2)
        self.fc2 = nn.Linear(hidden2, 1)
        self.alpha_fc = nn.Linear(hidden1 * 2, 1)
        nn.init.constant_(self.alpha_fc.weight, 0.0)
        nn.init.constant_(self.alpha_fc.bias, 0.0)

    def forward(self, x_vol, x_orderbook, x_garch, return_alpha=False):
        _, (h_vol, _) = self.lstm_vol(x_vol)
        _, (h_ord, _) = self.lstm_orderbook(x_orderbook)
        combined = torch.cat((h_vol[-1], h_ord[-1]), dim=1)
        combined = self.dropout(combined)
        lstm_out = torch.relu(self.fc2(torch.relu(self.fc1(combined))))
        raw_alpha = self.alpha_fc(combined)
        alpha = torch.sigmoid(raw_alpha)
        final_pred = alpha * x_garch + (1 - alpha) * lstm_out

        if return_alpha:
            return final_pred, alpha
        return final_pred

    def train_model(self, x_vol, x_ord, x_garch, y,
                    learning_rate=0.001, weight_decay=0.0001,
                    epochs=5, batch_size=32):

        device = self.get_device()
        self.to(device)
        dataset = torch.utils.data.TensorDataset(x_vol, x_ord, x_garch, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        opt = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            for vol, ord, garch, target in loader:
                vol, ord, garch, target = vol.to(device), ord.to(device), garch.to(device), target.to(device)
                opt.zero_grad()
                pred = self.forward(vol, ord, garch)
                loss = loss_fn(pred, target)
                loss.backward()
                opt.step()

    @staticmethod
    def get_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def evaluate(model, x_vol, x_ord, x_garch, y_true):
        model.eval()
        device = model.get_device()
        x_vol, x_ord, x_garch, y_true = x_vol.to(device), x_ord.to(device), x_garch.to(device), y_true.to(device)
        with torch.no_grad():
            y_pred, alphas = model.forward(x_vol, x_ord, x_garch, return_alpha=True)

        y_pred_np = y_pred.cpu().numpy().flatten()
        y_true_np = y_true.cpu().numpy().flatten()
        alpha_np = alphas.cpu().numpy().flatten()

        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        mae = mean_absolute_error(y_true_np, y_pred_np)

        return rmse, mae, y_true_np, y_pred_np, alpha_np


    @staticmethod
    def prepare_data(df, lb_garch=76, orderbook_timesteps=30, future_horizon=1):
        features = ['ask_depth', 'bid_depth', 'bid_volume', 'spread', 'volume_difference', 'vol']
        df_values = df[features].values
        garch_vals = df['garch_vol'].values
        vol_vals = df['vol'].values

        total_samples = len(df) - lb_garch - future_horizon
        X_vol = np.zeros((total_samples, orderbook_timesteps, 1))
        X_orderbook = np.zeros((total_samples, orderbook_timesteps, 5))
        X_garch = garch_vals[:total_samples].reshape(-1, 1)
        y_volatility = vol_vals[lb_garch + future_horizon: lb_garch + future_horizon + total_samples].reshape(-1, 1)

        for i in range(total_samples):
            seq = df_values[i + lb_garch - orderbook_timesteps: i + lb_garch]
            X_vol[i] = seq[:, -1].reshape(-1, 1)  
            X_orderbook[i] = seq[:, :-1] 

        return (
            torch.tensor(X_vol, dtype=torch.float32),
            torch.tensor(X_orderbook, dtype=torch.float32),
            torch.tensor(X_garch, dtype=torch.float32),
            torch.tensor(y_volatility, dtype=torch.float32)
        )


    @classmethod
    def grid_search(cls, param_grid
, X_train_returns, X_train_orderbook, X_train_garch, y_train,
                    X_val_returns, X_val_orderbook, X_val_garch, y_val,
                    epochs=5, batch_size=32):
        best_rmse = float('inf')
        best_params = None
        best_model = None

        for params in ParameterGrid(param_grid):
            model = cls(
                hidden1=params['hidden_size_1'],
                hidden2=params['hidden_size_2'],
                dropout_rate=params['dropout_rate']
            )

            model.train_model(X_train_returns, X_train_orderbook, X_train_garch, y_train,
                              learning_rate=params['learning_rate'],
                              weight_decay=params['weight_decay'],
                              epochs=epochs,
                              batch_size=batch_size)

            mse_val = cls.evaluate(model, X_val_returns, X_val_orderbook, X_val_garch, y_val)[0]

            if mse_val < best_rmse:
                best_rmse = mse_val
                best_params = params
                best_model = model

        return best_model, best_params, best_rmse

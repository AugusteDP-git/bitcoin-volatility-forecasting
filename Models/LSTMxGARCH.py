import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd



class DualLSTMModel(nn.Module):
    def __init__(self, hidden_size_1=256, hidden_size_2=256, dropout_rate=0.3):
        super(DualLSTMModel, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

      
        self.lstm_vol = nn.LSTM(1, hidden_size_1, batch_first=True)
       
        self.lstm_orderbook = nn.LSTM(5, hidden_size_1, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size_1 * 2, hidden_size_2)
        self.fc2 = nn.Linear(hidden_size_2, 1)

    def forward(self, X_vol, X_orderbook):
       
        _, (h_vol, _)   = self.lstm_vol(X_vol)
        _, (h_order, _) = self.lstm_orderbook(X_orderbook)

        h_combined = torch.cat((h_vol[-1], h_order[-1]), dim=1)
        h_combined = self.dropout(h_combined)

        h_dense = torch.relu(self.fc1(h_combined))
        output  = self.fc2(h_dense)
        return output

    def train_model(self, x_volatility, x_orderbook, y_train,
                    learning_rate=0.001, weight_decay=0.0001,
                    epochs=5, batch_size=32, criterion=nn.MSELoss()):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        x_volatility = x_volatility.clone().detach().float().to(device)
        x_orderbook  = x_orderbook.clone().detach().float().to(device)
        y_train      = y_train.clone().detach().float().view(-1, 1).to(device)

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        dataset   = torch.utils.data.TensorDataset(x_volatility, x_orderbook, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch_vol, batch_ord, batch_y in dataloader:
                optimizer.zero_grad()
                preds = self(batch_vol, batch_ord)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

    @staticmethod
    def prepare_data(df_metrics, orderbook_timesteps=15, future_horizon=1):
       
      
        df = df_metrics.dropna().reset_index(drop=True)
        features = ['ask_depth', 'bid_depth', 'bid_volume', 'spread', 'volume_difference', 'vol']
        df_features = df[features]

        X_orderbook = []
        X_vol       = []
        y_volatility = []
        ts_list = []

        for i in range(len(df) - orderbook_timesteps - future_horizon):
            ts_pred = df['timestamp'].iloc[i + orderbook_timesteps - 1 + future_horizon]
            ts_list.append(ts_pred)

            seq = df_features.iloc[i : i + orderbook_timesteps].copy()
            future_vol = df['vol'].iloc[i + orderbook_timesteps - 1 + future_horizon]
            y_volatility.append(future_vol)

            orderbook_seq = seq.drop(columns=['vol']).values
            vol_seq       = seq['vol'].values.reshape(-1, 1)

            X_orderbook.append(orderbook_seq)
            X_vol.append(vol_seq)

        X_orderbook = torch.tensor(np.array(X_orderbook), dtype=torch.float32)
        X_vol       = torch.tensor(np.array(X_vol),       dtype=torch.float32)
        y_volatility = torch.tensor(np.array(y_volatility), dtype=torch.float32).view(-1, 1)

        return X_vol, X_orderbook, y_volatility, ts_list

    def evaluate(self, df_metrics, orderbook_timesteps=15, future_horizon=1):
        
        X_vol, X_ord, y_true, ts_list = DualLSTMModel.prepare_data(
            df_metrics, orderbook_timesteps, future_horizon
        )
        if X_vol.numel() == 0:
            return pd.DataFrame(columns=['timestamp', 'pred'])

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        X_vol = X_vol.to(device)
        X_ord = X_ord.to(device)

        with torch.no_grad():
            y_pred = self(X_vol, X_ord)

        preds_np = y_pred.cpu().numpy().flatten()
        return pd.DataFrame({'timestamp': ts_list, 'pred': preds_np})

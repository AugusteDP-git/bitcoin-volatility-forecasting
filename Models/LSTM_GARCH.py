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
        self.fc1 = nn.Linear(hidden1*2, hidden2) 
        self.fc2 = nn.Linear(hidden2, 1)
        self.alpha_fc = nn.Linear(hidden1*2, 1)
        


    def forward(self, x_vol, x_orderbook, x_garch,return_alpha=False):
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
        
       

    def train_model(self, x_vol, x_ord, x_garch, y, learning_rate = 0.001, weight_decay=0.0001, epochs=5, batch_size=32):
        device = get_device()
        self.to(device)
        dataset = torch.utils.data.TensorDataset(x_vol, x_ord, x_garch, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        opt = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            losses = []
            for vol, ord, garch, target in loader:
                vol, ord, garch, target = vol.to(device), ord.to(device), garch.to(device), target.to(device)
                opt.zero_grad()
                pred = self.forward(vol, ord, garch)
                loss = loss_fn(pred, target)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            #print(f"Epoch {epoch+1}/{epochs}, Loss: {np.mean(losses):.6f}")


def prepare_data_from_metrics_garch(df, orderbook_timesteps=30, future_horizon=1):

    features = ['ask_depth', 'bid_depth', 'bid_volume', 'spread', 'volume_difference','vol']

    df_features = df[features]

    X_vol, X_orderbook, X_garch, y_volatility = [], [], [], []

    for i in range(len(df) - future_horizon - 79):
        
        seq = df_features.iloc[i+79-orderbook_timesteps : i+79] 
        future_vol = df['vol'].iloc[i +future_horizon + 79] #79 is the best window we found
        X_orderbook.append(seq.drop(columns=['vol']).values) 
        X_vol.append(seq['vol'])  
        y_volatility.append(future_vol) 
        X_garch.append(df['garch_vol'].iloc[i])

    return (
    torch.tensor(np.array(X_vol), dtype=torch.float32).view(-1, orderbook_timesteps, 1),
    torch.tensor(np.array(X_orderbook), dtype=torch.float32),
    torch.tensor(np.array(X_garch), dtype=torch.float32).view(-1, 1),
    torch.tensor(np.array(y_volatility), dtype=torch.float32).view(-1, 1)
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


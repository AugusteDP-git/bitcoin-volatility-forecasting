def prepare_data_from_metrics(df_metrics, orderbook_timesteps=15, future_horizon=1):

    df = df_metrics.dropna().reset_index(drop=True)
    print(df.shape)

    features = ['spread', 'bid_depth', 'ask_depth', 'difference',
                'bid_volume', 'ask_volume', 'volume_difference',
                'weighted_spread', 'bid_slope', 'ask_slope', 'vol']
    df_features = df[features]


    X_orderbook = []
    X_vol = []
    y_volatility = []

    for i in range(len(df) - orderbook_timesteps - future_horizon):
        seq_features = df_features.iloc[i:i+orderbook_timesteps].copy()
        future_vol = df['vol'].iloc[i + orderbook_timesteps - 1 + future_horizon]


        orderbook_seq = seq_features.drop(columns=['vol']).values
        vol_seq = seq_features['vol'].values.reshape(-1, 1)

        X_orderbook.append(orderbook_seq)
        X_vol.append(vol_seq)
        y_volatility.append(future_vol)

    X_orderbook = torch.tensor(np.array(X_orderbook), dtype=torch.float32)
    X_vol = torch.tensor(np.array(X_vol), dtype=torch.float32)
    y_volatility = torch.tensor(np.array(y_volatility), dtype=torch.float32).view(-1, 1)

    return X_vol, X_orderbook, y_volatility


class DualLSTMModel(nn.Module):


    def __init__(self, hidden_size_1, hidden_size_2, dropout_rate):
        super(DualLSTMModel, self).__init__()
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.lstm_vol = nn.LSTM(1, hidden_size_1, batch_first=True)
        self.lstm_orderbook = nn.LSTM(10, hidden_size_1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size_1 * 2, hidden_size_2)
        self.fc2 = nn.Linear(hidden_size_2, 1)

   
    def forward(self, X_vol, X_orderbook):
        _, (h_vol, _) = self.lstm_vol(X_vol)
        _, (h_order, _) = self.lstm_orderbook(X_orderbook)
        h_combined = torch.cat((h_vol[-1], h_order[-1]), dim=1)
        h_combined = self.dropout(h_combined)
        h_dense = torch.relu(self.fc1(h_combined))
        output = torch.relu(self.fc2(h_dense)) 
        return output



 
    def train_model(self, x_volatility, x_orderbook, y_train, learning_rate, weight_decay, epochs=5, batch_size=32,criterion = nn.MSELoss()):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)  

        optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        #x_volatility = torch.tensor(x_volatility, dtype=torch.float32).to(device)
        #x_orderbook = torch.tensor(x_orderbook, dtype=torch.float32).to(device)
        #y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)

        x_volatility = x_volatility.clone().detach().float().to(device)
        x_orderbook = x_orderbook.clone().detach().float().to(device)
        y_train = y_train.clone().detach().float().view(-1, 1).to(device)


        dataset = torch.utils.data.TensorDataset(x_volatility, x_orderbook, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for batch_vol, batch_ord, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self(batch_vol, batch_ord)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.6f}")


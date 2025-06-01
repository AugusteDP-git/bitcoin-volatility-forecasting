from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error



param_grid = {
    'hidden_size_1': [32, 64, 128, 256],
    'hidden_size_2': [32, 64, 128, 256],
    'learning_rate': [0.0005, 0.001],
    'weight_decay': [0.0001, 0.001],
    'dropout_rate': [0.2, 0.4, 0.6,0.8]
}


def grid_search_LSTM(param_grid, X_train_returns, X_train_orderbook, y_train, X_val_returns, X_val_orderbook, y_val, epochs=5, batch_size=32):
    best_rmse = float('inf')
    best_params = None
    best_model = None


    for params in ParameterGrid(param_grid):
        #print(f"Training with params: {params}")


        model = DualLSTMModel(
            hidden_size_1=params['hidden_size_1'],
            hidden_size_2=params['hidden_size_2'],
            dropout_rate=params['dropout_rate']
        )


        model.train_model(X_train_returns, X_train_orderbook, y_train,
                          learning_rate=params['learning_rate'],
                          weight_decay=params['weight_decay'],
                          epochs=epochs,
                          batch_size=batch_size)


        mse_val = evaluate_model_mse(model, X_val_returns, X_val_orderbook, y_val)[0]


        if mse_val < best_rmse:
            best_rmse = mse_val
            best_params = params
            best_model = model

    return best_model, best_params, best_rmse

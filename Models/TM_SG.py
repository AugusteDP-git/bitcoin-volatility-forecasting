import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm




class TM_SG(nn.Module):
    def __init__(self, n_features, lv=5, lb=30, ls = 2, step_ahead = 1, learning_rate=0.01, lambda_reg=0.01, alpha=0.1):
        """
        Gaussian Temporal Mixture Model w. sentiment

        Args:
        - n_features: Dimension of order book feature vector
        - lv: Time horizon for volatility history (autoregression order)
        - lb: Time horizon for order book features (in minutes)
        - learning_rate: Learning rate for optimizer
        - lambda_reg: L2 regularization coefficient
        - alpha: Hinge loss regularization coefficient
        """

        super(TM_SG, self).__init__()
        self.step_ahead = step_ahead

        self.lv = lv
        self.lb = lb
        self.ls = ls
        self.n_features = n_features
        self.lambda_reg = lambda_reg
        self.alpha = alpha

        self.phi = nn.Parameter(torch.empty(lv))
        self.psi = nn.Parameter(torch.empty(ls))

        self.U = nn.Parameter(torch.empty(n_features))
        self.V = nn.Parameter(torch.empty(lb))

        self.theta = nn.Parameter(torch.empty(lv))
        self.xi = nn.Parameter(torch.empty(ls))
        self.A = nn.Parameter(torch.empty(n_features))
        self.B = nn.Parameter(torch.empty(lb))

        for param in [self.phi, self.psi, self.U, self.V ,self.theta, self.xi, self.A, self.B]:
            if param.ndim == 1:
                param.data = torch.nn.init.xavier_uniform_(param.unsqueeze(0)).squeeze(0)
            else:
                torch.nn.init.xavier_uniform_(param)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, vol_history, order_book_feats, sentiment):
        """
        Forward pass to compute predicted volatility and components

        Args:
        - vol_history: Tensor of shape (lv,)
        - order_book_feats: Tensor of shape (n_features, lb)

        Returns:
        - mixture_mean: Predicted volatility
        - component_means: Tuple (mu_h0, mu_h1)
        - gating: Gating weight for history component
        """


        mu_h0 = torch.sum(self.phi * vol_history, dim=1)
        bilinear = self.U @ order_book_feats @ self.V
        mu_h1 = bilinear

        mu_h2 = torch.sum(self.psi * sentiment, dim = 1)

        g_hist = F.softplus(torch.sum(self.theta * vol_history, dim=1))
        g_order = F.softplus(self.A.T @ order_book_feats @ self.B)
        g_sentiment = F.softplus(torch.sum(self.xi * sentiment, dim=1))

        Z = g_hist + g_order + g_sentiment + 1e-6
        g_hist = g_hist / Z
        g_order = g_order / Z
        g_sentiment = g_sentiment / Z



        mixture_mean = g_hist * mu_h0 + g_order * mu_h1 + g_sentiment * mu_h2

        return mixture_mean, mu_h0, mu_h1, mu_h2, g_hist, g_order, g_sentiment

    def loss_fn(self, vol_target, vol_history, order_book_feats, sentiment):
        """
        Compute total loss (negative log likelihood + regularization)
        """


        mixture_mean, mu_h0, mu_h1, mu_h2, g_hist, g_order, g_sentiment = self.forward(vol_history, order_book_feats, sentiment)

        norm_0 = torch.distributions.Normal(mu_h0, 1)
        norm_1 = torch.distributions.Normal(mu_h1, 1)
        norm_2 = torch.distributions.Normal(mu_h2, 1)

        comp_0 = g_hist * norm_0.log_prob(vol_target)
        comp_1 = g_order * norm_1.log_prob(vol_target)
        comp_2 = g_sentiment * norm_2.log_prob(vol_target)

        likelihood = torch.logsumexp(torch.stack([comp_0, comp_1, comp_2]), dim=0)

        neg_log_likelihood = -torch.sum(likelihood)

        l2_reg = self.lambda_reg * sum(torch.sum(p ** 2) for p in self.parameters())
        hinge_reg_u = self.alpha * torch.sum(F.relu(-mu_h0) + F.relu(-mu_h1) + F.relu(-mu_h2))

        total_loss = neg_log_likelihood + l2_reg + hinge_reg_u

        return total_loss

    def fit_step(self, vol_target, vol_history, order_book_feats, sentiment):
        """
        Single optimization step
        """
        self.optimizer.zero_grad()
        loss = self.loss_fn(vol_target, vol_history, order_book_feats, sentiment)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()
        return loss.item()
    
    def prepare_train_data(self, df):
        lv, ls, lb = self.lv, self.ls, self.lb
        L = max(lv, ls, lb)

        OB_feats = np.stack([
                    df.iloc[j - lb:j].drop(columns=["vol", 'sentiment']).values.T
                    for j in range(L, len(df)-self.step_ahead)
        ])

        vol_history = np.stack([
            df["vol"].iloc[j - lv:j].values for j in range(L, len(df)-self.step_ahead)
        ], axis = 0)

        sentiment = np.stack([
            df["sentiment"].iloc[j - ls:j].values for j in range(L, len(df)-self.step_ahead)
        ], axis = 0)

        vol_target = df["vol"].iloc[L:].reset_index(drop=True).shift(-self.step_ahead).dropna().reset_index(drop=True)
        
        return (
            torch.tensor(vol_history, dtype = torch.float32),
            torch.tensor(vol_target, dtype = torch.float32),
            torch.tensor(OB_feats, dtype = torch.float32),
            torch.tensor(sentiment, dtype = torch.float32)
        )
        

    def train_model(self, train_data, epochs=10):
        """
        Train the TMM model without batching.

        Args:
        - train_data (pd.DataFrame): DataFrame with order book (OB) features and 'returns' column.
        - epochs (int): Number of training epochs.
        """
        vol_history, vol_target, OB_feats, sentiment = self.prepare_train_data(train_data)
        self.train()
        losses = []
        qbar = tqdm(range(epochs))
        for epoch in qbar:
            loss = self.fit_step(vol_target, vol_history, OB_feats, sentiment)
            losses.append(loss)

        return losses

    def prepare_data(self, df, feature_cols, lv, lb, step_ahead):
        df = df.copy()
        df = df.dropna().reset_index(drop=True)
        L = max(lv, lb, self.ls)

        vol_history = np.stack([
            df["vol"].iloc[j - lv:j].values
            for j in range(L, len(df) - step_ahead)
        ])
        order_book_feats = np.stack([
            df.iloc[j - lb:j][feature_cols].values.T
            for j in range(L, len(df) - step_ahead)
        ])
        vol_target = df["vol"].iloc[L:].reset_index(drop=True)

        sentiment = np.stack([
            df["sentiment"].iloc[j - self.ls:j].values
            for j in range(L, len(df) - step_ahead)
        ])

        min_len = min(len(vol_history), len(order_book_feats), len(vol_target), len(sentiment))
        return (
            torch.tensor(vol_history[:min_len], dtype=torch.float32),
            torch.tensor(order_book_feats[:min_len], dtype=torch.float32),
            torch.tensor(vol_target[:min_len], dtype=torch.float32),
            torch.tensor(sentiment[:min_len], dtype = torch.float32)
        )

    def predict(self, test_data):

        """
        Predict future volatility given a DataFrame of historical data.

        Args:
        - train_data (pd.DataFrame): DataFrame with order book (OB) features and 'returns' column.

        Returns:
        - Predicted volatility (float)

        """
        self.eval()
        lv = self.lv
        lb = self.lb
        step_ahead = self.step_ahead

        feature_cols = test_data.columns.difference(['vol', 'sentiment'])

        vol_history, order_book_feats, y_true, sentiment = self.prepare_data(test_data, feature_cols, lv, lb, step_ahead)

        with torch.no_grad():
          y_pred, _, _, _, g_hist, g_order, g_sentiment = self.forward(vol_history, order_book_feats, sentiment)
          y_pred = np.array(y_pred.detach().numpy())
          y_true = np.array(y_true)


        return y_pred, y_true, g_hist, g_order, g_sentiment
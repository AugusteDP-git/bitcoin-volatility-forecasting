import pandas as pd
import numpy as np
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import InverseGamma
from tqdm import tqdm


class TM_IG(nn.Module):
    def __init__(self, n_features, lv=30, lb=30, step_ahead = 5, learning_rate=0.01, lambda_reg=0.01, alpha = 10):
        """
        Inverse Gamma Temporal Mixture Model

        Args:
        - n_features: Dimension of order book feature vector
        - lv: Time horizon for volatility history (autoregression order)
        - lb: Time horizon for order book features (in minutes)
        - learning_rate: Learning rate for optimizer
        - lambda_reg: L2 regularization coefficient
        - alpha: Hinge loss regularization coefficient
        """

        super(TM_IG, self).__init__()
        self.step_ahead = step_ahead
        self.alpha = alpha

        self.lv = lv
        self.lb = lb
        self.n_features = n_features
        self.lambda_reg = lambda_reg

        self.phi = nn.Parameter(torch.empty(lv))
        self.xi = nn.Parameter(torch.empty(lv))

        self.U = nn.Parameter(torch.empty(n_features))
        self.V = nn.Parameter(torch.empty(lb))

        self.U_s = nn.Parameter(torch.empty(n_features))
        self.V_s = nn.Parameter(torch.empty(lb))

        self.theta = nn.Parameter(torch.empty(lv))
        self.A = nn.Parameter(torch.empty(n_features))
        self.B = nn.Parameter(torch.empty(lb))

        for param in [self.phi, self.xi, self.U, self.V, self.U_s, self.V_s, self.theta, self.A, self.B]:
            torch.nn.init.xavier_uniform_(param.unsqueeze(1) if param.ndim == 1 else param)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, vol_history, order_book_feats):
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
        epsilon = 1e-6

        alpha_0 = 1 + F.softplus(torch.sum(self.phi * vol_history, dim=1))
        beta_0 =  epsilon + F.softplus(torch.sum(self.xi * vol_history, dim=1))

        alpha_1 = 1 + F.softplus(self.U @ order_book_feats @ self.V ) 
        beta_1 = epsilon + F.softplus(self.U_s @ order_book_feats @ self.V_s) 

        g_hist = F.softplus((torch.sum(self.theta * vol_history, dim=1)))
        g_order = F.softplus((self.A.T @ order_book_feats @ self.B))
        gating = g_hist / (g_hist + g_order + epsilon)

        mixture_mean = gating * beta_0 / (alpha_0 - 1 + epsilon) + (1 - gating) * beta_1 / (alpha_0 - 1)

        return mixture_mean, (alpha_0, beta_0), gating, (alpha_1, beta_1)

    def loss_fn(self, vol_target, vol_history, order_book_feats):
        """
        Compute total loss (negative log likelihood + regularization)

        Args:
        - vol_target: True volatility value (scalar)
        """

        mixture_mean, (alpha_0, beta_0), gating, (alpha_1, beta_1) = self.forward(vol_history, order_book_feats)

        vol_target = torch.clamp(vol_target, min=1e-6)


        IG_0 = InverseGamma(alpha_0, beta_0)
        IG_1 = InverseGamma(alpha_1, beta_1)

        comp_0 = gating * IG_0.log_prob(vol_target)
        comp_1 = (1 - gating) * IG_1.log_prob(vol_target)
        likelihood = torch.logsumexp(torch.stack([comp_0, comp_1]), dim=0)
        neg_log_likelihood = -torch.sum(likelihood)

        l2_reg = self.lambda_reg * sum(torch.sum(p ** 2) for p in self.parameters())


        total_loss = neg_log_likelihood  +l2_reg

        return total_loss

    def fit_step(self, vol_target, vol_history, order_book_feats):
        """
        Single optimization step
        """
        self.optimizer.zero_grad()
        loss = self.loss_fn(vol_target, vol_history, order_book_feats)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        self.optimizer.step()
        return loss.item()
    
    def prepare_train_data(self, df):
        lv, lb = self.lv, self.lb


        OB_feats = np.stack([
                    df.iloc[j - lb:j].drop(columns=["vol"]).values.T
                    for j in range(lb, len(df)-self.step_ahead)
        ])

        vol_history = np.stack([
            df["vol"].iloc[j - lv:j].values for j in range(lv, len(df)-self.step_ahead)
        ], axis = 0)

        vol_target = df["vol"].iloc[lv:].reset_index(drop=True).shift(-self.step_ahead).dropna().reset_index(drop=True)

        return (
            torch.tensor(vol_history, dtype = torch.float32),
            torch.tensor(vol_target, dtype = torch.float32),
            torch.tensor(OB_feats, dtype = torch.float32)
            
        )

    def train_model(self, train_data, epochs=10):
        """
        Train the TMM model without batching.

        Args:
        - train_data (pd.DataFrame): DataFrame with order book (OB) features and 'returns' column.
        - epochs (int): Number of training epochs.
        """
        vol_history, vol_target, OB_feats = self.prepare_train_data(train_data)
        self.train()
        losses = []
        qbar = tqdm(range(epochs))
        for epoch in qbar:
            loss = self.fit_step(vol_target, vol_history, OB_feats)
            qbar.set_postfix(loss=f"{loss:.6f}")
            losses.append(loss)

        return losses
    
    
    def prepare_data(self, df, feature_cols, lv, lb, step_ahead):
        df = df.copy()
        df = df.dropna().reset_index(drop=True)

        vol_history = np.stack([
            df["vol"].iloc[j - lv:j].values
            for j in range(lv, len(df) - step_ahead)
        ])
        order_book_feats = np.stack([
            df.iloc[j - lb:j][feature_cols].values.T
            for j in range(lb, len(df) - step_ahead)
        ])
        vol_target = df["vol"].iloc[max(lv, lb):].shift(-self.step_ahead).dropna().reset_index(drop=True)

        min_len = min(len(vol_history), len(order_book_feats), len(vol_target))
        return (
            torch.tensor(vol_history[:min_len], dtype=torch.float32),
            torch.tensor(order_book_feats[:min_len], dtype=torch.float32),
            torch.tensor(vol_target[:min_len], dtype=torch.float32)
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

        feature_cols = test_data.columns.difference(['vol'])

        vol_history, order_book_feats, y_true = self.prepare_data(test_data, feature_cols, lv, lb, step_ahead)
        lv = self.lv
        lb = self.lb


        with torch.no_grad():
          y_pred, _, _, gating = self.forward(vol_history, order_book_feats)

          y_pred = np.array(y_pred.detach().numpy())
          y_true = np.array(y_true)

        return y_pred, y_true, gating

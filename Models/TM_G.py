import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm



class TM_G(nn.Module):
    def __init__(self, n_features, lv = 5, lb=30, step_ahead = 1, learning_rate=0.01, lambda_reg=0.01, alpha = 10, lambda_entropy = 1):
        """
        Gaussian Temporal Mixture Model

        Args:
        - n_features: Dimension of order book feature vector
        - lv: Horizon for volatility history
        - lb: Horizon for order book features 
        - learning_rate: Learning rate for optimizer
        - lambda_reg: L2 regularization coefficient
        - alpha: Hinge loss regularization coefficient
        """

        super(TM_G, self).__init__()
        self.step_ahead = step_ahead

        self.lv = lv
        self.lb = lb
        self.n_features = n_features
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.lambda_entropy = lambda_entropy

        self.phi = nn.Parameter(torch.empty(lv))

        self.U = nn.Parameter(torch.empty(n_features))
        self.V = nn.Parameter(torch.empty(lb))

        self.theta = nn.Parameter(torch.empty(lv))
        self.A = nn.Parameter(torch.empty(n_features))
        self.B = torch.zeros(lb)
        self.B[0] = 1

        for param in [self.phi,  self.U, self.V ,self.theta, self.A, self.B]:
            if param.ndim == 1:
                param.data = torch.nn.init.xavier_uniform_(param.unsqueeze(0)).squeeze(0)
            else:
                torch.nn.init.xavier_uniform_(param)
        self.optimizer = optim.RAdam(self.parameters(), lr=learning_rate)

        
        
        

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


        mu_h0 = F.relu(torch.sum(self.phi * vol_history, dim=1))
        bilinear = self.U @ order_book_feats @ self.V
        mu_h1 = bilinear

        g_hist = F.softplus(torch.sum(self.theta * vol_history, dim=1), beta = 1)
        g_order = F.softplus(self.A.T @ order_book_feats @ self.B, beta = 1)

        gating = g_hist / (g_hist + g_order + 1e-6)

        mixture_mean = gating * mu_h0 + (1 - gating) * mu_h1

        return mixture_mean, mu_h0, mu_h1, gating

    def loss_fn(self, vol_target, vol_history, order_book_feats):
        """
        Compute total loss (negative log likelihood + regularization)
        """


        mixture_mean, mu_h0, mu_h1, gating = self.forward(vol_history, order_book_feats)

        norm_0 = torch.distributions.Normal(mu_h0, 1)
        norm_1 = torch.distributions.Normal(mu_h1, 1)

        comp_0 = gating * norm_0.log_prob(vol_target)
        comp_1 = (1 - gating) * norm_1.log_prob(vol_target)


        likelihood = torch.logsumexp(torch.stack([comp_0, comp_1]), dim=0)

        neg_log_likelihood = -torch.sum(likelihood)
        entropy = - (gating * torch.log(gating + 1e-8) + (1 - gating) * torch.log(1 - gating + 1e-8))
        entropy_reg = self.lambda_entropy * torch.mean(entropy)


        l2_reg = self.lambda_reg * sum(torch.sum(p ** 2) for p in self.parameters())
        hinge_reg_u = self.alpha * torch.sum(F.relu(-mu_h1))

        total_loss = neg_log_likelihood + l2_reg + hinge_reg_u - entropy_reg

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
                    for j in range(max(lv,lb), len(df)-self.step_ahead)
        ])

        vol_history = np.stack([
            df["vol"].iloc[j - lv:j].values for j in range(max(lv,lb), len(df)-self.step_ahead)
        ], axis = 0)

        vol_target = df["vol"].iloc[max(lv,lb):].reset_index(drop=True).shift(-self.step_ahead).dropna().reset_index(drop=True)

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
            for j in range(max(lv,lb), len(df) - step_ahead)
        ])
        order_book_feats = np.stack([
            df.iloc[j - lb:j][feature_cols].values.T
            for j in range(max(lv,lb), len(df) - step_ahead)
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

        with torch.no_grad():
          y_pred, _, _, gating = self.forward(vol_history, order_book_feats)

          y_pred = np.array(y_pred.detach().numpy())
          y_true = np.array(y_true)


        return y_pred, y_true, gating



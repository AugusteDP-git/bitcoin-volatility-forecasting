# Mixture Models for Bitcoin Volatility Forecasting

This repository contains the source code and data references for our paper: **"Mixture Models for Bitcoin Volatility Forecasting"**

## 🧠 Abstract

Forecasting financial market volatility is central to many areas of quantitative finance, including risk management, algorithmic trading, and derivative pricing. In this work, we investigate the modeling of **realized volatility for Bitcoin**, a highly volatile and speculative asset.

We compare a range of models—both traditional and modern—including:

- **Temporal Mixture Models** (Gaussian, Inverse Gamma, and Sentiment-augmented)
-  **HAR and Flexible HAR models**
-   **GARCH**
- **LSTM and LSTM-GARCH hybrids**
- **XGBoost and XGBoost-GARCH hybrids**

We evaluate how **order book features** and **news sentiment** contribute to predictive performance and model calibration.

## 📂 Repository Structure

```
.
├── Data/
│   ├── ADF.ipynb              # Augmented Dickey-Fuller test
│   ├── Data_cleaning.py       # Processing Aisot order book data
│   ├── DMF.ipy                # Data manipulation functions
│   ├── GCT.ipynb              # Granger Causality Test
│   ├── KPSS.ipynb             # Kwiatkowski-Phillips-Schmidt-Shin test
│   └── SPP.py                 # Sentiment preprocessing pipeline
├── Model_eval/
│   ├── LSTM_eval.py           # LSTM model evaluation
│   └── TM_eval.py             # Temporal Mixture Model evaluation
├── Models/
│   ├── CVal_LSTM.py           # Cross-validation for LSTM
│   ├── CVal.py                # General cross-validation utilities
│   ├── GARCH.py               # GARCH model implementation
│   ├── HAR.py                 # HAR model
│   ├── HARX.py                # HAR model with exogenous variables
│   ├── LSTM_GARCH.py          # LSTM-GARCH hybrid model
│   ├── LSTM.py                # LSTM implementation
│   ├── TM_G.py                # Gaussian Temporal Mixture Model
│   ├── TM_IG.py               # Inverse Gamma Temporal Mixture Model
│   ├── TM_SG.py               # Sentiment-augmented Temporal Mixture Model
│   └── XGBoost.py             # XGBoost model
├── main.ipynb                 # Main notebook for running experiments
└── README.md                  # Project documentation
```


## 📈 Key Findings

## 📊 Datasets

- **Order Book**: BTC/USD limit order book data from Bitstamp, sampled every 30 seconds.
- **Sentiment**: 
  - Crypto Fear and Greed Index (FNG)
  - Crypto Bull & Bear Index (CBI)

Available at:
- [LLMs-Sentiment-Augmented-Bitcoin-Dataset](https://huggingface.co/datasets/danilocorsi/LLMs-Sentiment-Augmented-Bitcoin-Dataset)
- [Aisot BTC Order Book Trades](https://huggingface.co/datasets/AisotTechnologies/aisot_btc_lob_trades)
## 📬 Contact

For questions, suggestions, or collaboration inquiries, feel free to reach out to the authors:

- **Auguste Darracq Paries** – [adarracq@ethz.ch](mailto:adarracq@ethz.ch)
- **Yann Adorno** – [yadorno@ethz.ch](mailto:yadorno@ethz.ch)
- **Laurent Martini–Bonnet** – [martinil@ethz.ch](mailto:martinil@ethz.ch)

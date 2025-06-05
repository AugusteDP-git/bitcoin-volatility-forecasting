import pandas as pd
from DMF import *

PATH = 'Enter path to Aisot data here'
SAVE_PATH = 'Enter path to save data here'
raw = pd.read_csv(PATH)

df = GMD(raw, SAVE_PATH).sort_values(by = 'timestamp').reset_index(drop = True)

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

df.set_index('timestamp', inplace=True)
df = df[~df.index.duplicated(keep='first')]

df.index = df.index.tz_localize(None)

full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='min')

df = df.reindex(full_index)
df.index.name = 'timestamp'
df = df.ffill()

df.reset_index(inplace=True)
df.rename(columns={'index': 'timestamp'}, inplace=True)

_, dfH,_ ,_ = GOD(df, '/content/drive/MyDrive/Bitcoin Volatility prediction/DATA/')

df['returns'] = np.log(df['price'] / df['price'].shift(1))
df = df.dropna().reset_index(drop = True)

hourly_vol = []
i = 0

while i < df.shape[0]:
  returns = df['returns'][i: i+60]
  hourly_vol.append(np.sqrt(np.sum(returns ** 2)) * 100 * np.sqrt(365))
  i += 60

dfH['vol'] = hourly_vol

missing_rows = [i for i in range(dfH.shape[0]) if dfH['returns'].iloc[i] == 0]

dfH = dfH.drop(missing_rows, axis = 0).reset_index(drop = True)
dfH['returns'] = np.log(dfH['price'] / dfH['price'].shift(1))
dfH = dfH.dropna().reset_index(drop = True)
dfH_returns=dfH.copy()
dfH_returns['returns'] = np.log(dfH_returns['price'] / dfH_returns['price'].shift(1))
dfH_returns = dfH_returns.dropna().reset_index(drop = True)

dfH.to_csv(PATH + 'hourly_data_ff.csv', index = False)
dfH_returns.to_csv(PATH + 'hourly_data_ff_with_returns.csv', index = False)
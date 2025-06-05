import pandas as pd
from tqdm import tqdm
import numpy as np
class OrderBook:
  def __init__(self, time, bids=None, asks=None):
    self.time = time
    self.bids = bids if bids is not None else []
    self.asks = asks if asks is not None else []
    self.bid_depth = None
    self.ask_depth = None


  def add_bid(self, price, amount):
    self.bids.append((price, amount))
  def add_ask(self, price, amount):
    self.asks.append((price, amount))


  def max_bid(self):
    if len(self.bids) == 0:
      return None
    return max(self.bids, key=lambda x: x[0])
  def min_ask(self):
    if len(self.asks) == 0:
      return None
    return min(self.asks, key=lambda x: x[0])

  def MergeOB(self, OB2):
    self.bids = self.bids + OB2.bids
    self.asks = self.asks + OB2.asks
    return self

  def Condense(self):
    bid_volume_map = {}
    for price, amount in self.bids:
        if price in bid_volume_map:
            bid_volume_map[price] += amount
        else:
            bid_volume_map[price] = amount

    self.bids = [(price, amount) for price, amount in bid_volume_map.items()]
    self.bids.sort(key=lambda x: x[0], reverse=True)


    ask_volume_map = {}
    for price, amount in self.asks:
        if price in ask_volume_map:
            ask_volume_map[price] += amount
        else:
            ask_volume_map[price] = amount

    self.asks = [(price, amount) for price, amount in ask_volume_map.items()]
    self.asks.sort(key=lambda x: x[0])

    return self
  



def D2MOB(df):
    df['interval'] = (df['time'] // 60) * 60
    grouped = df.groupby('interval')

    order_books = {}

    for interval_start, group in tqdm(grouped):

        ob = OrderBook(time=interval_start)

        for _, row in group.iterrows():
            if row['type'] == 'b':
                ob.add_bid(row['price'], row['amount'])
            elif row['type'] == 'a':
                ob.add_ask(row['price'], row['amount'])
        ob.Condense()

        if not ob.bids:
          print('No bids before cleaning')
          continue
        if not ob.asks:
          print('No asks before cleaning')
          continue

        min_ask_price = ob.min_ask()[0]


        ob.bids = [bid for bid in ob.bids if bid[0] <= min_ask_price]

        if not ob.bids:
          print('No bids after imposing bids less than minimum ask.')
          continue

        max_bid_price = ob.max_bid()[0]
        bid_cutoff = 0.75 * max_bid_price
        ob.bids = [bid for bid in ob.bids if bid[0] >= bid_cutoff]

        if not ob.bids:
          print('No bids after keeping bids greater than 0.75 * max bid.')
          continue

        min_ask_price = ob.min_ask()[0]
        ask_cutoff = 1.25 * min_ask_price
        ob.asks = [ask for ask in ob.asks if ask[0] <= ask_cutoff]

        if not ob.asks:
          print('No asks after imposing asks less than 1.25 * min ask.')
          continue

        ob.bid_depth=len(ob.bids)
        ob.ask_depth=len(ob.asks)
        order_books[interval_start] = ob


    return order_books
  
  
def GMD(df, PATH):

    order_books = D2MOB(df)

    data = []
    prev_mid_price = None

    for timestamp, ob in order_books.items():
        max_bid = ob.max_bid()
        min_ask = ob.min_ask()
        spread = (min_ask[0] - max_bid[0]) if max_bid and min_ask else None


        bid_depth = ob.bid_depth
        ask_depth = ob.ask_depth


        depth_difference = ask_depth - bid_depth


        bid_volume = sum(amount for _, amount in ob.bids)
        ask_volume = sum(amount for _, amount in ob.asks)


        volume_difference = ask_volume - bid_volume


        bid_threshold = int(0.1 * bid_depth)
        considered_bids = [bid for bid in ob.bids[:bid_threshold]]
        bid_cum_price = np.sum([bid[0] * bid[1] for bid in considered_bids])

        ask_threshold = int(0.1 * ask_depth)
        considered_asks = [ask for ask in ob.asks[:ask_threshold]]
        ask_cum_price = np.sum([ask[0] * ask[1] for ask in considered_asks])

        bid_slope = np.sum([bid[1] for bid in considered_bids])
        ask_slope = np.sum([ask[1] for ask in considered_asks])

        weighted_spread = bid_cum_price - ask_cum_price

        mid_price = (max_bid[0] + min_ask[0]) / 2 if max_bid and min_ask else None


        data.append([timestamp, spread, bid_depth, ask_depth, depth_difference,bid_volume,ask_volume,volume_difference, weighted_spread, bid_slope, ask_slope,mid_price])


    df_metrics = pd.DataFrame(data, columns=['timestamp', 'spread', 'bid_depth', 'ask_depth','difference','bid_volume','ask_volume','volume_difference','weighted_spread','bid_slope','ask_slope','price'])
    df_metrics.to_csv(PATH + 'minute_data.csv', index = False)
    return df_metrics
  

def GOD(df, PATH):
  data5 = []
  dataH = []
  dataD = []
  for i in tqdm(range(len(df))):
    if i % 5 == 0:
      data5.append(df.iloc[i].values)
    if i % 60 == 0:
      dataH.append(df.iloc[i].values)
    if i % 1440 == 0:
      dataD.append(df.iloc[i].values)
  df5 = pd.DataFrame(data5, columns=df.columns)
  dfH = pd.DataFrame(dataH, columns=df.columns)
  dfD = pd.DataFrame(dataD, columns=df.columns)

  df5.to_csv(PATH + '5minute_data.csv', index = False)
  dfH.to_csv(PATH + 'hourly_data.csv', index = False)
  dfD.to_csv(PATH + 'daily_data.csv', index = False)
  return df5, dfH, dfD
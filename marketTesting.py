from library.market_modelsM import lob_market, real_stock_lob
import pandas as pd
df = pd.read_csv("data/2019_05_19BTX_O.csv",low_memory = False)#,names = ["index","time","bid1","bidSize1","ask1","askSize1"])#, names=["Instrument", "Time", "Bid", "Ask"]) # Load .csv

df.columns = ["index","time","bid","bidSize","ask","askSize"]
print(df.head())
stock = real_stock_lob(df)
market = lob_market(stock,4)
market.
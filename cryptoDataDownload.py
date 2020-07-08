import qpython.qconnection
import pandas as pd
q = qpython.qconnection.QConnection('3.9.195.248', 41822, pandas=True)
print("Is connected ",q.is_connected())
q.open()
print("Is connected ",q.is_connected())
#df = q('select from orderBook where date=2019.05.18')

#(for example)


#The data may be OK for kdb+/q but too big for Python, so I would define the tick as

#dfO = q('select time, bid1, bidSize1, bid2, bidSize2, ask1, askSize1, ask2, askSize2 from orderBook where date=2019.05.19, (bid1 <> prev bid1) or (ask1 <> prev ask1) or (ask2 <> prev ask2) or (bid2 <> prev bid2) or (bidSize1 <> prev bidSize1) or (askSize1 <> prev askSize1) or (askSize2 <> prev askSize2) or (bidSize2 <> prev bidSize2)')
#dfO = q('select time, bid1, bidSize1, ask1, askSize1 from orderBook where date=2019.05.19, (bid1 <> prev bid1) or (ask1 <> prev ask1) or (bidSize1 <> prev bidSize1) or (askSize1 <> prev askSize1)')
download_orderbook = False
day = 19
if download_orderbook:
	dfO = q(f'select time, bid1, bidSize1, ask1, askSize1 from orderBook where date=2019.05.{day}, (bid1 <> prev bid1) or (ask1 <> prev ask1) or (bidSize1 <> prev bidSize1) or (askSize1 <> prev askSize1)')
	print(dfO.head())
	dfO.to_csv(f"2019_05_{day}BTX_O.csv")
	print("Orders done")
dfT = q(f'select time, side, size, price from trade where date=2019.05.{day}')

print(dfT.head())
dfT.to_csv(f"2019_05_{day}BTX_TF.csv")
print(f"{day} All Done")


import qpython.qconnection
import pandas as pd
q = qpython.qconnection.QConnection('3.9.195.248', 41822, pandas=True)
print("Is connected ",q.is_connected())
q.open()
print("Is connected ",q.is_connected())
df = q('select from orderBook where date=2020.05.21')

#(for example)


#The data may be OK for kdb+/q but too big for Python, so I would define the tick as

#df = q('select from orderBook where date=2019.06.21, where (bid1<>prev bid1) or (ask1<>prev ask1)')
df.head()
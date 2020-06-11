# External Packages
import numpy as np

# Internal Library
from library.market_modelsM import market, bs_stock, signal_stock, real_stock
from library.agents.distAgentsWIP2 import QRAgent, C51Agent
from library.agents.baseAgents import TWAPAgent
#from library.agents.distAgents import C51Agent
import library.simulations
import pandas as pd

# Define setup
params = {
    "terminal" : 3600,
    "num_trades" : 10,
    "position" : 1,
    "batch_size" : 32,
    "action_values" : [0.05,0.075,0.09,0.1,0.11,0.15,0.2] #[0.05,0.075,0.1,0.15,0.2] 
}
state_size = 2
action_size = len(params["action_values"])
n_hist_prices = 32

# Define Agents
quentin = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "Quentin",C=0, alternative_target = False,UCB=False,UCBc = 1,tree_horizon = 4,market_data_size=n_hist_prices)
brian = library.agents.distAgentsWIP2.C51Agent(state_size, params["action_values"], "Brian Appl 1hr md32",C=100, alternative_target = True,UCB=True,UCBc = 100,tree_horizon = 4,market_data_size=n_hist_prices)
tim = library.agents.baseAgents.TWAPAgent(3,"TWAP_APPL", len(params["action_values"]))
#print(brian.model.summary())
agents = [
    quentin
]
brian.learning_rate = 0.0001

# NOTE: Cosine basis for Isabelle results in a lot of params...

# Initialise Simulator
# BS market
'''
simple_stock = bs_stock(1,0,0.0005) # No drift, 0.0005 vol
simple_market = market(simple_stock,num_strats = len(agents))
my_simulator = library.simulations.simulator(simple_market,agents,params,test_name = "QR Testing")
'''

# real stock testing
# Retrieve data

df1 = pd.read_csv("data/2020_05_04_SPX_yFinance") # Load .csv
df1 = df1["Adj Close.3"][2:]
df2 = pd.read_csv("data/2020_05_16_SPX_yFinance") # Load .csv
df2 = df2["Adj Close.3"][2:]
df3 = pd.read_csv("data/2020_05_22_SPX_yFinance") # Load .csv
df3 = df3["Adj Close.3"][2:]
df4 = pd.read_csv("data/2020_05_28_SPX_yFinance") # Load .csv
df4 = df4["Adj Close.3"][2:]

appl_data = pd.concat([df1,df2,df3,df4])
print("Warning: dropping",sum(pd.isnull(appl_data)), "nan value(s)")
appl_data = appl_data.dropna()
appl_data = appl_data.values # Extract APPL as np array
appl_data = appl_data.astype(float) # convert any rouge strings to floats
appl_stock = real_stock(appl_data,n_steps = params["terminal"],recycle = True,n_train=10) # create stock - traded once per 6 minutes and recycled
appl_market = market(appl_stock,num_strats = len(agents),n_hist_prices = n_hist_prices)
my_simulator = library.simulations.simulator(appl_market,agents,params,test_name = "Apple Stock Testing")
my_simulator.train(10000)

# Signal market
'''
signal_stock = signal_stock(1,0.0005,0.0005,0.0005) # initial 1, vol 0.0005, signal vol 0.0005, signal reversion 0.0005
signal_market = market(signal_stock,num_strats = len(agents))
signal_simulator = library.simulations.simulator(signal_market,agents,params,test_name = "Signal Testing 1")
'''
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
    "terminal" : 1,
    "num_trades" : 10,
    "position" : 10,
    "batch_size" : 32,
    "action_values" : [0.09,0.095,0.1,0.105,0.11] #[0.05,0.075,0.09,0.1,0.11,0.15,0.2] #[0.05,0.075,0.1,0.15,0.2] 
}
state_size = 2
action_size = len(params["action_values"])
n_hist_prices = 16

# Define Agents
quentin = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "Quentin 300",C=100, alternative_target = True,UCB=True,UCBc = 300,tree_horizon = 4,market_data_size=n_hist_prices)
brian = library.agents.distAgentsWIP2.C51Agent(state_size, params["action_values"], "Brian Appl 1hr md32",C=100, alternative_target = True,UCB=True,UCBc = 100,tree_horizon = 4,market_data_size=n_hist_prices)
tim = library.agents.baseAgents.TWAPAgent(2,"TWAP_APPL", len(params["action_values"]))
#print(brian.model.summary())
agents = [
    quentin
]
quentin.learning_rate = 0.0005
quentin.reward_scaling = True

# NOTE: Cosine basis for Isabelle results in a lot of params...

# Initialise Simulator
# BS market

df = pd.read_csv("data/EURUSD-2019-01C.csv",low_memory = False, names=["Instrument", "Time", "Bid", "Ask"]) # Load .csv
df = df["Bid"]
print("Warning: dropping",sum(pd.isnull(df)), "nan value(s)")
#appl_data = appl_data.dropna()
appl_data = df.values # Extract APPL as np array
print("Using",len(appl_data),"values")
appl_data = appl_data.astype(float) # convert any rogue strings to floats
appl_stock = real_stock(appl_data,n_steps = 10,recycle = True,n_train=100) # create stock - traded once per 6 minutes and recycled
appl_market = market(appl_stock,n_hist_prices = n_hist_prices)
### Micro intervals ###
# Assume we are executing over 30s intervals, k should therefore be 120 times smaller (both vol and time / 120 but vol appears twice)

appl_market.k /= 240
my_simulator = library.simulations.simulator(appl_market,agents,params,test_name = "EURUSD Testing")
my_simulator.train(60000)

# End
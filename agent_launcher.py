# External Packages
import numpy as np

# Internal Library
from library.market_modelsM import market, bs_stock, signal_stock
from library.agents.distAgentsWIP import QRAgent, C51Agent
#from library.agents.distAgents import C51Agent
import library.simulations

# Define setup
params = {
    "terminal" : 1,
    "num_trades" : 10,
    "position" : 10,
    "batch_size" : 32,
    "action_values" : [0.05,0.075,0.1,0.15,0.2] 
}
state_size = 2
action_size = len(params["action_values"])

# Define Agents
quentin = library.agents.distAgentsWIP.QRAgent(state_size, params["action_values"], "Quentin",C=0, alternative_target = False,UCB=False,UCBc = 1,tree_horizon = 1)
brian = library.agents.distAgentsWIP.C51Agent(state_size, params["action_values"], "Brian sup0_8",C=100, alternative_target = True,UCB=True,UCBc = 100,tree_horizon = 4)
#print(brian.model.summary())
agents = [
    quentin
]

# NOTE: Cosine basis for Isabelle results in a lot of params...

# Initialise Simulator
simple_stock = bs_stock(1,0,0.0005) # No drift, 0.0005 vol
simple_market = market(simple_stock,num_strats = len(agents))
my_simulator = library.simulations.simulator(simple_market,agents,params,test_name = "QR Testing")

my_simulator.train(8000)

# Signal market
'''
signal_stock = signal_stock(1,0.0005,0.0005,0.0005) # initial 1, vol 0.0005, signal vol 0.0005, signal reversion 0.0005
signal_market = market(signal_stock,num_strats = len(agents))
signal_simulator = library.simulations.simulator(signal_market,agents,params,test_name = "Signal Testing 1")
'''
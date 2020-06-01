# External Packages
import numpy as np

# Internal Library
from library.market_modelsM import market, bs_stock
from library.agents.distAgents import IQNAgent
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
brian = library.agents.distAgents.IQNAgent(state_size, params["action_values"], "Isabelle",C=0, alternative_target = False,UCB=False,UCBc = 1,tree_horizon = 3)

agents = [
    brian
]

# Initialise Simulator
simple_stock = bs_stock(1,0,0.0005) # No drift, 0.0005 vol
simple_market = market(simple_stock,num_strats = len(agents))
my_simulator = library.simulations.simulator(simple_market,agents,params,test_name = "IQN Testing")

my_simulator.train(500)
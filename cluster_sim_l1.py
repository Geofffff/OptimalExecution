import pandas as pd
merged = pd.read_csv("cluster_data/cluster_BTX_15s_comb.csv",index_col = "time",low_memory = False)
#merged = merged.values

import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents, library.market_modelsM

# SWEEP
import random
r = random.randint(1,2)
if r == 1:
	UCBc = 200
else:
	UCBc = 100

r = random.randint(1,3)
	C = 200
elif r == 2:
	C = 100
else:
	C = 50

r = random.randint(1,2)
if r == 1:
	n_hist_data = 32
else:
	n_hist_data = 64

r = random.randint(1,2)
if r == 1:
	lr = 0.00005
else:
	lr = 0.000025


params = {
    "terminal" : 1,
    "num_trades" : 10,
    "position" : 1,
    "batch_size" : 32,
    "action_values" : [0.95,0.98,0.99,1,1.01,1.02,1.05]
}
state_size = 2
harry = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "10T10 QRDQN BTX4",C=C, N=200,alternative_target = True,UCB=True,UCBc = UCBc,tree_horizon = 4,n_hist_data=n_hist_data,n_hist_inputs=5,orderbook =False)
tim = library.agents.baseAgents.TWAPAgent(3,"BTX TWAP",11)
agent = harry

agent.learning_rate = lr

stock = library.market_modelsM.real_stock(merged,n_steps=10,n_train=80)
market = library.market_modelsM.market(stock,n_hist_data)
market.k = 0.004
market.b = 0.00004

my_simulator = library.simulations2.simulator(market,agent,params,test_name = "MOMD2",orderbook = False)
my_simulator.train(70000,epsilon_decay =0.9999)
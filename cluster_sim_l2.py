import pandas as pd
merged = pd.read_csv("cluster_data/cluster_BTX_5s_comb.csv",index_col = "time",low_memory = False)
#merged = merged.values

import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents, library.market_modelsM

# SWEEP
import random
r = 2#random.randint(1,2)
if r == 1:
	UCBc = 100
else:
	UCBc = 200


C = 50

r = random.randint(1,3)
if r == 1:
	n_hist_data = 32
elif r == 2:
	n_hist_data = 64
elif r == 3:
	n_hist_data = 128

r = 1#random.randint(1,2)
if r == 1:
	lr = 0.00005
else:
	lr = 0.000025

n_steps = 100
params = {
    "terminal" : 1,
    "num_trades" : n_steps,
    "position" : 10000,
    "batch_size" : 64,
    "action_values" : [[0.5,5],[1,0],[2,0],
                       [0.25,2],[0.5,2],[1,1],
                       [0,0.5],[0,1],[0,2],
                       [1,4],[1,3]]
}



state_size = 3
harry = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], f"{n_steps}T{n_steps} QRDQN BTX LO",C=C, N=200,alternative_target = True,UCB=True,UCBc = UCBc,tree_horizon = n_steps,n_hist_data=n_hist_data,n_hist_inputs=7,orderbook =True)
tim = library.agents.baseAgents.TWAPAgent(1,"TWAP",11)
agent = harry

stock = library.market_modelsM.real_stock_lob(merged,n_steps=n_steps,n_train=20)
market = library.market_modelsM.lob_market(stock,n_hist_data)

agent.learning_rate = lr

agent.expected_range = 0.002
agent.expected_mean = 0.99

market.k = 0.005 / 10000#params["position"]**2
#market.b = 0.005

my_simulator = library.simulations2.simulator(market,agent,params,test_name = "MOMD2",orderbook = True)
my_simulator.train(10000,epsilon_decay =0.9999)



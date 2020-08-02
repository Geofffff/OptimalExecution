import pandas as pd
merged = pd.read_csv("cluster_data/cluster_BTX_15s_19.csv",index_col = "time",low_memory = False)

import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents, library.market_modelsM

n_hist_data = 32

params = {
    "terminal" : 1,
    "num_trades" : 50,
    "position" : 1,
    "batch_size" : 64,
    "action_values" : [[0.99,0],[1,0],[1.01,0]]
}
state_size = 3
harry = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "50T50 QRDQN MD",C=50, N=200,alternative_target = True,UCB=True,UCBc = 150,tree_horizon = 4,n_hist_data=n_hist_data,n_hist_inputs=7,orderbook =True)
tim = library.agents.baseAgents.TWAPAgent(1,"50T50 TWAP",11)
agent = harry

agent.learning_rate = 0.000025

stock = library.market_modelsM.real_stock_lob(merged,n_steps=50,n_train=20)
market = library.market_modelsM.lob_market(stock,n_hist_data)
market.k = 0.004
market.b = 0.00004

my_simulator = library.simulations2.simulator(market,agent,params,test_name = "MO MD Testing",orderbook = True)
my_simulator.train(70000,epsilon_decay =0.9999)
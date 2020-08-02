import pandas as pd
merged = pd.read_csv("cluster_data/cluster_BTX_15s_19.csv",index_col = "time",low_memory = False)

import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents, library.market_modelsM

params = {
    "terminal" : 1,
    "num_trades" : 10,
    "position" : 1,
    "batch_size" : 64,
    "action_values" : [[0.95,0],[0.96,0],[0.97,0],[0.98,0],[0.99,0],[1,0],[1.01,0],[1.02,0],[1.03,0],[1.04,0],[1.05,0]]
}
state_size = 3
harry = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "T QRDQN MD2",C=50, alternative_target = True,UCB=True,UCBc = 150,tree_horizon = 4,n_hist_data=32,n_hist_inputs=7,orderbook =True)
tim = library.agents.baseAgents.TWAPAgent(5,"TWAP",11)
agent = harry

agent.learning_rate = 0.00005

stock = library.market_modelsM.real_stock_lob(merged,n_steps=10,n_train=300)
market = library.market_modelsM.lob_market(stock,32)
market.k = 0.02
market.b = 0.005

my_simulator = library.simulations2.simulator(market,agent,params,test_name = "MO MD Testing",orderbook = True)
my_simulator.train(70000,epsilon_decay =0.9999)
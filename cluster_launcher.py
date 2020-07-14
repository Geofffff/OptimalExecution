import pandas as pd
import numpy as np 
import library.market_modelsM
import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents


merged = pd.read_csv("cluster_data/cluster_BTX_19.csv",index_col = "time",low_memory = False)


params = {
    "terminal" : 1,
    "num_trades" : 30,
    "position" : 10000,
    "batch_size" : 32,
    "action_values" : [[0.5,0],[1,0],[2,0],
                       [0.25,2],[0.5,2],[1,1],
                       [0,0.5],[0,1],[0,2],
                       [0,3],[1,3]]
}
state_size = 7
harry = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "CX Harry11 U 10p10",C=100, alternative_target = True,UCB=True,UCBc = 50,tree_horizon = 15,orderbook =True)#,market_data_size=n_hist_prices)
#tim = library.agents.baseAgents.TWAPAgent(1,"TWAP",9)
agent = harry


stock = library.market_modelsM.real_stock_lob(merged,n_steps=30,n_train=300)
market = library.market_modelsM.lob_market(stock,0)
market.k *= 10

my_simulator = library.simulations2.simulator(market,agent,params,test_name = "LOB 10s Testing",orderbook = True)
my_simulator.train(80000,epsilon_decay =0.9999)

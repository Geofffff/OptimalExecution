import pandas as pd
merged = pd.read_csv("cluster_data/cluster_EURUSD_15s_Feb.csv",index_col = "time",low_memory = False)
#merged = merged.values
from imp import reload
import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents, library.market_modelsM
reload(library.agents.distAgentsWIP2)
reload(library.simulations2)
reload(library.market_modelsM)

n_hist_data = 32

params = {
    "terminal" : 1,
    "num_trades" : 50,
    "position" : 1,
    "batch_size" : 64,
    "action_values" : [0.98,0.99,1,1.01,1.02]
}
state_size = 2
harry = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "50T50 QRDQN FX MA",C=50, N=200,alternative_target = True,UCB=True,UCBc = 100,tree_horizon = 4,n_hist_data=n_hist_data,n_hist_inputs=1,orderbook =False)
tim = library.agents.baseAgents.TWAPAgent(1,"50T50 TWAP",11)
agent = tim

agent.learning_rate = 0.000025

stock = library.market_modelsM.real_stock(merged,n_steps=50,n_train=50)
market = library.market_modelsM.market(stock,n_hist_data)
market.k = 0.004
market.b = 0.00004

my_simulator = library.simulations2.simulator(market,agent,params,test_name = "MOMD2",orderbook = False)
my_simulator.train(70000,epsilon_decay =0.9999)
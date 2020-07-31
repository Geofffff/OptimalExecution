import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents, library.market_modelsM, library.agents.valueAgents
lr = 0.0001
ucbc = 150
th = 4
tl = 50
N = 200

params = {
    "terminal" : 1,
    "num_trades" : 10,
    "position" : 1,
    "batch_size" : 32,
    "action_values" : [0.5,0.75,1,1.25,1.5,2]
                        # [0.001,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35]
}
state_size = 2
harry1 = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "Sim QR LK VSlr",C=tl,N=N, alternative_target = True,UCB=True,UCBc = ucbc,tree_horizon = th,n_hist_data=0,n_hist_inputs=0,orderbook =False)#,market_data_size=n_hist_prices)
alice = library.agents.valueAgents.DDQNAgent(state_size, len(params["action_values"]), "Sim DDQN Mod",C=tl, alternative_target = True,tree_horizon=th)
tim = library.agents.baseAgents.TWAPAgent(2,"TWAP Test",21)
agent = alice
#agent.learning_rate = lr


simple_stock = library.market_modelsM.bs_stock(1,0,0.0005,n_steps = 10) # No drift, 0.0005 vol
simple_market = library.market_modelsM.market(simple_stock)
simple_market.k = 0.0186 # 0.02
simple_market.b = 0.0 # 0.01

my_simulator = library.simulations2.simulator(simple_market,agent,params,test_name = "DDQNTesting",orderbook = False)
my_simulator.train(100000,epsilon_decay =0.9999)
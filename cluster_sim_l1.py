import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents, library.market_modelsM

lr = 0.0002
ucbc = 80
th = 4
tl = 50
N = 51

params = {
    "terminal" : 1,
    "num_trades" : 10,
    "position" : 1,
    "batch_size" : 32,
    "action_values" : [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]
                        # [0.001,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35]
}
state_size = 2
harry = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], f"10s lr {lr} UCBc {ucbc} tl {tl} th {th} N{N}",C=tl, alternative_target = True,UCB=True,UCBc = ucbc,tree_horizon = th,n_hist_data=0,n_hist_inputs=0,orderbook =False)#,market_data_size=n_hist_prices)
tim = library.agents.baseAgents.TWAPAgent(7,"TWAP",21)
agent = harry
agent.learning_rate = lr
agent.N = N

simple_stock = library.market_modelsM.bs_stock(1,0,0.0005,n_steps = 10) # No drift, 0.0005 vol
simple_market = library.market_modelsM.market(simple_stock)
simple_market.k = 0.0025

my_simulator = library.simulations2.simulator(simple_market,agent,params,test_name = "Test3",orderbook = False)
my_simulator.train(20000,epsilon_decay =0.9999)
import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents, library.market_modelsM


params = {
    "terminal" : 1,
    "num_trades" : 10,
    "position" : 1,
    "batch_size" : 32,
    "action_values" : [0.2,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8]
                        # [0.001,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35]
}
state_size = 2
harry = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "Sim QRDQN",C=20, alternative_target = True,UCB=True,UCBc = 40,tree_horizon = 3,n_hist_data=0,n_hist_inputs=0,orderbook =False)#,market_data_size=n_hist_prices)
tim = library.agents.baseAgents.TWAPAgent(10,"TWAP",21)
agent = harry

#agent.learning_rate = 0.00025

simple_stock = library.market_modelsM.bs_stock(1,0,0.0005,n_steps = 240) # No drift, 0.0005 vol
simple_market = library.market_modelsM.market(simple_stock)
simple_market.k *= 100000

my_simulator = library.simulations2.simulator(simple_market,agent,params,test_name = "Simulted Results",orderbook = False)
my_simulator.train(20000,epsilon_decay =0.9999)
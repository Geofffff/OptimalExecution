
import library.agents.distAgentsWIP2, library.simulations2, library.agents.baseAgents, library.market_modelsM, library.agents.valueAgents

lr = 0.00005
ucbc = 100
th = 4
tl = 50
N = 200

params = {
    "terminal" : 1,
    "num_trades" : 10,
    "position" : 1,
    "batch_size" : 32,
    "action_values" : [0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]
                        # [0.001,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35]
}
state_size = 2
harry = library.agents.distAgentsWIP2.QRAgent(state_size, params["action_values"], "Sim QR Mod2",C=tl,N=N, alternative_target = True,UCB=True,UCBc = ucbc,tree_horizon = th,n_hist_data=0,n_hist_inputs=0,orderbook =False)#,market_data_size=n_hist_prices)
alice = library.agents.valueAgents.DDQNAgent(state_size, len(params["action_values"]), "Sim DDQN Mod2",C=tl, alternative_target = True,tree_horizon=th)
tim = library.agents.baseAgents.TWAPAgent(5,"TWAP Test",21)
agent = harry
#agent.learning_rate = lr


simple_stock = library.market_modelsM.bs_stock(1,0,0.0017,n_steps = 10) # No drift, 0.0005 vol
simple_market = library.market_modelsM.market(simple_stock)
simple_market.k = 0.0186 # 0.02
simple_market.b = 0.0 # 0.01

my_simulator = library.simulations2.simulator(simple_market,agent,params,test_name = "QRDQN RESULTS",orderbook = False)
my_simulator.train(10,epsilon_decay =0.9999)

agent.model.save(os.path.join(wandb.run.dir, "model.h5"))
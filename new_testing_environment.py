import os # for creating directories
import matplotlib
matplotlib.use('TkAgg')
import numpy as np

from library.market_modelsM import market
#from library.local_environments import agent_environmentM
from library.market_modelsM import bs_stock
import library.agents
import library.simulations
import library.WIPAgents
import library.WIPAgentsv2


params = {
    "terminal" : 1,
    "num_trades" : 20,
    "position" : 10,
    "batch_size" : 32
}

state_size = 2
action_size = 7 # This is NOT dynamic (and probably should be)

fred = library.agents.DQNAgent(state_size, action_size,"Fred") # initialise agent
greg = library.agents.DQNAgent(state_size, action_size,"Greg15",C=15) # Second agent
greta = library.agents.DQNAgent(state_size, action_size,"Greta4",C=4,alternative_target = True) # Second agent

alice = library.agents.DDQNAgent(state_size, action_size,"Alice")
alice2 = library.agents.DDQNAgent(state_size, action_size,"Alice2")

tim = library.agents.TWAPAgent(5,"Tim")
rob = library.agents.randomAgent(4,"Rob",action_size = action_size)

amanda = library.agents.DDQNAgent(state_size, action_size,"Amanda30",C=30) # Second agent
agnes = library.agents.DDQNAgent(state_size, action_size,"Agnes10",C=10,alternative_target = True) # Second agent

david30 = library.WIPAgents.distAgent("David30",C=30)
daisy = library.WIPAgentsv2.distAgent("Daisy")

agents = [
    daisy,
    tim,
    rob
]
#alice.learning_rate = 0.01

simple_stock = bs_stock(1,0,0.001) # No drift, 0.5 vol
simple_market = market(simple_stock,num_strats = len(agents))

#epsilon_decays = [0.9992,0.999,1,1]

my_simulator = library.simulations.simulator(simple_market,agents,params)
my_simulator.plot_title = "Linear temporary impact alpha test with DDQN dist Agent w/ target"


# Training
my_simulator.train(n_episodes = 1000)

from matplotlib import pyplot as plt
bar_agent = daisy
#plt.plot(my_simulator.eval_rewards_mean[:,1],label = "act " + str(1))
#print(my_simulator.eval_rewards_mean)
#print(my_simulator.env.reset())
#episode_actions = np.zeros((7,3))
#print(my_simulator.env.step([1,5,6,6]))
#state = my_simulator.env.reset()[0]
#print(my_simulator.env.cash)
state = [1,-1] 
state = np.reshape(state, [1, 2])
state1 = [0,0] 
state1 = np.reshape(state1, [1, 2])
predictions = bar_agent.predict(state)
#episode_actions[:,0] = predictions
#print(predictions)
plt.bar(bar_agent.z,bar_agent.probs(state)[4][0])
#print(bar_agent.probs(state))
#print(david30.predict(state))
print(state,daisy.predict(state))
print(state1,daisy.predict(state1))
#state = np.reshape(np.append(state,1), [1, 3])
#print(state)
#print(daisy.model.get_weights())
#print("state probs", daisy.probs(state))
print("pred probs", daisy.model.predict(state1)[1]-daisy.model.predict(state)[1])


print(my_simulator.eval_rewards_mean.shape)
twap_stat = np.mean(my_simulator.eval_rewards_mean[:,1])
plt.plot([0, len(my_simulator.train_actions[:,0])], [twap_stat, twap_stat], 'k--')

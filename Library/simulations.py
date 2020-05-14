import time
from library.local_environments import agent_environmentM
import numpy as np
from matplotlib import pyplot as plt

class simulator:

	def __init__(self,market_,agents, params = None):
		
		# Default params
		if params is None:
			params = params = {"terminal" : 1, "num_trades" : 50, "position" : 10, "batch_size" : 32 }
			print("Initialising using default parameters")

		self.terminal = params["terminal"]
		self.num_steps = params["num_trades"]
		self.batch_size = params["batch_size"]
		self.agents = agents
		self.n_agents = len(self.agents)

		self.m = market_
		self.env = agent_environmentM(self.m,
									 params["position"],
									 params["num_trades"],
									 params["terminal"],
									 [0,0.001,0.005,0.01,0.02,0.05,0.1],
									 self.n_agents
									)
		
		

		# Stats
		self.final_timestep = []
		self.train_rewards = np.zeros((0,self.n_agents))
		self.eval_rewards = np.zeros((0,self.n_agents))
		
		

	def __moving_average(self,a, n=300):
		ret = np.cumsum(a, dtype=float)
		ret[n:] = ret[n:] - ret[:-n]
		return ret[n - 1:] / n



	def train(self,n_episodes = 10000, epsilon = None, epsilon_decay = None,show_details = True, evaluate = False):
		# TODO: different training parameters
		
		# Number of agents to be trained
		
		### Live Plots ###
		fig = plt.figure()
		ax = fig.add_subplot(111)
		plt.ion()

		fig.show()
		fig.canvas.draw()
		### Live Plots ###
		
		# Default training parameters if not provided
		if epsilon is None:
			epsilon = [1] * self.n_agents
			
		if epsilon_decay is None:
			epsilon_decay = [0.998] * self.n_agents
		
		if evaluate:
			train = [False] * self.n_agents
		else:
			train = [True] * self.n_agents
			
		# TEMPORARY? #
		self.code_time_i = np.zeros((0,2))
		self.code_time_o = np.zeros((0,3))
		# TEMPORARY? #

		# Evaluatory Stats
		current_training_step = len(self.train_rewards) #
		
		n_correct = 0#
		total_reward = np.zeros(self.n_agents)#
		
		pcnt_opt = []#
		
		

		

		# Set up the agents:
		for i ,agent in enumerate(self.agents):
			agent.update_paramaters(epsilon = epsilon[i], epsilon_decay = epsilon_decay[i])
			##### TEMP ######
			print(agent.epsilon_decay)
			
		# Setup action list
		actions = [-1] * self.n_agents
		
		timer_o = []
		
		for e in range(n_episodes): # iterate over new episodes of the game
			
			#Time code
			timer = []
			timer_o = []
			start_time_o = time.time()
			
			states = self.env.reset() # reset state at start of each new episode of the game
			states = np.reshape(states, [self.n_agents,1, self.env.state_size])
			
			correct_action = 0.0
			
			done = np.zeros(self.n_agents) # Has the episode finished
			inactive = np.zeros(self.n_agents) # Agents which are still trading
			
			total_reward.fill(0)
			
			time_now = time.time()
			timer_o.append(time_now - start_time_o)
			start_time_o = time.time()
			
			
			for t in range(self.num_steps):
				timer = []
				start_time = time.time()
				# Get actions for each agent
				for i, agent in enumerate(self.agents):
					# Agents action only updated if still active
					if not inactive[i]:
						actions[i] = agent.act(states[i])
				
				next_states, rewards, done = self.env.step(actions)
				
				time_now = time.time()
				timer.append(time_now - start_time)
				start_time = time_now
				
				rewards = (1 - done) * rewards
				
				next_states = np.reshape(next_states, [self.n_agents,1, self.env.state_size])
				total_reward += rewards
				#print(total_reward)
				for i, agent in enumerate(self.agents):
					if not inactive[i] and train[i]:
						agent.remember(states[i], actions[i], rewards[i], next_states[i], done[i])
				
				time_now = time.time()
				timer.append(time_now - start_time)
				#if e % 100 == 0:
					#print("time", t, "Actions ", actions[0], "Rewards ", rewards[0], states[0],next_states[0])

				states = next_states
				
				if actions[0] == 4:
					correct_action += 1
					
					
				if all(done): 
					percent_optimal = correct_action / self.num_steps
					pcnt_opt.append(percent_optimal)
					break # exit loop
					
				inactive = inactive + done
				
				self.code_time_i = np.vstack((self.code_time_i,timer))   

			if not all(done):
				print("We have a problem.")
				
			
			time_now = time.time()
			timer_o.append(time_now - start_time_o)
			start_time_o = time.time()
			
			if evaluate:
				self.eval_rewards = np.vstack((self.eval_rewards,total_reward))
			else:
				self.train_rewards = np.vstack((self.train_rewards,total_reward))
			
			for i, agent in enumerate(self.agents):
				if len(agent.memory) > self.batch_size and train[i]:
					agent.replay(self.batch_size) # train the agent by replaying the experiences of the episode

			time_now = time.time()
			timer_o.append(time_now - start_time_o)
			self.code_time_o = np.vstack((self.code_time_o,timer_o)) 
			if e % 100 == 0:
				#self.total_training_steps += 100
				if show_details and not evaluate:
					ax.clear()
					for i in range(self.train_rewards.shape[1]):
						ax.plot(self.__moving_average(self.train_rewards[current_training_step:,i],n=500), label  = self.agents[i].agent_name)
					plt.pause(0.0001)
					plt.draw()
		if not evaluate:
			self.show_stats(trained_from = current_training_step)       
		
	def evaluate(self,n_episodes = 500):
		start_iteration = len(self.eval_rewards)
		epsilon = [0] * self.n_agents
		self.train(n_episodes = n_episodes, epsilon = epsilon, show_details = False,evaluate = True)
		self.show_stats(trained_from = start_iteration,training = False)

	def show_stats(self,trained_from = 0,trained_to = None,moving_average = 400,training = True):
		
		if training:
			if trained_to is None:
				trained_to = len(self.train_rewards)
			for i in range(self.train_rewards.shape[1]):
				plt.plot(self.__moving_average(self.train_rewards[trained_from:trained_to,i],n=moving_average), label  = self.agents[i].agent_name)
		else:
			if trained_to is None:
				trained_to = len(self.eval_rewards)
			for i in range(self.eval_rewards.shape[1]):
				plt.plot(self.__moving_average(self.eval_rewards[trained_from:trained_to,i],n=moving_average), label  = self.agents[i].agent_name)
		plt.legend()
		
		
	#def test_convergence(self,)

	def execute(self,agent):
		# Currently just one strat
		position = []
		cash = []
		states = self.env.reset() # reset state at start of each new episode of the game
		states = np.reshape(states, [len(training_agents),1, self.env.state_size])
			
		for t in range(self.num_steps):

			action = agent.act(states)
			next_state, reward, done = self.env.step(action)
			next_states = np.reshape(next_states, [len(training_agents),1, self.env.state_size])
			total_reward += rewards
			#print(total_reward)
			for i, agent in enumerate(training_agents):
				# Note this happens when its been done for more than one step
				training_agents[agent].remember(states[i], actions[i], rewards[i], next_states[i], done[i])
			states = next_states

			if all(done): 
				break 
		
		
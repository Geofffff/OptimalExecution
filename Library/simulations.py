import time
from library.local_environments import agent_environmentM
import numpy as np
from matplotlib import pyplot as plt
import random
import wandb
class simulator:

	def __init__(self,market_,agents, params = None, test_name = 'undefined'):
		
		# Default params
		if params is None:
			params = params = {"terminal" : 1, "num_trades" : 50, "position" : 10, "batch_size" : 32,"action_values" : [0,0.001,0.005,0.01,0.02,0.05,0.1] }
			print("Initialising using default parameters")

		self.terminal = params["terminal"]
		self.num_steps = params["num_trades"]
		self.batch_size = params["batch_size"]
		self.agents = agents
		self.n_agents = len(self.agents)

		self.m = market_
		self.possible_actions = params["action_values"]#[0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]
		self.env = agent_environmentM(self.m,
									 params["position"],
									 params["num_trades"],
									 params["terminal"],
									 self.possible_actions,
									 self.n_agents
									)
		
		
		self.intensive_training = False
		self.test_name = test_name

		# Stats
		self.final_timestep = [] # Inactive
		self.train_rewards = np.zeros((0,self.n_agents))
		self.eval_rewards = np.zeros((1,self.n_agents)) 
		self.eval_rewards_mean = np.zeros((0,self.n_agents)) 
		self.eval_window = 40
		self.plot_title = "Unlabelled Performance Test"

		# Record actions
		self.train_actions = np.zeros((0,len(self.possible_actions),self.n_agents))
		self.episode_actions = np.zeros((len(self.possible_actions),self.n_agents))
		self.record_frequency = 200
		self.plot_y_lim = (0.96,0.99)
		self.episode = 0

		# Wandb record parameters
		self.wandb_agents = []
		for agent in self.agents:
			new_run = wandb.init(project="OptEx",name = agent.agent_name,reinit=True)
			new_run.group = self.test_name
			new_run.config.update({"num_trades": self.num_steps,
			 "batch_size": self.batch_size,
			 "action_size": len(self.possible_actions),
			 "target_lag": agent.C,
			 "alt_target": agent.alternative_target,
			 "tree_horizon": agent.tree_n
			 })
			if agent.UCB:
				new_run.config.UCBc = agent.UCBc
			else:
				new_run.config.epsilon_min = agent.epsilon_min
				new_run.config.epsilon_decay = agent.epsilon_decay

			# Agent specifics
			self.wandb_agents.append(new_run)


	def _moving_average(self,a, n=300):
		ret = np.cumsum(a, dtype=float)
		ret[n:] = ret[n:] - ret[:-n]
		return ret[n - 1:] / n

	def pretrain(self,n_samples = 2000,n_iterations = 500):
		pretain_position = True
		pretrain_time = False
		for i in range(n_samples):
			## Pretrain for state where position is 0 ##
			
			# Randomly sample transformed t in the time interval [-1,1] and action from space
			if pretain_position:
				t = random.uniform(-1,1)
				a = random.randrange(len(self.possible_actions))
				state = [-1,t]
				next_time = max(1,t + 2 / self.num_steps)
				next_state = [-1,next_time]
				state = np.reshape(state, [1, self.env.state_size])
				next_state = np.reshape(next_state, [1, self.env.state_size])
				for agent in self.agents:
					agent.remember(state, a, 0, next_state, True)

			## Pretrain for state where time is 0 ##
			if pretrain_time:
				# Randomly sample transformed position in the time interval [-1,1] and action from space
				p = random.uniform(-1,1)
				a = random.randrange(len(self.possible_actions))
				state = [p,1]
				state = np.reshape(state, [1, self.env.state_size])
				for agent in self.agents:
					agent.remember(state, a, 0, state, True)

		for i in range(n_iterations):
			for agent in self.agents:
				agent.replay(self.batch_size)

		# Clear the memory
		for agent in self.agents:
			agent.memory.clear()
		print("Pretraining Complete")
			







	def train(self,n_episodes = 10000, epsilon = None, epsilon_decay = None,show_details = True, evaluate = False):
		# TODO: different training parameters
		
		# Number of agents to be trained
		
		### Live Plots ###
		if not evaluate:
			fig = plt.figure(0)
			ax = fig.add_subplot(111)
			plt.ion()
			ax.grid(b=True, which='major', axis='both')
			fig.show()
			fig.canvas.draw()
		### Live Plots ###
		
		# Default training parameters if not provided
		if epsilon is None:
			epsilon = [1] * self.n_agents
			
		if epsilon_decay is None:
			epsilon_decay = [0.998] * self.n_agents

		# Evaluatory Stats
		#current_training_step = len(self.eval_rewards) # CHANGED TO EVAL
		
		# Set up the agents:
		for i ,agent in enumerate(self.agents):
			agent.update_paramaters(epsilon = epsilon[i], epsilon_decay = epsilon_decay[i])
			
		# Setup action list
		actions = [-1] * self.n_agents

		agent_reward_dists = []
		
		for e in range(n_episodes): # iterate over new episodes of the game
			
			### Record Probabilities 4 times throughout training ###
			# Currently only record for the first dist agent
			if e % n_episodes / 5 == 0 and e>0:
				for agent in self.agents:
					if agent.agent_type == "dist":
						self.dist_agent_for_plot = agent
						init_state = [1,-1] 
						init_state = np.reshape(init_, [1, 2])
						agent_reward_dists.append(agent.probs(init_state))
						break

			# Record the initial action values if training
			#self.episode_actions.fill(0)
			
			self.episode(actions = actions, evaluate = evaluate)
			if not self.intensive_training:
				for i, agent in enumerate(self.agents):
					if len(agent.memory) > self.batch_size and not evaluate:
						agent.replay(self.batch_size) # train the agent by replaying the experiences of the episode
						agent.step() # Update target network if required


			if e % self.record_frequency == 0 and e>0:
				#self.total_training_steps += 100
				if show_details and not evaluate:
					current_training_step = len(self.eval_rewards_mean) # CHANGED TO EVAL
					self.evaluate(self.eval_window,show_stats=False)
					#ax.clear()
					#print(self.eval_rewards_mean)
					
					for i in range(self.eval_rewards_mean.shape[1]):
						self.wandb_agents[i].log({'episode': self.eval_rewards_mean.shape[0] * self.record_frequency, 'eval_rewards': self.eval_rewards_mean[-1,i]})
						#y_vals = self._moving_average(self.eval_rewards_mean[:,i],n=3)
						#x_vals = np.arange(len(y_vals)) * self.record_frequency
						#agent_label = self.agents[i].agent_name + "(" + str(round(self.agents[i].epsilon,3)) + ")"
						#ax.plot(x_vals,y_vals, label  = agent_label )
					#plt.legend()
					#plt.ylim(self.plot_y_lim) # Temporary
					#plt.title(self.plot_title)
					#plt.grid(b=True, which='major', axis='both')
					### TEMPORARY ###
					#twap_stat = 9.849
					#if len(x_vals) > 0:
						#plt.plot([0, x_vals[-1]], [twap_stat, twap_stat], 'k--')
					### TEMPORARY ###
					#plt.pause(0.0001)
					#plt.draw()
		if not evaluate:
			self.show_stats(trained_from = current_training_step) 
			for i, d in enumerate(agent_reward_dists):
				self.show_dist(self.dist_agent_for_plot,d,figure = i + 1)
		else:
			self.eval_rewards_mean = np.vstack((self.eval_rewards_mean,self.eval_rewards / self.eval_window))
			self.eval_rewards = np.zeros((1,self.n_agents))
		for run in wandb_agents:
			run.join()

	def episode(self,actions, verbose = False,evaluate = False):
		states = self.env.reset() # reset state at start of each new episode of the game
		states = np.reshape(states, [self.n_agents,1, self.env.state_size])

		if not evaluate:
			for i, agent in enumerate(self.agents):
				self.episode += 1
				self.wandb_agents[i].log({'episode': self.episode, 'action_values': agent.predict(states[i])})
			self.train_actions = np.concatenate((self.train_actions,[self.episode_actions]))
					
		done = np.zeros(self.n_agents) # Has the episode finished
		inactive = np.zeros(self.n_agents) # Agents which are still trading
					
		total_reward = np.zeros(self.n_agents)

		for t in range(self.num_steps):
			timer = []
			start_time = time.time()
			# Get actions for each agent
			for i, agent in enumerate(self.agents):
				# Agents action only updated if still active
				if not inactive[i]:
					actions[i] = agent.act(states[i])
				else:
					actions[i] = -1 # Could speed up (only need to change once)
			
			next_states, rewards, done = self.env.step(actions)
			
			#rewards = (1 - done) * rewards
			
			next_states = np.reshape(next_states, [self.n_agents,1, self.env.state_size])
			total_reward += rewards
			#print(total_reward)
			if not evaluate:
				for i, agent in enumerate(self.agents):
					if not inactive[i]:
						agent.remember(states[i], actions[i], rewards[i], next_states[i], done[i])

			if verbose:
				print("Agent 0 predict", self.agents[0].predict(states[0]))
				print("State[0]: ",states[0], "Actions[0]: ", actions[0], "Rewards[0]: ", rewards[0], "Next_states[0]: ", next_states[0], "Done[0]: ", done[0])
				print("Agent 0 next predict", self.agents[0].predict(next_states[0]))
			states = next_states
				
			if all(done): 
				break # exit loop
				
			inactive = inactive + done

			if self.intensive_training:
				for i, agent in enumerate(self.agents):
					if len(agent.memory) > self.batch_size and not evaluate:
						agent.replay(self.batch_size) # train the agent by replaying the experiences of the episode
						agent.step() # Update target network if required

		if not all(done):
			print("We have a problem.")
		
		if evaluate:
			self.eval_rewards += total_reward

	def evaluate(self,n_episodes = 200,show_stats = True):
		epsilon_old = []
		epsilon_decay_old = []
		# Get current epsilon values
		for agent in self.agents:
			epsilon_old.append(agent.epsilon)
			epsilon_decay_old.append(agent.epsilon_decay)

		epsilon = [0] * self.n_agents
		self.train(n_episodes = n_episodes, epsilon = epsilon, show_details = False,evaluate = True)
		# Return agent epsilons to their original values:
		for i, agent in enumerate(self.agents):
			agent.update_paramaters(epsilon = epsilon_old[i],epsilon_decay = epsilon_decay_old[i])

		if show_stats:
			start_iteration = len(self.eval_rewards)
			self.show_stats(trained_from = start_iteration,training = False)

	def show_stats(self,trained_from = 0,trained_to = None,moving_average = 400,training = True):
		
		if training:
			if trained_to is None:
				trained_to = len(self.train_rewards)
			for i in range(self.train_rewards.shape[1]):
				plt.plot(self._moving_average(self.train_rewards[trained_from:trained_to,i],n=moving_average), label  = self.agents[i].agent_name)
		else:
			if trained_to is None:
				trained_to = len(self.eval_rewards)
			for i in range(self.eval_rewards.shape[1]):
				plt.plot(self._moving_average(self.eval_rewards[trained_from:trained_to,i],n=moving_average), label  = self.agents[i].agent_name)
		plt.legend()

	def show_dist(self, dist_agent, data,figure = 1, actions = [5,6]):
		plt.figure(figure)
		for a in actions:
			plt.bar(dist_agent.z,data[a][0],alpha = 0.4,width = 0.25,label = f"action {bar_act}")
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
		
		
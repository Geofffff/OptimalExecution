import time
from library.local_environments import agent_environment, orderbook_environment
import numpy as np
from matplotlib import pyplot as plt
import random
import wandb


class simulator:

	def __init__(self,market_,agent, params = None, test_name = 'undefined',orderbook = False):
		
		# Default params
		if params is None:
			params = params = {"terminal" : 1, "num_trades" : 50, "position" : 10, "batch_size" : 32,"action_values" : [0,0.001,0.005,0.01,0.02,0.05,0.1] }
			print("Initialising using default parameters")

		self.terminal = params["terminal"]
		self.num_steps = params["num_trades"]
		self.batch_size = params["batch_size"]
		self.agent = agent
		self.orderbook = orderbook

		self.m = market_
		self.possible_actions = params["action_values"]
		if orderbook:
			self.env = orderbook_environment(self.m,
								 params["position"],
								 params["num_trades"],
								 self.possible_actions
								)
		else:
			self.env = agent_environment(self.m,
								 params["position"],
								 params["num_trades"],
								 self.possible_actions
								)
		self.trade_freq = self.m.stock.n_steps / self.num_steps
		
		# TAG: Depreciate?
		self.intensive_training = False

		self.test_name = test_name

		self.eval_freq = 500
		self.train_stat_freq = 100
		self.eval_window = 25
		self.episode_n = 0 # Number of training episodes completed

		self.logging_options = set(["count","value","position","event","reward","lo","lotime"])
		# Try replacing below with logging matplotlib
		#self.position_granularity = 4

		# Wandb initialise and congigure
		self.plot_position = False
		self.new_run = wandb.init(project="OptEx",name = self.agent.agent_name,group = self.test_name,reinit=True)
		self.new_run.config.update({"num_trades": self.num_steps,
		 "batch_size": self.batch_size,
		 "action_size": len(self.possible_actions),
		 "state_size": self.env.state_size,
		 "temp_impact": self.env.m.k,
		 "perm_impact": self.env.m.b,
		 "stock": type(self.env.m.stock).__name__,
		 "orderbook": orderbook
		 })
		if type(self.env.m.stock).__name__ == "bs_stock":
			self.new_run.config.update({"stock_vol": self.env.m.stock.vol})
		if self.agent.agent_type != "basic":
			self.new_run.config.update({"target_lag": self.agent.C,
			 "alt_target": self.agent.alternative_target,
		 	 "tree_horizon": self.agent.tree_n,
		 	 "buffer_size": self.agent.replay_buffer_size,
		 	 "learning_rate": self.agent.learning_rate,
		 	 "reward_scaling": self.agent.reward_scaling,
		 	 "action_input" : self.agent.action_as_input
		 	})
			
		if self.agent.agent_type == "dist":
			self.new_run.config.update({"twap_scaling": self.agent.twap_scaling})
			if type(agent).__name__ == "C51Agent":
				self.new_run.config.update({"support_range": self.agent.V_max - self.agent.V_min})
			if type(agent).__name__ == "QRAgent":
				self.new_run.config.update({"n_quantiles": self.agent.N})
				self.new_run.config.update({"UCB_optimistic": self.agent.optimisticUCB})
			if self.agent.UCB:
				self.new_run.config.update({"UCBc": self.agent.c})
			else:
				self.new_run.config.update({"epsilon_min": self.agent.epsilon_min})
				self.new_run.config.update({"epsilon_decay": self.agent.epsilon_decay})
		else:
			self.new_run.config.update({"epsilon_min": self.agent.epsilon_min})
			self.new_run.config.update({"epsilon_decay": self.agent.epsilon_decay})

		if self.agent.agent_type == "DQN":
			self.new_run.config.update({"model_layers": self.agent.model_layers,
										"model_units": self.agent.model_units
									})

	def __str__(self):
		return f"{type(agent).__name__} exiting position {self.env.initial_position} over period of {self.m.stock.n_steps} seconds, changing trading rate every {self.trade_freq} seconds."
	@staticmethod
	def _moving_average(a, n=300):
		ret = np.cumsum(a, dtype=float)
		ret[n:] = ret[n:] - ret[:-n]
		return ret[n - 1:] / n

	def _pretrain_position(self):
		t = random.uniform(-1,1)
		a = random.randrange(len(self.possible_actions))
		state = [-1,t]
		next_time = max(1,t + 2 / self.num_steps)
		next_state = [-1,next_time]
		state = np.reshape(state, [1, self.env.state_size])
		next_state = np.reshape(next_state, [1, self.env.state_size])
		self.agent.remember(state, a, 0, next_state, True)

	# TAG: overhaul
	def pretrain(self,n_samples = 2000,n_iterations = 500):
		raise "This function has not been updated for version 2"
		pretain_position = True
		pretrain_time = False
		for i in range(n_samples):
			## Pretrain for state where position is 0 ##
			
			# Randomly sample transformed t in the time interval [-1,1] and action from space
			if pretain_position:
				self._pretrain_position()

			## Pretrain for state where time is 0 ##
			if pretrain_time:
				# Randomly sample transformed position in the time interval [-1,1] and action from space
				p = random.uniform(-1,1)
				a = random.randrange(len(self.possible_actions))
				state = [p,1]
				state = np.reshape(state, [1, self.env.state_size])
				self.agent.remember(state, a, 0, state, True)

		for i in range(n_iterations):
			self.agent.replay(self.batch_size)

		# Clear the memory
		self.agent.memory.clear()
		print("Pretraining Complete")

	def _train(self,n_episodes):
		self.agent.evaluate = False
		for e in range(n_episodes):
			if e % self.train_stat_freq == 0:
				track = self.episode(evaluate = False,record = ["count","value"])
				for j in range(len(self.possible_actions)):
					self.new_run.log({'episode': self.episode_n, ('act_count' + str(j)): track["count"].count(j)})
			else:
				self.episode(evaluate = False)
			# Train agent
			if not self.intensive_training:
				if len(self.agent.memory) > self.batch_size:
					self.agent.replay(self.batch_size) # train the agent by replaying the experiences of the episode
					self.agent.step() # Update target network if required

			self.episode_n += 1


	def _evaluate(self,n_episodes):
		self.agent.evaluate = True
		total_count = []
		total_reward = 0
		total_position = [0] * self.num_steps
		total_lo_value = 0
		for e in range(n_episodes):
			record = ["count","reward","position"]
			if self.orderbook:
				record.append("lo")
			track = self.episode(evaluate = False,record = record)
			total_count += track["count"]
			total_reward += track["reward"]
			#print("position ",total_position,track["position"])
			total_position = [total_position[i] + (track["position"][i] if i < len(track["position"]) else 0) for i in range(self.num_steps)]
			if self.orderbook:
				total_lo_value += track["lo"]
		for j in range(len(self.possible_actions)):
			self.new_run.log({'episode': self.episode_n, ('eval_act_count' + str(j)): total_count.count(j) / n_episodes})
		self.new_run.log({'episode': self.episode_n, 'eval_rewards': total_reward / n_episodes})
		if self.orderbook:
			self.new_run.log({'episode': self.episode_n, 'lo_value': total_lo_value / n_episodes})
		if self.plot_position:
			plt.plot(np.arange(self.num_steps) ,np.array(total_position) / (n_episodes * self.env.initial_position))
			plt.ylabel("Percentage of Position")
			#print(np.arange(self.num_steps) / self.num_steps,np.array(total_position) / n_episodes)
			self.new_run.log({'episode': self.episode_n, 'position': plt})
		else:
			for j in range(len(total_position)):
				self.new_run.log({'episode': self.episode_n, ('position' + str(j)): total_position[j] / n_episodes})
			
	
	def train(self,n_episodes = 10000, epsilon = None, epsilon_decay = None,show_details = True, evaluate = False):
		
		# Default training parameters if not provided
		if epsilon is not None:
			self.agent.epsilon = epsilon

		if epsilon_decay is not None:
			self.agent.epsilon_decay = epsilon_decay

		# TAG: Deprecate
		# Setup action list
		action = -1

		# TAG: Deprecate
		agent_reward_dists = []
		initial_episode = self.episode_n
		while self.episode_n - initial_episode < n_episodes:
			self._train(self.eval_freq)
			self._evaluate(self.eval_window)

		
		
	def episode(self, verbose = False,evaluate = False, record = None):
		recording = record is not None and len(record) > 0
		if recording:	
			assert set(record).issubset(self.logging_options), "Undefined recording parameters"
		
		state = self.env.reset(training = (not evaluate)) # reset state at start of each new episode of the game

		# TAG: Depreciate
		#track_action_p = False
		
		track = {}
		if recording:
			# Log action values at t=0
			if "value" in record:
				for j in range(len(self.possible_actions)):
					#print("state (sim)",state)
					predicts = self.agent.predict(state)[0]
					self.new_run.log({'episode': self.episode_n, ('act_val' + str(j)): predicts[j]})

			for stat in record:
				if not stat == "value" and not stat == "reward" and not stat == "lo":
					track[stat] = []
				if stat == "reward":
					track[stat] = 0
				if stat == "lo":
					assert self.orderbook, "Limit orders can only be recorded in orderbook environments"
					track[stat] = 0
		
		done = False # Has the episode finished

		for t in range(self.num_steps):
			# Get actions for each agent
			action = self.agent.act(state)
			
			next_state, reward, done = self.env.step(action)

			if recording:
				if "count" in record and t < (self.num_steps - 1):
					# The final action doesn't matter
					track["count"].append(action)
				
				if "reward" in record:
					track["reward"] += reward

				if "event" in record:
					print("WARNING: Track events has not been implemented")
					track_events = False

				if "position" in record:
					track["position"].append(self.env.position)


			if not evaluate:
				self.agent.remember(state, action, reward, next_state, done)

			if verbose:
				print("Predict", self.agent.predict(state))
				print("State: ",state, "Actions: ", action, "Rewards: ", reward, "Next_states: ", next_state, "Done: ", done)
				# For final step print the predicted rewards for 0 position
				if t == self.num_steps - 1:
					print("Next predict", self.agent.predict(next_state))
			
			state = next_state
				
			if done: 
				break # exit loop
			
			# TAG: Depreciate?
			if self.intensive_training:
				for i, agent in enumerate(self.agents):
					if len(agent.memory) > self.batch_size and not evaluate:
						agent.replay(self.batch_size) # train the agent by replaying the experiences of the episode
						agent.step() # Update target network if required

		if not done:
			print(state)
			print("We have a problem.")

		if recording and "lo" in record:
			track["lo"] = self.env.m.lo_value
		
		return track

	def evaluate(self,n_episodes = 200,show_stats = True):
		raise "This function has not been updated to version 2"
		self.train(n_episodes = n_episodes, show_details = False,evaluate = True)
		# Return agent epsilons to their original values:
		for i, agent in enumerate(self.agents):
			agent.evaluate = False

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
		

	def execute(self,agent):
		# Currently just one strat
		raise "Depreciated function"
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

		
		
# Agents
from tensorflow.random import set_seed
set_seed(84)
import numpy as np
np.random.seed(84)

from collections import deque
from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.layers import Softmax
from keras.optimizers import Adam
from keras import Input
from keras import Model
import random
random.seed(84)

class replayMemory:
	'''Container for recalling agents interactions with the environment'''
	def __init__(self, max_size):
		self.buffer = [None] * max_size
		self.max_size = max_size
		self.index = 0
		self.size = 0

	def append(self, obj):
		self.buffer[self.index] = obj
		self.size = min(self.size + 1, self.max_size)
		self.index = (self.index + 1) % self.max_size

	def sample(self, batch_size):
		indices = random.sample(range(self.size), batch_size)
		return [(index,self.buffer[index]) for index in indices]

	def __len__(self):
		return self.size

	def __getitem__(self, index):
		return self.buffer[index]

class basicAgent:
	'''Base class for a deterministic agent'''
	def __init__(self,action, agent_name, action_size = None):
		self.agent_name = agent_name
		# Currently this agent must be provided with the correct action
		self.action = action
		self.memory = [1,2]
		self.epsilon_decay = 0
		self.epsilon = 1
		self.action_size = action_size
		
	def remember(self, state, action, reward, next_state, done):
		pass
	
	def replay(self, batch_size):
		pass
	
	# 'Virtual' function
	def act(self, state):
		raise "act() must be overriden by the child class"

	def step(self):
		pass

	# 'Virtual' function
	def predict(self, state):
		#raise "predict() must be overriden by the child class"
		return 0

	def update_paramaters(self,epsilon = 1.0,epsilon_decay = 0.9992,gamma = 1.0, epsilon_min = 0.01):
		pass

class learningAgent:
	def __init__(self, state_size, action_size, agent_name,C=0, alternative_target=False,agent_type="Undefined",tree_horizon=1):
		# Default Params: state_size, action_size
		self.agent_type = agent_type
		self.agent_name = agent_name
		self.state_size = state_size
		self.action_size = action_size
		# double-ended queue; acts like list, but elements can be added/removed from either end:
		self.replay_buffer_size = 3000
		self.memory = replayMemory(max_size=self.replay_buffer_size)
		self.n_since_updated = 0
		self.geometric_decay = True
		self.alternative_target = alternative_target
		self.C = C
		self.tree_n = tree_horizon
		self.epsilon_min = 0.01
		
		self.update_paramaters() # to defaults
		
		# rate at which NN adjusts models parameters via SGD to reduce cost:
		self.learning_rate = 0.001
		#self.model = self._build_model() # private method 

		self.model = self._build_model()
		self.n_since_updated = 0
		if self.C > 0:
			self.target_model = clone_model(self.model)

			if alternative_target:
				#self.prior_weights = deque(maxlen = C)
				self.prior_weights = self.model.get_weights()
	
	def remember(self, state, action, reward, next_state, done):
		'''Record the current environment for later replay'''
		self.memory.append((state, action, reward, next_state, done))
	 
	# 'Virtual' function    
	def predict(self,state):
		raise "No predict function available."

	# 'Virtual' function    
	def fit(self,state, action, reward, next_state, done,mem_index):
		raise "No fit function available."
		

	def act(self, state):
		# random action
		if np.random.rand() <= self.epsilon:
			rand_act = random.randrange(self.action_size)
			return rand_act#random.randrange(self.action_size)
		# Predict return
		act_values = self.predict(state)
		#print("act_values ",act_values)
		# Maximise return
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		'''Train with experiences sampled from memory'''
		# sample a minibatch from memory
		minibatch = self.memory.sample(batch_size)
		
		for mem_index, (state, action, reward, next_state, done) in minibatch:
			self.fit(state, action, reward, next_state, done,mem_index)
		
		if self.epsilon > self.epsilon_min: 
			if self.geometric_decay:
				self.epsilon *= self.epsilon_decay
			else:
				self.epsilon -= self.epsilon_decay#*= self.epsilon_decay

	def step(self):
		if self.UCB:
			self.t += 1 # For UCB method
		# Implementation described in Google Paper
		if not self.alternative_target:
			if self.C > 0:
				self.n_since_updated += 1
				if self.n_since_updated >= self.C: # Update the target network if C steps have passed
					if self.n_since_updated > self.C:
						print("target network not updated on time")
					#print("Debug: target network updated")
					self.n_since_updated = 0
					#self.target_model = clone_model(self.model)
					self.target_model.set_weights(self.model.get_weights())
					# Alternative Implementation with permenant lag
		else:
			if self.C > 0:
				self.n_since_updated += 1
				if self.n_since_updated >= self.C:
					self.n_since_updated = 0
					self.target_model.set_weights(self.prior_weights)
					self.prior_weights = self.model.get_weights()
				#if len(self.prior_weights) >= self.C: # Update the target network if at least C weights in memory
					#self.target_model.set_weights(self.prior_weights.pop())
					#print("DEBUG: prior weights: ",self.prior_weights)
				#self.prior_weights.appendleft(self.model.get_weights())

	def load(self, file_name):
		self.model.load_weights(file_name)

	def save(self, file_name):
		self.model.save_weights(file_name)

	def update_paramaters(self,epsilon = 1.0,epsilon_decay = 0.9992,gamma = 1, epsilon_min = 0.01):
		# No discounting but this can be enabled
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		#self.epsilon_min = epsilon_min

class randomAgent(basicAgent):
	def act(self,state):
		if self.action_size is None:
			raise "Action size must be specified when initialising the random agent"
		return random.randrange(self.action_size)

class TWAPAgent(basicAgent):
	def act(self, state):
		return self.action

class optimalSignalAgent(basicAgent):
	def __init__(self,action, agent_name, temp_impact_coef,gamma,dt,terminal,action_values):
		self.temp_impact_coef = temp_impact_coef
		self.gamma = gamma
		self.t = 0
		self.terminal = terminal
		self.dt = dt
		self.action_values = action_values
		super(optimalSignalAgent,self).__init__(action, agent_name, len(action_values))

	#https://link.springer.com/article/10.1007/s00780-019-00382-7
	def act(self, state):
		print("Unfinished - see notes")
		return
		optimal_rate = - state[0][2] / (2 * self.temp_impact_coef * self.gamma) * (1 - np.exp(- self.gamma * (self.terminal - self.t)))
		optimal_trade = optimal_rate * self.dt
		np.argmin()
		# ISSUE: we actually want to return the cloest action index to the optimal action
		return 

	def step(self,state):
		self.t += dt




## End		
# Agents
#from tensorflow.random import set_seed
#set_seed(84)
import numpy as np
#np.random.seed(84)

from collections import deque
from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.layers import Softmax, Conv1D, Flatten
from keras.optimizers import Adam
from keras import Input
from keras import Model
import random
#random.seed(84)

class replayMemory:
	'''Container for recalling an agents interactions with the environment'''
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

class stockProcessingNetwork(Model):
	'''Network for preprocessing of stock prices'''
	def __init__(self,input_dim,output_dim = None,units=16):
		super(stockProcessingNetwork,self).__init__()
		self.hidden1 = Conv1D(units,4,activation = 'relu')
		self.hidden2 = Conv1D(units,4,activation = 'relu')
		self.hidden3 = Conv1D(units,4,activation = 'relu')
		self.input_layer = Input(shape=(input_dim,))

	def call(self,inputs):
		res = self.input_layer(inputs)
		res = self.hidden1(res)
		res = self.hidden2(res)
		res = self.hidden3(res)
		return res

class basicAgent:
	'''Base class for a deterministic agent'''
	def __init__(self,action, agent_name, action_size):
		self.agent_name = agent_name
		# Currently this agent must be provided with the correct action
		self.action = action
		self.memory = [1,2]
		self.epsilon_decay = 0
		self.epsilon = 1
		self.epsilon_min = 0.01
		self.action_size = action_size
		self.agent_type = "basic"
		self.predicts = np.zeros(action_size)
		self.predicts.shape = (1,action_size)
		
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
		return self.predicts

	def update_paramaters(self,epsilon = 1.0,epsilon_decay = 0.9992,gamma = 1.0, epsilon_min = 0.01):
		pass

class learningAgent:
	'''Base class for trainable agents'''
	def __init__(self, 
		state_size, 
		action_size, 
		agent_name,
		C=0, 
		alternative_target=False,
		agent_type="Undefined",
		tree_horizon=1,
		n_hist_data=0,
		n_hist_inputs = 0,
		orderbook=False):
		
		# Agent identification
		self._agent_type = agent_type
		self._agent_name = agent_name

		# Input sizes
		self.state_size = state_size
		self.action_size = action_size
		if orderbook:
			self.action_space_size = 2
		else:
			self.action_space_size = 1
		

		# Replay buffer
		self.replay_buffer_size = 100000
		self.memory = replayMemory(max_size=self.replay_buffer_size)

		# Epsilon Greedy
		self.epsilon_greedy = True # Can be overridden by the subclass
		self.geometric_decay = True
		self.epsilon_min = 0.02
		self.epsilon_decay = 0.9992
		self.epsilon = 1

		# Target Network
		self.alternative_target = alternative_target
		self.C = C
		
		# Other parameters
		self.tree_n = tree_horizon
		self.learning_rate = 0.001
		self.gamma = 1
		self.reward_scaling = True
		self.action_as_input = False

		# Market data (currently only prices)
		self.n_hist_data = n_hist_data
		self.n_hist_inputs = n_hist_inputs
		if self.n_hist_data > 0:
			self.hist_model = self._build_hist_model(n_hist_data)

			if self.C > 0:
				self.hist_target_model = clone_model(self.hist_model)

			if alternative_target:
				self.hist_prior_weights = self.hist_model.get_weights()

		# Switch for agent evaluation mode
		self.evaluate = False
		
		self.model = self._build_model()

		# Target network
		self.n_since_updated = 0
		if self.C > 0:
			self.target_model = self._build_model(target = True)

			if alternative_target:
				self.prior_weights = self.model.get_weights()

	# TODO: This could be expanded
	def __str__(self):
		return agent_type + f", {self.action_space_size} primary inputs, {self.market_data_size} price inputs"

	@property
	def agent_name(self):
		return self._agent_name

	@property
	def agent_type(self):
		return self._agent_type
	
	def _build_hist_model(self,input_dim,units=16,depth=2,kernal_size=4):
		assert depth > 0 and units > 0 and input_dim > 0, "Invalid inputs"
		inputs = Input(shape=(input_dim,self.n_hist_inputs,))
		res = Conv1D(units,kernal_size,activation = 'relu')(inputs)
		for i in range(depth - 1):
			res = Conv1D(units,kernal_size,activation = 'relu')(res)
		
		res = Flatten()(res)
		model = Model(inputs=inputs,outputs=res)
		return model
		
	def remember(self, state, action, reward, next_state, done):
		'''Record the current environment for later replay'''
		#print(len(state),len(next_state))
		self.memory.append((state, action, reward, next_state, done))
	 
	# 'Virtual' function    
	def predict(self,state):
		raise "No predict function available."

	# 'Virtual' function    
	def fit(self,state, action, reward, next_state, done,mem_index):
		raise "No fit function available."
		

	def act(self, state):
		'''Choose action based on state'''
		# Note this applies only to epsilon greedy algorithms (function overridden for UCB)

		# Epsilon case
		if np.random.rand() <= self.epsilon and not self.evaluate:
			rand_act = random.randrange(self.action_size)
			return rand_act
		
		# Greedy case
		act_values = self.predict(state)
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		'''Train with experiences sampled from memory'''
		
		minibatch = self.memory.sample(batch_size)
		
		for mem_index, (state, action, reward, next_state, done) in minibatch:
			#print("replay state",state)
			self.fit(state, action, reward, next_state, done,mem_index)

	def step(self):
		'''Agent training update'''
		if self.epsilon_greedy:
			if self.epsilon > self.epsilon_min: 
				if self.geometric_decay:
					self.epsilon *= self.epsilon_decay
				else:
					self.epsilon -= self.epsilon_decay

		# Target network
		if self.C > 0:
			self.n_since_updated += 1

			if self.n_since_updated >= self.C:
				self.n_since_updated = 0
				if self.alternative_target:
					self.target_model.set_weights(self.prior_weights)
					self.prior_weights = self.model.get_weights()
					if self.n_hist_data > 0:
						self.hist_target_model.set_weights(self.hist_prior_weights)
						self.hist_prior_weights = self.hist_model.get_weights()
				else:
					self.target_model.set_weights(self.model.get_weights())
					if self.n_hist_data > 0:
						self.hist_target_model.set_weights(self.hist_model.get_weights())


	def load(self, file_name):
		self.model.load_weights(file_name)

	def save(self, file_name):
		self.model.save_weights(file_name)

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
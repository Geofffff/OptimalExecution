# Agents
from tensorflow.random import set_seed
set_seed(84)
import numpy as np
np.random.seed(84)

from collections import deque
from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.optimizers import Adam
import random
random.seed(84)



class basicAgent:
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
	def __init__(self, state_size, action_size, agent_name,agent_type):
		# Default Params: state_size, action_size
		self.agent_type = agent_type
		self.agent_name = agent_name
		self.state_size = state_size
		self.action_size = action_size
		# double-ended queue; acts like list, but elements can be added/removed from either end:
		self.memory = deque(maxlen=2000)
		self.n_since_updated = 0
		
		self.update_paramaters() # to defaults
		
		# rate at which NN adjusts models parameters via SGD to reduce cost:
		self.learning_rate = 0.001
		#self.model = self._build_model() # private method 
	
	def remember(self, state, action, reward, next_state, done):
		'''Record the current environment for later replay'''
		self.memory.append((state, action, reward, next_state, done))
	 
	# 'Virtual' function    
	def predict(self,state):
		raise "No predict function available."

	# 'Virtual' function    
	def fit(self,state, action, reward, next_state, done):
		raise "No fit function available."
		

	def act(self, state):
		# random action
		if np.random.rand() <= self.epsilon:
			rand_act = random.randrange(self.action_size)
			return rand_act#random.randrange(self.action_size)
		# Predict return
		act_values = self.predict(state)
		# Maximise return
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		'''Train with experiences sampled from memory'''
		# sample a minibatch from memory
		minibatch = random.sample(self.memory, batch_size)
		
		for state, action, reward, next_state, done in minibatch:
			self.fit(state, action, reward, next_state, done)
		
		if self.epsilon > self.epsilon_min: # TODO: unecessary?
			self.epsilon *= self.epsilon_decay

	def step(self):
		pass

	def load(self, file_name):
		self.model.load_weights(file_name)

	def save(self, file_name):
		self.model.save_weights(file_name)

	def update_paramaters(self,epsilon = 1.0,epsilon_decay = 0.9992,gamma = 1, epsilon_min = 0.01):
		# No discounting but this can be enabled
		self.gamma = gamma
		self.epsilon = epsilon
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min

class DQNAgent(learningAgent):
	'''Standard Deep Q Agent, network dimensions pre specified'''
	def __init__(self, state_size, action_size, agent_name, C = 0,alternative_target = False):
		learningAgent.__init__(self,state_size,action_size,agent_name,agent_type = "DQN")
		self.model = self._build_model() # private method 
		self.C = C
		self.alternative_target = alternative_target
		if self.C > 0:
			self.target_model = clone_model(self.model)

			if alternative_target:
				self.prior_weights = deque(maxlen = C)
	
	def _build_model(self):
		set_seed(84)
		# neural net to approximate Q-value function:
		model = Sequential()
		model.add(Dense(5, input_dim=self.state_size, activation='relu')) # 1st hidden layer; states as input
		model.add(Dense(5, activation='relu')) # 2nd hidden layer
		model.add(Dense(self.action_size, activation='linear')) # 2 actions, so 2 output neurons: 0 and 1 (L/R)
		model.compile(loss='mse',
						optimizer=Adam(lr=self.learning_rate))
		return model

	def step(self):
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
		# Alternative Implementation
		else:
			if self.C > 0:
				if len(self.prior_weights) >= self.C: # Update the target network if at least C weights in memory
					self.target_model.set_weights(self.prior_weights.pop())
					#print("DEBUG: prior weights: ",self.prior_weights)
				self.prior_weights.appendleft(self.model.get_weights())

	# Override predict and fit functions
	def predict(self,state,target = False):
		# Note that this predict function (without for_fitting) should only be used once per step!
		if self.C > 0 and target:
			return self.target_model.predict(state)

		return self.model.predict(state)
 
	def fit(self,state, action, reward, next_state, done):
		target = reward
		# if not done then returns must incorporate predicted (discounted) future reward
		if not done:
			target = (reward + self.gamma * 
						np.amax(self.predict(next_state,target = True)[0])) 
			#print("target ", target, ", reward ", reward)
		target_f = self.predict(state,target = True) # predicted returns for all actions
		target_f[0][action] = target 
		# Change the action taken to the reward + predicted max of next states
		self.model.fit(state, target_f,epochs=1, verbose=0) # Single epoch?

	def update_paramaters(self,epsilon = 1.0,epsilon_decay = 0.9992,gamma = 1.0, epsilon_min = 0.01):
		super(DQNAgent,self).update_paramaters(epsilon, epsilon_decay,gamma, epsilon_min)

class DDQNAgent_v1(learningAgent):
	''' Double Deep Q Agent, network dimensions pre specified'''
	def __init__(self, state_size, action_size, agent_name):
		learningAgent.__init__(self,state_size,action_size,agent_name,agent_type = "DDQN")
		self.model1 = self._build_model() # private method
		self.model2 = self._build_model() # private method

	def _build_model(self):
		set_seed(84)
		# neural net to approximate Q-value function:
		model = Sequential()
		model.add(Dense(5, input_dim=self.state_size, activation='relu')) # 1st hidden layer; states as input
		model.add(Dense(5, activation='relu')) # 2nd hidden layer
		model.add(Dense(self.action_size, activation='linear')) # 2 actions, so 2 output neurons: 0 and 1 (L/R)
		model.compile(loss='mse',
						optimizer=Adam(lr=self.learning_rate)) 
		return model

	# Override predict and fit functions
	def predict(self,state):
		# Return the average of the two networks
		return (self.model1.predict(state) + self.model2.predict(state)) / 2
 
	def fit(self,state, action, reward, next_state, done):
		target = reward
		# Update either of the two models with equal probability
		if np.random.rand() < 0.5:
			modelA = self.model1 
			modelB = self.model2
		else:
			modelA = self.model2
			modelB = self.model1

		if not done:
			max_act = np.argmax(modelA.predict(next_state))
			target = (reward + self.gamma * 
						modelB.predict(next_state)[0,max_act])
		target_f = modelA.predict(state) # predicted returns for all actions
		target_f[0][action] = target 
			# Change the action taken to the reward + predicted max of next states
		modelA.fit(state, target_f,epochs=1, verbose=0) # Single epoch?

	def update_paramaters(self,epsilon = 1.0,epsilon_decay = 0.9992,gamma = 1.0, epsilon_min = 0.01):
		super(DDQNAgent,self).update_paramaters(epsilon, epsilon_decay,gamma, epsilon_min)
		#epsilon_decay = np.sqrt(epsilon_decay)

class DDQNAgent(DQNAgent):

	def fit(self,state, action, reward, next_state, done):
		target = reward
		# if not done then returns must incorporate predicted (discounted) future reward
		if not done:
			max_act = np.argmax(self.predict(next_state,target = False))
			target = (reward + self.gamma * 
						self.predict(next_state,target = True)[0,max_act])
			#print("target ", target, ", reward ", reward)
		target_f = self.predict(state,target = True) # predicted returns for all actions
		target_f[0][action] = target 
		# Change the action taken to the reward + predicted max of next states
		self.model.fit(state, target_f,epochs=1, verbose=0) # Single epoch?	

class TWAPAgent(basicAgent):
	def act(self, state):
		return self.action

class randomAgent(basicAgent):
	def act(self,state):
		if self.action_size is None:
			raise "Action size must be specified when initialising the random agent"
		return random.randrange(self.action_size)










#End

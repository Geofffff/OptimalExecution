# Agents

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np


class learningAgent:
	def __init__(self, state_size, action_size, agent_name,agent_type):
		# Default Params: state_size, action_size
		self.agent_type = agent_type
		self.agent_name = agent_name
		self.state_size = state_size
		self.action_size = action_size
        # double-ended queue; acts like list, but elements can be added/removed from either end:
		self.memory = deque(maxlen=2000)
        
        # No discounting but this can be enabled
		self.gamma = 1 

        # Default epsilon params
		self.epsilon = 1.0
		self.epsilon_decay = 0.998 
		self.epsilon_min = 0.01

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
	def fit(self,state):
		raise "No fit function available."
        

	def act(self, state):
        # random action
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
        
        # Predict return
		act_values = self.predict(state)
        # Maximise return
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		'''Train with experiences sampled from memory'''
        
        # sample a minibatch from memory
		minibatch = random.sample(self.memory, batch_size)
        
		for state, action, reward, next_state, done in minibatch:
			# if done (boolean whether game ended or not, i.e., whether final state or not), then target = reward:
			target = reward

			# if not done then returns must incorporate predicted (discounted) future reward
			if not done:
				target = (reward + self.gamma * 
							np.amax(self.predict(next_state)[0])) 
			target_f = self.predict(state) # predicted returns for all actions
			target_f[0][action] = target 
            # Change the action taken to the reward + predicted max of next states
			self.fit(state, target_f) # Single epoch?
		if self.epsilon > self.epsilon_min: # TODO: unecessary?
			self.epsilon *= self.epsilon_decay

	def load(self, file_name):
		self.model.load_weights(file_name)

	def save(self, file_name):
		self.model.save_weights(file_name)

class DQNAgent(learningAgent):
	def __init__(self, state_size, action_size, agent_name):
		learningAgent.__init__(self,state_size,action_size,agent_name,agent_type = "DQN")
		self.model = self._build_model() # private method 
    
	def _build_model(self):
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
		return self.model.predict(state)
 
	def fit(self,state,target):
		self.model.fit(state, target, epochs=1, verbose=0)


class TWAPAgent:
	def __init__(self, action , agent_name):
		self.agent_name = agent_name
		# Currently this agent must be provided with the correct action
		self.action = action
		self.memory = [1,2]
        
	def remember(self, state, action, reward, next_state, done):
		pass
    
	def replay(self, batch_size):
		pass
    
	def act(self, state):
		return self.action











#End

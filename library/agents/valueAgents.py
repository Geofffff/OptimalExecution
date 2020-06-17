# Agents
#from tensorflow.random import set_seed
#set_seed(84)
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

if __name__ == "__main__":
	from baseAgents import learningAgent, replayMemory
else:
	from library.agents.baseAgents import learningAgent, replayMemory

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
		# Alternative Implementation with permenant lag
		else:
			if self.C > 0:
				if len(self.prior_weights) >= self.C: # Update the target network if at least C weights in memory
					self.target_model.set_weights(self.prior_weights.pop())
					#print("DEBUG: prior weights: ",self.prior_weights)
				self.prior_weights.appendleft(self.model.get_weights())

	# Override predict and fit functions
	def predict(self,state,target = False):
		if self.reward_scaling:
			scaling_factor = state[0][0]/2 + 0.5
		else:
			scaling_factor = 0

		if self.C > 0 and target:
			return self.target_model.predict(state) + scaling_factor

		return self.model.predict(state) + scaling_factor
 
	def fit(self,state, action, reward, next_state, done):
		
		target = reward
		# if not done then returns must incorporate predicted (discounted) future reward
		if not done:
			target = (reward + self.gamma * 
						np.amax(self.predict(next_state,target = True)[0])) 
			#print("target ", target, ", reward ", reward)
		target_f = self.predict(state,target = True) # predicted returns for all actions
		target_f[0][action] = target 
		if self.reward_scaling:
			target_f -= state[0][0]/2 + 0.5
		# Change the action taken to the reward + predicted max of next states
		self.model.fit(state, target_f,epochs=1, verbose=0) # Single epoch?

	def update_paramaters(self,epsilon = 1.0,epsilon_decay = 0.9992,gamma = 1.0, epsilon_min = 0.01):
		super(DQNAgent,self).update_paramaters(epsilon, epsilon_decay,gamma, epsilon_min)

class DDQNAgent(DQNAgent):

	def project(self,reward,next_state,done,horizon,mem_index):
		tree_success = False
		predict = 0
		if not done:
			next_action_index = np.argmax(self.predict(next_state,target = False)[0])
			if horizon > 1 and mem_index < (self.memory.size - 1):
				state1, action1, reward1, next_state1, done1 = self.memory[mem_index + 1]
				if next_action_index == action1:
					predict = self.project(reward1,next_state1,done1,horizon - 1,mem_index + 1)
					tree_success = True
			if not tree_success:
				predict = self.predict(next_state,target = True)[0][next_action_index]
			
		return reward + self.gamma * predict

	def fit(self,state, action, reward, next_state, done,mem_index = -1):
		target = self.project(reward,next_state,done,self.tree_n,mem_index)
		target_f = self.predict(state,target = True) # predicted returns for all actions
		target_f[0][action] = target 
		if self.reward_scaling:
			target_f -= state[0][0]/2 + 0.5
		# Change the action taken to the reward + predicted max of next states
		self.model.fit(state, target_f,epochs=1, verbose=0) # Single epoch?	

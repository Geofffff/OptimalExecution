import numpy as np
from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.layers import Softmax
from keras import Input
from keras import Model
from keras.optimizers import Adam
from collections import deque

if __name__ == "__main__":
	from agents import learningAgent
	DEBUG = True
else:
	from library.agents import learningAgent
	DEBUG = False

# Define support for the value distribution model
# Note that V_min and V_max should be dynamic and depend on vol - how would this work on real data (max historical?)
# Is V_min for the total value? In this case it can be capped by the remaining position
V_min = 0; V_max = 10

#N = 2 # This could be dynamic depending on state?
# This granularity is problematic - can we do this without discretisation?
# Especially if V_min and V_max are not dynamic
# Paper: increasing N always increases returns
#dz = (V_max - V_min) / (N - 1)
#z = np.array(range(N)) * (V_max - V_min) / (N - 1) + V_min
#theta = np.ones(N)
gamma = 1 # Discount factor
#learning_rate = 0.01
# This would result in uniform prob (not sure if this is the right approach)
state_size = 2
action_values = np.array([0,0.001,0.005,0.01,0.02,0.05,0.1])
action_values = action_values * 10

class distAgent(learningAgent):

	def __init__(self,action_size, agent_name,N=51,C = 0,alternative_target = False):
		self.V_min = 0; self.V_max = 15
		self.agent_type = "dist" 

		self.N = N # This could be dynamic depending on state?
		# This granularity is problematic - can we do this without discretisation?
		# Especially if V_min and V_max are not dynamic
		# Paper: increasing N always increases returns
		self.dz = (self.V_max - self.V_min) / (self.N - 1)
		self.z = np.array(range(self.N)) * (self.V_max - self.V_min) / (self.N - 1) + self.V_min
		#theta = np.ones(N)
		self.gamma = 1 # Discount factor
		self.learning_rate = 0.001
		# This would result in uniform prob (not sure if this is the right approach)
		self.state_size = 2

		self.memory = deque(maxlen=2000)
		self.action_size = action_size
		self.model = self._build_model()
		self.agent_name = agent_name
		self.epsilon = 1
		self.epsilon_decay = 0.998

		# Target networks
		self.C = C
		self.alternative_target = alternative_target
		self.n_since_updated = 0
		if self.C > 0:
			self.target_model = clone_model(self.model)

			if alternative_target:
				self.prior_weights = deque(maxlen = C)

	def probs(self,state,target=False):
		#action = self._transform_action(action_index)
		#state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1]) #np.reshape(action, [1, 1])#
		if DEBUG:
			#print("probs of ",state_action,"are",self.model.predict(state_action))
			pass
		if target and self.C>0:
			return self.target_model.predict(state)

		return self.model.predict(state)
		#return np.exp(theta(i,x,a)/np.sum(np.exp(theta(i,x,a))))

	def predict(self,state,target = False):
		dist = self.probs(state,target = target)
		res = np.sum(dist * self.z, axis = 2).flatten()
		res.shape = (1,len(res))
		return res

	def predict_act(self,state,action_index,target = False):
		#state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1])
		#print("predicting ", state_action)
		dist = self.probs(state,target = target)[action_index][0]
		return np.sum(dist * self.z)

	def vpredict(self,state,action_indices,target = False):
		return np.vectorize(self.predict_act,excluded=['state'] )(state = state,action_index = action_indices,target = target)

	def Tz(self,reward):
		Tz = reward + self.gamma * self.z
		return Tz

	# Think of how to do this in a more numpy way
	# Note this ALWAYS uses the target network
	# DDQN enabled (!!!)
	def projTZ(self,reward,next_state,done):
		res = []
		if not done:
			next_action_index = np.argmax(self.predict(next_state,target = False)[0])
			#next_action = self.action_values[next_action_index]
			all_probs = self.probs(next_state,target = True)[next_action_index][0]
			for i in range(self.N):
				res.append(np.sum(self._bound(1 - np.abs(self._bound(self.Tz(reward),self.V_min,self.V_max) - self.z[i])/self.dz,0,1) * all_probs))
		else:
			#reward_v = np.ones(N) * reward
			for i in range(self.N):
				res.append(self._bound(1 - np.abs(self._bound(reward,self.V_min,self.V_max) - self.z[i])/self.dz,0,1))
				#print("reward ", self._bound(reward,self.V_min,self.V_max), " dz ", self.dz, " z[i] ", self.z[i], " append ",(self._bound(reward,self.V_min,self.V_max) - self.z[i])/self.dz)
		return res

	def _bound(self,vec,lower,upper):
		return np.minimum(np.maximum(vec,lower),upper)

	def _build_model(self):
		# Using Keras functional API
		state_in = Input(shape=(self.state_size,))
		hidden1 = Dense(8, activation='relu')(state_in)
		hidden2 = Dense(8, activation='relu')(hidden1)
		hidden3 = Dense(30, activation='relu')(hidden2)
		outputs =[]
		for i in range(self.action_size):
			outputs.append(Dense(self.N, activation='softmax')(hidden3))
		model = Model(inputs=state_in, outputs=outputs)
		model.compile(loss='categorical_crossentropy',
						optimizer=Adam(lr=self.learning_rate))
		return model


	def fit(self,state, action_index, reward, next_state, done):
		#action = self._transform_action(action_index)
		#state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1])#np.reshape(action, [1, 2])#
		target = self.projTZ(reward,next_state,done)
		target_f = self.probs(state,target = True)
		#if DEBUG:
			#print("target_f ",target_f[action_index][0], "target ", target)
		debug_target_f = target_f[action_index][0].copy()
		target_f[action_index][0] = target
		#if DEBUG:
		#print("fitting state:", state,",action:",action_index,",reward:",reward, "target_f ",target_f[action_index][0]-debug_target_f)
		self.model.fit(state, target_f,epochs=1, verbose=0)

	# Temporary Experiment
	def variance(self,state,target = False):
		pass

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
# Testing the code
if __name__ == "__main__":
	myAgent = distAgent(7,"TonyTester")
	state = [1,-1] 
	state = np.reshape(state, [1, 2])
	state1 = [0,0] 
	state1 = np.reshape(state1, [1, 2])
	next_state = [0.8,-0.9] 
	next_state = np.reshape(state, [1, 2])
	myAgent.epsilon_min = 0.01
	
	if False:
		#print(bound(Tz(1),0,10))
		#print("test_pred ", predict_act(state,1))
		#print(np.vectorize(predict_act,excluded=['state'])(state = state,action = [0,1,2]))

		#old_predict = myAgent.predict(state)
		#old_probs = myAgent.probs(state,1)
		#print("target ",projTZ(1.0,next_state,True))
		DEBUG = False
		print("state ", state,"predict ",myAgent.predict(state) ,"probs(",1,") ", myAgent.probs(state)[6][0], "probs(",0,")", myAgent.probs(state,0)[0][0])
		for i in range(200):
			myAgent.remember(state, 6, 7, next_state, False)
			myAgent.remember(state, 5, 10, next_state, False)
			myAgent.remember(state, 4, 6, next_state, False)
			myAgent.remember(state, 3, 3, next_state, False)
			myAgent.remember(state, 2, 0, next_state, False)
			myAgent.remember(state1, 6, 3, next_state, True)
		for i in range(200):
			myAgent.replay(6)
		print("probs[0] ", myAgent.probs(state)[0][0])
		#print("predict change ",myAgent.predict(state) - old_predict ,"probs change ", myAgent.probs(state,6) - old_probs)
		print("state ", state,"predict ",myAgent.predict(state) , "state ", state1,"predict ",myAgent.predict(state1))
		#print("state ", state,"predict ",myAgent.predict(state) ,"probs(",1,") ", myAgent.probs(state)[6][0], "probs(",0,")", myAgent.probs(state,0)[0][0])
	if False:
		myAgent.epsilon = 0
		print(myAgent.act(state))
		print("predict change ",myAgent.predict(state)  ,"probs ")#, probs(state,6))

	if True:
		print("Tz:",myAgent.Tz(0.5))
		print("Next State Value:",myAgent.predict(next_state)[0])
		my_next_action = np.argmax(myAgent.predict(next_state)[0])
		print("choose action:",my_next_action)
		print("... and the probs:",myAgent.probs(next_state)[my_next_action][0])
		print("resulting in projTZ ", myAgent.projTZ(0.5,next_state,False))








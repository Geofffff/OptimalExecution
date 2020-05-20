import numpy as np
from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

if __name__ == "__main__":
	from agents import learningAgent
else:
	from library.agents import learningAgent

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
learning_rate = 0.01
# This would result in uniform prob (not sure if this is the right approach)
state_size = 2
action_values = np.array([0,0.001,0.005,0.01,0.02,0.05,0.1])
action_values = action_values * 10

class distAgent(learningAgent):

	def __init__(self):
		self.V_min = 0; self.V_max = 10 

		self.N = 2 # This could be dynamic depending on state?
		# This granularity is problematic - can we do this without discretisation?
		# Especially if V_min and V_max are not dynamic
		# Paper: increasing N always increases returns
		self.dz = (self.V_max - self.V_min) / (self.N - 1)
		self.z = np.array(range(self.N)) * (self.V_max - self.V_min) / (self.N - 1) + self.V_min
		#theta = np.ones(N)
		self.gamma = 1 # Discount factor
		self.learning_rate = 0.01
		# This would result in uniform prob (not sure if this is the right approach)
		self.state_size = 2
		self.action_values = np.array([0,0.001,0.005,0.01,0.02,0.05,0.1])
		self.action_values = self.action_values * 10

		self.memory = deque(maxlen=2000)
		self.action_size = len(self.action_values)
		self.model = self._build_model()
		self.agent_name = "David"
		self.epsilon = 1
		self.epsilon_decay = 0.998

		# Transformations
		self.trans_a = 2 / (np.amax(self.action_values) - np.amin(self.action_values))
		self.trans_b = -self.trans_a * np.amin(self.action_values) - 1

	def probs(self,state,action_index):
		action = self._transform_action(action_index)
		state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1]) #np.reshape(action, [1, 1])#
		print("probs of ",state_action,"are",self.model.predict(state_action))
		return self.model.predict(state_action)
		#return np.exp(theta(i,x,a)/np.sum(np.exp(theta(i,x,a))))

	def predict(self,state):
		res = self.vpredict(state,range(len(self.action_values)))
		return np.reshape(res, [1, len(res)])

	def predict_act(self,state,action_index):
		#state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1])
		#print("predicting ", state_action)
		dist = self.probs(state,action_index)
		return np.sum(dist * self.z)

	def vpredict(self,state,action_indices):
		return np.vectorize(self.predict_act,excluded=['state'] )(state = state,action_index = action_indices)

	def Tz(self,reward):
		Tz = reward + self.gamma * self.z
		return Tz

	# Think of how to do this in a more numpy way
	def projTZ(self,reward,next_state,done):
		res = []
		if not done:
			next_action_index = np.argmax(self.predict(next_state)[0])
			next_action = self.action_values[next_action_index]
			all_probs = self.probs(next_state,next_action)
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
		model = Sequential()
		# Input dim self.state_size + 1
		model.add(Dense(5, input_dim=(3), activation='relu')) # 1st hidden layer; states as input
		model.add(Dense(5, activation='relu')) # 2nd hidden layer
		model.add(Dense(self.N, activation='softmax')) 
		model.compile(loss='categorical_crossentropy',
						optimizer=Adam(lr=self.learning_rate))
		return model

	# CURRENTLY: Action index goes in - transformed action value out
	def _transform_action(self,action_index):
		return action_values[action_index] * self.trans_a + self.trans_b

	def fit(self,state, action_index, reward, next_state, done):
		action = self._transform_action(action_index)
		state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1])#np.reshape(action, [1, 2])#
		target = self.projTZ(reward,next_state,done)
		target_f = np.reshape(target, [1, self.N])
		
		print("fitting ", state_action," target_f ",target_f)
		self.model.fit(state_action, target_f,epochs=100, verbose=0)

	def step(self):
		# Temporarily pass this
		pass

# Testing the code
if __name__ == "__main__":
	myAgent = distAgent()
	state = [1,-1] 
	state = np.reshape(state, [1, 2])
	next_state = [0.8,-0.9] 
	next_state = np.reshape(state, [1, 2])
	
	if True:
		#print(bound(Tz(1),0,10))
		#print("test_pred ", predict_act(state,1))
		#print(np.vectorize(predict_act,excluded=['state'])(state = state,action = [0,1,2]))

		#old_predict = myAgent.predict(state)
		old_probs = myAgent.probs(state,1)
		#print("target ",projTZ(1.0,next_state,True))
		for i in range(2):
			myAgent.fit(state,6,10.0,next_state,True)
			myAgent.fit(state,0,0,next_state,True)
		#print("predict change ",myAgent.predict(state) - old_predict ,"probs change ", myAgent.probs(state,6) - old_probs)
		print("state ", state,"predict ",myAgent.predict(state) ,"probs(",1,") ", myAgent.probs(state,6), "probs(",0,")", myAgent.probs(state,0))
	if False:
		myAgent.epsilon = 0
		print(myAgent.act(state))
		print("predict change ",myAgent.predict(state)  ,"probs ")#, probs(state,6))








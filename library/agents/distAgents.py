import numpy as np
from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.layers import Softmax
from keras.layers import Multiply
from keras import Input
from keras import Model
from keras.optimizers import Adam
from keras.losses import huber_loss 
from collections import deque
import random
import keras.backend as K

if __name__ == "__main__":
	from baseAgents import learningAgent, replayMemory
	DEBUG = True
else:
	from library.agents.baseAgents import learningAgent, replayMemory
	DEBUG = False

class distAgent(learningAgent):
	def __init__(self, state_size, action_size, agent_name,C, alternative_target,UCB=False,UCBc = 1,tree_horizon = 3):
		super(distAgent,self).__init__(state_size, action_size, agent_name,C, alternative_target,"dist",tree_horizon)
		self.UCB = UCB
		self.c = UCBc
		self.geometric_decay = True

		if self.UCB:
			self.t = 1

	def act(self, state):
		# No eps greedy required for 'UCB' type update
		if self.UCB:
			self.ct = self.c * np.sqrt(np.log(self.t) / self.t)
			act_values = self.predict(state)
			return np.argmax(act_values[0] + self.ct * np.sqrt(self.variance(state)))
		# random action
		if np.random.rand() <= self.epsilon:
			rand_act = random.randrange(self.action_size)
			return rand_act#random.randrange(self.action_size)
		# Predict return
		act_values = self.predict(state)		
		return np.argmax(act_values[0])


class C51(distAgent):

	def __init__(self,state_size, action_size, agent_name,N=51,C = 0,alternative_target = False,UCB = False,UCBc = 1,tree_horizon = 3):
		distAgent.__init__(self,state_size, action_size, agent_name,C, alternative_target,UCB,UCBc,tree_horizon)
		self.V_max = 0.1
		self.V_min = -0.15

		self.N = N # This could be dynamic depending on state?
		# This granularity is problematic - can we do this without discretisation?
		# Especially if V_min and V_max are not dynamic
		# Paper: increasing N always increases returns
		self.dz = 2 * self.V_max / (self.N - 1) # (self.V_max - self.V_min)
		self.z = np.array(range(self.N)) * (self.V_max - self.V_min) / (self.N - 1) + self.V_min
		
		self.reward_mapping = True # Purely for Wandb config purposes 

	def return_mapping(self,state,ret,inverse = False):
		if inverse:
			return ret - state[0][0] * self.result_scaling_factor

		return ret + state[0][0] * self.result_scaling_factor

	def probs(self,state,action_index,target=False):
		# SHOULD transform state_action triple as in Ning
		action = self._transform_action(action_index)
		state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1]) #np.reshape(action, [1, 1])#
		if DEBUG:
			#print("probs of ",state_action,"are",self.model.predict(state_action))
			pass
		if target and self.C>0:
			res = self.target_model.predict(state_action)
		else:
			res = self.model.predict(state_action)
		return res

	def predict(self,state,target = False):
		res = self.vpredict(state,range(len(self.action_values)),target = target)
		return np.reshape(res, [1, len(res)])

	def predict_act(self,state,action_index,target = False):
		#state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1])
		#print("predicting ", state_action)
		dist = self.probs(state,action_index,target = target)
		return np.sum(dist * self.mapped_z(state))

	def vpredict(self,state,action_indices,target = False):
		return np.vectorize(self.predict_act,excluded=['state'] )(state = state,action_index = action_indices,target = target)

	def Tz(self,state,reward):
		Tz = reward + self.gamma * self.mapped_z(state)
		return Tz

	def mapped_z(self,state):
		return self.z + (state[0][0] / 2 + 0.5)

	def mapped_dz(self,state):
		return self.dz

	def mapped_bounds(self,state):
		return (self.V_min + (state[0][0] / 2 + 0.5), self.V_max + (state[0][0] / 2 + 0.5))

	# Get unmapped results using basis: STATE
	# Could be made more efficient
	def projTZ_nTree(self,state,reward,next_state,done,horizon,mem_index):
		res = []
		tree_success = False
		V_min_s, V_max_s = self.mapped_bounds(state)
		if not done:
			next_action_index = np.argmax(self.predict(next_state,target = False)[0])
			# Check there is a valid next state and that the tree is not at a leaf
			if horizon > 1 and mem_index < (self.memory.size - 1):
				state1, action1, reward1, next_state1, done1 = self.memory[mem_index + 1]
				if next_action_index == action1:
					#print("Tree Sucess",state1,horizon)
					all_probs = self.projTZ_nTree(next_state,reward1,next_state1,done1,horizon - 1,mem_index + 1)
					tree_success = True
			#next_action = self.action_values[next_action_index]
			if not tree_success:
				all_probs = self.probs(next_state,next_action_index,target = True)[0]
			for i in range(self.N):
				res.append(np.sum(self._bound(1 - np.abs(self._bound(self.Tz(next_state,reward),V_min_s,V_max_s) - self.mapped_z(state)[i])/self.mapped_dz(state),0,1) * all_probs))
		else:
			#reward_v = np.ones(N) * reward
			for i in range(self.N):
				res.append(self._bound(1 - np.abs(self._bound(reward,V_min_s,V_max_s) - self.mapped_z(state)[i])/self.mapped_dz(state),0,1))
				#print("reward ", self._bound(reward,self.V_min,self.V_max), " dz ", self.dz, " z[i] ", self.z[i], " append ",(self._bound(reward,self.V_min,self.V_max) - self.z[i])/self.dz)
		return res

	
	def _bound(self,vec,lower,upper):
		return np.minimum(np.maximum(vec,lower),upper)


	def _build_model(self):
		# Using Keras functional API
		state_in = Input(shape=(self.state_size + 1,))
		hidden1 = Dense(32, activation='relu')(state_in)
		#hidden1_n = BatchNormalization() (hidden1)
		hidden2 = Dense(32, activation='relu')(hidden1)
		skip_layer = Add()([hidden1, hidden2])
		#hidden3 = Dense(5, activation='relu')(hidden2)
		outputs = Dense(self.N, activation='softmax')(skip_layer)
		model = Model(inputs=state_in, outputs=outputs)
		model.compile(loss='categorical_crossentropy',
						optimizer=Adam(lr=self.learning_rate))
		return model

	# CURRENTLY: Action index goes in - transformed action value out
	def _transform_action(self,action_index):
		return self.action_values[action_index] * self.trans_a + self.trans_b

	def fit(self,state, action_index, reward, next_state, done,mem_index = -1):
		action = self._transform_action(action_index)
		state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1])#np.reshape(action, [1, 2])#
		if self.tree_n > 1:
			target = self.projTZ_nTree(state,reward,next_state,done,self.tree_n,mem_index)
		else:
			target = self.projTZ(reward,next_state,done)
		target_f = np.reshape(target, [1, self.N])
		#if DEBUG:
			#print("target_f ",target_f[action_index][0], "target ", target)
		#if DEBUG:
		#print("fitting state:", state,",action:",action_index,",reward:",reward, " delta target_f ",target_f[action_index][0]-debug_target_f,"done:",done)
		
		self.model.fit(state_action, target_f,epochs=1, verbose=0)


	def variance(self,state,target = False):
		res = self.vvar(state,range(len(self.action_values)),target = target) - np.power(self.predict(state,target),2)
		if not all(np.round(res[0],5) >= 0):
			print(res)
		assert all(np.round(res[0],5) >= 0)
		return np.max(res,0)#np.reshape(res, [1, res.shape[0]])

	def var_act(self,state,action_index,target = False):
		#state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1])
		#print("predicting ", state_action)
		dist = self.probs(state,action_index,target = target)[0]
		return np.sum(dist * np.power(self.mapped_z(state),2))

	def vvar(self,state,action_indices,target = False):
		return np.vectorize(self.var_act,excluded=['state'] )(state = state,action_index = action_indices,target = target)


class IQNAgent(distAgent):
	def __init__(self,state_size, action_size, agent_name,C, alternative_target = False,UCB=False,UCBc = 1,tree_horizon = 3):
		self.N = 8
		self.N_p = 8
		self.embedding_dim = 64
		self.state_model_size_out = 8
		self.kappa = 2 # What should this be?
		self.selected_qs = None
		print("Sucessfully initialised")
		super(IQNAgent,self).__init__(state_size, action_size, agent_name,C, alternative_target,UCB,UCBc,tree_horizon)
		
	
	# https://stackoverflow.com/questions/55445712/custom-loss-function-in-keras-based-on-the-input-data
	@staticmethod
	def huber_loss_quantile(self,tau):
		def loss(yTrue,yPred):
			bellman_errors = yTrue - yPred
			return (K.abs(tau - K.to_float(bellman_errors < 0)) * huber_loss(yTrue,yPred)) / self.kappa
		return loss
		

	def _build_model(self):
		# Using Keras functional API
		state_in = Input(shape=(self.state_size + 1,))
		state_hidden1 = Dense(8, activation='relu')(state_in)
		state_hidden2 = Dense(self.state_model_size_out, activation='relu')(state_hidden1)
		#hidden3 = Dense(30, activation='relu')(hidden2)
		
		quantile_in = Input(shape=(self.N,self.embedding_dim))
		# This needs fixing - transforming to tf functions etc.
		
		q_hidden = Dense(self.state_model_size_out, activation='relu')(quantile_in)

		# Full Model
		main_hidden1 = Multiply(q_hidden, state_hidden2)
		main_hidden2 = Dense(30, activation='relu')(main_hidden1)
		outputs = Dense(self.N, activation='linear')(hidden2)
		main_model = Model(inputs=(state_in,quantile_in), outputs=outputs)

		main_model.compile(loss = huber_loss_quantile(quantiles_in),
						optimizer=Adam(lr=self.learning_rate))

		return main_model

	def predict(self,state,target = False):
		quantiles_selected = np.random(self.N)

		quantiles_selected.shape = (1,self.N)
		if self.C > 0 and target:
			return np.add.reduce(self.target_model.predict(state,embedded_qs))

		return np.add.reduce(self.model.predict(state,embedded_qs))

	def fit(self,state, action_index, reward, next_state, done):
		quantiles_selected = np.random(self.N)
		quantile_in = self.process_quantiles(quantiles_selected)
		predicts = self.target_model.predict(next_state,quantiles_selected)
		
		target_f = np.reshape(target, [1, self.N])
		#if DEBUG:
			#print("target_f ",target_f[action_index][0], "target ", target)
		#if DEBUG:
		#print("fitting state:", state,",action:",action_index,",reward:",reward, " delta target_f ",target_f[action_index][0]-debug_target_f,"done:",done)
		
		self.model.fit(state_action, target_f,epochs=1, verbose=0)


	def process_quantiles(self,quantiles_selected):
		# Move to class init (why reinitialise?)
		embedded_range = np.arange(self.embedding_dim) + 1 # Note Chainer and dopamine implementation
		embedded_range.shape = (self.embedding_dim,1)
		return np.cos(np.dot(embedded_range, quantiles_selected) * np.pi)

# Testing the code
if __name__ == "__main__":
	myAgent = distAgent(5,"TonyTester",N=5)
	state = [-0.8,0.8] 
	state = np.reshape(state, [1, 2])
	state1 = [0,0] 
	state1 = np.reshape(state1, [1, 2])
	next_state = [-1,0.9] 
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
		print("Tz:",myAgent.Tz(state,0.1))
		print("Next State Value:",myAgent.predict(next_state)[0])
		my_next_action = np.argmax(myAgent.predict(next_state)[0])
		print("choose action:",my_next_action)
		print("... and the probs:",myAgent.probs(next_state,my_next_action)[0])
		print("resulting in projTZ ", myAgent.projTZ_nTree(state,0.1,next_state,False,0,1))
		myAgent.fit(state,2,0.1,next_state,False)








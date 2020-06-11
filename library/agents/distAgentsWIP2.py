import numpy as np
from keras.models import Sequential
from keras.models import clone_model
# Sort this out...
from keras.layers import Dense, Softmax, Multiply, Add, Input, ReLU, Lambda, Layer, concatenate
from keras.initializers import RandomNormal

#from keras import Input
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
	def __init__(self, state_size, action_values, agent_name,C, alternative_target,UCB=False,UCBc = 1,tree_horizon = 3,market_data_size=0):
		self.action_size = len(action_values)
		self.action_values = action_values
		super(distAgent,self).__init__(state_size, self.action_size, agent_name,C, alternative_target,"dist",tree_horizon,market_data_size=market_data_size)
		self.UCB = UCB
		self.c = UCBc
		self.geometric_decay = True

		# Transformations
		self.trans_a = 2 / (np.amax(self.action_values) - np.amin(self.action_values))
		self.trans_b = -self.trans_a * np.amin(self.action_values) - 1

		if self.UCB:
			self.t = 1
			self.epsilon_greedy = False

	def step(self):
		if self.UCB:
			self.t += 1

		super(distAgent,self).step()

	def act(self, state):
		# No eps greedy required for 'UCB' type update

		# Predict return
		act_values = self.predict(state)

		if self.evaluate:
			return np.argmax(act_values[0])
		
		if self.UCB:
			if self.t < 50:
				act = random.randrange(self.action_size)
			else:
				self.ct = self.c * np.sqrt(np.log(self.t) / self.t)
				#act_values = self.predict(state)
				#print("var",self.variance(state))
				#print("act_values",act_values)
				act = np.argmax(act_values[0] + self.ct * np.sqrt(self.variance(state)))
				#print("Act",act)
			return act
		# random action
		if np.random.rand() <= self.epsilon:
			rand_act = random.randrange(self.action_size)
			return rand_act#random.randrange(self.action_size)
				
		return np.argmax(act_values[0])

	# THIS FUNCTION HAS REPLACED TRASNFORM ACTION - NEEDS PROPIGATING AND TESTING
	# PROBABLY ALSO NEED TO RESHAPE NEW MARKET STATE
	def _process_state_action(self,state,action_index):
		#print("state",state)
		if self.market_data_size > 0:
			local_state, market_state = state
		else:
			local_state = state

		action = self.action_values[action_index] * self.trans_a + self.trans_b
		local_state_action = np.reshape(np.append(local_state,action), [1, len(local_state[0]) + 1])
		if self.market_data_size > 0:
			# market_state[0] since *market_state collects all in list
			#print(market_state)
			market_state = np.reshape(market_state, [1, len(market_state),1])
			#print(local_state_action,market_state)
			return [local_state_action, market_state]

		return local_state_action

	# 'Virtual' Function
	def variance(self,state):
		assert False, "Variance must be overwritten by child"

class C51Agent(distAgent):

	def __init__(self,state_size, action_values, agent_name,N=51,C = 0,alternative_target = False,UCB = False,UCBc = 1,tree_horizon = 3,market_data_size=0):
		self.V_max = 0.05
		self.V_min = -0.1

		self.N = N # This could be dynamic depending on state?
		# This granularity is problematic - can we do this without discretisation?
		# Especially if V_min and V_max are not dynamic
		# Paper: increasing N always increases returns
		self.dz = (self.V_max - self.V_min) / (self.N - 1) # (2 * self.V_max)
		self.z = np.array(range(self.N)) * (self.V_max - self.V_min) / (self.N - 1) + self.V_min
		
		distAgent.__init__(self,state_size, action_values, agent_name,C, alternative_target,UCB,UCBc,tree_horizon,market_data_size=market_data_size)

	def return_mapping(self,state,ret,inverse = False):
		if inverse:
			return ret - state[0][0] * self.result_scaling_factor

		return ret + state[0][0] * self.result_scaling_factor

	def probs(self,state,action_index,target=False):
		state_action = self._process_state_action(state,action_index)
		
		if DEBUG:
			#print("probs of ",state_action,"are",self.model.predict(state_action))
			pass
		if target and self.C>0:
			res = self.target_model.predict(state_action)
		else:
			res = self.model.predict(state_action)
		return res

	def predict(self,state,target = False):
		#print("predict state",state)
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
		
		# If using market data
		if self.market_data_size > 0:
			input_layer = concatenate([state_in,self.stock_model.output])
		else:
			input_layer = state_in

		hidden1 = Dense(32, activation='relu')(input_layer)
		#hidden1_n = BatchNormalization() (hidden1)
		hidden2 = Dense(32, activation='relu')(hidden1)
		skip_layer = Add()([hidden1, hidden2])
		#hidden3 = Dense(5, activation='relu')(hidden2)
		outputs = Dense(self.N, activation='softmax')(skip_layer)
		if self.market_data_size > 0:
			model = Model(inputs=[state_in,self.stock_model.input], outputs=outputs)
		else:
			model = Model(inputs=state_in, outputs=outputs)
		model.compile(loss='categorical_crossentropy',
						optimizer=Adam(lr=self.learning_rate))
		return model


	def fit(self,state, action_index, reward, next_state, done,mem_index = -1):
		state_action = self._process_state_action(state,action_index)
		#print("fit",next_state)
		if self.tree_n > 1:
			target = self.projTZ_nTree(state,reward,next_state,done,self.tree_n,mem_index)
		else:
			target = self.projTZ(reward,next_state,done)
		target_f = np.reshape(target, [1, self.N])
		#if DEBUG:
			#print("target_f ",target_f[action_index][0], "target ", target)
		#if DEBUG:
		#print("fitting state:", state,",action:",action_index,",reward:",reward, " delta target_f ",target_f[action_index][0]-debug_target_f,"done:",done)
		# Needs reworking
		#local,market = state_action

		self.model.fit(state_action, target_f,epochs=1, verbose=0)


	def variance(self,state,target = False):
		res = self.vvar(state,range(len(self.action_values)),target = target) - np.power(self.predict(state,target),2)
		if not all(np.round(res[0],5) >= 0):
			print("WARNING")
			print(res)
			print(np.power(self.predict(state,target),2))
			print(self.t)
			print("mapped_z",self.mapped_z(state))
			print("probs0",self.probs(state,0))
		#assert all(np.round(res[0],5) >= 0)
		return np.max(res,0)#np.reshape(res, [1, res.shape[0]])

	def var_act(self,state,action_index,target = False):
		#state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1])
		#print("predicting ", state_action)
		dist = self.probs(state,action_index,target = target)[0]
		return np.sum(dist * np.power(self.mapped_z(state),2))

	def vvar(self,state,action_indices,target = False):
		return np.vectorize(self.var_act,excluded=['state'] )(state = state,action_index = action_indices,target = target)

# Build the network seperately
class IQNNetwork(Model):
	def __init__(self,state_size,N,state_model_size_out,embedding_dim):
		super(IQNNetwork,self).__init__()
		self.N = N
		self.embedding_dim = embedding_dim
		self.state_size = state_size
		self.state_model_size_out = state_model_size_out
		self.state_hidden1 = Dense(8, activation='relu')
		self.state_hidden2 = Dense(self.state_model_size_out, activation='relu')
		self.q_hidden = Dense(self.state_model_size_out, activation='relu')
		self.main_hidden1 = Multiply()
		self.main_hidden2 = Dense(30, activation='relu')
		self.outputs = Dense(self.N, activation='linear')
		self.kappa = 2
		
		

		# State processing model
		state_input  = Input(shape=(self.state_size,))
		state_hidden1 = Dense(8, activation='relu')(state_input)
		self.process_state = Dense(self.state_model_size_out, activation='relu')(state_hidden1)
		
		self.phi = Dense(self.state_model_size_out, activation='relu')
		
		#state_hidden1 = Dense(8, activation='relu')(state_in)
		#state_hidden2 = Dense(self.state_model_size_out, activation='relu')(state_hidden1)



	def call(self,inputs):
		
		state_action,quantiles_selected = inputs
		embedded_quantiles = np.cos(np.dot(embedded_range, self.quantiles) * np.pi)
		embedded_quantiles.shape = (self.N,1)
		processed_state = self.process_state(state_action)
		combined = self.main_hidden1([processed_state,embedded_quantiles])
		combined = self.main_hidden2(combined)
		combined = self.outputs(combined)
		return combined

# https://www.tensorflow.org/guide/keras/custom_layers_and_models
class CosineBasisLayer(Layer):
	def __init__(self,units,input_dim,):
		super(CosineBasisLayer, self).__init__()
		# Needs looking at - how do we want to initialise weights in general?
		self.w = self.add_weight(shape=(input_dim, units,),initializer='random_normal',trainable=True)
		self.b = self.add_weight(shape=(input_dim,),initializer='zeros',trainable=True)

		self.units = units # Previously embedding_dim
		self.input_dim = input_dim

	def call(self, inputs):
		# Should these lines be converted to TF?
		#quantiles.shape = (1,len(self.quantiles))

		embedding_range = K.arange(1,self.units + 1) # Note this is DIFFERENT to paper
		# dot does not do automatic type conversion so...
		embedding_range = K.cast(embedding_range,dtype = "float32")
		embedding_range = K.reshape(embedding_range,(self.units,1))

		embedded_inputs = K.cos(K.dot(embedding_range, inputs) * K.constant(np.pi))
		embedded_inputs = K.reshape(embedded_inputs,(1,self.units,self.input_dim))

		res = K.dot(inputs, self.w) #+ self.b
		res = K.sum(res, axis = 1) + self.b
		return ReLU()(res)

# Temporarily swtiched to QRAgent
class QRAgent(distAgent):
	def __init__(self,state_size, action_values, agent_name,C, alternative_target = False,UCB=False,UCBc = 1,tree_horizon = 3,market_data_size=0):
		self.N = 32
		self.N_p = 8
		self.embedding_dim = 3
		self.state_model_size_out = 8
		#self.kappa = 2 # What should this be? Moved to loss fun
		#self.selected_qs = None
		#self.embedded_range = np.arange(self.embedding_dim) + 1 # Note Chainer and dopamine implementation
		#self.embedded_range.shape = (self.embedding_dim,1)

		# Temporarily uniformly parition [0,1] for the quantiles
		self.quantiles = np.arange(1,self.N + 1) / (self.N+1) # np.random.rand(self.N) This should be a random partition of [0,1]
		#print(self.quantiles)
		self.qi = 1 / (self.N)#self.quantiles[1] - self.quantiles[0] # WARNING: SHOULD BE DYNAMIC
		self.quantiles.shape = (1,len(self.quantiles))
		#self.embedded_quantiles = np.cos(np.dot(self.embedded_range, self.quantiles) * np.pi)
		#self.embedded_quantiles.shape = (1,self.embedding_dim,self.N)
		self.kappa = 1
		self.optimisticUCB = False
		super(QRAgent,self).__init__(state_size, action_values, agent_name,C, alternative_target,UCB,UCBc,tree_horizon,market_data_size)
			
	# https://stackoverflow.com/questions/55445712/custom-loss-function-in-keras-based-on-the-input-data
	@staticmethod
	def huber_loss_quantile(tau,kappa):
		#kappa = 2
		def loss(yTrue,yPred):
			bellman_errors =   yTrue - yPred
			#tau = np.array(quantile_in)
			#print("loss",(K.abs(tau - K.cast(bellman_errors < 0,"float32")) * huber_loss(yTrue,yPred)) / kappa)
			if kappa > 0:
				return (K.abs(tau - K.cast(bellman_errors < 0,"float32")) * huber_loss(yTrue,yPred)) / kappa
			return (K.abs(tau - K.cast(bellman_errors < 0,"float32")) * bellman_errors)
		return loss
		

	def _build_model(self):
		# Using Keras functional API
		
		state_in = Input(shape=(self.state_size + 1,))
		state_hidden1 = Dense(32, activation='relu')(state_in)
		state_hidden2 = Dense(32, activation='relu')(state_hidden1)
		#hidden3 = Dense(30, activation='relu')(hidden2)
		

		#quantiles_in = []
		#quantile_in = Input(shape=(self.N,))
		# This needs fixing - transforming to tf functions etc.

		#cosine_layer = CosineBasisLayer(64,input_dim = self.N)(quantile_in)
		#Lambda( lambda x: K.sum(x, axis=0), input_shape=(self.embedding_dim,self.N))(quantile_in)
		#quantile_col = ReLU()(quantile_col)
		
		# Full Model
		#main_hidden1 = Multiply()([cosine_layer, state_hidden2])
		main_hidden2 = Dense(32, activation='relu')(state_hidden2) #main_hidden1
		outputs = Dense(self.N, activation='linear')(main_hidden2)
		#main_model = Model(inputs=state_in, outputs=outputs)

		if self.market_data_size > 0:
			model = Model(inputs=[state_in,self.stock_model.input], outputs=outputs)
		else:
			model = Model(inputs=state_in, outputs=outputs)

		model.compile(loss = self.huber_loss_quantile(self.quantiles,self.kappa),
						optimizer=Adam(lr=self.learning_rate))
		'''
		main_model = IQNNetwork(self.state_size + 1,self.N,self.state_model_size_out,self.embedding_dim)

		main_model.compile(loss = self.huber_loss_quantile(quantiles_selected,self.kappa),
						optimizer=Adam(lr=self.learning_rate))
		'''
		return model

	def predict_action(self,state,action_index,target = False,above_med = False):
		#print("quantiles",self.quantiles)
		#print(np.dot(self.embedded_range, self.quantiles))
		#print("action index",action_index)
		state_action = self._process_state_action(state,action_index)
		#quantile_in = self.process_quantiles(quantiles_selected)
		#print("state action",state_action,"embedded_quantiles",self.embedded_quantiles)
		#print(self.model.summary())
		predict = self.predict_quantiles(state_action,target)
		#if self.C > 0 and target:
			#assert not above_med, "Why is variance being used with target?"
			#return np.add.reduce(self.qi * predict,1)
		#print("pre reduce predict_action output",np.add.reduce(self.model.predict([state_action,self.quantiles]),1))
		#print("predict_action output",self.model.predict([state_action,self.quantiles]))
		#predict = self.model.predict(state_action)[0]
		if above_med:
			predict = predict[0][int(len(predict)/2):]
			np.reshape(predict,[1,len(predict)])
			#return np.add.reduce(self.qi * predict,1)
		return np.add.reduce(self.qi * predict,1)

	# This function needs to be rolled into predict_action
	def predict_quantiles(self,state_action,target = False):
		#state_action = self._process_state_action(state,action_index)
		if DEBUG:
			print("predict state action", state_action)
		#state_action.shape = (1,len(state_action))
		if self.C > 0 and target:
			return self.target_model.predict(state_action) + state_action[0][0] / 2 + 0.5
		return self.model.predict(state_action) + state_action[0][0] / 2 + 0.5


	# predict function could be moved to distAgent and transitioned to np
	def predict(self,state,target = False):
		res = []
		for i in range(self.action_size):
			res.append(self.predict_action(state,i,target = target))

		res = np.array(res)
		res.shape = (1,len(res))
		return res

	def project(self,reward,next_state,done,horizon,mem_index):
		tree_success = False
		predict = []
		if not done:
			next_action_index = np.argmax(self.predict(next_state)[0])
			if horizon > 1 and mem_index < (self.memory.size - 1):
				state1, action1, reward1, next_state1, done1 = self.memory[mem_index + 1]
				if next_action_index == action1:
					predict = self.project(reward1,next_state1,done1,horizon - 1,mem_index + 1)
					tree_success = True
			if not tree_success:
				next_state_action = self._process_state_action(next_state,next_action_index)
				predict = self.predict_quantiles(next_state_action,target = True)
		else:
			predict = np.zeros(self.N)
			predict = np.reshape(predict, [1, self.N])

		return reward + self.gamma * predict

	def fit(self,state, action_index, reward, next_state, done,mem_index = -1):
		state_action = self._process_state_action(state,action_index)
		if DEBUG:
			print("State Action",state_action)
		# For Double Deep
		#print("predictions",self.predict(next_state,quantiles_selected = quantiles_selected))
		
		
		

		target = self.project(reward,next_state,done,self.tree_n,mem_index) - state[0][0] / 2 - 0.5
		#print(target)

			
		if DEBUG:
			print("Target", target)
		#print("target shape",target.shape)
		target_f = target #np.reshape(target, [1, self.N])
		#print("predicted", self.predict_quantiles(state,action_index,self.quantiles,target = False),"target",target_f)
		#print("loss", self.huber_loss_quantile(self.quantiles,1)(target_f,self.predict_quantiles(state,action_index,self.quantiles,target = False)))
		
		self.model.fit(state_action, target_f,epochs=1, verbose=0)


	def process_quantiles(self,quantiles_selected):
		# Move to class init (why reinitialise?)
		assert False

	def action_variance(self,state,action_index):
		state_action = self._process_state_action(state,action_index)
		predict = self.predict_quantiles(state_action)[0]
		if self.optimisticUCB:
			predict = predict[int(len(predict) / 2):]
		#print("action_var",np.add.reduce(self.qi * predict * predict))
		return np.add.reduce(self.qi * predict * predict)

	def variance(self,state):
		res = []
		#print("00",np.power(self.predict_action(state,0,above_med = self.optimisticUCB),2))
		for i in range(self.action_size):
			res.append(self.action_variance(state,i) - np.power(self.predict_action(state,i,above_med = self.optimisticUCB)[0],2))
		return res

# Testing the code
if __name__ == "__main__":
	pass

	'''
	myAgent = distAgent(5,"TonyTester",N=5)
	state = [-0.8,0.8] 
	state = np.reshape(state, [1, 2])
	state1 = [0,0] 
	state1 = np.reshape(state1, [1, 2])
	next_state = [-1,0.9] 
	next_state = np.reshape(state, [1, 2])
	myAgent.epsilon_min = 0.01
	
	myAgent = QRAgent(2,[0.1,0.5,1.0],"TonyTester",C=0,market_data_size = 16)
	state = [-1,1] 
	state = np.reshape(state, [1, 2])
	myAgent
	DEBUG = True
	myAgent.learning_rate = 0.0001
	state = [-1,1] 
	state = np.reshape(state, [1, 2])
	state1 = [-0.5,0.9] 
	state1 = np.reshape(state1, [1, 2])
	next_state = [-1,1] 
	next_state = np.reshape(state, [1, 2])
	print("state ", state,"predict ",myAgent.predict(state) ,"quantiles(",0,") ", myAgent.predict_quantiles(state,0,myAgent.quantiles_selected))
	for i in range(200):
		myAgent.fit(state,0,0.1,next_state,True)
		myAgent.fit(state,2,-0.2,next_state,True)
		myAgent.fit(state,0,0.5,next_state,True)
		myAgent.fit(state1,0,-0.5,next_state,False)
		myAgent.fit(state1,2,0.5,next_state,False)
	print("state ", state,"predict ",myAgent.predict(state) ,"quantiles(",0,") ", myAgent.predict_quantiles(state,0,myAgent.quantiles_selected),"quantiles(",2,") ", myAgent.predict_quantiles(state,2,myAgent.quantiles_selected))
	print("state ", state1,"predict ",myAgent.predict(state1) ,"quantiles(",0,") ", myAgent.predict_quantiles(state1,0,myAgent.quantiles_selected),"quantiles(",2,") ", myAgent.predict_quantiles(state1,2,myAgent.quantiles_selected))


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

	if False:
		print("Tz:",myAgent.Tz(state,0.1))
		print("Next State Value:",myAgent.predict(next_state)[0])
		my_next_action = np.argmax(myAgent.predict(next_state)[0])
		print("choose action:",my_next_action)
		print("... and the probs:",myAgent.probs(next_state,my_next_action)[0])
		print("resulting in projTZ ", myAgent.projTZ_nTree(state,0.1,next_state,False,0,1))
		myAgent.fit(state,2,0.1,next_state,False)
	'''







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
#from keras.losses import QuantileHuber # Doesn't exist

if __name__ == "__main__":
	from WIPAgentsv4 import distAgent
	DEBUG = True
else:
	from library.WIPAgentsv4 import distAgent
	DEBUG = False


class IQNAgent(distAgent):

	def __init__(self):
		self.N = 8
		self.N_p = 8
		self.embedding_dim = 64
		self.state_model_size_out = 8
		self.kappa = 2 # What should this be?
		self.selected_qs = None
		self.model = self._build_network()
	
	# https://stackoverflow.com/questions/55445712/custom-loss-function-in-keras-based-on-the-input-data
	def huber_loss_quantile(self,tau):
		def loss(yTrue,yPred):
			bellman_errors = yTrue - yPred
	    	return (K.abs(tau - K.to_float(bellman_errors < 0)) * huber_loss(yTrue,yPred)) / self.kappa
	    return loss
		

	def _build_network(self):
		# Using Keras functional API
		state_in = Input(shape=(self.state_size + 1,))
		state_hidden1 = Dense(8, activation='relu')(state_in)
		state_hidden2 = Dense(self.state_model_size_out, activation='relu')(state_hidden1)
		#hidden3 = Dense(30, activation='relu')(hidden2)

		quantiles_in = Input(shape=(self.N,))
		selected_qs.shape = (1,self.N)
		embedded_range = np.arange(self.embedding_dim) + 1 # Note Chainer and dopamine implementation
		embedded_range.shape = (self.embedding_dim,1)

		embedded_qs = np.cos(np.dot(embedded_range, selected_qs) * np.pi)
		quantile_in = Input(shape=(self.N,self.embedding_dim))
		q_hidden = Dense(self.state_model_size_out, activation='relu')(quantile_in)

		# Full Model
		main_hidden1 = Multiply(q_hidden, state_hidden2)
		main_hidden2 = Dense(30, activation='relu')(main_hidden1)
		outputs = Dense(self.N, activation='linear')(hidden2)
		main_model = Model(inputs=(state_in,quantile_in), outputs=outputs)

		main_model.compile(loss = self.huber_loss_quantile(quantiles_in),
						optimizer=Adam(lr=self.learning_rate))

		return main_model

	def predict(self,state,target = False):
		if self.C > 0 and target:
			return self.target_model.predict(state)

		return self.model.predict(state)

	def fit(self,state, action_index, reward, next_state, done):
		quantiles_selected = np.random(self.N)
		predicts = self.target_model.predict(next_state,quantiles_selected)
		
		target_f = np.reshape(target, [1, self.N])
		#if DEBUG:
			#print("target_f ",target_f[action_index][0], "target ", target)
		#if DEBUG:
		#print("fitting state:", state,",action:",action_index,",reward:",reward, " delta target_f ",target_f[action_index][0]-debug_target_f,"done:",done)
		
		self.model.fit(state_action, target_f,epochs=1, verbose=0)

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


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


class IQNAgent(learningAgent):

	def __init__(self):
		self.N = 
		self.N_p = 
		self.embedding_dim = 64
		self.state_model_size_out = 8
		self.kappa = 2 # What should this be?
		self.selected_qs = None


	# For explanation see https://stackoverflow.com/questions/45961428/make-a-custom-loss-function-in-keras
	def set_huber_params(yTrue,yPred,selected_qs):
		bellman_errors = yTrue - yPred
		huber_loss = Huber()
	    return (K.abs(selected_qs - K.to_float(bellman_errors < 0)) * huber_loss(yTrue,yPred)) / self.kappa

	def huber_loss_quantile(selected_qs):
		def huber_loss_quantile_fin(yTrue,yPred):
	    	return set_huber_params(yTrue, yPred, selected_qs)
	    return huber_loss_quantile_fin


	def huber_loss_quantile(yTrue,yPred):
	    return K.sum(K.log(yTrue) - K.log(yPred))
	    q_loss = (K.abs(self.selected_qs - tf.stop_gradient(
        tf.to_float(bellman_errors < 0))) * huber_loss) / self.kappa


	def _build_network:
		# Using Keras functional API
		state_in = Input(shape=(self.state_size,))
		state_hidden1 = Dense(8, activation='relu')(state_in)
		state_hidden2 = Dense(self.state_model_size_out, activation='relu')(state_hidden1)
		#hidden3 = Dense(30, activation='relu')(hidden2)

		selected_qs = np.random.rand(self.N)
		selected_qs.shape = (1,self.N)
		embedded_range = np.arange(self.embedding_dim) + 1 # Note Chainer and dopamine implementation
		embedded_range.shape = (self.embedding_dim,1)

		embedded_qs = np.cos(np.dot(embedded_range, selected_qs) * np.pi)
		quantile_in = Input(shape=(self.N,self.embedding_dim))
		q_hidden = Dense(self.state_model_size_out, activation='relu')(quantile_in)

		# Full Model
		main_hidden1 = Multiply(q_hidden, state_hidden2)
		main_hidden2 = Dense(30, activation='relu')(main_hidden1)
		outputs = Dense(self.action_size, activation='linear')(hidden2)
		main_model = Model(inputs=(state_in,quantile_in), outputs=outputs)

		main_model.compile(loss='mse',
						optimizer=Adam(lr=self.learning_rate))


		

		quantile_model.
		return model
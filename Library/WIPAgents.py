import numpy as np
from keras.models import Sequential
from keras.models import clone_model
from keras.layers import Dense
from keras.optimizers import Adam

# Define support for the value distribution model
# Note that V_min and V_max should be dynamic and depend on vol - how would this work on real data (max historical?)
# Is V_min for the total value? In this case it can be capped by the remaining position
V_min = 0; V_max = 10 

N = 50 # This could be dynamic depending on state?
# This granularity is problematic - can we do this without discretisation?
# Especially if V_min and V_max are not dynamic
# Paper: increasing N always increases returns
dz = (V_max - V_min) / (N - 1)
z = np.array(range(N)) * (V_max - V_min) / (N - 1) + V_min
theta = np.ones(N)
gamma = 1 # Discount factor
learning_rate = 0.001
# This would result in uniform prob (not sure if this is the right approach)
state_size = 2



def probs(state,action):
	state_action = np.reshape(np.append(state,action), [1, len(state[0]) + 1])
	return model.predict(state_action)
	#return np.exp(theta(i,x,a)/np.sum(np.exp(theta(i,x,a))))

#def predict(state,)

def Tz(reward):
	Tz = reward + gamma * z
	return Tz

# Think of how to do this in a more numpy way
def projTZ(reward,next_state):
	next_action = np.argmax(model.predict(next_state)[0])
	res = []
	for i in range(N):
		res.append(np.sum(bound(1 - np.abs(bound(Tz(reward),V_min,V_max) - z)/dz,0,1) * prob(next_state,next_action)))
	return res

def bound(vec,lower,upper):
	return np.minimum(np.maximum(vec,lower),upper)

def _build_model():
	model = Sequential()
	model.add(Dense(5, input_dim=(state_size + 1), activation='relu')) # 1st hidden layer; states as input
	model.add(Dense(5, activation='relu')) # 2nd hidden layer
	model.add(Dense(N, activation='softmax')) 
	model.compile(loss='categorical_crossentropy',
					optimizer=Adam(lr=learning_rate))
	return model

# This is currently 1D - won't accept multiple agents
def fit(state, action, reward, next_state, done):
	pass

# Testing the code
if __name__ == "__main__":
	model = _build_model()
	print(bound(Tz(1),0,10))
	state = [1,-1] 
	state = np.reshape(state, [1, 2])

	print(probs(state,1))


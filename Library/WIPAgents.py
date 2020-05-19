import numpy as np

# Define support for the value distribution model
# Note that V_min and V_max should be dynamic and depend on vol - how would this work on real data (max historical?)
# Is V_min for the total value? In this case it can be capped by the remaining position
V_min = 0, V_max = 10, N = 50 # This could be dynamic depending on state?
# This granularity is problematic - can we do this without discretisation?
# Especially if V_min and V_max are not dynamic
# Paper: increasing N always increases returns
dZ = (V_max - V_min) / (N - 1)
theta = np.ones(N)
gamma = 1 # Discount factor
# This would result in uniform prob (not sure if this is the right approach)

def prob(i,x,a):
	assert(i >= 0 and i <= N)

	#return np.exp(theta(i,x,a)/np.sum(np.exp(theta(i,x,a))))

def fit(state, action, reward, next_state, done):
	Tz = reward + gamma


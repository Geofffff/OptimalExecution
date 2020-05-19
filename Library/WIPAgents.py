import numpy as np

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
# This would result in uniform prob (not sure if this is the right approach)

def prob(x,a):
	# Return vector of probs - if we ever need an individual prob then may be more efficient including i as a param
	assert(i >= 0 and i <= N)

	#return np.exp(theta(i,x,a)/np.sum(np.exp(theta(i,x,a))))	

def Tz(reward):
	Tz = reward + gamma * z
	return Tz

# Think of how to do this in a more numpy way
def projTZ(reward):
	res = []
	for i in range(N):
		res.append(np.sum(bound(1 - np.abs(bound(Tz(reward),V_min,V_max) - z)/dz,0,1) * prob(next_state,next_action)))
	return res
	
def bound(vec,lower,upper):
	return np.minimum(np.maximum(vec,lower),upper)

# This is currently 1D - won't accept multiple agents
def fit(state, action, reward, next_state, done):
	pass




# Testing the code
if __name__ == "__main__":
	print(bound(Tz(1),0,10))


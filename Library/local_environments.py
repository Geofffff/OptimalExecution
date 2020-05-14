#from numpy import exp
import numpy as np
    
class agent_environmentM:

    def __init__(self, market, position,num_steps,terminal,action_values_pct,n_strats):
        self.m = market
        self.initial = np.ones(n_strats) * position
        self.n_strats = n_strats
        self.reset()
        self.terminal = terminal
        self.step_size = terminal / num_steps
        # Possible amounts to sell: 0 - 10% of the total position
        self.action_values = np.array(action_values_pct) * position 
        self.num_actions = len(self.action_values)
        self.state_size = 2


    def sell(self,volumes):
        capped_volume = np.minimum(volumes,self.position)
        self.position -= capped_volume
        returns = self.m.sell(capped_volume,self.step_size) 
        self.cash += returns
        return returns

    def reset(self):
        self.position = self.initial.copy()
        self.cash = np.zeros(self.n_strats)
        self.time = 0
        self.m.reset()
        return self.state() # State not dynamic (full = False)

    def progress(self,dt):
        self.m.progress(dt)
        self.time += dt

    def state(self,full = False):

        times = np.ones(self.n_strats) * self.time
        # TODO: Store state as a seprate variable
        states = np.vstack((self.position,times))
        
        return np.vstack((self.position,times)).T
    
    def step(self,actions):
        self.progress(self.step_size)
        
        rewards = self.sell(self.action_values[actions])
        done = (self.position == 0) + (self.time >= self.terminal)
        done = np.array(done,dtype = bool)
            
		# Reward is currently just the returned cash / 100...
		# Not sure what the last value of the tuple should be??
        
        # Ufortunately this is no longer the format of the AI gym env
        # Ideally this would be flexible depending on the input (array vs scalar)
        return self.state(), rewards, done
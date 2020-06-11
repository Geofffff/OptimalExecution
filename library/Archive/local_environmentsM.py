#from numpy import exp
import numpy as np

class agent_environmentM:
    '''Local Environment for a trading agent'''
    def __init__(self, market, position,n_trades,terminal,action_values_pct,n_strats):
        
        # Parameters
        self.n_strats = n_strats
        self.state_size = 2 # ???

        # Market environment
        self.m = market
        self.market_data = (self.m.n_hist_prices > 0) # bool: use market data

        # Local environment
        self.initial_position = np.ones(n_strats) * position
        self.terminal = terminal # In seconds
        self.step_size = terminal / n_trades
        self.reset()
        
        # Possible amounts to sell: 0 - 10% of the total position
        self.action_values = np.array(action_values_pct) * position 
        self.num_actions = len(self.action_values)
        #self.reward_scaling = self.initial / (num_steps)


    def sell(self,volumes):
        capped_volume = np.minimum(volumes,self.position)
        #print("time ", self.time, "capped_volume ", capped_volume, "state ", self.state())
        self.position -= capped_volume
        returns = self.m.sell(capped_volume,self.step_size) 
        self.cash += returns
        #print("time ", self.time, "capped_volume ", capped_volume, "state ", self.state(),"returns ", returns)
        return returns, capped_volume

    def reset(self,training = True):
        self.position = self.initial_position.copy()
        self.cash = np.zeros(self.n_strats)
        self.time = 0
        self.m.reset(self.step_size,training)
        return self.state() # State not dynamic (full = False)

    def progress(self,dt):
        self.m.progress(dt)
        self.time += dt

    def state(self,full = False):

        times = np.ones(self.n_strats) * 2 * self.time / self.terminal - 1
        # TODO: Store state as a seprate variable
        res = np.vstack((2 * self.position/self.initial_position[0] - 1,times)) # Assuming initial position always the same
        if self.market_data:
            return res.T, self.m.state()
        
        return res.T
    
    def step(self,actions):
        self.progress(self.step_size)
        time_out = (round(self.time,7) >= self.terminal)
        if time_out:
            rewards, amount = self.sell(self.position)
            #print("Selling off",amount)
        else:
            rewards, amount = self.sell(self.action_values[actions])
            #print("selling ",amount)
        done = (self.position <= 0) + (round(self.time,7) >= self.terminal)
        if any(self.position < 0):
            print("Warning position is ",self.position)
        done = np.array(done,dtype = bool)

        rewards = self.scale_rewards(rewards,amount)
        #print("times ",self.time, self.terminal)
            
		# Reward is currently just the returned cash / 100...
		# Not sure what the last value of the tuple should be??
        
        # Ufortunately this is no longer the format of the AI gym env
        # Ideally this would be flexible depending on the input (array vs scalar)
        return self.state(), rewards, done


    def scale_rewards(self,rewards,amount):
        #print(type(self.m.stock.initial),self.m.stock.initial)
        return (rewards) / (self.initial_position[0] ) # /* self.m.stock.initial


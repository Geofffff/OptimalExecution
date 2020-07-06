import numpy as np

class agent_environment:
    '''Local Environment for a trading agent using market orders'''
    def __init__(self, 
        market, 
        position, # Position to exit
        n_trades, # Number of trades
        action_values_pct # Actions corresponding to the percentage of stock to sell
        ):
        
        # Parameters
        self.state_size = 2 # ???

        # Market environment
        self.m = market
        self.market_data = (self.m.n_hist_prices > 0) # bool: use market data

        # Local environment
        self.initial_position =  position
        self.step_size = 1 / n_trades
        self.reset()
        
        # Possible amounts to sell: 0 - 10% of the total position
        self.action_values = np.array(action_values_pct) * position / n_trades
        self.num_actions = len(self.action_values)
        #self.reward_scaling = self.initial / (num_steps)


    def sell(self,volume):
        capped_volume = np.minimum(volume,self.position)
        self.position -= capped_volume
        returns = self.m.sell(capped_volume,self.step_size) 
        self.cash += returns
        return returns, capped_volume

    def reset(self,training = True):
        self.position = self.initial_position
        self.cash = 0
        self.time = -1 # Time runs from -1 to 1
        self.m.reset(self.step_size,training)
        return self.state()

    def state(self,full = False):
        res = [2 * self.position/self.initial_position - 1,self.time]
        res = np.reshape(res,(1,len(res)))
        if self.market_data:
            market_state = self.m.state()
            market_state = np.reshape(market_state,(1,len(market_state),1)) - 1
            return [res, market_state]
        
        return res
    
    def step(self,action):
        self.m.progress(self.step_size)
        self.time += 2 * self.step_size

        time_out = (round(self.time,7) >= 1)
        
        if time_out:
            rewards, amount = self.sell(self.position)
        else:
            rewards, amount = self.sell(self.action_values[action])
        
        done = (self.position <= 0) + time_out
        if self.position < 0:
            print("Warning position is ",self.position)

        rewards = self.scale_rewards(rewards,amount)
        
        return self.state(), rewards, done


    def scale_rewards(self,rewards,amount):
        return (rewards) / (self.initial_position ) # /* self.m.stock.initial

class orderbook_environment(agent_environment):
    '''Local Environment for a trading agent using both limit orders and market orders'''
    def __init__(self, 
        market, 
        position, # Position to exit
        n_trades, # Number of trades
        mo_action_values_pct # market order actions corresponding to the percentage of stock to sell
        ):
        super(orderbook_environment,self).__init__(market,position,n_trades,mo_action_values_pct)
        self.state_size = 7 # position, time, bid, ask, bidSize, askSize, loPos
        #self.lo_action_values = np.array(lo_action_values_pct) * position / n_trades


    def place_limit_order(self,size):
        # WARNING order capping must take place at agent level
        returns = self.m.place_limit_order(size)
        self.cash -= returns
        return returns

    def state(self):
        '''Returns the current state of the agent as a tuple with the following values:
        position (scaled), time (scaled), bid (scaled by market), ask (scaled by market),
        askSize, bidSize, total value of agents limit orders'''

        # How should bidSize and askSize be scaled?
        # We need to scale them in this function, not at market level as they have a
        # directly interpretable value there.
        res = [2 * self.position/self.initial_position - 1,
                self.time,
                self.m.stock.bid, 
                self.m.stock.ask, 
                self.m.stock.bidSize, 
                self.m.stock.askSize,
                self.m.lo_total_pos]
        res = np.reshape(res,(1,len(res)))
        if self.market_data:
            market_state = self.m.state()
            market_state = np.reshape(market_state,(1,len(market_state),1)) - 1
            return [res, market_state]
        
        return res

    def sell(self,volume):
        # For the orderbook agent volume is a 2 tuple containing MO and LO volume
        # First update LOB...
        delta_position, returns = self.market.exectute_lob()
        self.position -= delta_position
        # ... then execute any market orders ...
        capped_mo_volume = np.minimum(volume[0],self.position)
        self.position -= capped_mo_volume
        returns += self.m.sell(capped_mo_volume,self.step_size) 
        # ... then add limit orders up to remaining position - currently standing LOs
        capped_lo_volume = np.max(np.minimum(volume[1],self.position - self.market.lo_total_pos),0)
        self.market.place_limit_order(capped_lo_volume)
        self.cash += returns

        # To avoid agents "remembering" order sizes wrongly we must adjust volume within act
        return returns, capped_mo_volume, capped_lo_volume




    


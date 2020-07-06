from random import gauss, randint, shuffle
from collections import deque
import numpy as np
np.random.seed(84)


class bs_stock:

	def __init__(self, initial, drift, vol, terminal = 1):
		self.initial = initial
		self.drift = drift
		self.vol = vol
		self.terminal = terminal
		self.reset()

	def generate_price(self,dt,St = None):
		dt = dt * self.terminal
		if St is None:
			St = self.price

		self.price = St * np.exp((self.drift - 0.5 * self.vol ** 2) * dt + self.vol * dt**0.5 * gauss(0,1))
		return self.price

	def reset(self,training=None):
		self.price = self.initial


class mean_rev_stock(bs_stock):
	def __init__(self, initial, drift, vol,reversion):
		bs_stock.__init__(self,initial,drift,vol)
		self.reversion = reversion
		self.alpha = 0
		self.eps = 0.05
		self.xi = 0.5
		self.lamb = 0.5
		self.beta = 0.01

	def generate_price(self,dt,St = None):
		if St is None:
			St = self.price

		jump = 0
		self.M = np.random.poisson(self.lamb * 2)
		# Note we are assuming here that dt is sufficiently small that >1 jumps is highly unlikely
		if Mp < dt:
			if random.random() < 0.5:
				jump = 1
			else:
				jump = -1

		self.alpha += - self.alpha * self.xi * dt + self.beta * dt**0.5 * gauss(0,1) + jump * self.eps * gauss(0,1)
		self.price = St * np.exp((self.drift - 0.5 * self.vol) * dt + self.vol * dt**0.5 * gauss(0,1))
		return self.price

class signal_stock(bs_stock):
	def __init__(self, initial, vol, gamma, drift_vol):
		self.gamma = gamma
		self.drift_vol = drift_vol
		super(signal_stock,self).__init__(initial, vol, 0)

	def generate_price(self,dt,St = None):
		if St is None:
			St = self.price

		# Mean reverting OU process for the signal
		self.signal = - self.gamma * self.signal * dt + self.signal_vol * dt**0.5 * gauss(0,1)

		self.price = St * np.exp((self.signal - 0.5 * self.vol ** 2) * dt + self.vol * dt**0.5 * gauss(0,1))
		return self.price

	def reset(self):
		self.price = self.initial
		self.signal = 0 # Always start with no signal (could improve this)

class real_stock:
	def __init__(self,data,n_steps = 60, data_freq = 6,recycle = True,n_train = 0):
		self.recycle = recycle
		self.n_steps = n_steps
		self.df_prices = data
		self.hist_buffer = 0
		self.data_freq = data_freq # 1M = 60
		
		# Partition data into training and testing data
		self.n_train = n_train
		self.partition_training = (self.n_train > 0)

		if not self.recycle:
			print("Assuming 1M frequency",("with" if recycle else "without"),"recycling")
			self.final_period = floor((len(data) - self.hist_buffer) / self.n_steps) - self.n_train
			self.available_periods = range(self.final_period)
			shuffle(self.available_periods)

		self.period_index = -1
		self.reset()

		#self.df_prices[self.data_index] # This will need changing with the format of input

	def reset(self,training=True):
		if (not training) and self.partition_training:
			self.data_index = randint(len(self.df_prices) - self.n_train * self.n_steps,len(self.df_prices) - self.n_steps - 1)
			#print(self.data_index,len(self.df_prices) - self.n_steps)
		else:
			if not self.recycle:
				self.period_index += 1
				assert self.period_index <= self.final_period, "Dataset finished"
				self.data_index = self.period_index * self.n_steps * self.hist_buffer
			else:
				self.data_index = randint(self.hist_buffer,len(self.df_prices) - self.n_steps * (1 + self.n_train))
		self.in_period_index = 0
		
		self.initial = self.df_prices[self.data_index]
		self.price = 1

	def _update_data_index(self,dt):
		index_update = dt * self.n_steps
		
		assert index_update.is_integer(), "Step size must be an integer unit of time"
		index_update = int(index_update)
		self.data_index += index_update
		self.in_period_index += index_update
		assert self.in_period_index <= self.n_steps, "Stock price requested outside of period"

	def generate_price(self,dt):
		self._update_data_index(dt)

		self.price = self.df_prices[self.data_index] / self.initial

		# WARNING: For now we return a scaled price (scaled by initial price at the start of every episode)
		error = np.isnan(self.price)
		assert not error, "Price must be a finite real number"

		return self.price

	def hist_price(self,n,dt):
		dt_adj = dt * self.n_steps
		assert dt_adj.is_integer(), "Time step must be an integer"
		dt_adj = int(dt_adj)
		res = []
		for i in range(n):
			res.append(self.df_prices[self.data_index + (- n + i + 1 ) * dt_adj])
		return np.array(res) / self.initial

class real_stock_lob(real_stock):

	def __init__(self,data,n_steps = 60, data_freq = 6,recycle = True,n_train = 100):
		self.data = data
		assert list(self.data.columns) == ["bid","bidSize","ask","askSize","buyMO"], "input data must be of the form [bid,bidSize,ask,askSize,buyMO]"
		print(type(self))
		super(real_stock_lob,self).__init__(data["bid"],n_steps, data_freq,recycle,n_train)

	def reset(self,training = True):
		super(real_stock_lob,self).reset(training)
		# Override the initial price with the mid price
		self.initial = self.df_prices[self.data_index]
		self.generate_price(first = True)

	def generate_price(self,dt = None,first = False):
		if not first:
			assert dt is not None, "dt argument required for non initial price"
			self._update_data_index(dt)

		self.price = self.df_prices[self.data_index] / self.initial

		# WARNING: For now we return a scaled price (scaled by initial price at the start of every episode)
		error = np.isnan(self.price)
		assert not error, "Price must be a finite real number"

		self.bid = self.data["bid"][self.data_index] / self.initial
		self.ask = self.data["ask"][self.data_index] / self.initial
		# TODO: how do we scale these?
		self.bidSize = self.data["bidSize"][self.data_index] 
		self.askSize = self.data["askSize"][self.data_index]
		self.market_orders = self.data["buyMO"][self.data_index]
		if not first:
			# Can this be depreciated?
			return self.price


# Need to rework to record n previous prices...
class market:
	'''Basic market model, base class for more complex models'''

	def __init__(self,stock_,n_hist_prices = 0):
		self.k = 0.000001 # I've scaled these to represent the fact that the position is now 100000 not 10
		self.b = 0.00000005
		self.stock = stock_
		self.stock.hist_buffer = n_hist_prices
		self.spread = 0
		self.price_adjust = 1
		self.n_hist_prices = n_hist_prices

	def sell(self,volume,dt):
		'''sell *volume* of stock over time window dt, volume is np array'''
		#print(volume / dt)
		self.price_adjust *= self.exp_g(volume)
		#print("rate ", volume/dt, "price_adjust ",self.price_adjust)        
		ret = (self._adjusted_price() - self.f(volume/dt) - 0.5 * self.spread) * volume 
		#print("return",ret)
		return ret

	def g(self,v):
		return v * self.b
    
	def exp_g(self,v):
		return np.exp(-self.g(v))

	def f(self,v):
		return v * self.k#0.00186 # Temporarily adjusting by 10 to account for non unit terminal
		# What should this be?
			# - HFT book (position = 1, terminal  = 1, k = 0.01)
			# Since position = 1 but terminal = 10 I've *10
			# NOW CHANGED TO 0.001 (V LOW) TO TEST STOCK PROCESSING NET

	def _adjusted_price(self):
		#print("price",self.stock.price,"adjust",self.price_adjust)
		return self.stock.price * self.price_adjust

	def reset(self,dt,training = True):
		self.stock.reset(training)
		self.price_adjust = 1

		for i in range(self.n_hist_prices):
			self.hist_prices = self.stock.hist_price(self.n_hist_prices,dt)
		#return self.hist_prices

	def progress(self,dt):
		self.stock.generate_price(dt)
		# MULTIPLE STRATS NOT SUPPORTED HERE
		if self.n_hist_prices > 0:
			self.hist_prices[:-1] = self.hist_prices[1:]; self.hist_prices[-1] = self._adjusted_price()

	def state(self):
		return self.hist_prices

class lob_market(market):

	def __init__(self,stock_,n_hist_prices):
		#self.stock = stock_
		super(lob_market,self).__init__(stock_,n_hist_prices)
		self.reset_lo()
		self.b = 0 # No permenant market impact
		self.lo_cap = 10 
		print("LOs capped at 10")
		# For now LOs can be made but not cancelled
		self.perc_fee = 0 # Fee charged for all LOs upon posting

	def place_limit_order(self,size):
		capped_size = max(min(self.lo_cap - self.lo_total_pos,size),0)
		fee = 0
		if not capped_size == 0:
			self.lo_size.append(capped_size)
			self.lo_position.append(self.stock.askSize)
			self.lo_total_pos += capped_size
			self.lo_adjust += capped_size
			print(self.lo_position)
			fee = size * self.perc_fee
		return fee

	def reset_lo(self):
		# Cancel all limit orders
		self.lo_position = []
		self.lo_total_pos = 0
		self.lo_size = []
		self.lo_adjust = 0
		self.lo_price = self.stock.ask # TODO: Implement
		self.warn_solo_price = False		

	def execute_lob(self):
		# Stock market orders in considered time window
		# NOTE: We are assuming that lo_position is monotonically increasing
		assert self._monotonic_increasing(self.lo_position), "Order positons should be increasing"
		#print("lo positions",self.lo_position)
		# Diagram letters in comments
		self.lo_position -= self.stock.market_orders
		#print("market orders",self.stock.market_orders)
		#print("new lo positions",self.lo_position)
		#print("lo size",self.lo_size)
		pos_plus_size = self.lo_position + self.lo_size #E
		#print("pos plus size",pos_plus_size)
		pos_lt_zero = (self.lo_position < 0) #D
		#print("pos lt zero",pos_lt_zero)
		fulfilled_sizes = (self.lo_size - np.maximum(pos_plus_size,0)) * pos_lt_zero #F
		#print("fulfilled_sizes",fulfilled_sizes)
		fulfilled_total = np.sum(fulfilled_sizes)
		self.lo_total_pos -= fulfilled_total
		#print("fulfilled total",fulfilled_total)

		# Now update lo_size and lo_position to reflect changes

		# First check that the top of book ask hasn't changed
		if self.lo_price != self.stock.ask:
			
			if self.lo_price < self.stock.ask:
				# Price has become more competitive
				self.reset_lo()
			else:
				# Price less competitive
				if self.lo_total_pos > 0:
					self.warn_solo_price = True
				else:
					self.warn_solo_price = False
					self.reset_lo()


		self.lo_size = self.lo_size * (1 - pos_lt_zero) + np.maximum(pos_plus_size,0) * pos_lt_zero
		# Remove orders where size = 0
		self.lo_size = self.lo_size[self.lo_size > 0]
		self.lo_position = np.maximum(self.lo_position,0)
		self.lo_position = self.lo_position[len(self.lo_position) - len(self.lo_size):]
		#print("new positions",self.lo_position)
		#print("new sizes",self.lo_size)

		# Return the volume * the ask price
		return fulfilled_total, fulfilled_total * self.stock.ask

	# Override state method
	# TODO: add in other market data
	def state(self):
		return self.hist_prices

	@staticmethod
	def _monotonic_increasing(x):
		dx = np.diff(x)
		return np.all(dx >= 0)











# End
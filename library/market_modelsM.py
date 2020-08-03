from random import gauss, randint, shuffle
from collections import deque
import numpy as np
#np.random.seed(84)
DEBUG = False
RARE_DEBUG = False # Print messages for rare events

class bs_stock:

	def __init__(self, initial, drift, vol, terminal = 1,n_steps=10):
		self.initial = initial
		self.drift = drift
		self.vol = vol
		self.terminal = terminal
		self.n_steps = n_steps
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
		self.data = data
		self.hist_buffer = self.n_steps
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

	def __str__(self):
		if self.recycle:
			recycling = "not"
		else:
			recycling = ""
		n_data = len(self.data["bid"])
		return f"Real Stock, using {n_data} data points, {recycling} recycling data points. Sampling over {self.n_steps} steps."

	def reset(self,training=True):
		if (not training) and self.partition_training:
			self.data_index = randint(len(self.data['bid']) - self.n_train * self.n_steps,len(self.data['bid']) - self.n_steps - 1)
			#print(self.data_index,len(self.df_prices) - self.n_steps)
		else:
			if not self.recycle:
				self.period_index += 1
				assert self.period_index <= self.final_period, "Dataset finished"
				self.data_index = self.period_index * self.n_steps * self.hist_buffer
			else:
				self.data_index = randint(self.hist_buffer,len(self.data['bid']) - self.n_steps * (1 + self.n_train))
		self.in_period_index = 0
		
		self.initial = self.data['bid'][self.data_index]
		self.price = 1

	def _update_data_index(self,dt):
		index_update = dt * self.n_steps
		
		assert index_update.is_integer(), "Step size must be an integer unit of time"
		index_update = int(index_update)
		if type(self).__name__ == "real_stock_lob":
			assert index_update == 1, "For real orderbook stocks trades must be made every second"
		self.data_index += index_update
		self.in_period_index += index_update
		assert self.in_period_index <= self.n_steps, "Stock price requested outside of period"
		#print("data index",self.data_index)

	def generate_price(self,dt):
		self._update_data_index(dt)

		self.price = self.data['bid'][self.data_index] / self.initial

		# WARNING: For now we return a scaled price (scaled by initial price at the start of every episode)
		error = np.isnan(self.price)
		assert not error, "Price must be a finite real number"

		return self.price

	def get_hist(self,n,dt,col):
		dt_adj = dt * self.n_steps
		assert dt_adj.is_integer(), "Time step must be an integer"
		dt_adj = int(dt_adj)
		res = []
		for i in range(n):
			res.append(self._scale(col,self.data_index + (- n + i + 1 ) * dt_adj))
		return np.array(res)

	def _scale(self,col,index):
		# Allows for columns to be scaled in a unique way
		if col == "bid" or col == "ask":
			return self.data[col][index] / self.initial 
		else:
			raise "Unknown column"

class real_stock_lob(real_stock):

	def __init__(self,data,n_steps = 60, data_freq = 6,recycle = True,n_train = 100):
		self.data = data
		assert set(self.data.columns).issubset({"bid","bidSize","ask","askSize","buyMO","sellMO","buySellImb","orderImb","spread"}), f'input columns {self.data.columns} must be a subset of ("bid","bidSize","ask","askSize","buyMO","sellMO","buySellImb","orderImb")'
		super(real_stock_lob,self).__init__(data["bid"],n_steps, data_freq,recycle,n_train)
		print("WARNING: Several market data inputs have been forced to 0 temporarily")

	def reset(self,training = True):
		super(real_stock_lob,self).reset(training)
		# Override the initial price with the mid price
		self.initial = (self.data["bid"][self.data_index] + self.data["ask"][self.data_index]) / 2
		self.initial_spread = self.data["spread"][self.data_index]
		self.generate_price(first = True)

	def generate_price(self,dt = None,first = False):
		if not first:
			assert dt is not None, "dt argument required for non initial price"
			self._update_data_index(dt)

		self.price = self.data['bid'][self.data_index] / self.initial

		# WARNING: For now we return a scaled price (scaled by initial price at the start of every episode)
		error = np.isnan(self.price)
		assert not error, "Price must be a finite real number"
		# Extract and rescale core data
		self.bid = self._scale("bid",self.data_index)
		self.ask = self._scale("ask",self.data_index)
		# TODO: how do we scale these?
		self.bidSize = self._scale("bidSize",self.data_index)
		self.askSize = self._scale("askSize",self.data_index)
		self.market_orders = self._scale("buyMO",self.data_index)

		# Extract and scale alt data (using buySellImb as proxy for presence of all alt data)
		
		if "buySellImb" in self.data.columns:
			self.buySellImb = self._scale("buySellImb",self.data_index)
			self.orderImb = self._scale("orderImb",self.data_index)

		if not first:
			# Can this be depreciated?
			return self.price

	def _scale(self,col,index,center = False):
		# Allows for columns to be scaled in a unique way
		if col == "bid" or col == "ask":
			return self.data[col][index] / self.initial - int(center)
		elif col == "askSize" or col == "bidSize" or col == "buyMO":
			return 0#self.data[col][index] - int(center)
		elif col ==  "buySellImb":
			res = self.data[col][index] 
			return res / max(self.data["buyMO"][index],self.data["sellMO"][index]) - 0.5 * int(center)
		elif col == "orderImb":
			res = self.data[col][index]
			return res / max(self.data["bidSize"][index],self.data["askSize"][index]) - 0.5 * int(center)
		elif col == "spread":
			return self.data[col][index] / self.initial_spread - int(center)
		else:
			raise "Unknown column"

# Need to rework to record n previous prices...
class market:
	'''Basic market model, base class for more complex models'''

	def __init__(self,stock_,n_hist_prices = 0):
		self.k = 0.0000000186 # I've scaled these to represent the fact that the position is now 100000 not 10
		self.b = 0.000000005
		self.stock = stock_
		self.stock.hist_buffer = n_hist_prices
		self.spread = 0
		self.price_adjust = 1
		self.n_hist_prices = n_hist_prices
		if n_hist_prices > 0:
			self.hist = {
				"bid" : []
			}

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
		if self.n_hist_prices > 0:
			for col in self.hist:
				self.hist[col] = self.stock.get_hist(self.n_hist_prices,dt,col = col)
		#print(list(self.hist.values()))
			
	def progress(self,dt):
		self.stock.generate_price(dt)
		if self.n_hist_prices > 0:
			for col in self.hist:
				self.hist[col][:-1] = self.hist[col][1:]; self.hist[col][-1] = self._adjusted_price()

	def state(self):
		#print(tuple(self.hist.values()))
		#print(tuple(self.hist.values()))
		return list(self.hist.values())

class lob_market(market):

	def __init__(self,stock_,n_hist_prices):
		#self.stock = stock_
		super(lob_market,self).__init__(stock_,n_hist_prices)
		self.reset_lo()
		self.b = 0 # No permenant market impact
		self.lo_cap = 100000 
		print(f"LOs capped at {self.lo_cap}")
		# For now LOs can be made but not cancelled
		self.perc_fee = 0 # Fee charged for all LOs upon posting
		self.hist = {
			"bid" : [],
			"ask" : [],
			"askSize" : [],
			"bidSize" : [],
			"buySellImb" : [],
			"orderImb" : [],
			"spread" : []
		}

	def place_limit_order(self,size):
		capped_size = max(min(self.lo_cap - self.lo_total_pos,size),0)
		fee = 0
		if not capped_size == 0:
			self.lo_size = np.append(self.lo_size,capped_size)
			if len(self.lo_position) > 0:
				end_of_queue = max(self.lo_position[-1],float(self.stock.askSize)) + self.lo_adjust
			else:
				end_of_queue = self.stock.askSize + self.lo_adjust

			self.lo_position = np.append(self.lo_position,end_of_queue)
			self.lo_total_pos += capped_size
			self.lo_adjust += capped_size
			if DEBUG:
				print(self.lo_position,self.stock.askSize + self.lo_adjust)
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
		self.lo_value = 0 # Purely for summary statistics purposes

	def execute_lob(self):
		# Stock market orders in considered time window
		if len(self.lo_position) == 0:
			return 0,0
		# NOTE: We are assuming that lo_position is monotonically increasing
		assert self._monotonic_increasing(self.lo_position), f"Order positons, {self.lo_position}, should be increasing"
		# Diagram letters in comments

		try:
			self.lo_position -= self.stock.market_orders
		except:
			assert False, f"something wrong above, pos {self.lo_position}, type {type(self.lo_position)} -= {type(float(self.stock.market_orders))}"

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
			
			if self.lo_price > self.stock.ask:
				# market price has become more competitive
				self.reset_lo()
				if RARE_DEBUG:
					print("Cancelling all limit orders")
			else:
				# market price less competitive
				if self.lo_total_pos > 0:
					if RARE_DEBUG:
						print("Agent offering is more competitive than the market")
					# Check overlapping LOs
					if self.stock.bid >= self.lo_price:
						fulfilled_total = self.lo_total_pos
						self.reset_lo()
						if RARE_DEBUG:
							print("Crossed Bid ask, fulfilling all LOs")
					else:
						self.warn_solo_price = True
						if RARE_DEBUG:
							print("Collapsing agents LOB")
						self.lo_position = np.array([0])
						self.lo_size = np.array([self.lo_total_pos])
				else:
					self.warn_solo_price = False
					self.reset_lo()
		else:
			# No top of book ask price change
			self.lo_size = self.lo_size * (1 - pos_lt_zero) + np.maximum(pos_plus_size,0) * pos_lt_zero
			if DEBUG:
				print("size",self.lo_size,"pos_lt",pos_lt_zero,"pos",self.lo_position)
			# Remove orders where size = 0
			self.lo_size = self.lo_size[self.lo_size > 0]
			self.lo_position = np.maximum(self.lo_position,0)
			self.lo_position = self.lo_position[len(self.lo_position) - len(self.lo_size):]
			

		# Now check that all limit orders are at minimum the market askSize
		if len(self.lo_position) > 0:
			# Check the final LO before preceeding
			order_delta = self.lo_position[-1] - self.stock.askSize
			# If the agents last limit order is now at the back then we can 
			# consolidate all "stranded" LOs past this point to one LO (equivalent)
			if self.stock.market_orders < order_delta:
				assert abs(self.lo_total_pos - np.sum(self.lo_size)) < 1, f"lo_position, {self.lo_size}, is not equal to the lo_total_pos, {self.lo_total_pos}, difference {abs(self.lo_total_pos - np.sum(self.lo_size))}"
				not_stranded = self.lo_position < self.stock.askSize
				self.lo_position = self.lo_position[not_stranded]
				if RARE_DEBUG:
					print("Collapsing some LOs")
				collapsed_lo_size = np.sum(self.lo_size * (1 - not_stranded.astype(int)))
				self.lo_position = np.append(self.lo_position,self.stock.askSize + self.lo_total_pos - collapsed_lo_size)
				self.lo_size = self.lo_size[not_stranded]
				self.lo_size = np.append(self.lo_size,collapsed_lo_size)

		#print("lob returns", fulfilled_total * self.stock.ask)
		assert fulfilled_total >= 0, "We can't have negative returns from LOs"
		self.lo_value += fulfilled_total
		return fulfilled_total, fulfilled_total * self.stock.ask


	@staticmethod
	def _monotonic_increasing(x):
		dx = np.diff(x)
		return np.all(dx >= 0)











# End
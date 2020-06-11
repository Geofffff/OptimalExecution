from random import gauss, randint, shuffle
from collections import deque
import numpy as np
np.random.seed(84)

class stock:
	
	def __init__(self, initial, drift, vol):
		self.initial = initial




class bs_stock:

	def __init__(self, initial, drift, vol):
		self.initial = initial
		self.drift = drift
		self.vol = vol
		self.reset()

	def generate_price(self,dt,St = None):
		if St is None:
			St = self.price

		self.price = St * np.exp((self.drift - 0.5 * self.vol ** 2) * dt + self.vol * dt**0.5 * gauss(0,1))
		return self.price

	def generate_path(self,T,grid_size):
		res = []
		self.reset()
		next_price = self.price
		res.append(next_price)
		dt = T / grid_size

		for i in range(grid_size):
			next_price = generate_price(dt)
			res.append(next_price)

	def reset(self,training=None):
		self.price = self.initial

	def __str__(self):
		print(f"Stock Price: {self.price} \n \
		 Black Scholes Dynamics, drift: {self.drift}, vol: {self.vol}")

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
	def __init__(self,data,n_steps = 3600, data_freq = 60,recycle = False,n_train = 0):
		self.recycle = recycle
		self.n_steps = n_steps
		self.df = data
		self.hist_buffer = 0
		self.data_freq = data_freq # 1M = 60
		
		# Partition data into training and testing data
		self.n_train = n_train
		self.partition_training = (self.n_train > 0)
		self.n_points_period = int(self.n_steps / self.data_freq)


		if self.recycle:
			pass
		else:
			print("Assuming 1M frequency",("with" if recycle else "without"),"recycling")
			self.final_period = floor((len(data) - self.hist_buffer) / self.n_points_period) - self.n_train
			self.available_periods = range(self.final_period)
			shuffle(self.available_periods)

		self.period_index = -1
		self.reset()

		#self.df[self.data_index] # This will need changing with the format of input

	def reset(self,training=True):
		if (not training) and self.partition_training:
			self.data_index = randint(len(self.df) - self.n_train * self.n_points_period,len(self.df) - self.n_points_period - 1)
			#print(self.data_index,len(self.df) - self.n_points_period)
		else:
			if not self.recycle:
				self.period_index += 1
				assert self.period_index <= self.final_period, "Dataset finished"
				self.data_index = self.period_index * self.n_points_period * self.hist_buffer
			else:
				self.data_index = randint(self.hist_buffer,len(self.df) - self.n_points_period * (1 + self.n_train))
		self.in_period_index = 0
		
		self.initial = self.df[self.data_index]
		self.price = 1


	def _scale_price(self,initial,price):
		pass

	def generate_price(self,dt):
		index_update = dt / self.data_freq
		assert index_update.is_integer(), "Step size must be an integer unit of time"
		index_update = int(index_update)
		self.data_index += index_update
		self.in_period_index += index_update

		self.price = self.df[self.data_index] / self.initial
		#print("period_index",self.period_index,"data_index",self.data_index)
		assert self.in_period_index <= self.n_points_period, "Stock price requested outside of period"
		
		# WARNING: For now we return a scaled price (scaled by initial price at the start of every episode)
		error = np.isnan(self.price)
		assert not error, "Price must be a finite real number"

		return self.price

	def hist_price(self,n,dt):
		dt_adj = dt / self.data_freq
		assert dt_adj.is_integer(), "Time step must be an integer"
		dt_adj = int(dt_adj)
		res = []
		for i in range(n):
			res.append(self.df[self.data_index + (- n + i + 1 ) * dt_adj])
		return np.array(res) / self.initial


# Need to rework to record n previous prices...
class market:
	'''Basic market model, base class for more complex models'''

	def __init__(self,stock_,num_strats = 1,n_hist_prices = 0):
		self.k = 0.00186
		self.stock = stock_
		self.stock.hist_buffer = n_hist_prices
		self.spread = 0
		self.price_adjust = np.ones(num_strats)
		self.n_hist_prices = n_hist_prices
		if self.n_hist_prices > 0:
			assert num_strats == 1, "Currently historical data does not support more than one strat"


	def sell(self,volume,dt):
		'''sell *volume* of stock over time window dt, volume is np array'''
		self.price_adjust *= np.vectorize(self.exp_g)(volume)
		#print("volume ", volume, "price_adjust ",self.price_adjust)        
		ret = (self._adjusted_price() - np.vectorize(self.f)(volume/dt) - 0.5 * self.spread) * volume 
		#print("return",ret)
		return ret

	def g(self,v):
		return v * 0.0005
    
	def exp_g(self,v):
		return np.exp(-self.g(v))

	def f(self,v):
		return v * self.k#0.00186 # Temporarily adjusting by 10 to account for non unit terminal
		# What should this be?
			# - HFT book (position = 1, terminal  = 1, k = 0.01)
			# Since position = 1 but terminal = 10 I've *10
			# NOW CHANGED TO 0.001 (V LOW) TO TEST STOCK PROCESSING NET

	def _adjusted_price(self):
		return self.stock.price * self.price_adjust

	def reset(self,dt,training = True):
		self.stock.reset(training)
		self.price_adjust = np.ones(len(self.price_adjust))
		for i in range(self.n_hist_prices):
			self.hist_prices = self.stock.hist_price(self.n_hist_prices,dt)
		#return self.hist_prices

	def progress(self,dt):
		self.stock.generate_price(dt)
		# MULTIPLE STRATS NOT SUPPORTED HERE
		if self.n_hist_prices > 0:
			self.hist_prices[:-1] = self.hist_prices[1:]; self.hist_prices[-1] = self._adjusted_price()[0]

	def state(self,n_prices = 1):
		return self.hist_prices


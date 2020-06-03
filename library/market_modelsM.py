from random import gauss, sample
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

	def reset(self):
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
	def __init__(self,data,n_steps = 60, recycle = False):
		self.recycle = recycle
		self.n_steps = n_steps
		self.df = data

		if self.recycle:
			pass
		else:
			print("Assuming 1M frequency",if recycle "with" else "without","recycling")
			self.final_period = floor(len(data) / self.terminal)
			self.available_periods = range(self.final_period)
			shuffle(self.available_periods)
			self.period_index = -1
		
		self.reset()

		self.price = self.df[self.data_index] # This will need changing with the format of input

	def reset(self):
		if self.recycle:
			self.period_index += 1
			assert self.period_index <= self.terminal, "Dataset finished"
			self.data_index = self.period_index * n_steps
		else:
			self.data_index = random.randint(len(data) - self.n_steps)

	def _scale_price(self,initial,price):
		pass




class market:
	'''Basic market model, base class for more complex models'''

	def __init__(self,stock_,num_strats = 1):
		self.stock = stock_
		self.spread = 0
		self.price_adjust = np.ones(num_strats)


	def sell(self,volume,dt):
		'''sell *volume* of stock over time window dt, volume is np array'''
		self.price_adjust *= np.vectorize(self.exp_g)(volume)
		#print("volume ", volume, "price_adjust ",self.price_adjust)        
		ret = (self.stock.price * self.price_adjust - np.vectorize(self.f)(volume/dt) - 0.5 * self.spread) * volume 
		return ret


	def g(self,v):
		return v * 0.0005
    
	def exp_g(self,v):
		return np.exp(-self.g(v))

	def f(self,v):
		return v * 0.00186

	def reset(self):
		self.stock.reset()
		self.price_adjust = np.ones(len(self.price_adjust))

	def progress(self,dt):
		self.stock.generate_price(dt)

	def state(self):
		return (self.stock.price)
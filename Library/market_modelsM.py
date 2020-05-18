from random import gauss
import numpy as np
np.random.seed(84)

class stock:
	
	def __init__(self, initial, drift, vol):
		self.initial = initial




class bs_stock:

	def __init__(self, initial, drift, vol):
		self.initial = initial
		self.price = initial
		self.drift = drift
		self.vol = vol

	def generate_price(self,dt,St = None):
		if St is None:
			St = self.price

		self.price = St * np.exp((self.drift - 0.5 * self.vol) * dt + self.vol * dt**0.5 * gauss(0,1))
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
		return v * 0.001
    
	def exp_g(self,v):
		return np.exp(-self.g(v))

	def f(self,v):
		return v * 0.001

	def reset(self):
		self.stock.reset()
		self.price_adjust = np.ones(len(self.price_adjust))

	def progress(self,dt):
		self.stock.generate_price(dt)

	def state(self):
		return (self.stock.price)
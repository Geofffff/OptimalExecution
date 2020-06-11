import unittest
from library.agents.distAgentsWIP2 import QRAgent, C51Agent
import numpy as np

def dummy_state():
	res = np.zeros(agent.state_size)
	res = np.reshape(res,[1,len(res)])
	return res

agent = QRAgent(2,[0.1,0.5,1.0],"TonyTester",C=0)

# Test output dims
class testAgent(unittest.TestCase):

	def test_outputs(self):
		state = dummy_state()
		predictions = agent.predict(state)
		self.assertEqual(predictions.shape,(1,agent.action_size),"Incorrect shape for predict function output")
		self.assertTrue(agent.act(state) < agent.action_size, "act output invalid")
		self.assertTrue(agent.act(state) >= 0,f"{agent.act(state)} act output invalid")

	def test_dist(self):
		state = dummy_state()
		self.assertTrue(agent.agent_type == "dist", "Can't test non distAgent")
		if type(agent).__name__ == "QRAgent":
			self.assertTrue(np.all(agent.quantiles <= 1), "Quantiles out of bounds")
			self.assertTrue(np.all(agent.quantiles >= 0), "Quantiles out of bounds")
		self.assertTrue(len(agent.variance(state)) == agent.action_size)
		#self.assertTrue(np.all(len(variance(state)) >= 0))
	
		

'''
class testEnvironment(unittest.TestCase):
	def __init__(self):
		self.env = env
		super(testEnvironment,self).__init__()

	def test_state(self):
		pass
'''
# Run training test

if __name__ == "__main__":
	unittest.main()
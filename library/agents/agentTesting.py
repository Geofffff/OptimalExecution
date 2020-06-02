import unittest
import numpy as np

# Test output dims
class testAgent(unittest.TestCase):
	def __init__(self,agent):
		self.agent = agent
		super(testAgent,self).__init__()

	def test_predict(self):
		# Dummy state
		state = np.zeros(agent.state_size)
		predictions = agent.predict(state)
		self.assertEqual(predictions.shape,(1,agent.action_size),"Incorrect shape for predict function output")
		

# Run training test
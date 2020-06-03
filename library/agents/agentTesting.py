import unittest
import numpy as np

# Test output dims
class testAgent(unittest.TestCase):
	def __init__(self,agent):
		self.agent = agent
		super(testAgent,self).__init__()

	def test_outputs(self):
		# Dummy state
		state = np.zeros(agent.state_size)
		predictions = agent.predict(state)
		self.assertEqual(predictions.shape,(1,agent.action_size),"Incorrect shape for predict function output")
		self.assertTrue(self.agent.act(state) < self.agent.action_size, "act output invalid")
		self.assertFalse(self.agent.act(state) >= 0,"act output invalid")

		

# Run training test
## Agents
### DDQN
- appears much more robust - dont see the training dip at the start
- theoretically slower to train? [Think about it?]
- sqrt of eps taken [test without doing this...] as each NN is updated approx every other step [check paper too...]

### Target Networks
- it seems that target networks are very important for this problem
	- faster convergence and higher prob of converging
	- testing harder version of the problem (linear temporary impact)
	- comment that the impact seems much greater than illustrated in [Minh15]? Is this true? Not necessarily, just say may be more important given random nature of the problem
	- Possible improvement, keep lag between target network and true (higher memory useage, could be a problem for larger networks)
	- random weights performed better(!)
	- Papers on target networks
	- Look at action value function progression with both implementations of target network
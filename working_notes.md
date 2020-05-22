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

- Set gamma = 1, cant see logic in non unit gamma

### Dist Agent
- Version 1.8: switched Bin X entropy to KL divergence
- Version 2: Revert to original value neural net?
- Version 2.2: Switched to DDQ with target

Things to Try:
- Lower learning rate
- Changing N
- Capping max reward to 10?
- Larger target lag (100)

### Simulator
Simulator v3:
- better plots

### NOTES

- practical considerations:
	- less data - no unlimited games like atari
	- more noise

## Results
- Daisy Test 4: NN with 3 hidden layers, 8 nodes each, N = 50, V_max = 15, Daisy50 lr=0.001 and Daisy20_l lr = 0.0001
- Large Scale Test 1:
	Parameters: linear temp impact k=0.001, eps_min = 0.04
	- 10 Agents:
		- DDQN C = 5, eps_decay = 0.999 
		- DDQN C = 50, eps_decay = 0.999  
		- Double DistAgent C = 5, eps_decay = 0.9992 
		- DistAgent C = 5, eps_decay = 0.9992 
		- DistAgent C = 20, eps_decay = 0.9992 
		- DistAgent C = 20, eps_decay = 0.999
		- DistAgent C = 50, eps_decay = 0.9998, lr = 0.0001
		- DistAgent C = 50 N = 61
		- Experimental Dist Agent WIPAgentsv3 with larger network and 





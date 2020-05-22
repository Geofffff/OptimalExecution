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
- Daisy Test 4: NN with 3 hidden layers, 8 nodes each, N = 50, V_max = 15, Daisy50 lr=0.001 and Daisy20_l lr = 0.0001, bsstock w/ mu = 0, sigma = 0.001
- Large Scale Test 1:
	Parameters: num_trades = 20, linear temp impact k=0.00186, b=0.0005, eps_min = 0.01, decay = 0.9992, lr = 0.001
	- 5 Agents:
		- DistAgent C=20
		- DistAgent C=10a
		- DDQN C = 10a, eps_decay = 0.999  
		- DQN C=4a
		- Experimental Dist Agent WIPAgentsv3 with larger network and epoch = 3

- Large Scale Test 2:
	Parameters (same unless specified): num_trades = 15, (N=51 for distAgents)
	- 5 Agents:
		- Daisy8a eps_d = 0.999, eps_min = 0.04
		- Daisy8a eps_d = 0.998, eps_min = 0.1
		- Paul8a eps_d = 0.999, eps_min = 0.04 (N=61) (epochs 4)
		- Amanda10a eps_d = 0.999, eps_min = 0.04
		- Daisy20a ??





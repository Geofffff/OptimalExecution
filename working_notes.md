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
	- theoretically, if following DDQN Opt Ex, the reward for every action could be evaluated?
		- Or, more optimally the same data could be trained on multiple times? - new perspective each time but risk of overfitting
- Think about theory - volatile agent at the start does explore more of the state space?
	- Agents seem to respond to similar triggers (dist agents more robust to this?)

- Pretraining: Could pretrain with executing remaining position at time t? - What would be a realistic stock price for this though? - might work if stock price were a feature
Pre training for time 0 not useful

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
	Parameters (same unless specified): num_trades = 20, (N=51 for distAgents)
	- 5 Agents:
		- Daisy8a eps_d = 0.999, eps_min = 0.04
		- Daisy8a eps_d = 0.998, eps_min = 0.1
		- Paul8a eps_d = 0.999, eps_min = 0.04 (N=61) (epochs 4)
		- Amanda10a eps_d = 0.999, eps_min = 0.04

- Small Test 3:
	Parameters (same unless specified): eps_d = 0.9992, pretraining (defaults)
	- Agents:
		- Daisy8a
		- Amanda10a

- Small Test 4:
	Parameters (same unless specified): larger range of action values
	- 

- Test 5:
	Action space now larger, number of trades: 10, 10 decisions to make
	- Agents:
		- Daisy8a eps_decay = 0.9994

- Test 6:
	Intensive training switched on
		- Daisy8aN51
		- Daisy8aN11
		- Amanda10a

- Test 7:
	Intensive training off
	- Agents:
		- Daisy8aN11 V_max = 15 (NN with 30 units in base layer)
		- Amanda10a

- Test 8:
	UCB method descibed in paper (c = 1) vs regular Daisy, lr = 0.00025

- Test 9:
	c = 10 lr identical






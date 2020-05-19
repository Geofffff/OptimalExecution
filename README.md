# ThesisCode
Code for optimal execution



## TODO:

1) Agents 
- Agents don't seem to converge so diagnose and fix this before adding / improving anything else [DONE]
- Trying making the problem simpler (higher temp impact and lower vol)
- Approach should possibly be adapted to the Jaimungal approach
- Find average trade sizes from data
- Incorporate changes from the paper (reread)
- Set seed to control reproducability [DONE] - add seed to stock price

### Agents
- The build model function should probably be broken out of the learning agents
- Fix slightly hacky solution to non generalised parameters
- Test target network concept [DONE]
- Add Sutton and Barto n step approach
- Prioritised Sampling
- Distributional Agent
	#### Distributional Agent
		- Dynamic V_min and V_max dependent on vol etc?

### Simulator
- Silent training option
- Major rework needed once its working
- Evaluation option to continue from random strategy [Think about this - not sure its necessary, problem is markov]
- [IMPORTANT] Track action values and calculate how they diverge from theoretical ones
- Investigate what this suboptimal strategy is that agents converge to

### Market Models and Environment
- Add new market models
- Expand the feature set

### Reading
- Sutton and Barto (p142)
- Next RL execution paper [DONE]
- Deep L in py book (buy)
- Double deep q paper [DONE]
- Prioritised replay
- Distributional approach


#### Other Reading
- Restricted Boltzman machines
- Random Forests
- Target Networks

### Writing
- Talk about DDQN in practise (halving epsilon decay?)
- Transformations of the features
- Case study on robustness and DDQ - could demonstate / prove the issues with the niave approach DQN

### Questions for Paul


Deep reinforcement learning hands on

### Paul Meeting 3
- Evaluation (run for 100 steps, evaluate for 40)
- 



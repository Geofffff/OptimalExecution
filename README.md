# ThesisCode
Code for optimal execution



## TODO:

Thursday:
- Simulator rebuild
	- Move over to Wandb
		- track action values [DONE]
		- track eval rewards [DONE]
		- track distributions [TODO]
	- Automatic test classification & model saving [PARTIALLY DONE]
- distAgent Cleanup
	- tidy up parameters [DONE]
	- tranisiton to wandb [DONE]
- Continue testing UCB models [DONE]
- Test and work on neural net - look at batch regulisation

Friday:
- Dynamic V_min / V_max depending on vol? [REPLACED WITH IQN]
- distAgent efficiency drive - currently slow to train [POSTPONED]
- Optimise V_max and V_min [DONE]
- Look at reducing plummeting behaviour [DONE?]
- Once optimised, start working on non TWAP optimal environments [POSTPONED]

Monday:
- Testing of new cleaned up code
- Get IQN Agent working
- Efficiency possibilities: there are a finite (and not huge) number of states considering only the agents personal environment so some precomputation could be done?
- Look at simulations with signal (HFT course notes)


### Agents
- The build model function should probably be broken out of the learning agents
- Fix slightly hacky solution to non generalised parameters
- Test target network concept [DONE]
- Add Sutton and Barto n step approach
- Prioritised Sampling
- Distributional Agent [DONE]


### Simulator
- Silent training option
- Major rework needed once its working
- Evaluation option to continue from random strategy [Think about this - not sure its necessary, problem is markov]
- Track action values and calculate how they diverge from theoretical ones [DONE]
- Investigate what this suboptimal strategy is that agents converge to [SEMI DONE]

#### Simulator Version 2
- Transition to Wandb:
	- test metric tracking
	- remove live plots
	- integrate agent hyperparameters
	- sweep integration


### Market Models and Environment
- Add new market models 
- Expand the feature set

### Reading
- Sutton and Barto (part 2)
- Next RL execution paper [DONE]
- Deep L in py book (buy)
- Double deep q paper [DONE]
- Prioritised replay
- Distributional approach
	- C51 [DONE]
	- IQN 
- Neural net papers / reading [PRIORITY]


#### Other Reading
- Restricted Boltzman machines [SCRAP]
- Random Forests [SCRAP]
- Target Networks [DONE]

### Writing
- Talk about DDQN in practise (halving epsilon decay?)
- Transformations of the features
- Case study on robustness and DDQ - could demonstate / prove the issues with the niave approach DQN
- How do we know when its converged outside of the simulated environment
- Semi Gradient decent discussion (convergence page 202 Sutton Barto)
- Distributional RL Theory

### Network Architecture
- Look at improving the architecture of the Neural Net

### General
- Need to cut down the number of training episodes:
	- pre training on simulated data?
	- Reuse training data
	- prioritised sweeping

### Questions for Paul


Deep reinforcement learning hands on

### Paul Meeting 3
- Evaluation (run for 100 steps, evaluate for 40)
- Takes many episodes to converge (if ever) and limited data available



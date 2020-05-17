# ThesisCode
Code for optimal execution



## TODO:

1) Agents 
- Agents don't seem to converge so diagnose and fix this before adding / improving anything else
- Trying making the problem simpler (higher temp impact and lower vol)
- Approach should probably be adapted to the Jaimungal approach

- Find average trade sizes from data
- Initial weights?
- Incorporate changes from the paper (reread)
- Set seed to control reproducability [DONE] - add seed to stock price

### Agents
- Clean up and standardise
- The build model function should probably be broken out of the learning agents
- Fix slightly hacky solution to non generalised parameters
- Test target network concept
- Add Sutton and Barto n step approach

### Simulator
- Silent training option
- Major rework needed once its working
- Evaluation option to continue from random strategy [Think about this - not sure its necessary, problem is markov]
- [IMPORTANT] Track action values and calculate how they diverge from theoretical ones

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
- The role of gamma in this case? - Risk aversion
- The different approach - estimating action values



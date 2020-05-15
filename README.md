# ThesisCode
Code for optimal execution

## TODO:
### Agents
- The build model function should probably be broken out of the learning agents
- Add DDQN Agent [TESTING]
- Fix slightly hacky solution to non generalised parameters
- Test target network concept
- Add random agent
- Add Sutton and Barto n step approach

### Simulator
- Efficiency
- Robust model saving and figure saving
- Silent training option
- Record model choices for diagnosis [IMPORTANT - why does it start at 9.4]
- Remove checks [DONE]
- Continuous evaluation (every 1000 steps evaluate for 100 steps?)

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
- Talk about DDQN in practise (halving epsilon decay)
- Transformations of the features
- Case study on robustness and DDQ - could demonstate / prove the issues with the niave approach DQN

### General
- Test training agents with a larger number of options and timesteps
- Find average trade sizes from data
- Initial weights?
- Incorporate changes from the paper (reread)
- Set seed to control reproducability [DONE]

### Questions for Paul
- The role of gamma in this case? - Risk aversion
- The different approach - estimating action values



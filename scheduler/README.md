## Migration Modeling and Learning Algorithms for Containers in Fog Computing 
QLearning.py implements the container migration scheduler from [Migration Modeling and Learning Algorithms for Containers in Fog Computing](https://ieeexplore.ieee.org/document/8338124) 

### Assumptions and adaptations
1. Every container has 1 mobile application
2. The hyperparameters of the scheduler, such as weights in formulae, use default values
3. Migration cost is approximated by adding up network delay of moving containers and difference in delay. 

### Experiment and results
|metric | value | 
|----------|-------------|
|Summation  numdestroyed  |  52 |
|Summation  nummigrations  |  98 |
|Summation  energytotalinterval  |  28871380.979745276 |
|Summation  avgresponsetime  |  24140.059269982452 |
|Summation  avgmigrationtime  |  3.5613087101111107 |
|Summation  slaviolations  |  0 |
|Summation  slaviolationspercentage  |  0.0 |
|Summation  waittime  |  52|
|Summation  energypercontainerinterval  |  1335081.6417352615|
|Average energy (sum energy interval / sum numdestroyed) |  555218.8649951015|

TODO

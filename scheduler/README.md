## Migration Modeling and Learning Algorithms for Containers in Fog Computing
QLearning.py implements the container migration scheduler from [Migration Modeling and Learning Algorithms for Containers in Fog Computing](https://ieeexplore.ieee.org/document/8338124)

### Assumptions and adaptations

1. Every container has 1 mobile application
2. The hyperparameters of the scheduler, such as weights in formulae, use default values
3. Migration cost is approximated by adding up network delay of moving containers and difference in delay.

### Experiment and results

| metric                                                  | GOBIScheduler      | QLearningScheduler |
| ------------------------------------------------------- | ------------------ | ------------------ |
| Summation numdestroyed                                  | 362                | 381                |
| Summation nummigrations                                 | 480                | 490                |
| Summation avgresponsetime                               | 227285.43379873672 | 239365.64326102476 |
| Summation avgmigrationtime                              | 31.256877305656094 | 29.536304415939153 |
| Summation slaviolations                                 | 74                 | 95                 |
| Summation slaviolationspercentage                       | 1809.4444444444446 | 2245.0             |
| Summation waittime                                      | 1494               | 1906               |
| Summation energypercontainerinterval                    | 3762901.4000540907 | 3606934.293326813  |
| Average energy (sum energy interval / sum numdestroyed) | 399089.5133896802  | 385872.16202946176 |

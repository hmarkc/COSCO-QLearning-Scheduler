## Migration Modeling and Learning Algorithms for Containers in Fog Computing
QLearning.py implements the container migration scheduler from [Migration Modeling and Learning Algorithms for Containers in Fog Computing](https://ieeexplore.ieee.org/document/8338124)

### Assumptions and adaptations

1. Every container has 1 mobile application
2. The hyperparameters of the scheduler, such as weights in formulae, use default values
3. Migration cost is approximated by adding up network delay of moving containers and difference in delay.

### Experiment and results

| metric                                                  | GOBIScheduler      | QLearningScheduler |
| ------------------------------------------------------- | ------------------ | ------------------ |
| Summation numdestroyed                                  | 362                | 363                |
| Summation nummigrations                                 | 480                | 455                |
| Summation avgresponsetime                               | 227285.43379873672 | 231174.47462357438 |
| Summation avgmigrationtime                              | 31.256877305656094 | 25.58503976023545  |
| Summation slaviolations                                 | 74                 | 50                 |
| Summation slaviolationspercentage                       | 1809.4444444444446 | 1132.9365079365082 |
| Summation waittime                                      | 1494               | 656                |
| Summation energypercontainerinterval                    | 3762901.4000540907 | 3828151.005934516  |
| Average energy (sum energy interval / sum numdestroyed) | 399089.5133896802  | 404380.5635811904  |

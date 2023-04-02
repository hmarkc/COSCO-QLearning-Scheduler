from .Scheduler import *
from .BaGTI.train import *
from .QL.DQL import *
import numpy as np 

class QLearningScheduler(Scheduler):
    def __init__(self, num_hosts, num_containers, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, replay_memory_size=10000, batch_size=32, target_update_frequency=100):
        super().__init__()
        self.target_update_frequency = target_update_frequency
        self.episode = 0
        self.total_reward = 0
        self.simulator_env = SimulatorEnv(num_hosts, num_containers) # Environment for simulator
        self.state = self.simulator_env.reset()

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
    
        input_size = 2 * num_containers + num_hosts * num_containers # number of possible states: number of elements in delay matrix + number of containers * 2
        output_size = num_hosts * num_containers # number of possible actions: number of containers * number of hosts

        self.agent = Agent(input_size, output_size, self.learning_rate, self.gamma, self.epsilon, self.epsilon_decay, self.min_epsilon, self.replay_memory_size, self.batch_size)

    # select container to be sheduled or migrated
    def selection(self):
        return []
    
    # place selected container to a host
    # def placement(self, containerIDs):
    #     decision = []
    #     for id in containerIDs:
    #         scores = [self.env.stats.runSimpleSimulation([(id, hostId)])[0] for hostId, _ in enumerate(self.env.hostlist)]
    #         decision.append((id, np.argmin(scores)))
    #     return decision
    def placement(self, containerlist):
        action = self.agent.act(self.state)
        next_state, reward, done, _, _ = self.simulator_env.step(action)
        self.agent.remember(self.state, action, reward, next_state, done)
        self.total_reward += reward
        self.state = next_state
        
        self.agent.learn()
        self.agent.decrease_epsilon()
        
        if self.episode % self.target_update_frequency == 0:
            self.agent.update_target_model()
            self.episode = 0
            self.total_reward = 0
        
        self.episode += 1
        print("Episode: {}, Total Reward: {}, Epsilon: {:.2f}".format(self.episode, self.total_reward, self.agent.epsilon))
        selected_host, selected_container = action 
        return [(container, selected_host) if container == selected_container else (container, -1) for container in containerlist] 
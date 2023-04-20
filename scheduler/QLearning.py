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
        self.state = None
        self.action_map = [(i, j) for i in range(num_hosts) for j in range(num_containers)]

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

    def setEnvironment(self, env):
        super().setEnvironment(env)
        self.simulator_env.setEnvironment(env)

    # select container to be sheduled or migrated
    def selection(self):
        return []
    
    def decode_action(self, action):
        return self.action_map[action]

    # place selected container to a host
    def placement(self, containerlist):
        if self.state is None:
            self.state = self.simulator_env.reset()
        action = self.agent.act(self.state)
        selected_host, selected_container = self.decode_action(action)

        # performs migration on only 1 container, the rest are placed on the same host
        decision = []
        for container in self.env.containerlist:
            if container is None:
                continue
            c = container.id
            if c == selected_container:
                decision.append((c, selected_host))
            else:
                host = random.randint(0, len(self.env.hostlist) - 1) if self.env.containerlist[c].hostid == -1 else self.env.containerlist[c].hostid
                decision.append((c, host))

        next_state, reward, done, _ = self.simulator_env.step((selected_host, selected_container))
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
        return decision
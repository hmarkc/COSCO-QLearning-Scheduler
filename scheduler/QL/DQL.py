import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces

class SimulatorEnv(gym.Env):
    def __init__(self, host_num, container_num):
        # state space
        self.state_num = 2 * container_num + host_num * container_num
        self.observation_space = spaces.Box(low=0, high=1, shape=(state_num,), dtype=float)
        # action space
        self.action_space = spaces.Tuple((spaces.Discrete(host_num), spaces.Discrete(container_num)))
        self.state = [random.uniform(0, 1) for _ in range(state_num)]
        self.reward_range = (-1, 1)
    
    def reset(self):
        # reset the state to a new random value
        self.state = [random.uniform(0, 1) for _ in range(self.state_num)]
        return self.state
    
    def step(self, action):
        # needs to update the delay matrix and container location as well as resource allocation
        host, container = action
        self.state = None
        reward = self._calculate_reward(self.state)
        return self.state, reward, False, {}
    
    def _calculate_reward(self, state):
        # calculate reward based on the formula provided by the paper
        pass

class DQL(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQL, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

class Agent():
    def __init__(self, input_size, output_size, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, replay_memory_size, batch_size):
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.replay_memory_size = replay_memory_size
        self.batch_size = batch_size
        
        self.replay_memory = []
        self.pred_model = DQL(input_size, output_size)
        self.target_model = DQL(input_size, output_size)
        self.optimizer = optim.Adam(self.pred_model.parameters(), lr=self.learning_rate)
        
        # exploitation optimization
        self.best_action_dict = {}
        
    def act(self, state):
        if random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.pred_model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randint(0, self.output_size - 1)
        return action
        
    def remember(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.pop(0)
        
    def learn(self):
        if len(self.replay_memory) < self.batch_size:
            return
        
        batch = random.sample(self.replay_memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1)
        
        q_values = self.pred_model(state_batch).gather(1, action_batch)
        target_q_values = self.target_model(next_state_batch).detach().max(1)[0].unsqueeze(1)
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * target_q_values
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.pred_model.state_dict())
        
    def decrease_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

def q_learning():
    env = gym.make('CartPole-v1')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    learning_rate = 0.001
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    replay_memory_size = 10000
    batch_size = 32
    target_update_frequency = 100

    agent = Agent(input_size, output_size, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, replay_memory_size, batch_size)

    num_episodes = 1000

    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            
            agent.learn()
            agent.decrease_epsilon()
            
            if episode % target_update_frequency == 0:
                agent.update_target_model()
            
        print("Episode: {}, Total Reward: {}, Epsilon: {:.2f}".format(episode, total_reward, agent.epsilon))
        
    env.close()

if __name__ == "__main__":
    q_learning()
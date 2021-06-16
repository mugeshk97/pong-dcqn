import numpy as np


class Memory(object):
    def __init__(self, max_size, input_shape, n_actions):
        
        self.mem_size = max_size
        self.mem_cntr = 65
        self.state = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action = np.zeros(self.mem_size, dtype=np.int64)
        self.reward = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal = np.zeros(self.mem_size, dtype=np.bool)


    def store_memory(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state[index] = state
        self.new_state[index] = state_
        self.action[index] = action
        self.reward[index] = reward
        self.terminal[index] = done
        self.mem_cntr += 1

    def sample_memory(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state[batch]
        actions = self.action[batch]
        rewards = self.reward[batch]
        states_ = self.new_state[batch]
        terminal = self.terminal[batch]

        return states, actions, rewards, states_, terminal

from agent_memory import Memory
from agent_network import network
import numpy as np
import tensorflow as tf

class Agent(object):
	"""docstring for Agent"""
	def __init__(self, n_actions, input_shape, gamma, epsilon = 1.0, epsilon_dec = 1e-5, epsilon_min = 0.1,
		batch_size = 64, memory_size = 10000, replace = 1000, eval_network_name = 'Model/eval_network.h5', next_network_name = 'Model/next_network.h5',):
		super(Agent, self).__init__()
		
		self.action_space = [i for i in range(n_actions)]
		self.gamma = gamma
		self.epsilon = epsilon
		self.eps_dec = epsilon_dec
		self.eps_min = epsilon_min
		self.batch_size = batch_size
		self.replace = replace
		self.learn_step = 0
		self.eval_network_name = eval_network_name
		self.next_network_name = next_network_name

		self.memory = Memory(max_size= memory_size, input_shape=input_shape, n_actions = n_actions)
		self.eval_network = network(input_shape, n_actions)
		self.next_network = network(input_shape, n_actions)

	def replace_target_network(self):
		if self.replace is not None and self.learn_step % self.replace == 0:

			self.next_network.set_weights(self.eval_network.get_weights())

	def store_memory(self, state, action, reward, new_state, done):

		self.memory.store_memory(state, action, reward, new_state, done)

    # epsilon greedy strategy based choosing the action
	def choose_action(self, observation):
		if np.random.random() < self.epsilon:
			action = np.random.choice(self.action_space)
		else:
			state = np.array([observation], copy=False, dtype=np.float32)
			actions = self.eval_network.predict(state)
			action = np.argmax(actions)

		return action

	# training the network
	def learn(self):
		if self.memory.mem_cntr > self.batch_size:

			state, action, reward, new_state, done = self.memory.sample_memory(self.batch_size)

			self.replace_target_network()

			q_eval = self.eval_network.predict(state)
			q_next = self.next_network.predict(new_state)

			q_target = q_eval[:]

			indices = np.arange(self.batch_size)

			q_target[indices, action] = reward + self.gamma*np.max(q_next, axis=1)*(1 - done)

			self.eval_network.train_on_batch(state, q_target)

			self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
			self.learn_step += 1

	def save_model(self):
		self.eval_network.save(self.eval_network_name)
		self.next_network.save(self.next_network_name)
		print('[INFO]  saving models')

	def load_model(self):
		self.q_eval = tf.keras.models.load_model(self.eval_network_name)
		self.q_nexdt = tf.keras.models.load_model(self.next_network_name)
		print('[INFO]  loading models')
from agent import Agent
from wrappers import make_env
import numpy as np
from tqdm import tqdm

env = make_env('PongNoFrameskip-v4')
num_games = 100
best_score = -21
load_checkpoint = True

agent = agent = Agent(n_actions = env.action_space.n, input_shape= env.observation_space.shape , gamma= 0.99)
if load_checkpoint:
	agent.load_model()

scores, eps_hist = [], []
n_steps = 0

for i in tqdm(range(num_games)):
	done = False
	observation = env.reset()
	score = 0
	while not done:
		action = agent.choose_action(observation)
		observation_, reward, done, info = env.step(action)
		n_steps += 1
		score += reward
		if not load_checkpoint:
			agent.store_memory(observation, action, reward, observation_, int(done))
			agent.learn()
		else:
			env.render()
		observation = observation_

	scores.append(score)

	avg_score = np.mean(scores[-100:])
	print('episode: ', i,'score: ', score,' average score %.3f' % avg_score,'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

	if avg_score > best_score:
		agent.save_model()
		best_score = avg_score

	eps_hist.append(agent.epsilon)
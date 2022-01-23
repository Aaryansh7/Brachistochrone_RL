import sys
sys.path.append('..')

from env.Brachistochrone import Brachistochrone_Env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ddpg.policies import MlpPolicy
import numpy as np 
import gym
import os


from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback

import matplotlib.pyplot as plt

env = Brachistochrone_Env()
print(env.target_pos)
env = make_vec_env(lambda: env, n_envs=1)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean = np.zeros(n_actions), 
								sigma=0.1*np.ones(n_actions))

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=5000,
                             deterministic=True, render=False)

model = DDPG('MlpPolicy', env =env, action_noise=action_noise, verbose=1,
			gamma= 0.9999, buffer_size = 200000, learning_starts = 50000,
			 gradient_steps = -1, learning_rate=0.001,
			policy_kwargs= dict(net_arch=[400, 300]), train_freq = (1, "episode"), tensorboard_log="./barchistostrone_ddpg_tensorboard/")

os.system('spd-say "your program has started"')


model.learn(total_timesteps=250000, log_interval=1, callback=eval_callback)

os.system('spd-say "your program has finished"')

max_epsiodes = 1
position= []
x = []
y = []

for ep in range(max_epsiodes):
	print("Episode {}".format(ep + 1))
	episodic_reward = []
	steps = 0

	obs = env.reset()
	print("Inital obs: "  +str(obs))
	while True:
		action, _ = model.predict(obs, deterministic = True)
		obs, reward, done, info = env.step(action)
		position.append((obs[0][0], obs[0][1]))
		x.append(obs[0][0])
		y.append(obs[0][1])
		episodic_reward.append(reward)

		steps+=1
		#print('obs=', obs, 'reward=', reward, 'done=', done)
		if done:
			print("Goal reached!", "episodic-reward=", np.sum(episodic_reward))
			print("Steps taken: " + str(steps))
			#print()
			break


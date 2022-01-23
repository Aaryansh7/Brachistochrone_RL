import sys
sys.path.append('..')

from env.Brachistochrone import Brachistochrone_Env
from stable_baselines3.common.env_checker import check_env
import math
import matplotlib
import matplotlib.pyplot as plt

env = Brachistochrone_Env()
check_env(env, warn=True)

#Testing the env

env = Brachistochrone_Env()
obs = env.reset()
#env.render()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())


theta = env.action_space.sample()
print('action=', theta)

n_steps =500
standard_x = []
standard_y = []
theta = math.atan(-6.0/8.0)
theta_deg = math.degrees(theta)
print('standard theta = ', theta_deg)

for step in range(n_steps):
	print("Step {}". format(step + 1))
	obs, reward, done, _  = env.step(theta_deg)
	print('obs =', obs, 'reward=', reward, 'done=', done)
	if done==False:
			standard_x.append(obs[0])
			standard_y.append(obs[1])
	#env.render()
	if done:
		print('Goal reached!', "reward=", reward)
		break
		
plt.plot(standard_x,standard_y)
plt.show()

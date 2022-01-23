import sys
sys.path.append('..')

from env.Brachistochrone import Brachistochrone_Env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ddpg.policies import MlpPolicy
import numpy as np 
import gym

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback

import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import newton
from scipy.integrate import quad

env = Brachistochrone_Env()

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean = np.zeros(n_actions), 
								sigma=0.1*np.ones(n_actions))

model = DDPG('MlpPolicy', env =env, action_noise=action_noise, verbose=1,
			gamma= 0.98, buffer_size = 200000, learning_starts = 50000,
			 gradient_steps = -1, learning_rate=0.001,
			policy_kwargs= dict(net_arch=[400, 300]), train_freq = (1, "episode"))

model = DDPG.load("logs/best_model", env=env)

def cycloid(x2, y2, N):
    """Return the path of Brachistochrone curve from (0,0) to (x2, y2).

    The Brachistochrone curve is the path down which a bead will fall without
    friction between two points in the least time (an arc of a cycloid).
    It is returned as an array of N values of (x,y) between (0,0) and (x2,y2).

    """

    # First find theta2 from (x2, y2) numerically (by Newton-Rapheson).
    def f(theta):
        return y2/x2 - (1-np.cos(theta))/(theta-np.sin(theta))
    theta2 = newton(f, np.pi/2, tol=10**(-10), maxiter=10000)

    # The radius of the circle generating the cycloid.
    R = y2 / (1 - np.cos(theta2))

    theta = np.linspace(0, theta2, N)
    x = R * (theta - np.sin(theta))
    y = R * (1 - np.cos(theta))

    # The time of travel
    T = theta2 * np.sqrt(R / 10.0)
    #print('T(cycloid) = {:.3f}'.format(T))
    return x, y, T

max_epsiodes = 1

position= []
x = []
y = []
steps = 0
for ep in range(max_epsiodes):
	#print("Episode {}".format(ep + 1))
	episodic_reward = []

	obs = env.reset()
	print("Inital obs: "  +str(obs))
	while True:
		action, _ = model.predict(obs, deterministic = True)
		obs, reward, done, info = env.step(action)
		position.append((obs[0], obs[1]))
		if done==False:
			x.append(obs[0])
			y.append(obs[1])
		episodic_reward.append(reward)

		steps+=1
		#print('obs=', obs, 'reward=', reward, 'done=', done)
		if done:
			print("Goal reached!", "episodic-reward=", np.sum(episodic_reward))
			print("Steps taken: " + str(steps))
			print("Time taken: " , steps/100)

			break
cycloid_x, cycloid_y, cycloid_T = cycloid(x[-1], -y[-1], steps)
print('Time taken by cycloid = ', cycloid_T)

time_taken = steps/100
plt.plot(x,y , label = 'Path by RL agent, time taken =  %fs' %time_taken)
plt.plot(cycloid_x, -cycloid_y, label='Brachistochrone, time taken =  %fs' %cycloid_T)
plt.scatter(x[-1], y[-1], color = 'green', s = 200.0)
plt.scatter(x[0], y[0], color = 'red', s = 200.0)
plt.annotate('Start', (x[0], y[0]))
plt.annotate('End', (x[-1], y[-1]))
plt.legend()
plt.xlabel('X-COORDINATE(m)')
plt.ylabel('Y-COORDINATE(m)')
plt.title('TRAJECTORY')
plt.show()




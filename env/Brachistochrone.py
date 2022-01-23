import numpy as np 
import math
import gym 
from gym import spaces
import random
import math

class Brachistochrone_Env(gym.Env):
	def __init__(self):
		super(Brachistochrone_Env, self).__init__()
		self.states = np.array([0.0, 0.0, 0.0 ,-45.0]) #pos_x, pos_y, vel, #theta from +ve x axis
		self.target_pos = np.array([+8.0, -6.0])
		self.g = -10.0
		self.reward = 0.0
		self.time_steps = 0
		self.target_steps = 200
		self.action_seq = []

		act_val_high = np.array([0.0])
		act_val_low = np.array([-90.0])
		self.action_space = spaces.Box(low = act_val_low, high = act_val_high, shape = (1,), dtype=np.float32)

		obs_val_high = np.array([100.0, 100.0, 1000.0, 0.0])
		obs_val_low = np.array([-100.0, -100.0, -1000.0,-90.0])
		self.observation_space = spaces.Box(low= obs_val_low, high = obs_val_high, shape=(4,), dtype=np.float32)

	def step(self, action):
		dt = 0.01
		# action corresponds to theta
		final_theta = action
		original_theta = self.states[3]

		original_vel = self.states[2]
		initial_vel_x = original_vel*math.cos(math.radians(original_theta))
		initial_vel_y = original_vel*math.sin(math.radians(original_theta))

		acceleration = self.g*math.sin(math.radians(final_theta)) # always positive
		acceleration_x = acceleration*math.cos(math.radians(final_theta)) #always positive
		acceleration_y = acceleration*math.sin(math.radians(final_theta)) # always negative
				
		displacement_x = (initial_vel_x*dt) + (0.5*acceleration_x*(dt**2))
		displacement_y = (initial_vel_y*dt) + (0.5*acceleration_y*(dt**2))
		displacement = math.sqrt(displacement_x**2 + displacement_y**2)
		
		final_vel_x = initial_vel_x + acceleration_x*dt
		final_vel_y = initial_vel_y + acceleration_y*dt

		final_x = self.states[0] + displacement_x
		final_y = self.states[1] + displacement_y
		final_vel = math.sqrt(final_vel_x**2 + final_vel_y**2)

		self.states[0] = final_x
		self.states[1] = final_y
		self.states[2] = final_vel
		self.states[3] = final_theta

		err_position = math.sqrt((self.states[0] - self.target_pos[0])**2 + (self.states[1] - self.target_pos[1])**2)
		self.reward = -0.1*err_position

		done=False	
		self.time_steps+=1
		if self.time_steps>=self.target_steps or err_position<=0.5:
			done = True
		if err_position<=0.5:
			self.reward  = 0.1*(self.target_steps  - self.time_steps)
		if self.states[0]>8.0 or self.states[0]<0.0 or self.states[1]>0.0 or self.states[1]<-6.0:
			self.reward = -2*(self.target_steps - self.time_steps)
			done = True

		info = {'g_component':acceleration}
		return self.states, self.reward, done, info

	def reset(self):
		self.time_steps = 0
		self.reward = 0.0
		self.states = np.array([0.0, 0.0, 0.0 ,-45.0])
		
		return self.states
		


import gym
import d4rl
import mujoco_py
import numpy as np
from mujoco_py import functions

from kinematics import Kinematics



def set_state(env, qpos, qvel):

	state = env.sim.get_state()
	for i in range(env.n_jnt):
		state[i] = qpos[i]
	for i in range(env.model.nq,env.model.nq+env.n_jnt):
		state[i] = qvel[i-env.model.nq]
	env.sim.set_state(state)
	env.sim.forward()




if __name__ == '__main__':
	env = gym.make('kitchen-complete-v0')
	env.reset()
	print("reset!")
	kine = Kinematics()
	U = np.zeros(9)
	print("!")
	observation, reward, done, info = env.step(U)
	init_state = observation[0:7]
	print(init_state)


import argparse
import d4rl
import gym
import numpy as np
import matplotlib.pyplot as plt

from frankx import Affine, Kinematics, NullSpaceHandling
import numpy as np



def generate_traj(current_pos):
    T = 1000
    target_pos = current_pos
    target_pos[0] += 0.1
    traj = np.zeros([T,6])
    traj[0,0:6] = current_pos
    delta_traj = (target_pos - current_pos)/(T-1)
    for t in range(1,T):
        traj[t,0:6] = traj[t-1,0:6] + delta_traj
    return traj


def init_env(env):
    env.reset()
    action = np.zeros(9)
    observation, reward, done, info = env.step(action)
    init_joint = observation[0:7]
    init_state = observation[0:9]
    init_pos = Affine(Kinematics.forward(init_joint))
    init_pos = np.array(str(init_pos).strip('][').split(', '), np.float64)
    return init_pos,init_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='kitchen-complete-v0')
    args = parser.parse_args()

    env = gym.make(args.env_name)

    dataset = env.get_dataset()
    np.set_printoptions(threshold=np.inf)

    rewards = dataset['rewards']
    actions = dataset['actions']
    observations = dataset['observations']

    init_pos,init_state = init_env(env)
    null_space = NullSpaceHandling(0, 0.14)  # Set elbow joint to 1.4
    init_joint = init_state[0:7]
    traj = generate_traj(init_pos)
    T = traj.shape[0]
    current_pos = init_pos
    current_state = init_state

    for t in range (0,T):
        target_pos = traj[t,:]

        print(init_pos,target_pos)

        current_joint = current_state[0:7]

        init_pos = Affine(Kinematics.forward(init_joint))
        init_pos = np.array(str(init_pos).strip('][').split(', '), np.float64)
        target_joint = Kinematics.inverse(init_pos, init_joint, null_space)
        print(init_joint,target_joint)






        actions = np.zeros(9)
        actions[0:7] = target_joint-current_joint
        observation, reward, done, info = env.step(actions)
        current_state = observation
        env.render()
        input('\n')

'''
generate the Uref given observation (Xref)
Uref is the the gravity compensate control for each time-step calculated by PID
'''



import numpy as np

import argparse
import copy

import d4rl
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import pybullet as p
import pybullet_data




def pid_control (env,target_state, init_action):
    '''
    calculated the uref_k given observation (xref_k),
    which is the gravity compensate control calculated by PID

    :param env: the environment
    :param target_state: np.ndarray (9,): the target steady state
    :param init_action: np.ndarray (9,)
    :return:
        action: np.ndarray (9,): the compensate control
    '''



    action = init_action
    P = 1
    D = 0.01
    I = 0.1
    old_state_error = 0
    sum_state_error = 0
    time_step = 0
    observation, reward, done, info = env.step(action)
    env.render()
    current_state = observation[0:9]
    error_threthold = 0.04
    while True:
        state_error = target_state-current_state
        sum_state_error += state_error
        delta_state_error = state_error-old_state_error
        # if np.linalg.norm(delta_state_error)<0.01:
        #     print("using delta state error\n")
        #     break
        old_state_error = state_error
        action = P*state_error + D*delta_state_error + I*sum_state_error
        # action[7] = 1
        # action[8] = 1
        if target_state[7] <0.01:
            action[7] = -1
        if target_state[8]<0.01:
            action[8]= -1
        observation, reward, done, info = env.step(action)
        env.render()
        current_state = observation[0:9]

        time_step += 1
        if (np.linalg.norm(current_state-target_state)<error_threthold):
            break
        if time_step%100 ==0:
            print(target_state,'\n',current_state,'\n',action,'\n',np.linalg.norm(current_state-target_state),np.linalg.norm(delta_state_error),'\n')
            error_threthold += 0.01
    return action


def direct_pid_control (env,target_state, init_action):

    action = init_action
    print(target_state)
    P = 1
    D = 0.01
    I = 0.1
    old_state_error = 0
    sum_state_error = 0
    time_step = 0
    observation, reward, done, info = env.step(action)
    env.render()
    current_state = observation[0:9]
    error_threthold = 0.03
    xref = []
    uref = []
    while True:
        state_error = target_state-current_state
        sum_state_error += state_error
        delta_state_error = state_error-old_state_error
        # if np.linalg.norm(delta_state_error)<0.01:
        #     print("using delta state error\n")
        #     break
        old_state_error = state_error
        action = P*state_error + D*delta_state_error + I*sum_state_error
        # action[7] = 1
        # action[8] = 1
        if target_state[7] <0.01:
            action[7] = -1
        if target_state[8]<0.01:
            action[8]= -1
        observation, reward, done, info = env.step(action)
        env.render()
        current_state = observation[0:9]
        xref.append(observation[0:18])
        uref.append(action)
        time_step += 1
        if (np.linalg.norm(current_state-target_state)<error_threthold):
            break
        if time_step%100 ==0:
            print(target_state,'\n',current_state,'\n',action,'\n',np.linalg.norm(current_state-target_state),np.linalg.norm(delta_state_error),'\n')
            error_threthold += 0.01
    return xref,uref



def init_env(env):
    env.reset()
    action = np.zeros(9)
    observation, reward, done, info = env.step(action)
    init_joint = observation[0:7]
    init_state = observation[0:9]

    return init_state




def calculate_uref():
    pass


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
    init_state = init_env(env)
    init_joint = init_state[0:7]

    templete_obs = np.load("data/trial1/obs.npy")

    templete_obs[20:29,7] = 0.04
    templete_obs[20:29,8] = 0.04
    templete_obs[29:45,7] = 0.002
    templete_obs[29:45, 8] = 0.002
    templete_obs[45:60,7] = 0.04
    templete_obs[45:60,8] = 0.04
    templete_obs[65:75, 7] = 0.002
    templete_obs[65:75, 8] = 0.002

    # actions = np.load("data/trial1/uref_rough_0-74.npy")

    # time_index = range(0,75)
    # target_joint_angle_trajectory = templete_obs[:,:]
    # init_action = np.zeros(9)
    # action = init_action
    # uref = np.zeros([120,9])
    # for i in range(0,120):
    #     if i<75:
    #         action = actions[i,:]
    #     target_state = target_joint_angle_trajectory[i,:]
    #     new_action = pid_control(env,target_state, action)
    #     action = new_action
    #     uref[i,:] = new_action
    #
    #     print(i,"accomplished\n")
    # np.save("uref.npy",uref)

    init_action = np.zeros(9)
    target_state = templete_obs[5,:]
    xref,uref = direct_pid_control(env,target_state,init_action)
    Xref = np.asarray(xref)
    print(Xref.shape)
    Uref = np.asarray(uref)
    print(Uref.shape)
    np.save("data/trial2/uref.npy",Uref)
    np.save("data/trial2/xref.npy", Xref)

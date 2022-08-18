import numpy as np

import argparse
import copy

import d4rl
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import pybullet as p
import pybullet_data

def forward_sim(env, Xref, Uref,tvlqr_x):
    # return cost
    import matplotlib.pyplot as plt
    cost = 0
    N = len(Xref) + 1

    X = [np.zeros(18) for k in range(0, N)]
    U = [np.zeros(9) for k in range(0, N - 1)]
    observation = env.reset()
    X[0] = observation[:18]

    for k in range(0, N - 1):
        U[k] = Uref[k]
        observation, reward, done, info = env.step(U[k])
        env.render()
        X[k + 1] = observation[:18]


    X = np.asarray(X)
    Xref = np.asarray(Xref)
    U = np.asarray(U)

    # for joint in range(0,7):
    #     plt.plot(tvlqr_x[:,joint],label='tvlqr trajectory')
    #     plt.plot(X[:,joint],label='ilqr tajectory')
    #     plt.legend()
    #     plt.xlabel("time step")
    #     plt.ylabel("joint "+str(joint))
    #     plt.show()

    return X,U

def init_env(env):
    env.reset()
    action = np.zeros(9)
    observation, reward, done, info = env.step(action)
    init_joint = observation[0:7]
    init_state = observation[0:9]

    return init_state




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

    U_ref = np.load('data/trial3/ilqr/u.npy')
    X_ref = np.load('data/trial3/ilqr/x.npy')
    tvlqr_x = np.load('data/trial3/tvlqr/x.npy')
    init_action = np.zeros(9)

    ilqr_x = X_ref
    # for joint in range(0,7):
    #     plt.plot(tvlqr_x[:,joint],label='tvlqr trajectory')
    #     plt.plot(ilqr_x[:,joint],label='ilqr tajectory')
    #     plt.legend()
    #     plt.xlabel("time step")
    #     plt.ylabel("joint "+str(joint))
    #     plt.show()

    for i in range(0,400):
        env.render()

    forward_sim(env,X_ref,U_ref,tvlqr_x)
    while True:
        env.render()

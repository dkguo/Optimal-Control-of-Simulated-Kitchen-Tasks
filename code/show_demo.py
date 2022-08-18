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
def pid_control (env,target_states, init_action):

    action = init_action
    N = len(target_states)
    # N = 100
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
    X = [np.zeros(9) for k in range(0, N*3+100)]
    Xref = [np.zeros(9) for k in range(0, N * 3 + 100)]

    while True:

        target_state = target_states[min(time_step//2,N-1),:9]
        Xref[time_step] = target_state
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
        # env.render()
        current_state = observation[0:9]
        X[time_step] = observation[:9]
        time_step += 1
        if (time_step//2 > (N-1)  and (np.linalg.norm(current_state-target_state)<error_threthold)):
            break
        if time_step%100 ==0:
            print(target_state,'\n',current_state,'\n',action,'\n',np.linalg.norm(current_state-target_state),np.linalg.norm(delta_state_error),'\n')
            error_threthold += 0.01

    X = np.asarray(X)
    Xref = np.asarray(Xref)
    #
    # for i in range(0,7):
    #     plt.plot(X[:2*N, i],label = 'tracked tajectory')
    #     plt.plot(Xref[:2*N,i],label = 'reference tajectory')
    #     plt.legend()
    #     plt.xlabel("time step")
    #     plt.ylabel("joint "+str(i))
    #     plt.show()

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


def set_state(env, qpos, qvel):
    state = env.sim.get_state()
    for i in range(env.n_jnt):
        state[i] = qpos[i]
    for i in range(env.model.nq, env.model.nq + env.n_jnt):
        state[i] = qvel[i - env.model.nq]
    env.sim.set_state(state)
    env.sim.forward()


def linearize(env, Xref, Uref):
    # import ipdb;ipdb.set_trace()
    N = Xref.shape[0]
    n, m = 18, 9
    env.reset()
    A = [np.zeros((n, n)) for k in range(0, N - 1)]
    B = [np.zeros((n, m)) for k in range(0, N - 1)]

    delta = 0.0001

    for step in range(N - 1):

        x = Xref[step]
        u = Uref[step]

        set_state(env, x[:9], x[9:18])

        # observation, reward, done, info = env.step(u)
        # forward sim
        for i in range(env.model.nu):
            env.sim.data.ctrl[i] = u[i]
        env.sim.step()

        fxu = np.concatenate((env.sim.data.qpos[:9], env.sim.data.qvel[:9]))

        for i in range(18):
            dx = np.zeros(18)
            dx[i] += delta

            set_state(env, x[:9] + dx[:9], x[9:18] + dx[9:])
            for i in range(env.model.nu):
                env.sim.data.ctrl[i] = u[i]
            env.sim.step()

            fdxu = np.concatenate((env.sim.data.qpos[:9], env.sim.data.qvel[:9]))
            A[step][:, i] = (fdxu - fxu) / delta

        for i in range(9):
            du = np.zeros(9)
            du[i] += delta

            set_state(env, x[:9], x[9:18])
            for i in range(env.model.nu):
                env.sim.data.ctrl[i] = (u + du)[i]
            env.sim.step()

            fxdu = np.concatenate((env.sim.data.qpos[:9], env.sim.data.qvel[:9]))
            B[step][:, i] = (fxdu - fxu) / delta
    # print(step)
    # np.save('A.npy', A)
    # np.save('B.npy', B)

    return A, B


def stage_cost(x, u, xref, uref, Q, R):
    """
    LQR cost at each knot point (depends on both x and u)
    """

    J = 0.0
    J = 0.5 * (x - xref).transpose() @ Q @ (x - xref) + 0.5 * (u).transpose() @ R @ (u)
    return J


def term_cost(x, xref, Qf):
    J = 0.0
    J = 0.5 * (x - xref).transpose() @ Qf @ (x - xref)
    return J


def trajectory_cost(X, U, Xref, Uref, Q, R, Qf):
    # calculate the cost of a given trajectory
    J = 0.0
    n = Xref.shape[0]
    for i in range(0, n - 1):
        J += stage_cost(X[i], U[i], Xref[i], Uref[i], Q, R)
    J += term_cost(X[n - 1], Xref[n - 1], Qf)
    return J


def tvlqr(A, B, Q, R, Qf):
    n, m = B[1].shape
    N = len(A) + 1
    K = [np.zeros((m, n)) for k in range(0, N - 1)]
    P = [np.zeros((n, n)) for k in range(0, N)]
    P[-1] = Qf
    for k in reversed(range(0, N - 1)):
        K[k] = np.linalg.pinv(R + B[k].T @ P[k + 1] @ B[k]) @ (B[k].T @ P[k + 1] @ A[k])
        # K[k] .= (R + B[k]'P[k+1]*B[k])\(B[k]'P[k+1]*A[k])
        P[k] = Q + A[k].T @ P[k + 1] @ A[k] - A[k].T @ P[k + 1] @ B[k] @ K[k]
    return K, P


# def cost()

def forward_sim(env, K, P, Xref, Uref, Q, R, Qf):
    # return cost
    import matplotlib.pyplot as plt
    cost = 0
    N = len(K) + 1

    X = [np.zeros(18) for k in range(0, N)]
    U = [np.zeros(9) for k in range(0, N - 1)]
    observation = env.reset()
    X[0] = observation[:18]

    pid_k = np.concatenate((np.identity(9) * 1, np.identity(9) * 0.01), axis=1)

    for k in range(0, N - 1):
        U[k] = Uref[k] - K[k] @ (X[k] - Xref[k])
        # U[k] = Uref[k] - pid_k @ (X[k] - Xref[k])

        # U[k] = clamp.(U[k], -u_bnd, u_bnd)
        observation, reward, done, info = env.step(U[k])
        # env.render()
        X[k + 1] = observation[:18]
    # X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)
    # cost += 0.5*(X[k]-Xref[k])@Q@((X[k]-Xref[k])) + 0.5*(U[k])@R@(U[k])
    # cost += 0.5*(X[N-1]-Xref[N-1])@Qf@((X[N-1]-Xref[N-1]))

    X = np.asarray(X)
    U = np.asarray(U)
    Xref = np.asarray(Xref)
    print(cost)
    # for joint in range(0,7):
    # 	plt.plot(X[:,joint],label='tracked tajectory')
    # 	plt.plot(Xref[:,joint],label='reference tajectory')
    # 	plt.legend()
    # 	plt.xlabel("time step")
    # 	plt.ylabel("joint "+str(joint))
    # 	plt.show()

    cost = trajectory_cost(X, U, Xref, Uref, Q, R, Qf)
    print(cost)
    return X, cost




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

    templete_obs = np.load("data/trial3/obs.npy")

    U_ref = np.load('data/trial3/Uref.npy')[:160]
    X_ref = np.load('data/trial3/obs.npy')[:160]

    X_ref[20:29, 7:9] = 0.04
    X_ref[29:45, 7:9] = 0.002
    X_ref[45:60, 7:9] = 0.04
    X_ref[65:75, 7:9] = 0.002
    U_ref[29:45, 7:9] = -1
    U_ref[65:75, 7:9] = -1

    init_action = np.zeros(9)

    for i in range(0,400):
        env.render()

    q = [10] * 9 + [1] * 9
    Q = np.diag(q)
    Qf = Q
    R = np.identity(9) * 10

    A, B = linearize(env, X_ref, U_ref)
    K, P = tvlqr(A, B, Q, R, Qf)
    # import ipdb;ipdb.set_trace()

    tvlqr_X, cost = forward_sim(env, K, P, X_ref, U_ref, Q, R, Qf)

    # N = len(templete_obs)
    # pid_control(env,X_ref,init_action)

    while True:
        env.render()

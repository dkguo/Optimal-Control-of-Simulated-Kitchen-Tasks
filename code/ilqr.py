"""
using

"""
import copy

import gym
import d4rl
import mujoco_py
import numpy as np
from mujoco_py import functions



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

    return A, B


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



def stage_cost(x, u, xref, uref,Q,R):
    """
    LQR cost at each knot point (depends on both x and u)
    """

    J = 0.0
    J = 0.5 * (x - xref).transpose()@ Q @(x-xref)+0.5*(u).transpose()@ R @ (u)
    return J



def term_cost(x, xref,Qf):
    J = 0.0
    J = 0.5 * (x - xref).transpose()@ Qf@(x-xref)
    return J

def trajectory_cost(X, U, Xref, Uref, Q,R,Qf):
# calculate the cost of a given trajectory
    J = 0.0
    n = Xref.shape[0]
    for i in range(0,n-1):
        J += stage_cost(X[i], U[i], Xref[i], Uref[i], Q,R)
    J += term_cost(X[n-1], Xref[n-1], Qf)
    return J


def backward_pass(env,X, U, Xref, Uref,A,B,Q,R,Qf):

    N = Xref.shape[0]
    nx = 18
    nu = 9

    P = [np.zeros((nx, nx)) for i in range(0,N)]  # cost to go quadratic term
    p = [np.zeros(nx) for i in range(0,N)]  # cost to go linear term
    d = [np.zeros(nu)  for i in range(0,N-1)]  # feedforward control
    K = [np.zeros((nu, nx)) for i in range(0,N-1)]  # feedback gain


    p[-1] = Qf @ (X[-1] - Xref[-1])
    P[-1] = Qf
    delta_J = 0.0


    A,B = linearize(env,np.asarray(X),np.asarray(U))
    for k in reversed(range(0, N - 1)):
        x = X[k]
        u = U[k]
        xref = Xref[k]
        uref = Uref[k]

        q = Q @ (x - xref)
        r = R @ (u- uref)


        gx = q + A[k].transpose() @ p[k+1]
        gu = r + B[k].transpose()@p[k+1]

        Gxx = Q + A[k].transpose()@P[k+1]@A[k]
        Guu = R + B[k].transpose()@P[k+1]@B[k]
        Gxu = A[k].transpose()@P[k+1]@B[k]
        Gux = B[k].transpose()@P[k+1]@A[k]

        d[k] = np.linalg.inv(Guu)@gu
        K[k] = np.linalg.inv(Guu)@Gux

        p[k] = gx - K[k].transpose()@gu + K[k].transpose() @ Guu @ d[k] - Gxu @ d[k]

        P[k] = Gxx + K[k].transpose()@Guu@K[k] - Gxu@K[k] - K[k].transpose() @ Gux
        delta_J += gu.transpose() @ d[k]
    return d, K, P, delta_J


def forward_pass(env, X, U, Xref, Uref, K, d, delta_J, Q,R,Qf, max_linesearch_iters = 10):

    Xn = copy.deepcopy(X)
    Un = copy.deepcopy(U)
    Jn = 0

    N = Xref.shape[0]

    alpha = 1
    J = trajectory_cost(X, U, Xref, Uref, Q,R,Qf)
    print("forward pass init cost", J, '\n')
    observation = env.reset()
    X[0] = observation[:18]

    for k in range(0,N-1):
        Un[k] = U[k] - alpha * d[k] - K[k] @ (Xn[k] - X[k])
        observation, reward, done, info = env.step(U[k])
        # env.render()
        Xn[k + 1] = observation[:18]


    Jn = trajectory_cost(Xn, Un, Xref, Uref, Q,R,Qf)
    print("forward pass cost", J, '\n')
    iter = 0
    while ( Jn > (J - 1e-2 * alpha * delta_J) and iter < max_linesearch_iters):
        alpha = 0.5 * alpha
        observation = env.reset()
        X[0] = observation[:18]
        for k in range(0,N-1):
            Un[k] = U[k] - alpha * d[k] - K[k] @ (Xn[k] - X[k])
            observation, reward, done, info = env.step(Un[k])
            # env.render()
            Xn[k + 1] = observation[:18]

        Jn = trajectory_cost(Xn, Un, Xref, Uref,Q,R,Qf)
        print(Jn,'\n')
        iter += 1


    return Xn, Un, Jn, alpha


def iLQR(env, U, Xref, Uref, A,B,Q,R,Qf,atol = 1e-5, max_iters = 1):


    N = Xref.shape[0]
    nx = 18
    nu = 9

    X = [np.zeros(18) for k in range(0, N)]
    observation = env.reset()
    X[0] = observation[:18]

    for i in range(1,N):
        observation, reward, done, info = env.step(U[i-1])
        # env.render()
        X[i] = observation[:18]

    K = [np.zeros((nu, nx)) for i in range(0,N-1)]
    P = [np.zeros((nx, nx)) for i in range(0,N)]
    iter = -1
    d = [np.ones(nu) for i in range(0,N-1)]
    while (max(np.linalg.norm(d,axis=1)) > atol) and (iter < max_iters):
        iter += 1
        d, K, P, delta_J = backward_pass(env,X, U, Xref, Uref,A,B,Q,R,Qf)
        X, U, J, alpha = forward_pass(env,X, U, Xref, Uref, K, d, delta_J,Q,R,Qf)
        print(iter,'\t',J,'\t',delta_J,'\t',max(np.linalg.norm(d,axis=1)),'\t',alpha,'\n')


    return X, U, K, P, iter


def forward_sim(env, K, P, Xref, Uref,Q,R,Qf):
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
        U[k] = Uref[k]  - K[k] @ (X[k] - Xref[k])
        # U[k] = Uref[k] - pid_k @ (X[k] - Xref[k])

        # U[k] = clamp.(U[k], -u_bnd, u_bnd)
        observation, reward, done, info = env.step(U[k])
        # env.render()
        X[k + 1] = observation[:18]
        # X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)
        # cost += 0.5 * (X[k] - Xref[k]) @ Q @ ((X[k] - Xref[k])) + 0.5 * (U[k]) @ R @ (U[k])
    # cost += 0.5 * (X[N - 1] - Xref[N - 1]) @ Q @ ((X[N - 1] - Xref[N - 1]))

    X = np.asarray(X)
    Xref = np.asarray(Xref)
    U = np.asarray(U)
    # plt.plot(X[:, 0], label='est')
    # plt.plot(Xref[:, 0], label='GT')
    # plt.legend()
    # plt.show()

    print(cost,'\n')
    print(trajectory_cost(X,U,Xref,Uref,Q,R,Qf))
    return U


if __name__ == '__main__':
    env = gym.make('kitchen-complete-v0')
    env.reset()
    # A = np.load('A.npy')
    # B = np.load('B.npy')
    U_ref = np.load('data/trial3/Uref.npy')[:160]
    X_ref = np.load('data/trial3/obs.npy')[:160]

    X_ref[20:29, 7:9] = 0.04
    X_ref[29:45, 7:9] = 0.002
    X_ref[45:60, 7:9] = 0.04
    X_ref[65:75, 7:9] = 0.002
    U_ref[29:45, 7:9] = -1
    U_ref[65:75, 7:9] = -1

    # import ipdb;ipdb.set_trace()

    q = [10] * 9 + [1] * 9
    Q = np.diag(q)
    R = np.identity(9) * 10

    U =copy.deepcopy(U_ref)

    A,B = linearize(env,X_ref,U_ref)

    print("TVLQR\n")
    K, P = tvlqr(A, B, Q, R, Q)
    U_ref = forward_sim(env, K, P, X_ref, U_ref,Q,R,Q)
    print("\n\n")
    print("iLQR\n")
    X, U, K, P, iter =  iLQR(env, U,X_ref,U_ref,A,B,Q,R,Q)

    np.save('data/trial3/ilqr/u.npy',U)
    np.save('data/trial3/ilqr/x.npy', X)

    forward_sim(env, K, P, X, U,Q,R,Q)

# while True:
# 	env.render()
# 	env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
# 	env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
# 	env.step(action)

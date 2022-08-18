"""


"""


import gym
import d4rl
import mujoco_py
import numpy as np
from mujoco_py import functions


def set_state(env, qpos, qvel):

	state = env.sim.get_state()
	for i in range(env.n_jnt):
		state[i] = qpos[i]
	for i in range(env.model.nq,env.model.nq+env.n_jnt):
		state[i] = qvel[i-env.model.nq]
	env.sim.set_state(state)
	env.sim.forward()


def linearize(env, Xref, Uref):

	# import ipdb;ipdb.set_trace()
	N = Xref.shape[0]
	n,m = 18, 9
	env.reset()
	A = [np.zeros((n,n)) for k in range(0,N-1)]
	B = [np.zeros((n,m)) for k in range(0,N-1)]

	delta = 0.0001

	for step in range(N-1):

		x = Xref[step]
		u = Uref[step]

		set_state(env,x[:9],x[9:18])

		# observation, reward, done, info = env.step(u)
		# forward sim
		for i in range(env.model.nu):
			env.sim.data.ctrl[i] = u[i]
		env.sim.step()

		fxu = np.concatenate((env.sim.data.qpos[:9],env.sim.data.qvel[:9]))

		for i in range(18):
			dx = np.zeros(18)
			dx[i] += delta

			set_state(env,x[:9]+ dx[:9],x[9:18]+ dx[9:])
			for i in range(env.model.nu):
				env.sim.data.ctrl[i] = u[i]
			env.sim.step()

			fdxu = np.concatenate((env.sim.data.qpos[:9],env.sim.data.qvel[:9]))
			A[step][:, i] = (fdxu - fxu) / delta

		for i in range(9):
			du = np.zeros(9)
			du[i] += delta

			set_state(env,x[:9],x[9:18])
			for i in range(env.model.nu):
				env.sim.data.ctrl[i] = (u+du)[i]
			env.sim.step()

			fxdu = np.concatenate((env.sim.data.qpos[:9],env.sim.data.qvel[:9]))
			B[step][:, i] = (fxdu - fxu) / delta
		# print(step)
	# np.save('A.npy', A)
	# np.save('B.npy', B)

	return A, B


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



def tvlqr(A,B,Q,R,Qf):

	n,m = B[1].shape
	N = len(A)+1
	K = [np.zeros((m,n)) for k in range(0,N-1)]
	P = [np.zeros((n,n)) for k in range(0,N)]
	P[-1] = Qf
	for k in reversed(range(0,N-1)):
		K[k] = np.linalg.pinv(R + B[k].T@P[k+1]@B[k]) @ (B[k].T@P[k+1]@A[k])
		# K[k] .= (R + B[k]'P[k+1]*B[k])\(B[k]'P[k+1]*A[k])
		P[k] = Q + A[k].T@P[k+1]@A[k] - A[k].T@P[k+1]@B[k]@K[k]
	return K,P

# def cost()

def forward_sim(env,K,P,Xref,Uref,Q,R,Qf):
	# return cost
	import matplotlib.pyplot as plt
	cost = 0
	N = len(K)+1

	X = [np.zeros(18) for k in range(0,N)]
	U = [np.zeros(9) for k in range(0,N-1)]
	observation = env.reset()
	X[0] = observation[:18]

	pid_k = np.concatenate((np.identity(9)*1,np.identity(9)*0.01),axis=1)

	for k in range(0,N-1):
		U[k] = Uref[k] - K[k]@(X[k]-Xref[k])
		# U[k] = Uref[k] - pid_k @ (X[k] - Xref[k])

		# U[k] = clamp.(U[k], -u_bnd, u_bnd)
		observation, reward, done, info = env.step(U[k])
		# env.render()
		X[k+1] = observation[:18]
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

	cost = trajectory_cost(X,U,Xref,Uref,Q,R,Qf)
	print(cost)
	return X,cost


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

	q = [10]*9+[1]*9
	Q = np.diag(q)
	Qf = Q
	R = np.identity(9)*10

	A,B = linearize(env,X_ref,U_ref)
	K,P = tvlqr(A,B,Q,R,Qf)
	# import ipdb;ipdb.set_trace()

	tvlqr_X,cost = forward_sim(env,K,P,X_ref,U_ref,Q,R,Qf)
	np.save('data/trial3/tvlqr/x.npy',tvlqr_X)
	# while True:
	# 	env.render()
	# 	env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
	# 	env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
	# 	env.step(action)

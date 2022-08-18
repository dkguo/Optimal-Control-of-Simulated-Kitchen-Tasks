import gym
import d4rl
import mujoco_py
import numpy as np
import time

def set_state(env, qpos, qvel):

	state = env.sim.get_state()
	for i in range(env.n_jnt):
		state[i] = qpos[i]
	for i in range(env.model.nq,env.model.nq+env.n_jnt):
		state[i] = qvel[i-env.model.nq]
	env.sim.set_state(state)
	env.sim.forward()
	env.sim.step()


def linearize(env, Xref, Uref):

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
		# env.sim.data.qpos[:9] = x[:9]
		# # env.sim.data.qvel[:9] = np.zeros(9)
		# env.sim.data.qvel[:9] = x[9:18]
		# env.sim.forward()
		observation, reward, done, info = env.step(u)

		fxu = observation[:18]

		for i in range(18):
			dx = np.zeros(18)
			dx[i] += delta
			# env.sim.data.qpos[:9] = x[:9] + dx[:9]
			# env.sim.data.qvel[:9] = x[9:18] + dx[9:]
			# env.sim.forward()
			set_state(env,x[:9],x[9:18])
			observation, reward, done, info = env.step(u)

			fdxu = observation[:18]
			A[step][:, i] = (fdxu - fxu) / delta

		for i in range(9):
			du = np.zeros(9)
			du[i] += delta
			# env.sim.data.qpos[:9] = x[:9]
			# # env.sim.data.qvel[:9] = np.zeros(9)
			# env.sim.data.qvel[:9] = x[9:18]
			# env.sim.forward()
			set_state(env,x[:9],x[9:18])
			observation, reward, done, info = env.step(u + du)

			fxdu = observation[:18]
			B[step][:, i] = (fxdu - fxu) / delta
		print(step)
	np.save('data/trial2/A.npy', A)
	np.save('data/trial2/B.npy', B)

	return A, B


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

def forward_sim(env,K,P,Xref,Uref):
	# return cost
	import matplotlib.pyplot as plt
	cost = 0
	N = len(K)+1

	X = [np.zeros(18) for k in range(0,N)]
	U = [np.zeros(9) for k in range(0,N-1)]
	observation = env.reset()
	X[0] = observation[:18]
	print(X[0])
	pid_k = np.concatenate((np.identity(9)*1,np.identity(9)*0.0),axis=1)

	for k in range(0,N-1):
		U[k] = Uref[k] - pid_k@(X[k]-Xref[k])

		if k>=29 and k<45:
			U[k][7] = -1
			U[k][8] = -1

		# U[k] = clamp.(U[k], -u_bnd, u_bnd)
		observation, reward, done, info = env.step(U[k])
		env.render()
		time.sleep(0.1)
		X[k+1] = observation[:18]
		# X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)
	X = np.asarray(X)
	Xref = np.asarray(Xref)

	# plt.plot(X[:,0])
	# plt.plot(Xref[:,0])
	# plt.show()

	return cost


def forward_ref_sim(env,Xref,Uref):
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
		# U[k] = clamp.(U[k], -u_bnd, u_bnd)
		observation, reward, done, info = env.step(U[k])
		env.render()
		X[k+1] = observation[:18]
		# X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)
	X = np.asarray(X)
	Xref = np.asarray(Xref)

	# plt.plot(X[:,0])
	# plt.plot(Xref[:,0])
	# plt.show()

	return cost



if __name__ == '__main__':
	env = gym.make('kitchen-complete-v0')
	env.reset()
	A = np.load('data/trial3/A.npy')
	B = np.load('data/trial3/B.npy')
	U_ref = np.load('data/trial3/Uref.npy')
	X_ref = np.load('data/trial3/obs.npy')

	q = [1]*18
	Q = np.diag(q)
	R = np.identity(9)*0.1

	# A,B = linearize(env,X_ref,U_ref)
	K,P = tvlqr(A,B,Q,R,Q)
	print(K)
	# import ipdb;ipdb.set_trace()
	for i in range(100):
		env.render()
	forward_sim(env,K,P,X_ref,U_ref)

	while True:
		env.render()
	# 	env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
	# 	env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
	# 	env.step(action)

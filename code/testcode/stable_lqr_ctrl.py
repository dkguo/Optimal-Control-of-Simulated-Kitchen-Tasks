import gym
import d4rl
import mujoco_py
import numpy as np


def set_state(env, qpos):

	state = env.sim.get_state()
	for i in range(env.n_jnt):
		state[i] = qpos[i]
	for i in range(env.model.nq,env.model.nq+env.n_jnt):
		state[i] = 0.0
	env.sim.set_state(state)
	env.sim.forward()
	env.sim.step()


def linearize(env, Xref, Uref):

	n,m = 18, 9
	env.reset()
	A = np.zeros((n,n))
	B = np.zeros((n,m))
	delta = 0.001


	x = Xref
	u = Uref
	set_state(env,x[:9])

	observation, reward, done, info = env.step(u)
	fxu = observation[:7]

	for i in range(7):
		dx = np.zeros(9)
		dx[i] += delta
		# env.sim.data.qpos[:9] = x[:9] + dx[:9]
		# env.sim.data.qvel[:9] = x[9:18] + dx[9:]
		# env.sim.forward()
		set_state(env,x[:9]+dx)
		observation, reward, done, info = env.step(u)

		fdxu = observation[:7]
		A[0:7, i] = (fdxu - fxu) / delta

	for i in range(7):
		du = np.zeros(9)
		du[i] += delta
		# env.sim.data.qpos[:9] = x[:9]
		# # env.sim.data.qvel[:9] = np.zeros(9)
		# env.sim.data.qvel[:9] = x[9:18]
		# env.sim.forward()
		set_state(env,x[:9])
		observation, reward, done, info = env.step(u + du)
		fxdu = observation[:7]
		B[0:7, i] = (fxdu - fxu) / delta

	np.save('data/trial1/stable_A.npy', A)
	np.save('data/trial1/stable_B.npy', B)

	return A, B


def tvlqr(A,B,Q,R,Qf):

	n,m = B.shape
	N = 1000
	K = [np.zeros((m,n)) for k in range(0,N-1)]
	P = [np.zeros((n,n)) for k in range(0,N)]
	P[-1] = Qf
	for k in reversed(range(0,N-1)):
		K[k] = np.linalg.pinv(R + B.T@P[k+1]@B) @ (B.T@P[k+1]@A)
		# K[k] .= (R + B[k]'P[k+1]*B[k])\(B[k]'P[k+1]*A[k])
		P[k] = Q + A.T@P[k+1]@A - A.T@P[k+1]@B@K[k]
		print(np.linalg.norm(K[k]))
	return K[1]

# def cost()

def forward_sim(env,K,Xref,Uref):
	# return cost
	import matplotlib.pyplot as plt
	cost = 0
	N = 1000

	X = [np.zeros(9) for k in range(0,N)]
	U = [np.zeros(9) for k in range(0,N-1)]
	observation = env.reset()
	X[0] = observation[:9]

	pid_k = np.concatenate((np.identity(9)*1,np.identity(9)*0.0),axis=1)

	print(K.shape)
	for k in range(0,N-1):
		U[k] = Uref - K[0:9,0:9]@(X[k]-Xref)
		# U[k] = clamp.(U[k], -u_bnd, u_bnd)
		observation, reward, done, info = env.step(U[k])
		env.render()
		X[k+1] = observation[:9]
		# X[k+1]  = true_dynamics_rk4(model, X[k], U[k], dt)
	X = np.asarray(X)
	Xref = np.asarray(Xref)

	plt.plot(X[:,0])
	plt.plot(Xref[:,0])
	plt.show()

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

	plt.plot(X[:,0])
	plt.plot(Xref[:,0])
	plt.show()

	return cost



if __name__ == '__main__':
	env = gym.make('kitchen-complete-v0')
	env.reset()
	# A = np.load('data/trial2/stable_A.npy')
	# B = np.load('data/trial2/stable_B.npy')
	U_ref = np.load('data/trial1/uref_rough_0-74.npy')
	X_ref = np.load('data/trial1/obs.npy')

	q = [0.1]*9+[0]*9
	Q = np.diag(q)
	R = np.identity(9)*0.0001

	A,B = linearize(env,X_ref[0,:],U_ref[0,:])
	K= tvlqr(A,B,Q,R,Q)
	print(K)
	# import ipdb;ipdb.set_trace()

	forward_sim(env,K,X_ref[0,:],U_ref[0,:])

	# while True:
	# 	env.render()
	# 	env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
	# 	env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
	# 	env.step(action)

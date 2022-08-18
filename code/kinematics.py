import time
import numpy as np
from sympy import lambdify
from numba import jit
from sympy import symbols, init_printing, Matrix, eye, sin, cos, pi


class Kinematics:
    def __init__(self):
        q1, q2, q3, q4, q5, q6, q7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
        joint_angles = [q1, q2, q3, q4, q5, q6, q7]

        dh_craig = [
            {'a':  0,      'd': 0.333, 'alpha':  0,  },
            {'a':  0,      'd': 0,     'alpha': -pi/2},
            {'a':  0,      'd': 0.316, 'alpha':  pi/2},
            {'a':  0.0825, 'd': 0,     'alpha':  pi/2},
            {'a': -0.0825, 'd': 0.384, 'alpha': -pi/2},
            {'a':  0,      'd': 0,     'alpha':  pi/2},
            {'a':  0.088,  'd': 0.107, 'alpha':  pi/2},
        ]

        DK = eye(4)

        for i, (p, q) in enumerate(zip(reversed(dh_craig), reversed(joint_angles))):
            d = p['d']
            a = p['a']
            alpha = p['alpha']
            ca = cos(alpha)
            sa = sin(alpha)
            cq = cos(q)
            sq = sin(q)

            transform = Matrix([
                    [cq, -sq, 0, a],
                    [ca * sq, ca * cq, -sa, -d * sa],
                    [sa * sq, cq * sa, ca, d * ca],
                    [0, 0, 0, 1],
            ])

            DK = transform @ DK


        DK.evalf(subs={
            'theta_1': 0,
            'theta_2': 0,
            'theta_3': 0,
            'theta_4': 0,
            'theta_5': 0,
            'theta_6': 0,
            'theta_7': 0,
        })

        A = DK[0:3, 0:4]  # crop last row
        A = A.transpose().reshape(12,1)  # reshape to column vector A = [a11, a21, a31, ..., a34]

        Q = Matrix(joint_angles)
        J = A.jacobian(Q)  # compute Jacobian symbolically
        self.A_lamb = jit(lambdify((q1, q2, q3, q4, q5, q6, q7), A, 'numpy'))
        self.J_lamb = jit(lambdify((q1, q2, q3, q4, q5, q6, q7), J, 'numpy'))

        # define joint limits for the Panda robot
        self.limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973)
        ]

        self.q_init = np.array([l+(u-l)/2 for l, u in self.limits], dtype=np.float64)
        self.A_init = self.A_lamb(*(self.q_init.flatten()))

    # @jit
    def fk(self, q):
        return self.A_lamb(*(q).flatten())

    def reshape_A(self, A):
        return A.reshape(3, 4, order='F')

    # @jit
    def incremental_ik(self, A_target, q=None, A=None, step=0.1, atol=1e-4):
        if q is not None and A is None:
            A = self.fk(q)
        if q is None:
            q = self.q_init
        if A is None:
            A = self.A_init
        while True:
            delta_A = (A_target - A)
            if np.max(np.abs(delta_A)) <= atol:
                break
            J_q = self.J_lamb(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
            J_q = J_q / np.linalg.norm(J_q)  # normalize Jacobian

            # multiply by step to interpolate between current and target pose
            delta_q = np.linalg.pinv(J_q) @ (delta_A*step)

            q = q + delta_q.reshape(7)
            A = self.A_lamb(q[0], q[1],q[2],q[3],q[4],q[5],q[6])
        return q



# K = Kinematics()
# q_init = np.array([0.1430575, -1.7539313, 1.8729757, -2.475513, 0.23822385, 0.7320934, 1.6475043])
# q_test = np.array([0.1330575, -1.7539313, 1.8729757, -2.475513, 0.23822385, 0.7320934, 1.6475043])
# A_target = K.fk(q_test)
#
# t = time.time()
# q_final = K.incremental_ik(A_target, q=q_init, atol=1e-6)
# print('time: ', time.time() - t)
# A_final = K.fk(q_final)
#
# print('test q: ', q_test)
# print('find q: ', q_final)
# print('test A: ', K.reshape_A(A_target))
# print('find A: ', K.reshape_A(A_final))

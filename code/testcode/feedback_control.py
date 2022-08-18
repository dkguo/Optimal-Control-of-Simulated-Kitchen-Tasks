import argparse
import copy

import d4rl
import gym
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import kinematics
from panda_robot import PandaRobot
import pybullet as p
import pybullet_data
def pid_control (env,target_states, init_action):

    action = init_action

    P = 1
    D = 0.01
    I = 0.1
    old_state_error = 0
    sum_state_error = 0
    N = target_states.shape[0]
    print(N)
    time_step = 0
    observation, reward, done, info = env.step(action)
    env.render()
    current_state = observation[0:9]
    while True:
        if time_step<N*100:
            target_state = target_states[time_step//100,:]
        else:
            target_state = target_states[-1,:]
        state_error = target_state-current_state
        sum_state_error += state_error
        delta_state_error = state_error-old_state_error
        old_state_error = state_error
        action = P*state_error + D*delta_state_error + I*sum_state_error
        observation, reward, done, info = env.step(action)
        env.render()
        current_state = observation[0:9]

        time_step += 1
        if (time_step>N and np.linalg.norm(current_state-target_state)<0.001):
            break
        if time_step%100 ==0:
            print(target_state,'\n',current_state,'\n',action,'\n',np.linalg.norm(current_state-target_state),'\n')
    print("completed!")
    return action



def generate_states_sequence (k,init_pos,init_joint,goal_pos,T):
    poss = np.zeros([T,12,1])
    poss[0,:,:] = init_pos
    poss[-1,:,:] = goal_pos
    delta_pos = (goal_pos - init_pos)/(T-1)
    for t in range(1,T):
        poss[t,:,:] = poss[t-1,:,:] + delta_pos

    states = np.zeros([T,9])
    states[0,0:7] = init_joint
    for t in range(1,T):
        print(t)
        states[t,0:7] = k.incremental_ik(poss[t,:,:], q=states[t-1,0:7], atol=1e-3)
    return states




def init_env(env):
    env.reset()
    action = np.zeros(9)
    observation, reward, done, info = env.step(action)
    # env.render()
    init_joint = observation[0:7]
    init_state = observation[0:9]

    return init_state



def rotation_matrixToQ(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()


def pos_from_matrix (pos_matrix):
    rotation_matrix = pos_matrix[0:9].reshape(3,3)
    pos_q = rotation_matrixToQ(rotation_matrix).reshape(4)
    pos = pos_matrix[9:12].reshape(3)
    return np.hstack([pos,pos_q])


def end_effector_from_state(kine,state):
    joint_angle  =state[0:7]
    pos_matrix  = kine.fk(joint_angle).reshape(12,1)
    rotation_matrix = pos_matrix[0:9].reshape(3,3)
    # rotation_matrix = np.identity(3)
    print(rotation_matrix)
    pos_q = rotation_matrixToQ(rotation_matrix).reshape(4)
    pos = pos_matrix[9:12].reshape(3)
    return np.hstack([pos,pos_q])






ll = [-1, -1.7628, -2.8973, -2.6, -2.8973, -1.6573, -2.8973]
#upper limits for null space
ul = [1, 1.7628, 2.8973, -2.3, 2.8973, 2.1127, 2.8973]
#
# #joint ranges for null space
jr = [2, 4, 5.8, 0.5, 5.8, 4, 6]


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
    kine = kinematics.Kinematics()
    init_state = init_env(env)
    init_joint = init_state[0:7]
    init_pos = end_effector_from_state(kine,init_state)

    clid = p.connect(p.SHARED_MEMORY)
    if (clid < 0):
        p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    kukaId = p.loadURDF("franka_panda/model_description/panda.urdf", useFixedBase=True)
    kukaEndEffectorIndex = 6

    target_xyz = init_pos[0:3]+[0,0,0.2]
    target_orn = init_pos[3:7]

    print("init_pos",init_pos,'\n')

    # target_joint_angle = panda_robot.calculate_inverse_kinematics(target_xyz, target_orn,
    #                                                               lowerLimits = ll,
    #                                                               upperLimits = ul,
    #                                                               # joint_range = jr,
    #                                                               restPoses = init_joint)
    print(init_joint.tolist())
    rp = [0.14351578, -1.76146432, 1.8522255, -2.47442022, 0.26120127, 0.71819886, 1.58290257]
    for i in range(7):
      p.resetJointState(kukaId, i, rp[i])
    target_joint_angle = p.calculateInverseKinematics(kukaId,kukaEndEffectorIndex,
                                                      target_xyz,
                                                      # target_orn,
                                                        lowerLimits = ll,
                                                        upperLimits = ul,
                                                        jointRanges = jr,
                                                        restPoses = rp)

    target_state = np.zeros([1,9])
    target_state[0,0:7] =target_joint_angle

    print("joint_angle", init_joint, '\n',target_joint_angle,'\n')
    input('\n')
    # goal_pos = copy.deepcopy(init_pos)
    # goal_pos[11] += 0.0
    # goal_pos[3] += 0.0
    # goal_pos[7] += 0.1
    # print("!!")
    # target_states = generate_states_sequence(kine,init_pos,init_joint,goal_pos,1)
    # print(target_states)

    # target_state[0,:] = init_state +[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
    print("trajectory generated!")
    init_action = np.zeros([9])
    pid_control (env,target_state, init_action)



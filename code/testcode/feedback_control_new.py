import argparse
import copy

import d4rl
import gym
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import kinematics

def pid_control (env,kine, target_states, init_action):

    action = init_action

    P = 10
    D = 0.2
    I = 0.1
    old_state_error = 0
    sum_state_error = 0
    N = target_states.shape[0]
    print(N)
    time_step = 0
    observation, reward, done, info = env.step(action)
    env.render()
    current_state = observation[0:9]
    Uref = []
    while True:
        if time_step<N:
            target_state = target_states[time_step,:]
        else:
            break
            # target_state = target_states[-1,:]
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
        # if time_step%100 ==0:
        print(time_step, current_state,np.linalg.norm(current_state-target_state))
        print(kine.reshape_A(kine.fk(current_state[:7])))
        Uref.append(action)

    np.save('Uref.npy', Uref)
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
        print("iv", t)
        states[t,0:7] = k.incremental_ik(poss[t,:,:], q=states[t-1,0:7], atol=1e-4)

    return states




def init_env(env,kine):
    env.reset()
    action = np.zeros(9)
    observation, reward, done, info = env.step(action)
    # env.render()
    init_joint = observation[0:7]
    init_state = observation[0:9]
    init_pos  = kine.fk(init_joint).reshape(12,1)
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
    kine = kinematics.Kinematics()
    init_pos,init_state = init_env(env,kine)
    init_joint = init_state[0:7]
    target_state = np.zeros([1,9])
    target_state[0,:] = init_state
    init_action = np.zeros(9)

    goal_pos = copy.deepcopy(init_pos)
    goal_pos[11] += 0.8
    goal_pos[3] += 0.0
    goal_pos[7] += 0.0

    # target_states = generate_states_sequence(kine,init_pos,init_joint,goal_pos,10)


    target_states = np.load('data/trial3/obs.npy')

    print(target_states.shape)
    # print("trajectory generated!")
    pid_control(env,kine, target_states[:, :9], init_action)

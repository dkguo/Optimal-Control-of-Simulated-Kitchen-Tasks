import numpy as np
import pybullet as p
import pybullet_data
import math
from panda_robot import PandaRobot
from scipy.spatial.transform import Rotation as R
import kinematics


INCLUDE_GRIPPER = True
DTYPE = 'float64'
SAMPLING_RATE = 1e-3  # 1000Hz sampling rate


def rotation_matrixToQ(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()



def main():
    """"""

    # Basic Setup of environment
    physics_client_id = p.connect(p.DIRECT)

    panda_robot = PandaRobot(include_gripper=INCLUDE_GRIPPER)


    init_joint = np.array([0.14351578,-1.76146432,1.8522255,-2.47442022,0.26120127,0.71819886,1.58290257])
    init_pos = np.array([0.1,0.1,0.1])
    init_pos = np.array([-0.01266822, 0.24700291, 0.22640632])
    init_orn = np.array([-0.4472884, 0.52046885, -0.51007255, 0.51852797])
    # int_orn = p.getQuaternionFromEuler([0, -math.pi, 0])
    kine = kinematics.Kinematics()

    target_pos = init_pos
    target_ori = init_orn

    for i in range(0,10):
        # target_pos += [0.01,0.01,0.01]
        print(target_pos)
        target_joint_angle = panda_robot.calculate_inverse_kinematics (target_pos,target_ori)
        print(target_joint_angle)
        target_joint_angle = np.array(target_joint_angle)
        pos = kine.fk(target_joint_angle).reshape(12, 1)
        init_rotation_matrix = pos[0:9].reshape(3, 3)
        init_xyz_pos = pos[9:12]
        q = rotation_matrixToQ(init_rotation_matrix)
        print([init_xyz_pos,q])


if __name__ == '__main__':
    main()
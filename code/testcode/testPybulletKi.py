import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data
from kinematics import Kinematics
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrixToQ(rotation_matrix):
    r = R.from_matrix(rotation_matrix)
    return r.as_quat()

def pos_from_matrix (pos_matrix):
    rotation_matrix = pos_matrix[0:9].reshape(3,3)
    pos_q = rotation_matrixToQ(rotation_matrix).reshape(4)
    pos = pos_matrix[9:12].reshape(3)
    return np.hstack([pos,pos_q])

clid = p.connect(p.SHARED_MEMORY)
if (clid < 0):
  p.connect(p.GUI)
  #p.connect(p.SHARED_MEMORY_GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
panda_kinematics = Kinematics()

kukaId = p.loadURDF("franka_panda/model_description/panda.urdf",useFixedBase=True)
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
print(numJoints)




ll = [-1, -1.7628, -2.8973, -2.6, -2.8973, -1.6573, -2]
#upper limits for null space
ul = [1, 1.7628, 2.8973, -2.3, 2.8973, 2.1127, 2]
#
# #joint ranges for null space
jr = [2, 4, 5.8, 0.5, 5.8, 4, 4]
#
rp = [0.14351578,-1.76146432,1.8522255,-2.47442022,0.26120127,0.71819886,1.58290257]



for i in range(numJoints):
  p.resetJointState(kukaId, i, rp[i])

input('\n')
pos = pos_from_matrix(panda_kinematics.fk(np.array(rp)))
print(pos)

jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos[0:3],
                                                    # pos[3:7],
                                                  lowerLimits=ll,
                                                  upperLimits=ul,
                                                  jointRanges=jr,
                                                  restPoses=rp)
print(jointPoses)
forward_pos = pos_from_matrix(panda_kinematics.fk(np.array(rp)))
print(forward_pos)
input('\n')

for i in range(numJoints):
  p.resetJointState(kukaId, i, jointPoses[i])

input('\n')
p.setGravity(0, 0, 0)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1

useOrientation = 0
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 1
useRealTimeSimulation = 0
ikSolver = 0
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15

i=0
while 1:
  i+=1
  #p.getCameraImage(320,
  #                 200,
  #                 flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
  #                 renderer=p.ER_BULLET_HARDWARE_OPENGL)
  if (useRealTimeSimulation):
    dt = datetime.now()
    t = (dt.second / 60.) * 2. * math.pi
  else:
    t = t + 0.01

  if (useSimulation and useRealTimeSimulation == 0):
    p.stepSimulation()
  for i in range(1):
    pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
    pos = [-0.01266822, 0.24700291, 0.22640632]
    #end effector points down, not up (in case useOrientation==1)
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])

    if (useNullSpace == 1):
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul,
                                                  jr, jd)

      else:
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  lowerLimits=ll,
                                                  upperLimits=ul,
                                                  jointRanges=jr,
                                                  restPoses=rp)
        print("!")
    else:
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  orn,
                                                  jointDamping=jd,
                                                  solver=ikSolver,
                                                  maxNumIterations=100,
                                                  residualThreshold=.01)
      else:
        jointPoses = p.calculateInverseKinematics(kukaId,
                                                  kukaEndEffectorIndex,
                                                  pos,
                                                  solver=ikSolver)
    print(jointPoses)
    if (useSimulation):
      for i in range(numJoints):
        p.setJointMotorControl2(bodyIndex=kukaId,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[i],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    else:
      #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
      for i in range(numJoints):
        p.resetJointState(kukaId, i, jointPoses[i])




  ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
  if (hasPrevPose):
    p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
    p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
  prevPose = pos
  prevPose1 = ls[4]
  hasPrevPose = 1
p.disconnect()
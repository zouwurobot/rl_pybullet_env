import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from . import kinova
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class KinovaGymEnv(gym.Env):
  metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second' : 50
  }

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               render=False,
               isDiscrete=False,
               maxSteps = 1000,
               hard_reset = False,
                ):
    #print("KukaGymEnv __init__")
    self._isDiscrete = isDiscrete
    self._timeStep = 1./240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = render
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40
    self._hard_reset = True
    self._robot_urdfRoot = '/home/kinova16/PycharmProjects/rl_kit_pytorch-my_pybullet/my_model/pybullet-models'

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid<0):
         cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.5,180,-41,[0.32,0.1,-0.00])
    else:
      p.connect(p.DIRECT)

    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self._seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
       action_dim = 3
       self._action_bound = 1
       action_high = np.array([self._action_bound] * action_dim)
       self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

    self._hard_reset = hard_reset  # This assignment need to be after reset()

  def _reset(self):
    #print("KukaGymEnv _reset")
    if self._hard_reset:

      p.resetSimulation()
      p.setPhysicsEngineParameter(numSolverIterations=150)
      p.setTimeStep(self._timeStep)

      # build gym env
      p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"),[0,0,-1])
      self.tableUid = p.loadURDF(os.path.join(self._urdfRoot,"table/table.urdf"), [0.5000000,0.00000,-.620000],
                 [0.000000,0.000000,0.0,1.0])

      xpos = 0.55 +0.12*random.random()
      ypos = 0
      ang = math.pi/2
      zpos = 0.25
      orn = p.getQuaternionFromEuler([ang,0,0])
      self.cubeUid = p.loadURDF(os.path.join(self._robot_urdfRoot,"jaco2/urdf/Cube.urdf"),[0.55,0,0.15],
                                                                [0,0,0,1],
                                                                  useFixedBase=True)

      self.teacupUid =p.loadURDF(os.path.join(self._robot_urdfRoot,"jaco2/urdf/TeaCup.urdf"), [xpos,ypos,zpos],
                                                                [orn[0],orn[1],orn[2],orn[3]],
                                                                  useFixedBase = False)
      p.setGravity(0,0,-9.81)

      #camera configuration
      self._CameraViewMatrix = p.computeViewMatrixFromYawPitchRoll([xpos,ypos,zpos], 1, -90, -60, 0, 2)
      # self._CameraProjMatrix = p.computeProjectionMatrix(-0.5000, 0.5000, -0.5000, 1.5000, 1.0000, 6.0000)
      self._CameraProjMatrix = p.computeProjectionMatrixFOV(80, 0.5, 0, 8)

      self.kinova = kinova.Kinova(p,
                                robot_type = 'j2s7s300',
                                urdfRootPath = self._robot_urdfRoot,
                                timeStep=self._timeStep,
                                building_env = False,  # use gym env
                                useInverseKinematics = True,  # IMPORTANCE! It determines the mode of the motion.
                                torque_control_enabled = False,
                                is_fixed = True )
    else:
      self.kinova.reset(reload_urdf=False)

    self.terminated = 0
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()

    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def _seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  # def getAction(self):
  #   cupPos, cupOrn = p.getBasePositionAndOrientation(self.teacupUid)
  #   EndEffector_pos, EndEffector_orn = self.kinova.EndEffectorObersavations()
  #   invEndEffector_pos, invEndEffector_orn = p.invertTransform(EndEffector_pos, EndEffector_orn)
  #   cupPosInEE, cupOrnInEE = p.multiplyTransforms(invEndEffector_pos, invEndEffector_orn, cupPos, cupOrn)
  #
  #   return cupPosInEE

  def getExtendedObservation(self):
     self._observation = []
     image_obs = p.getCameraImage(128, 128, self._CameraViewMatrix, self._CameraProjMatrix)
     cupPos, cupOrn = p.getBasePositionAndOrientation(self.teacupUid)
     kinovaState = self.kinova.GetObservation()

     # relative position
     EndEffector_pos, EndEffector_orn = self.kinova.EndEffectorObersavations()
     invEndEffector_pos, invEndEffector_orn = p.invertTransform(EndEffector_pos, EndEffector_orn)
     cupPosInEE, cupOrnInEE = p.multiplyTransforms(invEndEffector_pos, invEndEffector_orn, cupPos, cupOrn)

     self._observation.extend(image_obs)
     self._observation.extend(kinovaState)
     self._observation.extend([cupPos, cupOrn, cupPosInEE, cupOrnInEE])
     # gripperState  = p.getLinkState(self._kuka.kukaUid,self._kuka.kukaGripperIndex)
     # gripperPos = gripperState[0]
     # gripperOrn = gripperState[1]
     # blockPos,blockOrn = p.getBasePositionAndOrientation(self.blockUid)
     #
     # invGripperPos,invGripperOrn = p.invertTransform(gripperPos,gripperOrn)
     # gripperMat = p.getMatrixFromQuaternion(gripperOrn)
     # dir0 = [gripperMat[0],gripperMat[3],gripperMat[6]]
     # dir1 = [gripperMat[1],gripperMat[4],gripperMat[7]]
     # dir2 = [gripperMat[2],gripperMat[5],gripperMat[8]]
     #
     # gripperEul =  p.getEulerFromQuaternion(gripperOrn)
     # #print("gripperEul")
     # #print(gripperEul)
     # blockPosInGripper,blockOrnInGripper = p.multiplyTransforms(invGripperPos,invGripperOrn,blockPos,blockOrn)
     # projectedBlockPos2D =[blockPosInGripper[0],blockPosInGripper[1]]
     # blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
     # #print("projectedBlockPos2D")
     # #print(projectedBlockPos2D)
     # #print("blockEulerInGripper")
     # #print(blockEulerInGripper)
     #
     # #we return the relative x,y position and euler angle of block in gripper space
     # blockInGripperPosXYEulZ =[blockPosInGripper[0],blockPosInGripper[1],blockEulerInGripper[2]]
     #
     # #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir0[0],gripperPos[1]+dir0[1],gripperPos[2]+dir0[2]],[1,0,0],lifeTime=1)
     # #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir1[0],gripperPos[1]+dir1[1],gripperPos[2]+dir1[2]],[0,1,0],lifeTime=1)
     # #p.addUserDebugLine(gripperPos,[gripperPos[0]+dir2[0],gripperPos[1]+dir2[1],gripperPos[2]+dir2[2]],[0,0,1],lifeTime=1)
     #
     # self._observation.extend(list(blockInGripperPosXYEulZ))
     return self._observation

  def _step(self, action):
    if (self._isDiscrete):
      dv = 0.005
      dx = [0,-dv,dv,0,0,0,0][action]
      dy = [0,0,0,-dv,dv,0,0][action]
      da = [0,0,0,0,0,-0.05,0.05][action]
      f = 0.3
      realAction = [dx,dy,-0.002,da,f]
    else:
      #print("action[0]=", str(action[0]))
      dv = 0.005
      dx = action[0] * dv
      dy = action[1] * dv
      dz = action[2] * dv
      # da = action[2] * 0.05

      # Compute EndEffector Oritation: Oriented to the cup
      # cupPos, cupOrn = p.getBasePositionAndOrientation(self.teacupUid)
      # EndEffector_pos, EndEffector_orn = self.kinova.EndEffectorObersavations()
      # invEndEffector_pos, invEndEffector_orn = p.invertTransform(EndEffector_pos, EndEffector_orn)
      # cupPosInEE, cupOrnInEE = p.multiplyTransforms(invEndEffector_pos, invEndEffector_orn, cupPos, cupOrn)
      #Orn = [item * 0.05 for item in list(cupOrnInEE)]

      # Compute EndEffector Oritation: Oriented fixed direction
      f = 0.3

      realAction = [dx,dy,dz,0,f]
    return self.step2(realAction)

  def step2(self, action):
    for i in range(self._actionRepeat):
      self.kinova.ApplyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()

    #print("self._envStepCounter")
    #print(self._envStepCounter)

    done = self._termination()
    npaction = np.array([action[3]]) #only penalize rotation until learning works well [action[0],action[1],action[3]])
    actionCost = np.linalg.norm(npaction)*10.
    #print("actionCost")
    #print(actionCost)
    reward = self._reward()-actionCost
    #print("reward")
    #print(reward)

    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

  def _render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])
    
    base_pos,orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=base_pos,
        distance=self._cam_dist,
        yaw=self._cam_yaw,
        pitch=self._cam_pitch,
        roll=0,
        upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(
        fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
        nearVal=0.1, farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(
        width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
        projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        #renderer=self._p.ER_TINY_RENDERER)

        
    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
		
    rgb_array = rgb_array[:, :, :3]
    return rgb_array


  def _termination(self):
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self.kinova.kinovaUid,self.kinova.EndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter>self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005
    closestPoints_1 = p.getClosestPoints(self.teacupUid,self.kinova.kinovaUid,  maxDist)
    closestPoints_2 = p.getClosestPoints(self.tableUid, self.kinova.kinovaUid, maxDist)

    if (len(closestPoints_1)):#(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #print("terminating, closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range (100):
        graspAction = [0,0,0.0001,0,fingerAngle]
        self.kinova.ApplyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle-(0.3/100.)
        if (fingerAngle<0):
          fingerAngle=0

      for i in range (1000):
        graspAction = [0,0,0.001,0,fingerAngle]
        self.kinova.ApplyAction(graspAction)
        p.stepSimulation()
        cupPos,cupOrn=p.getBasePositionAndOrientation(self.teacupUid)
        if (cupPos[2] > 0.23):
          #print("BLOCKPOS!")
          #print(blockPos[2])
          break
        state = p.getLinkState(self.kinova.kinovaUid,self.kinova.EndEffectorIndex)
        actualEndEffectorPos = state[0] ##TODO COM need to modified
        if (actualEndEffectorPos[2]>0.5):
          break

      self._observation = self.getExtendedObservation()
      return True
    return False

  def _reward(self):
    maxDist = 0.003

    #rewards is height of target object
    cupPos,cupOrn=p.getBasePositionAndOrientation(self.teacupUid)
    closestPoints = p.getClosestPoints(self.teacupUid,self.kinova.kinovaUid,maxDist, -1, self.kinova.EndEffectorIndex)

    reward = -1000

    numPt = len(closestPoints)
    #print(numPt)
    if (numPt>0):
      #print("reward:")
      reward = -closestPoints[0][8]*10
    if (closestPoints):
      reward = reward+10000
      print("successfully grasped a cup!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      #print("reward")
      #print(reward)
     #print("reward")
    #print(reward)
    return reward

  if parse_version(gym.__version__)>=parse_version('0.9.6'):
    render = _render
    reset = _reset
    seed = _seed
    step = _step

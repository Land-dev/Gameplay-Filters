import pybullet as p
import os
import math
import numpy as np
from .utils import *
from scipy.spatial.transform import Rotation


class Go2:

  def __init__(self, client, height, orientation, envtype=None, payload=0, payload_max=0, **kwargs):
    self.client = client
    # self.urdf = "go2/go2_simplified_cp_improved.urdf" # make the effort and velocity larger, add damping and stiffness
    # self.urdf = "go2/go2_simplified_cp.urdf" # everything CoRL except for demo, realistic number
    # self.urdf = "go2/go2_simplified.urdf"
    self.urdf = "go2/go2_cp.urdf"
    self.height = height

    self.type = "physical"  # sim (harder), physical (easier)

    self.positionGain = [0.4, 0.8, 0.9]
    self.velocityGain = [0.5, 1.2, 1.1]

    ox = 0
    oy = 0

    if "ox" in kwargs.keys():
      ox = kwargs["ox"]
    if "oy" in kwargs.keys():
      oy = kwargs["oy"]

    if "dim_x" in kwargs.keys():
      self.dim_x = kwargs["dim_x"]
    else:
      self.dim_x = 33

    if "action_type" in kwargs.keys():
      self.action_type = kwargs["action_type"]
    else:
      self.action_type = "increment"

    if "center" in kwargs.keys():
      self.center = kwargs["center"]
    else:
      self.center = None

    if "target_list" in kwargs.keys():
      self.target_list = kwargs["target_list"]
    else:
      self.target_list = []

    if "safety_list" in kwargs.keys():
      self.safety_list = kwargs["safety_list"]
    else:
      self.safety_list = []

    if envtype != None:
      # TODO: create different env here
      pass

    f_name = os.path.join(os.path.dirname(__file__), self.urdf)

    self.id = p.loadURDF(
        fileName=f_name, basePosition=[ox, oy, self.height], baseOrientation=orientation, physicsClientId=client, flags=p.URDF_USE_SELF_COLLISION
    )

    self.joint_index = self.make_joint_list()
    self.toe_index = self.make_toe_joint_list()
    self.cp_index = self.make_cp_joint_list()

    self.obs = None

    corners = self.get_body_corners()
    corner_height = corners[2, :]

    elbows = self.get_elbows()
    elbow_height = elbows[2, :]

    self.last_contact = np.zeros(4, dtype=bool)
    self.feet_air_time = np.zeros(4)

    # self.corner_visualizer = []
    # self.elbow_visualizer = []

    # # draw visualizer where corner is
    # corner_coordinates = corners.T
    # for i, corner in enumerate(corner_coordinates):
    #     self.corner_visualizer.append(p.addUserDebugLine(corner-0.05, corner+0.05, lineColorRGB = [1,1,0] , lineWidth = 3))
    # print(corners[2, :])

    # elbow_coordinates = elbows.T
    # for i, elbow in enumerate(elbow_coordinates):
    #     self.elbow_visualizer.append(p.addUserDebugLine(elbow-0.05, elbow+0.05, lineColorRGB = [0,1,1] , lineWidth = 3))
    # print(elbows[2,:])

    # find the ground
    self.plane = None
    for i in range(p.getNumBodies()):
      if p.getBodyInfo(i)[1].decode("ascii") == "plane":
        self.plane = i
        break

  def get_ids(self):
    return self.id, self.client

  def reset(self, position, velocity=None):
    if velocity is None:
      for i in range(len(self.joint_index)):
        p.resetJointState(self.id, self.joint_index[i], position[i], physicsClientId=self.client)
    else:
      for i in range(len(self.joint_index)):
        p.resetJointState(
            self.id, self.joint_index[i], position[i], targetVelocity=velocity[i], physicsClientId=self.client
        )

  def apply_action(self, action):
    """
        Action is the angular increase for each of the joint wrt to the current position

        Args:
            action (_type_): angular positional increase
        """
    if self.action_type == "increment":
      new_angle = np.array(self.get_joint_position()) + np.array(action)
    elif self.action_type == "center_sampling":
      new_angle = np.array(self.center) + np.array(action)

    for i in range(len(self.joint_index)):
      info = p.getJointInfo(self.id, self.joint_index[i], physicsClientId=self.client)
      lower_limit = info[8]
      upper_limit = info[9]
      max_force = info[10]
      max_velocity = info[11]
      pos = min(max(lower_limit, new_angle[i]), upper_limit)

      # error = position_gain*(desired_position-actual_position)+velocity_gain*(desired_velocity-actual_velocity)
      # p.setJointMotorControl2(
      #     self.id, self.joint_index[i], p.POSITION_CONTROL, targetPosition=pos, positionGain=self.positionGain[i % 3],
      #     velocityGain=self.velocityGain[i % 3], force=max_force, maxVelocity=max_velocity, physicsClientId=self.client
      # )

      p.setJointMotorControl2(
          self.id, self.joint_index[i], p.POSITION_CONTROL, targetPosition=pos, 
          force=max_force, maxVelocity=max_velocity, physicsClientId=self.client
      )

  def apply_position(self, action):
    for i in range(len(self.joint_index)):
      info = p.getJointInfo(self.id, self.joint_index[i], physicsClientId=self.client)
      lower_limit = info[8]
      upper_limit = info[9]
      max_force = info[10]
      max_velocity = info[11]
      pos = min(max(lower_limit, action[i]), upper_limit)

      p.setJointMotorControl2(
          self.id, self.joint_index[i], p.POSITION_CONTROL, targetPosition=pos, positionGain=self.positionGain[i % 3],
          velocityGain=self.velocityGain[i % 3], force=max_force, maxVelocity=max_velocity, physicsClientId=self.client
      )

  def get_obs(self):
    """Get observation 32-D
            x_dot, y_dot, z_dot,
            roll, pitch,
            w_x, w_y, w_z,
            joint_pos x 12,
            joint_vel x 12

        Returns:
            observation (Tuple): 32-D observation
        """
    pos, ang = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)
    rotmat = Rotation.from_quat(ang).as_matrix()
    ang = p.getEulerFromQuaternion(ang, physicsClientId=self.client)
    linear_vel, angular_vel = p.getBaseVelocity(self.id, physicsClientId=self.client)
    # convert linear_vel and angular_vel to body frame coordinate
    #! TODO: DOUBLE CHECK TO SEE IF THIS TRANSFORMATION IS CORRECT
    robot_body_linear_vel = (np.linalg.inv(rotmat) @ np.array(linear_vel).T)
    robot_body_angular_vel = (np.linalg.inv(rotmat) @ np.array(angular_vel).T)
    joint_pos, joint_vel, joint_force, joint_torque = self.get_joint_state()

    if self.dim_x == 32:
      self.obs = (
          tuple(robot_body_linear_vel) + tuple(ang[:2]) + tuple(robot_body_angular_vel) + tuple(joint_pos)
          + tuple(joint_vel)
      )
    elif self.dim_x == 33:
      self.obs = (
          tuple([pos[2]]) + tuple(robot_body_linear_vel) + tuple(ang[:2]) + tuple(robot_body_angular_vel)
          + tuple(joint_pos) + tuple(joint_vel)
      )
    # elif self.dim_x == 36:
    #   # x, y, yaw
    #   self.obs = (
    #       tuple(pos) + tuple(robot_body_linear_vel) + tuple(ang) + tuple(robot_body_angular_vel) + tuple(joint_pos)
    #       + tuple(joint_vel)
    #   )
    elif self.dim_x == 36:
      # include foot contact information
      self.obs = (
          tuple(robot_body_linear_vel) + tuple(ang[:2]) + tuple(robot_body_angular_vel) + tuple(joint_pos)
          + tuple(joint_vel) + tuple(self.get_foot_contact())
      )
    else:
      print(self.dim_x)
      raise NotImplementedError

    return self.obs

  def map_to_spirit(self, position):
    """ 
    Map joint position from Go2 range to Spirit range
    """
    mapped_pos = []
    for i, j in enumerate(position):
      if i % 3 == 0:
        mapped_pos.append(j)
      elif i % 3 == 1:
        mapped_pos.append(np.arccos((2*j) / np.pi))
      elif i % 3 == 2:
        mapped_pos.append(np.arccos(-1 + (-2 * j) / np.pi))
    return mapped_pos

  def get_joint_state(self):
    joint_state = p.getJointStates(self.id, jointIndices=self.joint_index, physicsClientId=self.client)
    joint_pos = [state[0] for state in joint_state]
    joint_vel = [state[1] for state in joint_state]
    joint_force = [state[2] for state in joint_state]
    joint_torque = [state[3] for state in joint_state]
    # return self.map_to_spirit(joint_pos), joint_vel, joint_force, joint_torque
    return joint_pos, joint_vel, joint_force, joint_torque

  def get_joint_position(self):
    joint_state = p.getJointStates(self.id, jointIndices=self.joint_index, physicsClientId=self.client)
    position = [state[0] for state in joint_state]
    # return self.map_to_spirit(position)
    return position

  def get_joint_torque(self):
    joint_state = p.getJointStates(self.id, jointIndices=self.joint_index, physicsClientId=self.client)
    torque = [state[3] for state in joint_state]
    return torque

  def safety_margin(self, **kwargs):
    """
        Safety margin of the robot. 
        If the robot gets too close to the ground, or if any of the knee touches the ground (within an error margin)
        """
    # height, roll, pitch
    # rotate_margin = np.array([0.16, 0.16, 0.16]) * np.pi
    # dt = 0.008
    # new_obs = state[:9]
    # old_obs = state[9:18]
    # accel = (new_obs - old_obs)/dt
    # rotate_accel = accel[3:6]
    # rotate_error = abs(np.array(rotate_accel))  - np.array(rotate_margin)

    # return {
    #     "height_lower": 0.1 - state[2],
    #     "rotate_error": max(rotate_error)
    # }

    # NEW SAFETY MARGIN
    # Only consider if the robot flips

    """
      test stance
          0.0, 0.2, 1.0,
          0.0, 0.2, 1.0,
          0.0, 0.2, 1.0,
          0.0, 0.2, 1.0
      """

    state = self.get_obs()

    corners = self.get_body_corners()
    corner_height = corners[2, :]

    elbows = self.get_elbows()
    elbow_height = elbows[2, :]

    # draw visualizer where corners and elbows are
    # corner_coordinates = corners.T
    # for i, corner in enumerate(corner_coordinates):
    #     p.removeUserDebugItem(self.corner_visualizer.pop(0))
    #     self.corner_visualizer.append(p.addUserDebugLine(corner-0.05, corner+0.05, lineColorRGB = [1,1,0] , lineWidth = 3))

    # elbow_coordinates = elbows.T
    # for i, elbow in enumerate(elbow_coordinates):
    #     p.removeUserDebugItem(self.elbow_visualizer.pop(0))
    #     self.elbow_visualizer.append(p.addUserDebugLine(elbow-0.05, elbow+0.05, lineColorRGB = [0,1,1] , lineWidth = 3))

    #! TODO: Gain for corner height and elbow height. Corner x 4.0, Elbow x 8.0

    if self.dim_x == 32:
      roll = state[3]
      pitch = state[4]
    elif self.dim_x == 36:  # THIS IS NOT THE SAME AS WITH X, Y, Z, YAW information, this is with toe contact information
      roll = state[3]
      pitch = state[4]
    else:
      raise NotImplementedError

    if self.type == "physical":
      safeties = {
          "corner_height": min(corner_height) - 0.02,
          "elbow_height": min(elbow_height) - 0.01,
          "roll": math.pi * 0.5 - abs(roll),
          "pitch": math.pi * 0.5 - abs(pitch)
      }
    else:
      safeties = {
          "corner_height": min(corner_height) - 0.15,
          "elbow_height": min(elbow_height) - 0.05,
          "roll": math.pi * 0.5 - abs(roll),
          "pitch": math.pi * 0.5 - abs(pitch)
      }

    if len(self.safety_list) == 0:
      return safeties

    margin = {}
    for safety in self.safety_list:
      margin[safety] = safeties[safety]
    return margin

  def target_margin(self, **kwargs):
    """ 
        {
            body z no higher than 0.35 (m)
            all four feet below h_threshold (0.05 m) 
            and body angular velocity norm below w_threshold (10º/s) 
            and body velocity vector norm below v_threshold (0.2 m/s)
        }
        state = z, x_dot, y_dot, z_dot, roll, pitch, w_x, w_y, w_z, joint_pos x 12, joint_vel x 12
        """

    state = self.get_obs()
    toe_height = self.get_toes()
    corners = self.get_body_corners()
    corner_height = corners[2, :]
    elbows = self.get_elbows()
    elbow_height = elbows[2, :]

    if self.dim_x == 32:
      roll = state[3]
      pitch = state[4]
    elif self.dim_x == 36:  # THIS IS NOT THE SAME AS WITH X, Y, Z, YAW information, this is with toe contact information
      roll = state[3]
      pitch = state[4]
    else:
      raise NotImplementedError

    #! TODO: Gain for toe height, toe x 4.0
    if self.dim_x == 33:
      targets = {
          "corner_height": 0.4 - max(corner_height),
          "elbow_height": 0.2 - max(elbow_height),
          "toe_height": 0.05 - max(toe_height),
          "body_ang_x": 0.17444 - abs(state[6]),
          "body_ang_y": 0.17444 - abs(state[7]),
          "body_ang_z": 0.17444 - abs(state[8]),
          "x_dot": 0.2 - abs(state[1]),
          "y_dot": 0.2 - abs(state[2]),
          "z_dot": 0.2 - abs(state[3]),
          "roll": math.pi * 0.2 - abs(roll),
          "pitch": math.pi * 0.2 - abs(pitch)
      }
    elif self.dim_x == 32:
      targets = {
          "corner_height": 0.4 - max(corner_height),
          "elbow_height": 0.2 - max(elbow_height),
          "toe_height": 0.05 - max(toe_height),
          "body_ang_x": 0.17444 - abs(state[5]),
          "body_ang_y": 0.17444 - abs(state[6]),
          "body_ang_z": 0.17444 - abs(state[7]),
          "x_dot": 0.2 - abs(state[0]),
          "y_dot": 0.2 - abs(state[1]),
          "z_dot": 0.2 - abs(state[2]),
          "roll": math.pi * 0.2 - abs(roll),
          "pitch": math.pi * 0.2 - abs(pitch)
      }
    elif self.dim_x == 36:  # THIS IS NOT THE SAME AS WITH X, Y, Z, YAW information, this is with toe contact information
      targets = {
          "corner_height": 0.4 - max(corner_height),
          "elbow_height": 0.2 - max(elbow_height),
          "toe_height": 0.05 - max(toe_height),
          "body_ang_x": 0.17444 - abs(state[5]),
          "body_ang_y": 0.17444 - abs(state[6]),
          "body_ang_z": 0.17444 - abs(state[7]),
          "x_dot": 0.2 - abs(state[0]),
          "y_dot": 0.2 - abs(state[1]),
          "z_dot": 0.2 - abs(state[2]),
          "roll": math.pi * 0.2 - abs(roll),
          "pitch": math.pi * 0.2 - abs(pitch)
      }
    else:
      raise NotImplementedError

    if len(self.target_list) == 0:
      return targets

    margin = {}
    for target in self.target_list:
      margin[target] = targets[target]
    return margin

  def make_joint_list(self):
    damaged_legs = []
    joint_names = [
        b'FL_hip', b'FL_thigh', b'FL_calf', b'RL_hip', b'RL_thigh', b'RL_calf', b'FR_hip', b'FR_thigh', b'FR_calf',
        b'RR_hip', b'RR_thigh', b'RR_calf'
    ]
    joint_list = []
    for n in joint_names:
      joint_found = False
      for joint in range(p.getNumJoints(self.id, physicsClientId=self.client)):
        name = p.getJointInfo(self.id, joint, physicsClientId=self.client)[12]
        if name == n and name not in damaged_legs:
          joint_list += [joint]
          joint_found = True
        elif name == n and name in damaged_legs:
          p.changeVisualShape(1, joint, rgbaColor=[0.5, 0.5, 0.5, 0.5], physicsClientId=self.client)
      if joint_found is False:
        # if the joint is not here (aka broken leg case) put 1000
        # joint_list += [1000]
        continue
    return joint_list

  def make_toe_joint_list(self):
    joint_names = [b'FL_foot', b'FR_foot', b'RL_foot', b'RR_foot']
    joint_list = []
    for n in joint_names:
      for joint in range(p.getNumJoints(self.id, physicsClientId=self.client)):
        name = p.getJointInfo(self.id, joint, physicsClientId=self.client)[12]
        if name == n:
          joint_list += [joint]
    return joint_list

  def make_cp_joint_list(self):
    # LiDAR, hips, elbows, body corners
    joint_names = [
        b'cp0', b'cp1', b'cp2', b'cp3', b'cp4', b'cp5', b'cp6', b'cp7', b'cp8', b'corner1', b'corner2', b'corner3',
        b'corner4', b'corner5', b'corner6', b'corner7', b'corner8'
    ]

    # joint_names = [b'FL_hip', b'FR_hip', b'RL_hip', b'RR_hip', b'FL_calf', b'FR_calf', b'RL_calf', b'RR_calf']
    joint_list = []
    for n in joint_names:
      for joint in range(p.getNumJoints(self.id, physicsClientId=self.client)):
        name = p.getJointInfo(self.id, joint, physicsClientId=self.client)[12]
        if name == n:
          joint_list += [joint]
    return joint_list

  def get_elbows(self):
    """ 
    Return matrix shape (3, 4)
    """
    cp_states = p.getLinkStates(
        self.id, linkIndices=self.cp_index[5:9] + [self.cp_index[0]], physicsClientId=self.client
    )
    # cp_states = p.getLinkStates(self.id, linkIndices=self.cp_index[4:], physicsClientId=self.client)
    return np.concatenate([np.array(x[0]).reshape(3, 1) for x in cp_states], axis=1)

  def get_body_corners(self):
    """ 
    Return matrix shape (3, 4)
    """
    cp_states = p.getLinkStates(
        self.id, linkIndices=self.cp_index[9:] + self.cp_index[1:5], physicsClientId=self.client
    )
    # cp_states = p.getLinkStates(self.id, linkIndices=self.cp_index[0:4], physicsClientId=self.client)
    return np.concatenate([np.array(x[0]).reshape(3, 1) for x in cp_states], axis=1)

  def get_toes(self):
    toe_states = p.getLinkStates(self.id, linkIndices=self.toe_index, physicsClientId=self.client)
    return [x[0][2] for x in toe_states]

  def get_foot_contact(self):
    if self.plane is None:
      return [0, 0, 0, 0]

    contact_array = []
    # normal_forces = []
    for i in self.toe_index:
      contacts = p.getContactPoints(bodyA=self.id, bodyB=self.plane, linkIndexA=i, physicsClientId=self.client)
      if len(contacts) > 0:
        normal_force = 0.0
        for contact in contacts:
          normal_force += contact[9]
        # normal_forces.append(normal_force)
        if normal_force > 5:
          contact_array.append(1)
        else:
          contact_array.append(0)
      else:
        # normal_forces.append(0.0)
        contact_array.append(0)
    # return normal_forces
    return contact_array

import pybullet as p
import os
import math
import numpy as np
from .utils import *
from scipy.spatial.transform import Rotation


class Spirit:

  def __init__(self, client, height, orientation, envtype=None, payload=0, payload_max=0, **kwargs):
    self.client = client
    self.urdf = "spirit40.urdf"
    self.height = height

    self.type = "sim"

    # old gain when dt=0.008
    # self.positionGain = [1./12., 1./12., 1./12.]
    # self.velocityGain = [0.4, 0.4, 0.4]

    # no load, dt=0.05
    # self.positionGain = [0.4, 0.4, 0.3]
    # self.velocityGain = [0.5, 0.6, 0.6]

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
        fileName=f_name, basePosition=[ox, oy, self.height], baseOrientation=orientation, physicsClientId=client
    )

    self.joint_index = self.make_joint_list()
    self.toe_index = self.make_toe_joint_list()
    self.torque_gain = 10.0

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
    # # print(corners[2, :])

    # elbow_coordinates = elbows.T
    # for i, elbow in enumerate(elbow_coordinates):
    #     self.elbow_visualizer.append(p.addUserDebugLine(elbow-0.05, elbow+0.05, lineColorRGB = [0,1,1] , lineWidth = 3))
    # # print(elbows[2,:])

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
      p.setJointMotorControl2(
          self.id, self.joint_index[i], p.POSITION_CONTROL, targetPosition=pos, positionGain=self.positionGain[i % 3],
          velocityGain=self.velocityGain[i % 3], force=max_force, maxVelocity=max_velocity, physicsClientId=self.client
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
    elif self.dim_x == 36:
      # x, y, yaw
      self.obs = (
          tuple(pos) + tuple(robot_body_linear_vel) + tuple(ang) + tuple(robot_body_angular_vel) + tuple(joint_pos)
          + tuple(joint_vel)
      )
    else:
      print(self.dim_x)
      raise NotImplementedError

    return self.obs

  def get_joint_state(self):
    joint_state = p.getJointStates(self.id, jointIndices=self.joint_index, physicsClientId=self.client)
    joint_pos = [state[0] for state in joint_state]
    joint_vel = [state[1] for state in joint_state]
    joint_force = [state[2] for state in joint_state]
    joint_torque = [state[3] for state in joint_state]
    return joint_pos, joint_vel, joint_force, joint_torque

  def get_joint_position(self):
    joint_state = p.getJointStates(self.id, jointIndices=self.joint_index, physicsClientId=self.client)
    position = [state[0] for state in joint_state]
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
          "corner_height": min(corner_height) - 0.2,
          "elbow_height": min(elbow_height) - 0.1,
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

    #! TODO: Gain for toe height, toe x 4.0
    if self.dim_x == 33:
      targets = {
          "corner_height": 0.4 - max(corner_height),
          "toe_height": 0.05 - max(toe_height),
          "body_ang_x": 0.17444 - abs(state[6]),
          "body_ang_y": 0.17444 - abs(state[7]),
          "body_ang_z": 0.17444 - abs(state[8]),
          "x_dot": 0.2 - abs(state[1]),
          "y_dot": 0.2 - abs(state[2]),
          "z_dot": 0.2 - abs(state[3])
      }
    elif self.dim_x == 32:
      targets = {
          "corner_height": 0.4 - max(corner_height),
          "toe_height": 0.05 - max(toe_height),
          "body_ang_x": 0.17444 - abs(state[5]),
          "body_ang_y": 0.17444 - abs(state[6]),
          "body_ang_z": 0.17444 - abs(state[7]),
          "x_dot": 0.2 - abs(state[0]),
          "y_dot": 0.2 - abs(state[1]),
          "z_dot": 0.2 - abs(state[2])
      }

    if len(self.target_list) == 0:
      return targets

    margin = {}
    for target in self.target_list:
      margin[target] = targets[target]
    return margin

  def make_joint_list(self):
    damaged_legs = []
    joint_names = [
        b'hip0', b'upper0', b'lower0', b'hip1', b'upper1', b'lower1', b'hip2', b'upper2', b'lower2', b'hip3',
        b'upper3', b'lower3'
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
    joint_names = [b'toe0', b'toe1', b'toe2', b'toe3']
    joint_list = []
    for n in joint_names:
      for joint in range(p.getNumJoints(self.id, physicsClientId=self.client)):
        name = p.getJointInfo(self.id, joint, physicsClientId=self.client)[12]
        if name == n:
          joint_list += [joint]
    return joint_list

  def linc_get_joints_positions(self):
    """Return the actual position in the physics engine"""
    pos = np.zeros(len(self.joint_index))
    i = 0
    # be careful that the joint_list is not necessarily in the same order as
    # in bullet (see make_joint_list)
    for joint in self.joint_index:
      if joint != 1000:
        pos[i] = p.getJointState(self.id, joint, physicsClientId=self.client)[0]
        i += 1
    return pos

  def linc_get_pos(self):
    """
        Return the position list of 3 floats and orientation as list of
        4 floats in [x,y,z,w] order. Use pb.getEulerFromQuaternion to convert
        the quaternion to Euler if needed.
        """
    return p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)

  def linc_get_ground_contacts(self):
    leg_link_ids = [17, 14, 2, 5, 8, 11]
    descriptor = {17: [], 14: [], 2: [], 5: [], 8: [], 11: []}
    ground_contacts = np.zeros_like(leg_link_ids)

    # Get contact points between robot and world plane
    contact_points = p.getContactPoints(self.id, physicsClientId=self.client)
    link_ids = []  # list of links in contact with the ground plane
    if len(contact_points) > 0:
      for cn in contact_points:
        linkid = cn[3]  # robot link id in contact with world plane
        if linkid not in link_ids:
          link_ids.append(linkid)
    for l in leg_link_ids:
      cns = descriptor[l]
      if l in link_ids:
        cns.append(1)
      else:
        cns.append(0)
      descriptor[l] = cns

    for i, ll in enumerate(leg_link_ids):
      if ll in link_ids:
        ground_contacts[i] = 1

    return ground_contacts

  def linc_get_state(self, t):
    """Combine the elements of the state vector."""
    state = np.concatenate([[t],
                            list(sum(self.linc_get_pos(), ())),
                            self.linc_get_joints_positions(),
                            self.linc_get_ground_contacts()])
    return state

  def get_joint_position_wrt_body(self, alpha, beta):
    # get the joint position wrt to body (which joint is closer to the body), from this, the further a joint away from the body, the closer the joint to ground
    # leg length l1 = l2 = 0.206
    # alpha is the angle between upper link and body (upper joint)
    # beta is the angle between lower link and upper link (lower joint)
    # ------ O -- BODY ---------> HEAD
    #      |  \\         | h2
    #      |   \\       B
    #     h1    \\    //
    #      |      A //
    #----------- GROUND --------
    # for all legs, upper joint moving forward to the head will be to 3.14 (180 degree)
    l1 = 0.206
    l2 = 0.206
    h1 = math.sin(math.pi - alpha) * l1
    theta = math.pi * 1.5 - (math.pi - alpha) - beta
    OB = math.sqrt(l1*l1 + l2*l2 - 2 * l1 * l2 * math.cos(beta))
    if OB == 0:
      return h1, 0
    theta_1 = math.acos((l1**2 + OB**2 - l2**2) / (2*l1*OB))
    theta_2 = theta - theta_1
    h2 = math.cos(theta_2) * OB
    return h1, h2

  def calculate_ground_footing(self):
    joints = self.get_joint_position()
    leg0h1, leg0h2 = self.get_joint_position_wrt_body(joints[1], joints[2])
    leg1h1, leg1h2 = self.get_joint_position_wrt_body(joints[4], joints[5])
    leg2h1, leg2h2 = self.get_joint_position_wrt_body(joints[7], joints[8])
    leg3h1, leg3h2 = self.get_joint_position_wrt_body(joints[10], joints[11])

    return leg0h1, leg0h2, leg1h1, leg1h2, leg2h1, leg2h2, leg3h1, leg3h2

  def get_elbows(self):
    pos, ang = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)
    initial_pos = np.array(pos).reshape((3, 1))
    rotmat = Rotation.from_quat(ang).as_matrix()

    body_l = 0.335
    body_w = 0.24
    body_h = 0.104
    upper_link_l = 0.206
    hip_from_body_l = 0.2263
    hip_from_body_w = 0.07
    upper_from_hip_w = 0.1
    hip_w = 0.11
    hip_l = 0.08

    current_joint = self.get_joint_position()

    hip_FL = current_joint[0]
    hip_FR = current_joint[6]
    hip_BL = current_joint[3]
    hip_BR = current_joint[9]
    upper_FL = current_joint[1]
    upper_FR = current_joint[7]
    upper_BL = current_joint[4]
    upper_BR = current_joint[10]

    FL_elbow = initial_pos + rotmat @ (
        translate_x(hip_from_body_l) + rotate_x(-np.pi * 0.5) @ (
            translate_z(hip_from_body_w) + rotate_x(hip_FL) @ (
                translate_x(hip_l * 0.5) + translate_z(upper_from_hip_w)
                + rotate_z(np.pi - upper_FL) @ (translate_x(upper_link_l))
            )
        )
    )
    FR_elbow = initial_pos + rotmat @ (
        translate_x(hip_from_body_l) + rotate_x(np.pi * 0.5) @ (
            translate_z(hip_from_body_w) + rotate_x(hip_FR) @ (
                translate_x(hip_l * 0.5) + translate_z(upper_from_hip_w)
                + rotate_z(np.pi + upper_FR) @ (translate_x(upper_link_l))
            )
        )
    )
    BL_elbow = initial_pos + rotmat @ (
        translate_x(-hip_from_body_l) + rotate_x(-np.pi * 0.5) @ (
            translate_z(hip_from_body_w) + rotate_x(hip_BL) @ (
                translate_x(hip_l * 0.5) + translate_z(upper_from_hip_w)
                + rotate_z(np.pi - upper_BL) @ (translate_x(upper_link_l))
            )
        )
    )
    BR_elbow = initial_pos + rotmat @ (
        translate_x(-hip_from_body_l) + rotate_x(np.pi * 0.5) @ (
            translate_z(hip_from_body_w) + rotate_x(hip_BR) @ (
                translate_x(hip_l * 0.5) + translate_z(upper_from_hip_w)
                + rotate_z(np.pi + upper_BR) @ (translate_x(upper_link_l))
            )
        )
    )

    return np.concatenate([FL_elbow, FR_elbow, BL_elbow, BR_elbow], axis=1)

  def get_body_corners(self):
    pos, ang = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)
    initial_pos = np.array(pos).reshape((3, 1))
    rotmat = Rotation.from_quat(ang).as_matrix()

    # 0.335 0.24 0.104

    L = 0.464  # 0.36
    W = 0.33  # 0.35
    # L = 0.36
    # W = 0.35
    H = 0.104

    FL_top = initial_pos + rotmat @ (
        translate_x(L * 0.5)
        + rotate_x(-np.pi * 0.5) @ (translate_z(W * 0.5) + rotate_x(np.pi * 0.5) @ translate_z(H * 0.5))
    )
    FL_bot = initial_pos + rotmat @ (
        translate_x(L * 0.5)
        + rotate_x(-np.pi * 0.5) @ (translate_z(W * 0.5) + rotate_x(np.pi * 0.5) @ translate_z(-H * 0.5))
    )

    FR_top = initial_pos + rotmat @ (
        translate_x(L * 0.5)
        + rotate_x(np.pi * 0.5) @ (translate_z(W * 0.5) + rotate_x(-np.pi * 0.5) @ translate_z(H * 0.5))
    )
    FR_bot = initial_pos + rotmat @ (
        translate_x(L * 0.5)
        + rotate_x(np.pi * 0.5) @ (translate_z(W * 0.5) + rotate_x(-np.pi * 0.5) @ translate_z(-H * 0.5))
    )

    BL_top = initial_pos + rotmat @ (
        translate_x(-L * 0.5)
        + rotate_x(-np.pi * 0.5) @ (translate_z(W * 0.5) + rotate_x(np.pi * 0.5) @ translate_z(H * 0.5))
    )
    BL_bot = initial_pos + rotmat @ (
        translate_x(-L * 0.5)
        + rotate_x(-np.pi * 0.5) @ (translate_z(W * 0.5) + rotate_x(np.pi * 0.5) @ translate_z(-H * 0.5))
    )

    BR_top = initial_pos + rotmat @ (
        translate_x(-L * 0.5)
        + rotate_x(np.pi * 0.5) @ (translate_z(W * 0.5) + rotate_x(-np.pi * 0.5) @ translate_z(H * 0.5))
    )
    BR_bot = initial_pos + rotmat @ (
        translate_x(-L * 0.5)
        + rotate_x(np.pi * 0.5) @ (translate_z(W * 0.5) + rotate_x(-np.pi * 0.5) @ translate_z(-H * 0.5))
    )

    return np.concatenate([FL_top, FL_bot, FR_top, FR_bot, BL_top, BL_bot, BR_top, BR_bot], axis=1)

  def get_toes(self):
    toe_states = p.getLinkStates(self.id, linkIndices=self.toe_index, physicsClientId=self.client)
    return [x[0][2] for x in toe_states]

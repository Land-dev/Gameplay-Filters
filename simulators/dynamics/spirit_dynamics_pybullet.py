import numpy as np
import pybullet as p
from .base_pybullet_dynamics import BasePybulletDynamics
from typing import Optional, Tuple, Any
from .resources.spirit import Spirit
from .resources.force import Force
import time
import matplotlib.pyplot as plt
from jaxlib.xla_extension import DeviceArray
from scipy.spatial.transform import Rotation


class SpiritDynamicsPybullet(BasePybulletDynamics):

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    #! TODO: FIX THIS, SO THAT THERE WILL BE A SEPARATE DYNAMICS WHEN WE USE ISAACS (BaseDstbDynamics instead of BaseDynamics)
    # config here is config_agent

    if isinstance(config.obs_dim, dict):
      self.dim_x = config.obs_dim["actor_0"]
    else:
      self.dim_x = config.obs_dim

    self.obsrv_list = config.obsrv_list.ctrl
    self.action_type = config.action_type  # increment, center_sampling
    self.action_center = config.action_center

    self.target_list = config.target_margin
    self.safety_list = config.safety_margin

    self.obs_sequence = []

    if self.obsrv_list is not None:
      if "command" in self.obsrv_list:
        # state has command information
        self.command = [0.0, 0.0, 0.0]
      else:
        self.command = None

      if "obs" in self.obsrv_list:
        self.obs_sequence_length = self.obsrv_list.count("obs")
        # self.obs_sequence = [[0] * self.dim_x] * self.obs_sequence_length
        # find first occurence then pop all
        first_occur = self.obsrv_list.index("obs")
        for i in range(self.obs_sequence_length - 1):
          self.obsrv_list.pop(first_occur + 1)
      else:
        self.obs_sequence_length = 0
    else:
      self.command = None

    if isinstance(action_space, dict):
      super().__init__(config, action_space["ctrl"])

      # action range
      # The 12 joints are categorized into abduction, hip and knee
      # Joints of similar category share similar max, min
      # NOTE: This is not the joint range, this is the increment range constraint
      self.abduction_increment_min = action_space["ctrl"][0, 0]
      self.abduction_increment_max = action_space["ctrl"][0, 1]
      self.hip_increment_min = action_space["ctrl"][1, 0]
      self.hip_increment_max = action_space["ctrl"][1, 1]
      self.knee_increment_min = action_space["ctrl"][2, 0]
      self.knee_increment_max = action_space["ctrl"][2, 1]
      self.dim_u = len(action_space["ctrl"])
    else:
      super().__init__(config, action_space)
      self.abduction_increment_min = action_space[0, 0]
      self.abduction_increment_max = action_space[0, 1]
      self.hip_increment_min = action_space[1, 0]
      self.hip_increment_max = action_space[1, 1]
      self.knee_increment_min = action_space[2, 0]
      self.knee_increment_max = action_space[2, 1]
      self.dim_u = len(action_space)

    # TODO: Read this value from URDF, or pass from config
    #! This is hardware constraints, different from action range

    if self.action_type == "increment":
      self.abduction_min = -0.5
      self.abduction_max = 0.5
      self.hip_min = 0.5
      self.hip_max = 2.64
      self.knee_min = 0.5
      self.knee_max = 2.64
    elif self.action_type == "center_sampling":
      self.abduction_min = self.action_center[0] + self.abduction_increment_min
      self.abduction_max = self.action_center[0] + self.abduction_increment_max
      self.hip_min = self.action_center[1] + self.hip_increment_min
      self.hip_max = self.action_center[1] + self.hip_increment_max
      self.knee_min = self.action_center[2] + self.knee_increment_min
      self.knee_max = self.action_center[2] + self.knee_increment_max

    self.initial_height = None
    self.initial_rotation = None
    self.initial_joint_value = None

    self.rendered_img = None
    self.adv_debug_line_id = None
    self.shielding_status_debug_text_id = None
    self.adversarial_object = None

    #! TODO: PUT THIS IN CONFIG
    self.synthetic_symmetrical_dstb = True
    self.reset_count = 0
    self.dstb_array = []
    self.initial_linear_vel = None
    self.initial_angular_vel = None
    self.initial_height_reset_type = None
    self.initial_action = None
    self.initial_joint_vel = None

    self.robot = None

    self.reset()

  def reset(self, **kwargs):
    if self.reset_count % 2 == 0:
      self.dstb_array = []

      # rejection sampling until outside target set and safe set
      while True:
        #! SERIOUSLY NEED REFACTORING
        # use this to reset the robot exactly to this state, no rejection sampling, no standing up/dropping down
        if "initial_state" in kwargs.keys():
          state = kwargs["initial_state"]
        else:
          state = None
        if state is not None:
          """
              z, 
              x_dot, y_dot, z_dot,
              roll, pitch,
              w_x, w_y, w_z,
              joint_pos x 12,
              joint_vel x 12
          """
          # do not reset the simulation, just reset the robot state then go
          self.state = state
          if len(self.state) == 32:  # no z
            self.initial_height = 0.6

            self.initial_rotation = p.getQuaternionFromEuler([state[3], state[4],
                                                              0.0])  # yaw does not matter, there's no terrain anyway
            self.initial_joint_value = state[8:20]
            self.initial_joint_vel = state[20:]

            if self.robot is None:
              self.robot = Spirit(
                  self.client, self.initial_height, self.initial_rotation, dim_x=self.dim_x,
                  action_type=self.action_type, center=self.action_center, target_list=self.target_list,
                  safety_list=self.safety_list, **kwargs
              )
            else:
              p.resetBasePositionAndOrientation(
                  self.robot.id, np.array([0, 0, self.initial_height]), self.initial_rotation,
                  physicsClientId=self.client
              )
            self.robot.reset(self.initial_joint_value)
            # lower down the robot until there's a contact

            p.setGravity(0, 0, self.gravity * 0.2, physicsClientId=self.client)
            for i in range(100):
              toe_height = self.robot.get_toes()
              if min(toe_height) < 0.05:
                break
              p.stepSimulation(physicsClientId=self.client)
            p.setGravity(0, 0, self.gravity, physicsClientId=self.client)
            self.robot.reset(self.initial_joint_value, velocity=self.initial_joint_vel)

            rotmat = Rotation.from_quat(self.initial_rotation).as_matrix()
            self.initial_linear_vel = rotmat @ np.array(state[:3])
            self.initial_angular_vel = rotmat @ np.array(state[5:8])

            p.resetBaseVelocity(
                self.robot.id, linearVelocity=self.initial_linear_vel, angularVelocity=self.initial_angular_vel,
                physicsClientId=self.client
            )

            # if self.adversarial_object is None:
            #   if self.gui and self.force > 0:
            #     self.adversarial_object = Force(self.client)

            if "initial_action" in kwargs.keys():
              self.initial_action = kwargs["initial_action"]
              self.robot.apply_action(self.initial_action)
              p.stepSimulation(physicsClientId=self.client)
            break
          elif len(self.state) == 33:
            self.initial_height = state[0]
            # self.initial_linear_vel = np.array(state[1:4])
            self.initial_rotation = p.getQuaternionFromEuler([state[4], state[5],
                                                              0.0])  # yaw does not matter, there's no terrain anyway
            # self.initial_angular_vel = np.array(state[6:9])
            self.initial_joint_value = state[9:21]
            self.initial_joint_vel = state[21:]

            if self.robot is None:
              self.robot = Spirit(
                  self.client, self.initial_height, self.initial_rotation, dim_x=self.dim_x,
                  action_type=self.action_type, center=self.action_center, target_list=self.target_list,
                  safety_list=self.safety_list, **kwargs
              )
            else:
              p.resetBasePositionAndOrientation(
                  self.robot.id, np.array([0, 0, self.initial_height]), self.initial_rotation,
                  physicsClientId=self.client
              )
            self.robot.reset(self.initial_joint_value, velocity=self.initial_joint_vel)

            rotmat = Rotation.from_quat(self.initial_rotation).as_matrix()
            self.initial_linear_vel = rotmat @ np.array(state[1:4])
            self.initial_angular_vel = rotmat @ np.array(state[6:9])

            p.resetBaseVelocity(
                self.robot.id, linearVelocity=self.initial_linear_vel, angularVelocity=self.initial_angular_vel,
                physicsClientId=self.client
            )

            # if self.adversarial_object is None:
            #   if self.gui and self.force > 0:
            #     self.adversarial_object = Force(self.client)

            if "initial_action" in kwargs.keys():
              self.initial_action = kwargs["initial_action"]
              self.robot.apply_action(self.initial_action)
              p.stepSimulation(physicsClientId=self.client)
            break
          else:
            raise NotImplementedError

        else:
          if "initial_height" in kwargs.keys():
            height = kwargs["initial_height"]
          else:
            height = None

          if "initial_rotation" in kwargs.keys():
            rotate = kwargs["initial_rotation"]
          else:
            rotate = None

          if "initial_joint_value" in kwargs.keys():
            random_joint_value = kwargs["initial_joint_value"]
          else:
            random_joint_value = None

          if "initial_linear_vel" in kwargs.keys():
            random_linear_vel = kwargs["initial_linear_vel"]
          else:
            random_linear_vel = None

          if "initial_angular_vel" in kwargs.keys():
            random_angular_vel = kwargs["initial_angular_vel"]
          else:
            random_angular_vel = None

          if "initial_height_reset_type" in kwargs.keys():
            initial_height_reset_type = kwargs["initial_height_reset_type"]
          else:
            initial_height_reset_type = None

          if "initial_action" in kwargs.keys():
            initial_action = kwargs["initial_action"]
          else:
            initial_action = self.get_random_joint_increment()

          if "is_rollout_shielding_reset" in kwargs.keys():
            is_rollout_shielding_reset = kwargs["is_rollout_shielding_reset"]
          else:
            is_rollout_shielding_reset = False

          if height is None:
            if self.height_reset == "both":
              height = np.random.choice([0.4 + np.random.rand() * 0.2, 0.6], p=[0.5, 0.5])
            elif self.height_reset == "drop":
              height = 0.4 + np.random.rand() * 0.2
            else:
              height = 0.6
          self.initial_height = height

          if rotate is None:
            if self.rotate_reset:  # Resets the yaw, pitch, roll.
              rotate = p.getQuaternionFromEuler((np.random.rand(3) - 0.5) * np.pi * 0.25)
            else:
              rotate = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
          self.initial_rotation = rotate

          if self.robot is None:
            super().reset(**kwargs)
            self.robot = Spirit(
                self.client, self.initial_height, self.initial_rotation, dim_x=self.dim_x,
                action_type=self.action_type, center=self.action_center, target_list=self.target_list,
                safety_list=self.safety_list, **kwargs
            )
          else:
            p.resetBasePositionAndOrientation(
                self.robot.id, np.array([0, 0, self.initial_height]), self.initial_rotation,
                physicsClientId=self.client
            )
          # if self.gui and self.force > 0:
          #   self.adversarial_object = Force(self.client)

          if not is_rollout_shielding_reset:
            if random_joint_value is None:
              random_joint_value = self.get_random_joint_value()
            self.initial_joint_value = random_joint_value

            if initial_height_reset_type is None:
              if self.height_reset == "both":
                initial_height_reset_type = np.random.choice(["drop", "stand"], p=[0.5, 0.5])
              else:
                initial_height_reset_type = self.height_reset
            self.initial_height_reset_type = initial_height_reset_type

            if self.initial_height_reset_type == "drop":
              # drop from the air
              self.robot.reset(random_joint_value)
              self.robot.apply_position(random_joint_value)
              p.setGravity(0, 0, self.gravity * 0.2, physicsClientId=self.client)
              for t in range(0, 100):
                p.stepSimulation(physicsClientId=self.client)
              p.setGravity(0, 0, self.gravity, physicsClientId=self.client)
            elif self.initial_height_reset_type == "stand":
              # standup from the ground
              self.robot.reset(np.zeros(12))
              self.robot.apply_position(np.zeros(12))
              p.stepSimulation(physicsClientId=self.client)
              traj = np.linspace(self.robot.get_joint_position(), self.initial_joint_value, 100)

              # lower the robot down to the ground before standing up
              p.setGravity(0, 0, self.gravity * 0.2, physicsClientId=self.client)
              for t in range(0, 100):
                p.stepSimulation(physicsClientId=self.client)
              p.setGravity(0, 0, self.gravity, physicsClientId=self.client)

              for i in range(100):
                # if self.reset_criterion == "reach-avoid":
                #   if min(self.robot.safety_margin().values()) > 0 and min(self.robot.target_margin().values()) < 0:
                #     break
                self.robot.apply_position(traj[i])
                p.stepSimulation(physicsClientId=self.client)

            # set random state (linear and angular velocity) to the robot
            if random_linear_vel is None:
              random_linear_vel = np.random.uniform(-0.1, 0.1, 3)
            self.initial_linear_vel = random_linear_vel

            if random_angular_vel is None:
              random_angular_vel = np.random.uniform(-0.15, 0.15, 3)
            self.initial_angular_vel = random_angular_vel

            p.resetBaseVelocity(
                self.robot.id, linearVelocity=random_linear_vel, angularVelocity=random_angular_vel,
                physicsClientId=self.client
            )

            # # set random initial action
            self.initial_action = initial_action
            self.robot.apply_action(self.initial_action)

          if self.command is not None:
            self.command = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-0.5, 0.5)]

          # self.state = np.array(self.robot.get_obs(), dtype=np.float32)
          base_state = self.robot.get_obs()
          if self.obsrv_list is not None:
            for o in self.obsrv_list:
              if o == "obs":
                self.obs_sequence = []
                for i in range(self.obs_sequence_length):
                  self.obs_sequence.append(list(self.robot.get_obs()))
                assert len(self.obs_sequence) == self.obs_sequence_length
                base_state = base_state + tuple(sum(self.obs_sequence, []))
              elif o == "prev_ctrl":
                base_state = base_state + tuple(initial_action)
              elif o == "command":
                base_state = base_state + tuple(self.command)
          self.state = np.array(base_state, dtype=np.float32)

          if is_rollout_shielding_reset:
            break

          if self.reset_criterion == "failure":  # avoidonly
            if min(self.robot.safety_margin().values()) > 0:
              break
          elif self.reset_criterion == "reach-avoid":  # outside of both failure set and target set
            if min(self.robot.safety_margin().values()) > 0 and min(self.robot.target_margin().values()) < 0:
              break
    else:
      if self.robot is None:
        super().reset(**kwargs)
        self.robot = Spirit(
            self.client, self.initial_height, self.initial_rotation, dim_x=self.dim_x, action_type=self.action_type,
            center=self.action_center, target_list=self.target_list, safety_list=self.safety_list, **kwargs
        )
      else:
        p.resetBasePositionAndOrientation(
            self.robot.id, np.array([0, 0, self.initial_height]), self.initial_rotation, physicsClientId=self.client
        )
      # if self.gui and self.force > 0:
      #   self.adversarial_object = Force(self.client)

      if self.initial_height_reset_type == "drop":
        # drop from the air
        self.robot.reset(self.initial_joint_value)
        self.robot.apply_position(self.initial_joint_value)
        p.setGravity(0, 0, self.gravity * 0.2, physicsClientId=self.client)
        for t in range(0, 100):
          p.stepSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.client)
      elif self.initial_height_reset_type == "stand":
        # standup from the ground
        self.robot.reset(np.zeros(12))
        self.robot.apply_position(np.zeros(12))
        p.stepSimulation(physicsClientId=self.client)
        traj = np.linspace(self.robot.get_joint_position(), self.initial_joint_value, 100)

        # lower the robot down to the ground before standing up
        p.setGravity(0, 0, self.gravity * 0.2, physicsClientId=self.client)
        for t in range(0, 100):
          p.stepSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.client)

        for i in range(100):
          if self.reset_criterion == "reach-avoid":
            if min(self.robot.safety_margin().values()) > 0 and min(self.robot.target_margin().values()) < 0:
              break
          self.robot.apply_position(traj[i])
          p.stepSimulation(physicsClientId=self.client)

      # set random state (linear and angular velocity) to the robot
      p.resetBaseVelocity(
          self.robot.id, linearVelocity=self.initial_linear_vel, angularVelocity=self.initial_angular_vel,
          physicsClientId=self.client
      )
      self.robot.apply_action(self.initial_action)
      # self.state = np.array(self.robot.get_obs(), dtype=np.float32)
      base_state = self.robot.get_obs()
      if self.obsrv_list is not None:
        for o in self.obsrv_list:
          if o == "obs":
            self.obs_sequence = []
            for i in range(self.obs_sequence_length):
              self.obs_sequence.append(list(self.robot.get_obs()))
            assert len(self.obs_sequence) == self.obs_sequence_length
            base_state = base_state + tuple(sum(self.obs_sequence, []))
          if o == "prev_ctrl":
            base_state = base_state + tuple(initial_action)
          elif o == "command":
            base_state = base_state + tuple(self.command)
      self.state = np.array(base_state, dtype=np.float32)

    # use self.reset_count to check if it's time to use the force sequence provided
    if self.synthetic_symmetrical_dstb:
      self.reset_count = (self.reset_count + 1) % 2
    else:
      self.reset_count = 0  # always enforce 0 for reset_count so that we go into the correct reset sequence

  def get_constraints(self):
    return self.robot.safety_margin()

  def get_target_margin(self):
    return self.robot.target_margin()

  def get_random_joint_value(self):
    if self.action_type == "increment":
      return (
          np.random.uniform(self.abduction_min, self.abduction_max), 1.0 + np.random.uniform(-0.5, 0.5),
          1.9 + np.random.uniform(-0.5, 0.5), np.random.uniform(self.abduction_min, self.abduction_max),
          1.0 + np.random.uniform(-0.5, 0.5), 1.9 + np.random.uniform(-0.5, 0.5),
          np.random.uniform(self.abduction_min, self.abduction_max), 1.0 + np.random.uniform(-0.5, 0.5),
          1.9 + np.random.uniform(-0.5, 0.5), np.random.uniform(self.abduction_min, self.abduction_max),
          1.0 + np.random.uniform(-0.5, 0.5), 1.9 + np.random.uniform(-0.5, 0.5)
      )
    elif self.action_type == "center_sampling":
      return np.array(self.action_center) + np.array(self.get_random_joint_increment())

  def get_random_joint_increment_from_current(self):
    return np.array(self.robot.get_joint_position()) + np.array(self.get_random_joint_increment())

  def get_random_joint_increment(self):
    increment = (
        np.random.uniform(self.abduction_increment_min, self.abduction_increment_max),
        np.random.uniform(self.hip_increment_min,
                          self.hip_increment_max), np.random.uniform(self.knee_increment_min, self.knee_increment_max),
        np.random.uniform(self.abduction_increment_min, self.abduction_increment_max),
        np.random.uniform(self.hip_increment_min,
                          self.hip_increment_max), np.random.uniform(self.knee_increment_min, self.knee_increment_max),
        np.random.uniform(self.abduction_increment_min, self.abduction_increment_max),
        np.random.uniform(self.hip_increment_min,
                          self.hip_increment_max), np.random.uniform(self.knee_increment_min, self.knee_increment_max),
        np.random.uniform(self.abduction_increment_min, self.abduction_increment_max),
        np.random.uniform(self.hip_increment_min,
                          self.hip_increment_max), np.random.uniform(self.knee_increment_min, self.knee_increment_max)
    )

    return increment

  def integrate_forward(
      self, state: np.ndarray, control: np.ndarray, num_segment: Optional[int] = 1, noise: Optional[np.ndarray] = None,
      noise_type: Optional[str] = 'unif', adversary: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
        Use the Pybullet physics simulation engine to do 1-step integrate_forward.

        Args:
            state (np.ndarray): Dummy data, the system state will be maintained by Pybullet instead of passing in.
            control (np.ndarray): _description_
            num_segment (Optional[int], optional): _description_. Defaults to 1.
            noise (Optional[np.ndarray], optional): _description_. Defaults to None.
            noise_type (Optional[str], optional): _description_. Defaults to 'unif'.
            adversary (Optional[np.ndarray], optional): The adversarial action, this is the force vector, force applied position, and terrain information. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: _description_
        """

    # pop the oldest obs, add the current obs before applying action
    if len(self.obs_sequence) > 0:
      self.obs_sequence.pop()
      self.obs_sequence.insert(0, list(self.robot.get_obs()))
      assert len(self.obs_sequence) == self.obs_sequence_length

    if adversary is not None:
      has_adversarial = True
    else:
      has_adversarial = False

    # get the current state of the robot
    spirit_cur_joint_pos = np.array(self.robot.get_joint_position(), dtype=np.float32)

    # check clipped control
    clipped_control = []
    # TODO: The current control is a nested array, that is the result of running multiprocessing. Fix this to work with Pybullet
    for i, j in enumerate(control):
      if i % 3 == 0:
        increment = np.clip(j, self.abduction_increment_min, self.abduction_increment_max)
        if self.abduction_min <= spirit_cur_joint_pos[i] + increment <= self.abduction_max:
          clipped_control.append(increment)
        else:
          clipped_control.append(
              np.clip(spirit_cur_joint_pos[i] + increment, self.abduction_min, self.abduction_max)
              - spirit_cur_joint_pos[i]
          )
      elif i % 3 == 1:
        increment = np.clip(j, self.hip_increment_min, self.hip_increment_max)
        if self.hip_min <= spirit_cur_joint_pos[i] + increment <= self.hip_max:
          clipped_control.append(increment)
        else:
          clipped_control.append(
              np.clip(spirit_cur_joint_pos[i] + increment, self.hip_min, self.hip_max) - spirit_cur_joint_pos[i]
          )
      elif i % 3 == 2:
        increment = np.clip(j, self.knee_increment_min, self.knee_increment_max)
        if self.knee_min <= spirit_cur_joint_pos[i] + increment <= self.knee_max:
          clipped_control.append(increment)
        else:
          clipped_control.append(
              np.clip(spirit_cur_joint_pos[i] + increment, self.knee_min, self.knee_max) - spirit_cur_joint_pos[i]
          )

    # TODO: check clipped adversarial control
    self.robot.apply_action(clipped_control)
    if has_adversarial and not self.replace_adv_with_dr:
      if self.reset_count % 2 == 1 and self.synthetic_symmetrical_dstb:
        if len(self.dstb_array) > 0:
          adversary = self.dstb_array.pop(0)
          adversary[1] = -adversary[1]  # mirror y-axis
      force_vector = adversary[0:3]
      position_vector = adversary[3:]
      self._apply_adversarial_force(force_vector=force_vector, position_vector=position_vector)
    else:
      self._apply_force()

    p.stepSimulation(physicsClientId=self.client)

    if self.gui:
      if has_adversarial and not self.replace_adv_with_dr:
        if self.adv_debug_line_id is not None:
          p.removeUserDebugItem(self.adv_debug_line_id)
        if self.link_name is not None:
          self.adv_debug_line_id = p.addUserDebugLine(
              self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector,
              lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id,
              parentLinkIndex=self.robot.get_link_id(self.link_name)
          )
        else:
          self.adv_debug_line_id = p.addUserDebugLine(
              self.force_applied_position_vector, self.force_applied_position_vector + self.force_applied_force_vector,
              lineColorRGB=[0, 0, 1], lineWidth=2.0, physicsClientId=self.client, parentObjectUniqueId=self.robot.id
          )

        # just for visualization purpose, make the arrow always start slightly on top of the robot
        # if self.adversarial_object is not None:
        #   pos, ang = p.getBasePositionAndOrientation(self.robot.id)
        #   self.adversarial_object.resetAB(
        #       pos + np.array([0, 0, 0.05]), pos + np.array([0, 0, 0.05]) + self.force_applied_force_vector
        #   )
      time.sleep(self.dt)

      if self.video_output_file is not None:
        self._save_frames()

      self.debugger.cam_and_robotstates(self.robot.id)
    elif self.gui_imaginary:
      self.render()

    # self.state = np.array(self.robot.get_obs(), dtype=np.float32)
    base_state = self.robot.get_obs()
    if self.obsrv_list is not None:
      for o in self.obsrv_list:
        if o == "obs":
          assert len(self.obs_sequence) == self.obs_sequence_length
          base_state = base_state + tuple(sum(self.obs_sequence, []))
        elif o == "prev_ctrl":
          base_state = base_state + tuple(clipped_control)
        elif o == "command":
          base_state = base_state + tuple(self.command)
    self.state = np.array(base_state, dtype=np.float32)
    self.cnt += 1

    if has_adversarial:
      if self.replace_adv_with_dr:
        if self.force != 0:
          adversary = np.concatenate(
              (self.force_applied_force_vector / self.force, self.force_applied_position_vector), axis=0
          )
          if self.reset_count % 2 == 0 and self.synthetic_symmetrical_dstb:
            self.dstb_array.append(adversary)
          return self.state, clipped_control, adversary
        else:
          adversary = np.concatenate((np.zeros(3), self.force_applied_position_vector), axis=0)
          if self.reset_count % 2 == 0 and self.synthetic_symmetrical_dstb:
            self.dstb_array.append(adversary)
          return self.state, clipped_control, adversary
      else:
        if self.reset_count % 2 == 0 and self.synthetic_symmetrical_dstb:
          self.dstb_array.append(adversary)
        return self.state, clipped_control, adversary
    else:
      return self.state, clipped_control

  def render(self):
    if self.rendered_img is None:
      self.rendered_img = plt.imshow(np.zeros((200, 200, 4)))

    # Base information
    robot_id, client_id = self.robot.get_ids()
    proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1, nearVal=0.01, farVal=100, physicsClientId=self.client)
    pos, ori = [list(l) for l in p.getBasePositionAndOrientation(robot_id, client_id)]

    pos[0] += 1.0
    pos[1] -= 1.0
    pos[2] += 0.7
    ori = p.getQuaternionFromEuler([0, 0.2, np.pi * 0.8])

    # Rotate camera direction
    rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
    camera_vec = np.matmul(rot_mat, [1, 0, 0])
    up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
    view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

    # Display image
    frame = p.getCameraImage(200, 200, view_matrix, proj_matrix, physicsClientId=self.client)[2]
    frame = np.reshape(frame, (200, 200, 4))
    self.rendered_img.set_data(frame)
    plt.draw()
    plt.axis('off')
    plt.title("Rollout imagine env")
    plt.pause(.00001)

  def integrate_forward_jax(self, state: DeviceArray, control: DeviceArray) -> Tuple[DeviceArray, DeviceArray]:
    return super().integrate_forward_jax(state, control)

  def _integrate_forward(self, state: DeviceArray, control: DeviceArray) -> DeviceArray:
    return super()._integrate_forward(state, control)

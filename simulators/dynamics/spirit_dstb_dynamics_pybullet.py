import numpy as np
import pybullet as p
from .base_pybullet_dynamics import BasePybulletDynamics
from typing import Optional, Tuple, Any
from .resources.spirit import Spirit
import time
import matplotlib.pyplot as plt
from jaxlib.xla_extension import DeviceArray
import torch
import copy


class SpiritDstbDynamicsPybullet(BasePybulletDynamics):

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    if isinstance(action_space, dict):
      super().__init__(config, action_space["dstb"])
      self.dim_u = len(action_space["dstb"])
    else:
      super().__init__(config, action_space)
      self.dim_u = len(action_space)

    # self.dim_x = 45 # 33 + 12 control signal from performance controller
    if isinstance(config.obs_dim, dict):
      # self.dim_x = config.OBS_DIM["actor_0"] + 12 #! OR config.OBS_DIM["action_1"]?
      raise NotImplementedError
    else:
      self.dim_x = config.obs_dim

    self.abduction_min = -0.5
    self.abduction_max = 0.5
    self.hip_min = 0.5
    self.hip_max = 2.64
    self.knee_min = 0.5
    self.knee_max = 2.64

    self.initial_height = None
    self.initial_rotation = None
    self.initial_joint_value = None

    self.rendered_img = None
    self.adv_debug_line_id = None
    self.shielding_status_debug_text_id = None

    self.performance_controller = None
    self.safety_policy = None
    self.ctrl_type = None  # performance, safety, shield_value, shield_rollout
    self.epsilon = None
    self.critic = None
    self.dstb_policy = None  # used for value shielding
    self.gameplay_solver = None
    self.env_gameplay = None
    self.gameplay_horizon = None

    self.initial_linear_vel = None
    self.initial_angular_vel = None
    self.initial_height_reset_type = None
    self.initial_action = None

    self.target_list = config.target_margin
    self.safety_list = config.safety_margin

    # self.reset()

  def reset(self, **kwargs):
    # rejection sampling until outside target set and safe set
    while True:
      super().reset(**kwargs)

      if "performance" in kwargs.keys():
        self.performance_controller = kwargs["performance"]

      if "safety" in kwargs.keys():
        self.safety_policy = kwargs["safety"]

      if "dstb" in kwargs.keys():
        self.dstb_policy = kwargs["dstb"]

      if "ctrl_type" in kwargs.keys():
        # performance, safety, shield_value, shield_rollout
        self.ctrl_type = kwargs["ctrl_type"]

      if "epsilon" in kwargs.keys():
        self.epsilon = kwargs["epsilon"]

      if "gameplay_solver" in kwargs.keys():
        self.gameplay_solver = kwargs["gameplay_solver"]

      if "env_gameplay" in kwargs.keys():
        self.env_gameplay = kwargs["env_gameplay"]

      if "gameplay_horizon" in kwargs.keys():
        self.gameplay_horizon = kwargs["gameplay_horizon"]

      if "critic" in kwargs.keys():
        self.critic = kwargs["critic"]

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
          rotate = p.getQuaternionFromEuler((np.random.rand(3) - 0.5) * np.pi * 0.125)
        else:
          rotate = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
      self.initial_rotation = rotate

      # self.robot = Spirit(self.client, height, rotate, dim_x = self.dim_x-12, **kwargs)
      self.robot = Spirit(
          self.client, height, rotate, dim_x=self.dim_x, target_list=self.target_list, safety_list=self.safety_list,
          **kwargs
      )

      if not is_rollout_shielding_reset:
        if random_joint_value is None:
          if self.ctrl_type == "performance" or self.ctrl_type == "shield_value" or self.ctrl_type == "shield_rollout":
            if self.performance_controller is not None:
              random_joint_value = self.performance_controller.get_action()
            else:
              random_joint_value = self.get_random_joint_value()
          elif self.ctrl_type == "safety":
            # there's no state yet, so just get a randomized joint value
            random_joint_value = self.get_random_joint_value()
          else:
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
          traj = np.linspace(self.robot.get_joint_position(), self.initial_joint_value, 100)
          for i in range(100):
            if self.reset_criterion == "reach-avoid":
              if min(self.robot.safety_margin().values()) > 0 and min(self.robot.target_margin().values()) < 0:
                break
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

      # self.state = np.concatenate((np.array(self.robot.get_obs(), dtype = np.float32),np.zeros(12)), axis=0)
      self.state = np.array(self.robot.get_obs(), dtype=np.float32)

      if is_rollout_shielding_reset:
        break

      if self.reset_criterion == "failure":  # avoidonly
        if min(self.robot.safety_margin().values()) > 0:
          break
      elif self.reset_criterion == "reach-avoid":  # outside of both failure set and target set
        if min(self.robot.safety_margin().values()) > 0 and min(self.robot.target_margin().values()) < 0:
          break

  def get_constraints(self):
    return self.robot.safety_margin()

  def get_target_margin(self):
    return self.robot.target_margin()

  def get_random_joint_value(self):
    return (
        np.random.uniform(self.abduction_min,
                          self.abduction_max), 1.0 + np.random.uniform(-0.5, 0.5), 1.9 + np.random.uniform(-0.5, 0.5),
        np.random.uniform(self.abduction_min,
                          self.abduction_max), 1.0 + np.random.uniform(-0.5, 0.5), 1.9 + np.random.uniform(-0.5, 0.5),
        np.random.uniform(self.abduction_min,
                          self.abduction_max), 1.0 + np.random.uniform(-0.5, 0.5), 1.9 + np.random.uniform(-0.5, 0.5),
        np.random.uniform(self.abduction_min,
                          self.abduction_max), 1.0 + np.random.uniform(-0.5, 0.5), 1.9 + np.random.uniform(-0.5, 0.5)
    )

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
    # get the current state of the robot
    spirit_cur_joint_pos = np.array(self.robot.get_joint_position(), dtype=np.float32)
    if self.ctrl_type == "performance":
      if self.performance_controller is not None:
        robot_control = self.performance_controller.get_action() - spirit_cur_joint_pos
      else:
        raise NotImplementedError
    elif self.ctrl_type == "safety":
      if self.safety_policy is not None:
        #! TODO: READ DIMENSION FROM CONFIG OF SAFETY POLICY
        safety_control = self.safety_policy(self.state[:32])
        # check clipped control
        robot_control = []
        # TODO: The current control is a nested array, that is the result of running multiprocessing. Fix this to work with Pybullet
        for i, j in enumerate(safety_control):
          if i % 3 == 0:
            increment = np.clip(j, -0.5, 0.5)
            if self.abduction_min <= spirit_cur_joint_pos[i] + increment <= self.abduction_max:
              robot_control.append(increment)
            else:
              robot_control.append(
                  np.clip(spirit_cur_joint_pos[i] + increment, self.abduction_min, self.abduction_max)
                  - spirit_cur_joint_pos[i]
              )
          elif i % 3 == 1:
            increment = np.clip(j, -0.5, 0.5)
            if self.hip_min <= spirit_cur_joint_pos[i] + increment <= self.hip_max:
              robot_control.append(increment)
            else:
              robot_control.append(
                  np.clip(spirit_cur_joint_pos[i] + increment, self.hip_min, self.hip_max) - spirit_cur_joint_pos[i]
              )
          elif i % 3 == 2:
            increment = np.clip(j, -0.5, 0.5)
            if self.knee_min <= spirit_cur_joint_pos[i] + increment <= self.knee_max:
              robot_control.append(increment)
            else:
              robot_control.append(
                  np.clip(spirit_cur_joint_pos[i] + increment, self.knee_min, self.knee_max) - spirit_cur_joint_pos[i]
              )
      else:
        raise NotImplementedError
    elif self.ctrl_type == "shield_value":
      if self.epsilon is None:
        raise NotImplementedError
      if self.critic is None:
        raise NotImplementedError
      if self.performance_controller is None:
        raise NotImplementedError
      if self.safety_policy is None:
        raise NotImplementedError

      # DO VALUE SHIELDING
      robot_control = self.performance_controller.get_action() - spirit_cur_joint_pos
      if self.dstb_policy is None:
        critic_q = max(self.critic(self.state[:32], robot_control))
      else:
        if self.dstb_policy.obsrv_dim == 32:
          critic_q = max(
              self.critic(self.state[:32], robot_control, torch.FloatTensor(self.dstb_policy(self.state[:32])))
          )
        elif self.dstb_policy.obsrv_dim == 44:
          critic_q = max(
              self.critic(
                  self.state[:32], robot_control, torch.FloatTensor(self.dstb_policy(self.state[:32], robot_control))
              )
          )

      if critic_q < self.epsilon:
        # check to see if we are in target set
        if min(self.get_target_margin().values()) > 0:
          stable_stance = np.array([0.1, 0.8, 1.4, 0.4, 0.7, 1.9, -0.1, 0.8, 1.4, -0.4, 0.7, 1.9])
          safety_control = stable_stance - spirit_cur_joint_pos
        else:
          safety_control = self.safety_policy(self.state[:32])
        # check clipped control
        robot_control = []
        # TODO: The current control is a nested array, that is the result of running multiprocessing. Fix this to work with Pybullet
        for i, j in enumerate(safety_control):
          if i % 3 == 0:
            increment = np.clip(j, -0.5, 0.5)
            if self.abduction_min <= spirit_cur_joint_pos[i] + increment <= self.abduction_max:
              robot_control.append(increment)
            else:
              robot_control.append(
                  np.clip(spirit_cur_joint_pos[i] + increment, self.abduction_min, self.abduction_max)
                  - spirit_cur_joint_pos[i]
              )
          elif i % 3 == 1:
            increment = np.clip(j, -0.5, 0.5)
            if self.hip_min <= spirit_cur_joint_pos[i] + increment <= self.hip_max:
              robot_control.append(increment)
            else:
              robot_control.append(
                  np.clip(spirit_cur_joint_pos[i] + increment, self.hip_min, self.hip_max) - spirit_cur_joint_pos[i]
              )
          elif i % 3 == 2:
            increment = np.clip(j, -0.5, 0.5)
            if self.knee_min <= spirit_cur_joint_pos[i] + increment <= self.knee_max:
              robot_control.append(increment)
            else:
              robot_control.append(
                  np.clip(spirit_cur_joint_pos[i] + increment, self.knee_min, self.knee_max) - spirit_cur_joint_pos[i]
              )
    elif self.ctrl_type == "shield_rollout":
      if self.gameplay_solver is None:
        raise NotImplementedError
      if self.performance_controller is None:
        raise NotImplementedError
      if self.env_gameplay is None:
        raise NotImplementedError

      # DO ROLLOUT SHIELDING
      # reset imaginary_solver to the current state
      self.env_gameplay.agent.dyn.robot.type = "imaginary"
      robot_control = self.performance_controller.get_action() - spirit_cur_joint_pos
      gameplay_ctrl = torch.FloatTensor(copy.copy(robot_control)).to(self.gameplay_solver.device)
      gameplay_state = self.env_gameplay.reset(cast_torch=True, initial_state=np.concatenate(([self.robot.linc_get_pos()[0][2]], self.state[:32])))[:32]
      if self.gameplay_solver.cfg_arch.actor_1.obsrv_dim == 32:
        gameplay_s_dstb = gameplay_state
      elif self.gameplay_solver.cfg_arch.actor_1.obsrv_dim == 44:
        gameplay_s_dstb = torch.cat((gameplay_state, gameplay_ctrl), dim=-1)
      gameplay_dstb = self.gameplay_solver.dstb.net(gameplay_s_dstb.float().to(self.gameplay_solver.device))
      gameplay_action = {'ctrl': gameplay_ctrl.cpu().detach().numpy(), 'dstb': gameplay_dstb.cpu().detach().numpy()}
      gameplay_state, r, done, info = self.env_gameplay.step(gameplay_action, cast_torch=True)
      gameplay_state = gameplay_state.to(self.gameplay_solver.device)

      if info["g_x"] > 0:
        gameplay_failed = False
      else:
        gameplay_failed = True

      if not gameplay_failed:
        for i in range(self.gameplay_horizon - 1):
          gameplay_ctrl = self.gameplay_solver.ctrl.net(gameplay_state.float().to(self.gameplay_solver.device))
          if self.gameplay_solver.cfg_arch.actor_1.obsrv_dim == 32:
            gameplay_s_dstb = gameplay_state
          elif self.gameplay_solver.cfg_arch.actor_1.obsrv_dim == 44:
            gameplay_s_dstb = torch.cat((gameplay_state, gameplay_ctrl), dim=-1)
          gameplay_dstb = self.gameplay_solver.dstb.net(gameplay_s_dstb.float().to(self.gameplay_solver.device))
          gameplay_action = {
              'ctrl': gameplay_ctrl.cpu().detach().numpy(),
              'dstb': gameplay_dstb.cpu().detach().numpy()
          }
          gameplay_state, r, done, info = self.env_gameplay.step(gameplay_action, cast_torch=True)
          gameplay_state = gameplay_state.to(self.gameplay_solver.device)

          if self.gameplay_solver.cfg_solver.rollout_end_criterion == "failure":
            if info["g_x"] <= 0:
              gameplay_failed = True
              break
          elif self.gameplay_solver.cfg_solver.rollout_end_criterion == "reach-avoid":
            if info["g_x"] <= 0:
              gameplay_failed = True
              break
            elif info["l_x"] > 0:
              break
          # self.env_gameplay.agent.dyn.render()

      if self.gameplay_solver.cfg_solver.rollout_end_criterion == "reach-avoid":
        if info["l_x"] <= 0:
          gameplay_failed = True

      if gameplay_failed:
        if min(self.get_target_margin().values()) > 0:
          stable_stance = np.array([0.1, 0.8, 1.4, 0.4, 0.7, 1.9, -0.1, 0.8, 1.4, -0.4, 0.7, 1.9])
          safety_control = stable_stance - spirit_cur_joint_pos
          # safety_control = np.clip(stable_stance - spirit_cur_joint_pos, -np.ones(12) * 0.1, np.ones(12) * 0.1)
        else:
          safety_control = self.gameplay_solver.ctrl.net(self.state[:32])
        # check clipped control
        robot_control = []
        # TODO: The current control is a nested array, that is the result of running multiprocessing. Fix this to work with Pybullet
        for i, j in enumerate(safety_control):
          if i % 3 == 0:
            increment = np.clip(j, -0.5, 0.5)
            if self.abduction_min <= spirit_cur_joint_pos[i] + increment <= self.abduction_max:
              robot_control.append(increment)
            else:
              robot_control.append(
                  np.clip(spirit_cur_joint_pos[i] + increment, self.abduction_min, self.abduction_max)
                  - spirit_cur_joint_pos[i]
              )
          elif i % 3 == 1:
            increment = np.clip(j, -0.5, 0.5)
            if self.hip_min <= spirit_cur_joint_pos[i] + increment <= self.hip_max:
              robot_control.append(increment)
            else:
              robot_control.append(
                  np.clip(spirit_cur_joint_pos[i] + increment, self.hip_min, self.hip_max) - spirit_cur_joint_pos[i]
              )
          elif i % 3 == 2:
            increment = np.clip(j, -0.5, 0.5)
            if self.knee_min <= spirit_cur_joint_pos[i] + increment <= self.knee_max:
              robot_control.append(increment)
            else:
              robot_control.append(
                  np.clip(spirit_cur_joint_pos[i] + increment, self.knee_min, self.knee_max) - spirit_cur_joint_pos[i]
              )

    self.robot.apply_action(robot_control)

    # force_vector = control[0:3]
    # position_vector = control[3:]
    if not self.replace_adv_with_dr:
      force_vector = adversary[0:3]
      position_vector = adversary[3:]
      self._apply_adversarial_force(force_vector=force_vector, position_vector=position_vector)
    else:
      self._apply_force()

    p.stepSimulation(physicsClientId=self.client)

    if self.gui:
      if self.adv_debug_line_id is not None:
        p.removeUserDebugItem(self.adv_debug_line_id, physicsClientId=self.client)
      if self.link_name is not None:
        self.adv_debug_line_id = p.addUserDebugLine(
            self.force_applied_position_vector,
            self.force_applied_position_vector + self.force_applied_force_vector * self.force, lineColorRGB=[0, 0, 1],
            lineWidth=2.0, lifeTime=0.1, physicsClientId=self.client, parentObjectUniqueId=self.robot.id,
            parentLinkIndex=self.robot.get_link_id(self.link_name)
        )
      else:
        self.adv_debug_line_id = p.addUserDebugLine(
            self.force_applied_position_vector,
            self.force_applied_position_vector + self.force_applied_force_vector * self.force, lineColorRGB=[0, 0, 1],
            lineWidth=2.0, lifeTime=0.1, physicsClientId=self.client, parentObjectUniqueId=self.robot.id
        )
      time.sleep(self.dt)

      if self.video_output_file is not None:
        self._save_frames()

      self.debugger.cam_and_robotstates(self.robot.id)
    elif self.gui_imaginary:
      self.render()

    # self.state = np.concatenate((np.array(self.robot.get_obs(), dtype = np.float32), np.array(robot_control, dtype = np.float32)), axis=0)
    self.state = np.array(self.robot.get_obs(), dtype=np.float32)

    self.cnt += 1

    # return self.state, control
    if self.force != 0:
      adversary = np.concatenate((self.force_applied_force_vector / self.force, self.force_applied_position_vector),
                                 axis=0)
    else:
      adversary = np.concatenate((np.zeros(3), self.force_applied_position_vector), axis=0)

    return self.state, robot_control, adversary

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

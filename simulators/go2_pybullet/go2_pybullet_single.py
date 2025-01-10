# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Dict, Tuple, Any, Optional, Union
from gym import spaces
import numpy as np
import torch
import matplotlib

from ..base_single_env import BaseSingleEnv


class Go2PybulletSingleEnv(BaseSingleEnv):

  def __init__(self, cfg_env, cfg_agent, cfg_cost) -> None:
    super().__init__(cfg_env, cfg_agent)
    self.obsrv_dim = cfg_agent.obs_dim
    # dummy observation space for Spirit
    self.observation_space = spaces.Box(low=np.zeros(self.obsrv_dim,), high=np.zeros(self.obsrv_dim,))
    self.seed(cfg_env.seed)

  def seed(self, seed: int = 0):
    super().seed(seed)

  def reset(self, state: Optional[np.ndarray] = None, cast_torch: bool = False,
            **kwargs) -> Union[np.ndarray, torch.FloatTensor]:
    BaseSingleEnv.reset(self, state, cast_torch, **kwargs)
    self.agent.dyn.reset(**kwargs)
    obs = self.get_obsrv(None)

    self.state = obs.copy()

    if cast_torch:
      obs = torch.FloatTensor(obs)
    return obs

  def get_cost(
      self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray, constraints: Optional[dict] = None
  ) -> float:
    #TODO: need to have a way to encode this in config file
    if len(state) == 47:
      # if True:
      vel = state[:3]
      roll = state[3]
      pitch = state[4]
      ang_vel = state[5:8]
      jpos = state[8:20]
      jvel = state[20:32]
      prev_action = state[32:44]
      command = state[44:47]
      state_sequence = state[47:]

      v_x_bar = command[0]
      v_y_bar = command[1]
      ang_z_bar = command[2]

      # walk in the park
      # if vel[0] <= 2.0 * vel_target and vel[0] >= vel_target:
      #   rv = 1.0
      # elif vel[0] < -vel_target or vel[0] > 4.0 * vel_target:
      #   rv = 0.0
      # else:
      #   rv = 1.0 - (abs(vel[0] - vel_target) / 2.0 * vel_target)
      # cost = 0.1 * ang_vel[2]**2 - rv

      # barkour Google
      cost = 0
      # track linear vel
      cost += -1.5 * np.exp(-4 * np.square(vel[0] - v_x_bar) - 4 * np.square(vel[1] - v_y_bar))
      # track angular vel
      cost += -0.8 * np.exp(-4 * np.square(ang_z_bar - ang_vel[2]))
      # penalize z vel
      cost += 2.0 * np.square(vel[2])
      # penalize wx, wy
      cost += 0.05 * np.sum(np.square(ang_vel[:2]))
      # penalize roll, pitch
      cost += 5.0 * (roll**2 + pitch**2)
      # action delta
      cost += 0.01 * np.sum(np.square(action - prev_action))
      # action magnitude
      # cost += 0.02*np.sum(np.square(action))
      # give reward for feet airtime
      foot_contact_z = np.array(self.agent.dyn.robot.get_toes())
      contact = foot_contact_z < 0.01
      contact_filter = contact | self.agent.dyn.robot.last_contact
      first_contact = (self.agent.dyn.robot.feet_air_time > 0) * contact_filter
      self.agent.dyn.robot.feet_air_time += self.agent.dyn.dt
      rew_air_time = np.sum((self.agent.dyn.robot.feet_air_time - 0.1) * first_contact)
      # cost += -0.2*sum(self.agent.dyn.robot.get_toes())
      cost += -0.2 * rew_air_time
      # torque
      cost += 0.0002 * np.sum(np.square(self.agent.dyn.robot.get_joint_torque()))

      # state management for feet airtime
      self.agent.dyn.robot.feet_air_time *= ~contact_filter
      self.agent.dyn.robot.last_contact = contact
    else:
      cost = 0

    return cost

  def get_constraints(self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray) -> Dict:
    return self.agent.dyn.get_constraints()

  def get_constraints_all(self, states: np.ndarray, actions: np.ndarray) -> Dict:
    return self.agent.dyn.get_constraints()

  def get_target_margin(self, state: np.ndarray, action: np.ndarray, state_nxt: np.ndarray) -> Dict:
    return self.agent.dyn.get_target_margin()

  def get_done_and_info(
      self, state: np.ndarray, constraints: Dict, targets: Dict, final_only: bool = True,
      end_criterion: Optional[str] = None
  ) -> Tuple[bool, Dict]:
    if end_criterion is None:
      end_criterion = self.end_criterion

    done = False
    done_type = "not_raised"
    if self.cnt >= self.timeout:
      done = True
      done_type = "timeout"

    g_x = min(list(constraints.values()))
    l_x = min(list(targets.values()))
    binary_cost = 1. if g_x < 0. else 0.

    # Gets done flag
    #! TODO: override the g_x (if failure) and g_x and l_x (if reach-avoid) with a large value (e.g.: 0.1-0.2, empirical choose by sweeping/looking at largest value)
    if end_criterion == 'failure':
      failure = g_x < 0
      if failure:
        done = True
        done_type = "failure"
        g_x = -0.2
    elif end_criterion == 'reach-avoid':
      failure = g_x < 0.
      success = not failure and l_x >= 0.

      if success:
        done = True
        done_type = "success"
      elif failure:
        done = True
        done_type = "failure"
        g_x = -0.2
    elif end_criterion == 'timeout':
      pass
    else:
      raise ValueError("End criterion not supported!")

    # Gets info
    info = {"done_type": done_type, "g_x": g_x, "l_x": l_x, "binary_cost": binary_cost}

    return done, info

  def get_obsrv(self, state: np.ndarray) -> np.ndarray:
    return self.agent.dyn.state

  def render(self):
    return super().render()

  def report(self):
    print("Spirit Pybullet initialized")

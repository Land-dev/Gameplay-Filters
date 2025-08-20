# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from typing import Dict, Tuple, Any, Optional, Union
import numpy as np
import torch
from gym import spaces

from ..base_zs_env import BaseZeroSumEnv
from ..utils import ActionZS


class SpiritPybulletZeroSumEnv(BaseZeroSumEnv):

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
    BaseZeroSumEnv.reset(self, state, cast_torch, **kwargs)
    self.agent.dyn.reset(**kwargs)
    obs = self.get_obsrv(None)

    self.state = obs.copy()

    if cast_torch:
      obs = torch.FloatTensor(obs)
    return obs

  def get_cost(
      self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray, constraints: Optional[Dict] = None
  ) -> float:
    return 0

  def get_constraints(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
    return self.agent.dyn.get_constraints()

  def get_constraints_all(self, states: np.ndarray, actions: np.ndarray) -> Dict:
    return self.agent.dyn.get_constraints()

  def get_target_margin(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
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
    if end_criterion == 'failure':
      failure = g_x < 0
      if failure:
        done = True
        done_type = "failure"
        # g_x = -0.2
    elif end_criterion == 'reach-avoid':
      failure = g_x < 0.
      success = not failure and l_x >= 0.

      if success:
        done = True
        done_type = "success"
      elif failure:
        done = True
        done_type = "failure"
        # g_x = -0.2
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

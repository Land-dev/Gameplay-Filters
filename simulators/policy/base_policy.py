# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A parent class for (control) policy.

This file implements a parent class for (control) policy. A child class should
implement `get_action()`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
import torch


class BasePolicy(ABC):
  obsrv_list: Optional[List]

  @property
  @abstractmethod
  def is_stochastic(self) -> bool:
    raise NotImplementedError

  def __init__(self, id: str, obsrv_list: Optional[List] = None, **kwargs) -> None:
    super().__init__()
    self.id = id
    self.obsrv_list = obsrv_list

  @abstractmethod
  def get_action(
      self, obsrv: Union[np.ndarray, torch.Tensor], agents_action: Optional[Dict[str, np.ndarray]] = None, **kwargs
  ) -> Tuple[np.ndarray, dict]:
    """Gets the action to execute.

    Args:
        obsrv (np.ndarray): current observation.
        agents_action (Optional[Dict]): other agents' actions that are
            observable to the ego agent.

    Returns:
        np.ndarray: the action to be executed.
        dict: info for the solver, e.g., processing time, status, etc.
    """
    raise NotImplementedError

  @staticmethod
  def combine_actions(obsrv_list: List, agents_action: Dict[str, np.ndarray]) -> np.ndarray:
    """Combines the observed other agents' actions.

    Args:
        obsrv_list (List): list of observed agents' names.
        agents_action (Dict): other agents' actions.

    Returns:
        np.ndarray: the combined observation.
    """
    if len(obsrv_list) == 1:
      return agents_action[obsrv_list[0]]
    else:
      return np.concatenate([agents_action[name] for name in obsrv_list], axis=-1)

  def report(self):
    print(self.id)
    if self.obsrv_list is not None:
      print("  - The policy can observe:", end=' ')
      for i, k in enumerate(self.obsrv_list):
        print(k, end='')
        if i == len(self.obsrv_list) - 1:
          print('.')
        else:
          print(', ', end='')
    else:
      print("  - The policy can only access observation.")

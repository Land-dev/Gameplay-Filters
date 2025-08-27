# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import sys
import copy
import numpy as np
import argparse
from functools import partial
from shutil import copyfile
from omegaconf import OmegaConf
from agent import ISAACS
from utils.eval import evaluate_zero_sum
from simulators import PrintLogger, save_obj


def main(config_file):
  # Loads config.
  cfg = OmegaConf.load(config_file)

  os.makedirs(cfg.solver.out_folder, exist_ok=True)
  copyfile(config_file, os.path.join(cfg.solver.out_folder, 'config.yaml'))
  log_path = os.path.join(cfg.solver.out_folder, 'log.txt')
  if os.path.exists(log_path):
    os.remove(log_path)
  sys.stdout = PrintLogger(log_path)
  sys.stderr = PrintLogger(log_path)

  if cfg.solver.use_wandb:
    import wandb
    wandb.init(entity='saslab', project=cfg.solver.project_name, name=cfg.solver.name)
    tmp_cfg = {
        'environment': OmegaConf.to_container(cfg.environment),
        'solver': OmegaConf.to_container(cfg.solver),
        'arch': OmegaConf.to_container(cfg.arch),
    }
    wandb.config.update(tmp_cfg)

  if cfg.agent.dyn == "SpiritPybullet":
    from simulators import SpiritPybulletZeroSumEnv
    env_class = SpiritPybulletZeroSumEnv
    cfg.cost = None
  elif cfg.agent.dyn == "Go2Pybullet":
    from simulators import Go2PybulletZeroSumEnv
    env_class = Go2PybulletZeroSumEnv
    cfg.cost = None
  elif cfg.agent.dyn == "BicycleDstb5D":
    from simulators import RaceCarDstb5DEnv
    # from simulators.race_car.functions import visualize_dstbEnv as visualize
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    env_class = RaceCarDstb5DEnv
  elif cfg.agent.dyn == "Dubins6D":
      from simulators import DubinsPursuitEvasionEnv
      env_class = DubinsPursuitEvasionEnv
      cfg.cost = None  # Dubins environment handles cost internally
      import jax
      jax.config.update('jax_platform_name', 'cpu')
  else:
    raise ValueError("Dynamics type not supported!")

  # Constructs environment.
  print("\n== Environment information ==")
  env = env_class(cfg.environment, cfg.agent, cfg.cost)
  env.step_keep_constraints = False
  env.report()

  # Constructs solver.
  print("\n== Solver information ==")
  solver = ISAACS(cfg.solver, cfg.arch, cfg.environment.seed)
  env.agent.policy = copy.deepcopy(solver.ctrl)
  print('#params in ctrl: {}'.format(sum(p.numel() for p in solver.ctrl.net.parameters() if p.requires_grad)))
  print('#params in dstb: {}'.format(sum(p.numel() for p in solver.dstb.net.parameters() if p.requires_grad)))
  print('#params in critic: {}'.format(sum(p.numel() for p in solver.critic.net.parameters() if p.requires_grad)))
  print("We want to use: {}, and Agent uses: {}".format(cfg.solver.device, solver.device))
  print("Critic is using cuda: ", next(solver.critic.net.parameters()).is_cuda)

  if cfg.agent.dyn == "BicycleDstb5D":
    # Constructs visualization callback.
    # vel_list = [0.5, 1., 1.5]
    # yaw_list = [-np.pi / 3, -np.pi / 4, 0., np.pi / 6, np.pi * 80 / 180]
    # visualize_callback = partial(
    #     visualize, vel_list=vel_list, yaw_list=yaw_list, end_criterion=cfg.solver.eval.end_criterion,
    #     T_rollout=cfg.solver.eval.timeout, nx=cfg.solver.eval.cmap_res_x, ny=cfg.solver.eval.cmap_res_y,
    #     subfigsz_x=cfg.solver.eval.fig_size_x, subfigsz_y=cfg.solver.eval.fig_size_y, vmin=cfg.environment.g_x_fail,
    #     vmax=-cfg.environment.g_x_fail, markersz=40
    # )
    visualize_callback = None
  else:
    visualize_callback = None

  # Constructs evaluation callback.
  reset_kwargs_list = []  # Same initial states.
  for _ in range(int(cfg.solver.eval.num_trajectories)):
    env.reset()
    reset_kwargs_list.append({"state": np.copy(env.state)})
  eval_callback = partial(
      evaluate_zero_sum, num_trajectories=int(cfg.solver.eval.num_trajectories),
      end_criterion=cfg.solver.eval.end_criterion, timeout=int(cfg.solver.eval.timeout),
      reset_kwargs_list=reset_kwargs_list, visualize_callback=visualize_callback
  )

  print("\n== Learning starts ==")
  loss_record, eval_record, violation_record, episode_record, leaderboard = solver.learn(
      env, eval_callback=eval_callback
  )
  train_dict = {}
  train_dict['loss_record'] = loss_record
  train_dict['eval_record'] = eval_record
  train_dict['violation_record'] = violation_record
  train_dict['episode_record'] = episode_record
  train_dict['leaderboard'] = leaderboard
  save_obj(train_dict, os.path.join(cfg.solver.out_folder, 'train'))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str, default=os.path.join("config", "isaacs.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)

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
from agent import Actor, SACBestResponse
from agent.sac import SAC
from simulators.spirit_rl.inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from utils.eval import evaluate_zero_sum
from simulators import PrintLogger, save_obj
from utils.functions import load_solver
from utils.utils import get_model_index


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
    wandb.init(entity='safe-princeton', project=cfg.solver.project_name, name=cfg.solver.name)
    tmp_cfg = {
        'environment': OmegaConf.to_container(cfg.environment),
        'solver': OmegaConf.to_container(cfg.solver),
        'arch': OmegaConf.to_container(cfg.arch),
    }
    wandb.config.update(tmp_cfg)

  # Constructs solver.
  print("\n== Solver information ==")
  ctrl_policy = Actor(
      cfg=cfg.solver.ctrl, cfg_arch=cfg.arch.ctrl, verbose=False, device=cfg.solver.device,
      obsrv_list=cfg.agent.obsrv_list.ctrl
  )
  ctrl_policy.restore(step=cfg.solver.ctrl.step, model_folder=cfg.solver.ctrl.model_folder)
  solver = SACBestResponse(cfg.solver, cfg.arch, cfg.environment.seed, ctrl_policy=ctrl_policy)

  if cfg.agent.dyn == "SpiritDstbPybullet":
    from simulators import SpiritPybulletZeroSumEnv
    env_class = SpiritPybulletZeroSumEnv
    cfg.cost = None
    # load controller policies to pass to agent.dyn
    ctrl_type = cfg.solver.ctrl.ctrl_type
    # performance
    performance = InverseKinematicsController(dt=1. / 250, L=0.8, T=0.1, Xdist=0.464, Ydist=0.33)
    # value shielding
    safety_policy = None
    critic_policy = None
    dstb_policy = None  # in the case of ISAACS instead of L1
    epsilon = cfg.solver.ctrl.epsilon
    gameplay_solver = None
    env_gameplay = None

    # example of cfg.agent.PRETRAIN_CTRL - the entire path
    # train_result/spirit_naive_reachavoid_f0_failure_newStateDef2/spirit_naive_reachavoid_f0_failure_newStateDef2_00/model/actor/actor-4400000.pth
    print("Pretrained ctrl: {}, step {}".format(cfg.solver.ctrl.model_folder, cfg.solver.ctrl.step))
    if ctrl_type != "performance":
      if ctrl_type == "shield_value" or ctrl_type == "safety":
        if cfg.solver.ctrl.model_folder is not None:
          print("Loading pretrained models into ctrl")

          # find the config file
          safety_config_file = os.path.join("/".join(cfg.solver.ctrl.model_folder.split("/")[:-1]), "config.yaml")
          safety_cfg = OmegaConf.load(safety_config_file)
          if safety_cfg.solver.num_actors == 1:
            safety_solver = SAC(safety_cfg.solver, safety_cfg.arch, safety_cfg.environment.seed)
            print("\nRestore model information")
            ## load ctrl and critic
            safety_ctrl_step, safety_model_path = get_model_index(
                safety_cfg.solver.out_folder, safety_cfg.eval.model_type, safety_cfg.eval.step, type="ctrl",
                autocutoff=0.9
            )

            safety_solver.actor.restore(safety_ctrl_step, safety_model_path)
            safety_solver.critic.restore(safety_ctrl_step, safety_model_path)

            safety_policy = safety_solver.actor.net
            critic_policy = safety_solver.critic.net
          elif safety_cfg.solver.num_actors == 2:
            safety_solver, safety_cfg = load_solver(safety_config_file)
            safety_policy = safety_solver.ctrl.net
            critic_policy = safety_solver.critic.net
            dstb_policy = safety_solver.dstb.net

          else:
            raise NotImplementedError
        else:
          raise NotImplementedError
      # we only need GAMEPLAY_CONFIG if we want to train dstb against rollout shielding
      elif ctrl_type == "shield_rollout":
        if cfg.solver.ctrl.gameplay_config is not None:
          # initialize gameplay, with dstb, ctrl and critic (just like we are about to run evaluation)
          gameplay_solver, gameplay_cfg = load_solver(cfg.solver.ctrl.gameplay_config)

          if gameplay_cfg.agent.dyn == "SpiritPybullet":
            from simulators import SpiritPybulletZeroSumEnv
            env_gameplay_class = SpiritPybulletZeroSumEnv
            gameplay_cfg.cost = None

          # Constructs environment.
          print("\n== Environment information ==")
          env_gameplay = env_gameplay_class(gameplay_cfg.environment, gameplay_cfg.agent, gameplay_cfg.cost)
          env_gameplay.step_keep_constraints = False
          env_gameplay.report()

        else:
          raise NotImplementedError
      else:
        raise NotImplementedError

    reset_kwargs = {
        "safety": safety_policy,
        "performance": performance,
        "ctrl_type": cfg.solver.ctrl.ctrl_type,
        "critic": critic_policy,
        "dstb": dstb_policy,
        "epsilon": epsilon,
        "gameplay_solver": gameplay_solver,
        "env_gameplay": env_gameplay,
        "gameplay_horizon": cfg.solver.ctrl.gameplay_horizon
    }
  elif cfg.agent.dyn == "BicycleDstb5D":
    from simulators import RaceCarDstb5DEnv
    import jax
    from simulators.race_car.functions import visualize_dstbEnv as visualize
    jax.config.update('jax_platform_name', 'cpu')
    env_class = RaceCarDstb5DEnv
    reset_kwargs = {}
  else:
    raise ValueError("Dynamics type not supported!")

  # Constructs environment.
  print("\n== Environment information ==")
  env = env_class(cfg.environment, cfg.agent, cfg.cost)
  env.step_keep_constraints = False
  env.report()

  env.agent.policy = ctrl_policy
  print('#params in actor: {}'.format(sum(p.numel() for p in solver.actor.net.parameters() if p.requires_grad)))
  print('#params in critic: {}'.format(sum(p.numel() for p in solver.critic.net.parameters() if p.requires_grad)))
  print("We want to use: {}, and Agent uses: {}".format(cfg.solver.device, solver.device))
  print("Critic is using cuda: ", next(solver.critic.net.parameters()).is_cuda)

  ## RESTORE PREVIOUS RUN
  # print("\nRestore model information")
  # ## load ctrl and critic
  # actor_model_path = cfg.arch.actor_0.pretrained_path
  # critic_model_path = cfg.arch.critic_0.pretrained_path
  # actor_step = actor_model_path.split("/")[-1].split("-")[1].replace(".pth", "")
  # critic_step = critic_model_path.split("/")[-1].split("-")[1].replace(".pth", "")
  # model_path = "/".join(actor_model_path.split("/")[:-2])

  # solver.actor.restore(actor_step, model_path)
  # solver.critic.restore(critic_step, model_path)

  if cfg.agent.dyn == "BicycleDstb5D":
    # Constructs visualization callback.
    vel_list = [0.5, 1., 1.5]
    yaw_list = [-np.pi / 3, -np.pi / 4, 0., np.pi / 6, np.pi * 80 / 180]
    visualize_callback = partial(
        visualize, vel_list=vel_list, yaw_list=yaw_list, end_criterion=cfg.solver.eval.end_criterion,
        T_rollout=cfg.solver.eval.timeout, nx=cfg.solver.eval.cmap_res_x, ny=cfg.solver.eval.cmap_res_y,
        subfigsz_x=cfg.solver.eval.fig_size_x, subfigsz_y=cfg.solver.eval.fig_size_y, vmin=cfg.environment.g_x_fail,
        vmax=-cfg.environment.g_x_fail, markersz=40
    )
  else:
    visualize_callback = None

  # Constructs evaluation callback.
  reset_kwargs_list = []  # Same initial states.
  for _ in range(int(cfg.solver.eval.num_trajectories)):
    env.reset(**reset_kwargs)
    reset_kwargs_list.append({"state": np.copy(env.state)})
  eval_callback = partial(
      evaluate_zero_sum, num_trajectories=int(cfg.solver.eval.num_trajectories),
      end_criterion=cfg.solver.eval.end_criterion, timeout=int(cfg.solver.eval.timeout),
      reset_kwargs_list=reset_kwargs_list, visualize_callback=visualize_callback, **reset_kwargs
  )

  print("\n== Learning starts ==")
  loss_record, eval_record, violation_record, episode_record, pq_top_k = solver.learn(
      env, eval_callback=eval_callback, reset_kwargs=reset_kwargs
  )
  train_dict = {}
  train_dict['loss_record'] = loss_record
  train_dict['eval_record'] = eval_record
  train_dict['violation_record'] = violation_record
  train_dict['episode_record'] = episode_record
  train_dict['pq_top_k'] = list(pq_top_k.queue)
  save_obj(train_dict, os.path.join(cfg.solver.out_folder, 'train'))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="config file path", type=str, default=os.path.join("config", "pretrain_dstb.yaml")
  )
  args = parser.parse_args()
  main(args.config_file)

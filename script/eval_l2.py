# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import argparse
from omegaconf import OmegaConf
from agent import Actor, SACBestResponse
from agent.sac import SAC
from simulators import SpiritPybulletZeroSumEnv
from simulators.spirit_rl.inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from utils.functions import load_solver
from utils.utils import get_model_index
import pybullet as p
import torch


def main(args):
  config_file = args.config_file
  gui = args.gui
  dstb_step = args.dstb_step
  ctrl_type = args.ctrl_type

  # Loads config.
  cfg = OmegaConf.load(config_file)

  if ctrl_type != "default":
    cfg.solver.ctrl.ctrl_type = ctrl_type

  os.makedirs(cfg.solver.out_folder, exist_ok=True)

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
  else:
    raise ValueError("Dynamics type not supported!")

  # Constructs environment.
  print("\n== Environment information ==")

  # overwrite GUI flag from config if there's GUI flag from argparse
  if gui is True:
    cfg.agent.gui = True

  env = env_class(cfg.environment, cfg.agent, None)
  env.agent.policy = ctrl_policy
  print('#params in actor: {}'.format(sum(p.numel() for p in solver.actor.net.parameters() if p.requires_grad)))
  print('#params in critic: {}'.format(sum(p.numel() for p in solver.critic.net.parameters() if p.requires_grad)))
  print("We want to use: {}, and Agent uses: {}".format(cfg.solver.device, solver.device))
  print("Critic is using cuda: ", next(solver.critic.net.parameters()).is_cuda)

  ## RESTORE PREVIOUS RUN
  print("\nRestore model information")
  ## load ctrl and critic
  if dstb_step is None:
    dstb_step, model_path = get_model_index(
        cfg.solver.out_folder, cfg.eval.model_type, cfg.eval.step, type="dstb", autocutoff=0.9
    )
  else:
    model_path = os.path.join(cfg.solver.out_folder, "model")

  solver.actor.restore(dstb_step, model_path)
  solver.critic.restore(dstb_step, model_path)

  # evaluate
  s = env.reset(cast_torch=True, **reset_kwargs)
  while True:
    a_adv = solver.adv_policy.net(s.float().to(solver.device))  # ctrl
    s_dstb = [s.float().to(solver.device)]
    if cfg.agent.obsrv_list.dstb is not None:
      for i in cfg.agent.obsrv_list.dstb:
        if i == "ctrl":
          s_dstb.append(a_adv)
    a = solver.actor.net(*s_dstb)
    critic_q = max(solver.critic.net(s.float().to(solver.device), solver.combine_action(a, a_adv)))
    a = {'ctrl': a_adv.detach().numpy(), 'dstb': a.detach().numpy()}
    s_, r, done, info = env.step(a, cast_torch=True)
    s = s_
    if done:
      if p.getKeyboardEvents().get(49):
        continue
      else:
        env.reset()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-cf", "--config_file", help="config file path", type=str)
  parser.add_argument("--dstb_step", help="dstb policy model step", type=int, default=None)
  parser.add_argument("--gui", help="GUI for Pybullet", action="store_true")
  parser.add_argument(
      "--ctrl_type", help="overwrite control type", type=str, default="default",
      choices=["default", "performance", "safety", "shield_value", "shield_rollout"]
  )
  args = parser.parse_args()
  main(args)

# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import copy
import argparse
from omegaconf import OmegaConf
from agent import ISAACS
from utils.utils import get_model_index


def main(args):
  config_file = args.config_file
  gui = args.gui
  dstb_step = args.dstb_step
  ctrl_step = args.ctrl_step

  # Loads config.
  cfg = OmegaConf.load(config_file)

  os.makedirs(cfg.solver.out_folder, exist_ok=True)

  if cfg.agent.dyn == "SpiritPybullet":
    from simulators import SpiritPybulletZeroSumEnv
    env_class = SpiritPybulletZeroSumEnv
    import pybullet as p
  elif cfg.agent.dyn == "Go2Pybullet":
    from simulators import Go2PybulletZeroSumEnv
    env_class = Go2PybulletZeroSumEnv
    import pybullet as p
  else:
    raise ValueError("Dynamics type not supported!")

  # Constructs environment.
  print("\n== Environment information ==")

  # overwrite GUI flag from config if there's GUI flag from argparse
  if gui is True:
    cfg.agent.gui = True

  env = env_class(cfg.environment, cfg.agent, None)

  # Constructs solver.
  print("\n== Solver information ==")
  solver = ISAACS(cfg.solver, cfg.arch, cfg.environment.seed)
  env.agent.policy = copy.deepcopy(solver.ctrl)
  print('#params in ctrl: {}'.format(sum(p.numel() for p in solver.ctrl.net.parameters() if p.requires_grad)))
  print('#params in dstb: {}'.format(sum(p.numel() for p in solver.dstb.net.parameters() if p.requires_grad)))
  print('#params in critic: {}'.format(sum(p.numel() for p in solver.critic.net.parameters() if p.requires_grad)))
  print("We want to use: {}, and Agent uses: {}".format(cfg.solver.device, solver.device))
  print("Critic is using cuda: ", next(solver.critic.net.parameters()).is_cuda)

  ## RESTORE PREVIOUS RUN
  print("\nRestore model information")
  ## load ctrl and critic
  if dstb_step is None:
    dstb_step, model_path = get_model_index(
        cfg.solver.out_folder, cfg.eval.model_type[1], cfg.eval.step[1], type="dstb", autocutoff=0.9
    )
  else:
    model_path = os.path.join(cfg.solver.out_folder, "model")

  if ctrl_step is None:
    ctrl_step, model_path = get_model_index(
        cfg.solver.out_folder, cfg.eval.model_type[0], cfg.eval.step[0], type="ctrl", autocutoff=0.9
    )
  else:
    model_path = os.path.join(cfg.solver.out_folder, "model")

  solver.ctrl.restore(ctrl_step, model_path)
  solver.dstb.restore(dstb_step, model_path)
  solver.critic.restore(ctrl_step, model_path)

  # evaluate
  s = env.reset(cast_torch=True)
  while True:
    u = solver.ctrl.net(s.float().to(solver.device))
    s_dstb = [s.float().to(solver.device)]
    if cfg.agent.obsrv_list.dstb is not None:
      for i in cfg.agent.obsrv_list.dstb:
        if i == "ctrl":
          s_dstb.append(u)
    d = solver.dstb.net(*s_dstb)
    # critic_q = max(solver.critic.net(s.float().to(solver.device), solver.combine_action(u, d)))
    # print("\r{}".format(critic_q), end="")
    a = {'ctrl': u.detach().numpy(), 'dstb': d.detach().numpy()}
    s_, r, done, info = env.step(a, cast_torch=True)
    s = s_
    if done:
      if "Pybullet" in cfg.agent.dyn:
        if p.getKeyboardEvents().get(49):
          continue
        else:
          env.reset()
      elif "Mujoco" in cfg.agent.dyn:
        if chr(env.agent.dyn.keycode) == '1':
          continue
        else:
          env.reset()    
      else:
        env.reset()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-cf", "--config_file", help="config file path", type=str)
  parser.add_argument("--dstb_step", help="dstb policy model step", type=int, default=None)
  parser.add_argument("--ctrl_step", help="ctrl/critic policy model step", type=int, default=None)
  parser.add_argument("--gui", help="GUI for Pybullet", action="store_true")
  args = parser.parse_args()
  main(args)

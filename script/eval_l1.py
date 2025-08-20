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
from agent import SAC
from simulators import SpiritPybulletSingleEnv
from utils.utils import get_model_index
import pybullet as p


def main(args):
  config_file = args.config_file
  gui = args.gui
  ctrl_step = args.ctrl_step

  cfg = OmegaConf.load(config_file)

  os.makedirs(cfg.solver.out_folder, exist_ok=True)

  if cfg.agent.dyn == "SpiritPybullet":
    env_class = SpiritPybulletSingleEnv
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
  solver = SAC(cfg.solver, cfg.arch, cfg.environment.seed)
  env.agent.policy = copy.deepcopy(solver.actor)
  print('#params in actor: {}'.format(sum(p.numel() for p in solver.actor.net.parameters() if p.requires_grad)))
  print('#params in critic: {}'.format(sum(p.numel() for p in solver.critic.net.parameters() if p.requires_grad)))
  print("We want to use: {}, and Agent uses: {}".format(cfg.solver.device, solver.device))
  print("Critic is using cuda: ", next(solver.critic.net.parameters()).is_cuda)

  ## RESTORE PREVIOUS RUN
  print("\nRestore model information")
  ## load ctrl and critic
  if ctrl_step is None:
    ctrl_step, model_path = get_model_index(
        cfg.solver.out_folder, cfg.eval.model_type, cfg.eval.step, type="ctrl", autocutoff=0.9
    )
  else:
    model_path = os.path.join(cfg.solver.out_folder, "model")

  solver.actor.restore(ctrl_step, model_path)
  solver.critic.restore(ctrl_step, model_path)

  # evaluate
  s = env.reset(cast_torch=True)
  while True:
    a = solver.actor.net(s.float().to(solver.device))
    critic_q = max(solver.critic.net(s.float().to(solver.device), a.float().to(solver.device)))
    a = a.detach().numpy()
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
  parser.add_argument("--ctrl_step", help="ctrl/critic policy model step", type=int, default=None)
  parser.add_argument("--gui", help="GUI for Pybullet", action="store_true")
  args = parser.parse_args()
  main(args)

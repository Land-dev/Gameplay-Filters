# --------------------------------------------------------
# ISAACS: Iterative Soft Adversarial Actor-Critic for Safety
# https://arxiv.org/abs/2212.03228
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu, duyn@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import argparse
import pickle
from omegaconf import OmegaConf
from agent import SACBestResponse
from agent.base_block import Actor
from agent.sac import SAC
from simulators import SpiritPybulletZeroSumEnv
from simulators.spirit_rl.inverse_kinematics.inverse_kinematics_controller import InverseKinematicsController
from utils.functions import load_batch, load_solver
from utils.utils import get_model_index, save_obj
import pybullet as p
import torch
import pandas as pd
import numpy as np


def main(args):
  # general flag
  gui = args.gui
  log = args.log
  log_name = args.log_name
  exp_name = args.exp_name
  seed = args.seed

  # ctrl
  ctrl_type = args.ctrl_type
  ctrl_config = args.ctrl_config
  ctrl_step = args.ctrl_step
  epsilon = args.epsilon
  rollout_step = args.imaginary_horizon
  gameplay_config = args.gameplay_config

  # dstb
  dstb_type = args.dstb_type
  dstb_config = args.dstb_config
  dstb_step = args.dstb_step
  force = args.force

  # evaluation
  batch_path = args.batch_path
  batch_index = args.batch_index
  min_index, max_index = args.index_range
  eval_horizon = args.eval_horizon
  end_criterion = args.end_criterion
  rollout_end_criterion = args.rollout_end_criterion

  # set random seed for eval session
  np.random.seed(seed)

  print("Load the models")
  print(f"\tLoading control type: {ctrl_type}")
  safety_policy = None
  critic_policy = None
  performance = InverseKinematicsController(dt=1. / 250, L=0.8, T=0.1, Xdist=0.464, Ydist=0.33)
  gameplay_solver = None
  env_gameplay = None
  dstb_policy = None  # this policy is used to compute the critic_q, not the actual force/dstb that will attack the robot
  if ctrl_type == "performance":
    print(f"\t\t{performance} loaded")
  elif ctrl_type == "safety" or ctrl_type == "shield_value":
    assert ctrl_config is not None, "Error: Missing config file for control"
    ctrl_cfg = OmegaConf.load(ctrl_config)
    # check to see if it's SAC or ISAACS
    if ctrl_cfg.solver.num_actors == 1:
      print(f"\t\tDetect ctrl type SAC")
      ctrl_solver = SAC(ctrl_cfg.solver, ctrl_cfg.arch, ctrl_cfg.environment.seed)

      safety_ctrl_step, safety_model_path = get_model_index(
          ctrl_cfg.solver.out_folder, ctrl_cfg.eval.model_type, ctrl_cfg.eval.step, type="ctrl", autocutoff=0.9
      )

      if ctrl_step is not None:
        safety_ctrl_step = ctrl_step

      ctrl_solver.actor.restore(safety_ctrl_step, safety_model_path)
      ctrl_solver.critic.restore(safety_ctrl_step, safety_model_path)

      safety_policy = ctrl_solver.actor.net
      critic_policy = ctrl_solver.critic.net
    elif ctrl_cfg.solver.num_actors == 2:
      print(f"\t\tDetect ctrl type ISAACS")
      ctrl_solver, ctrl_cfg = load_solver(ctrl_config)
      safety_policy = ctrl_solver.ctrl.net
      critic_policy = ctrl_solver.critic.net
      dstb_policy = ctrl_solver.dstb.net
    else:
      raise NotImplementedError

    print(f"\t\t{safety_policy} and {critic_policy} loaded")
  elif ctrl_type == "shield_rollout":
    if gameplay_config is None:
      print("\t\tThe gameplay solver will use the same config as the ctrl")
      gameplay_config = ctrl_config

    assert ctrl_config is not None, "Error: Missing config file for control"
    gameplay_solver, gameplay_cfg = load_solver(gameplay_config)

    if gameplay_cfg.agent.dyn == "SpiritPybullet":
      env_gameplay_class = SpiritPybulletZeroSumEnv
      gameplay_cfg.cost = None

    gameplay_cfg.environment.seed = seed
    env_gameplay = env_gameplay_class(gameplay_cfg.environment, gameplay_cfg.agent, gameplay_cfg.cost)
    env_gameplay.step_keep_constraints = False

    print(f"\t\t{gameplay_solver} loaded")

  print(f"\tLoading dstb type: {dstb_type}")
  assert dstb_config is not None, "Error: Missing config file for dstb"
  dstb_cfg = OmegaConf.load(dstb_config)
  adv_policy = None

  if dstb_type == "adversary":
    if dstb_cfg.solver.num_actors == 1:
      print(f"\t\tDetect dstb type SAC_BR")
      ctrl_policy = Actor(
          cfg=dstb_cfg.solver.ctrl, cfg_arch=dstb_cfg.arch.ctrl, verbose=False, device=dstb_cfg.solver.device,
          obsrv_list=dstb_cfg.agent.obsrv_list.ctrl
      )
      dstb_solver = SACBestResponse(dstb_cfg.solver, dstb_cfg.arch, dstb_cfg.environment.seed, ctrl_policy=ctrl_policy)
      if dstb_step is None:
        dstb_step, model_path = get_model_index(
            dstb_cfg.solver.out_folder, dstb_cfg.eval.model_type, dstb_cfg.eval.step, type="dstb", autocutoff=0.9
        )
      else:
        model_path = os.path.join(dstb_cfg.solver.out_folder, "model")

      dstb_solver.actor.restore(dstb_step, model_path)
      dstb_solver.critic.restore(dstb_step, model_path)
      adv_policy = dstb_solver.actor.net

    elif dstb_cfg.solver.num_actors == 2:
      print(f"\t\tDetect dstb type ISAACS")
      dstb_solver, dstb_cfg = load_solver(dstb_config)
      adv_policy = dstb_solver.dstb.net

    else:
      raise NotImplementedError
    print(f"\t\t{dstb_solver} loaded")
  else:
    dstb_cfg.agent.force_type = dstb_type
    dstb_cfg.agent.replace_adv_with_dr = True

    print("\t\tRandom disturbance will be used")

  if gui:
    dstb_cfg.agent.gui = gui

  if end_criterion != "default":
    dstb_cfg.solver.eval.end_criterion = end_criterion
    dstb_cfg.environment.end_criterion = end_criterion

  if rollout_end_criterion != "default":
    dstb_cfg.solver.rollout_end_criterion = rollout_end_criterion

  if force is not None:
    dstb_cfg.agent.force = force

  if eval_horizon is None:
    eval_horizon = dstb_cfg.eval.eval_timeout
  else:
    dstb_cfg.eval.eval_timeout = eval_horizon
  # make sure dstb_cfg.solver.eval.timeout and dstb_cfg.environment.timeout is the same
  dstb_cfg.solver.eval.timeout = eval_horizon
  dstb_cfg.environment.timeout = eval_horizon

  if rollout_step is None:
    rollout_step = dstb_cfg.eval.imaginary_horizon
  else:
    dstb_cfg.eval.imaginary_horizon = rollout_step

  # make sure that the dstb_cfg.agent.dyn is SpiritDstbPybullet
  dstb_cfg.agent.dyn = "SpiritDstbPybullet"

  dstb_cfg.environment.seed = seed
  env = SpiritPybulletZeroSumEnv(dstb_cfg.environment, dstb_cfg.agent, None)

  reset_kwargs = {
      "safety": safety_policy,
      "performance": performance,
      "ctrl_type": ctrl_type,
      "critic": critic_policy,
      "dstb": dstb_policy,
      "epsilon": epsilon,
      "gameplay_solver": gameplay_solver,
      "env_gameplay": env_gameplay,
      "gameplay_horizon": rollout_step
  }

  # load batch information
  if batch_path is None:
    batch_df = None
  else:
    batch_condition_path = os.path.join(os.getcwd(), batch_path)
    print("Use batch condition: {}".format(batch_condition_path))
    with open(batch_condition_path, "rb") as f:
      batch_conditions = pickle.load(f)
    batch_df = pd.DataFrame(batch_conditions)
    if batch_index is not None:
      min_index = batch_index
      max_index = batch_index

  logger = []
  no_of_runs = max_index - min_index + 1
  safe_count = 0

  # begin running evaluation batch
  for i in range(min_index, max_index + 1):
    batch_condition = load_batch(batch_df=batch_df, batch_index=i, env=env)
    s = env.reset(cast_torch=True, **batch_condition, **reset_kwargs)
    state_array = [s]
    action_array = [None]
    reward_array = []
    info_array = []
    done_array = []

    # running single evaluation
    for j in range(eval_horizon):
      # print(f"\r{j+1}/{eval_horizon}", end="")

      # dummy control, will not affect anything
      # u = ctrl_policy.net(s.float().to('cpu'))  # ctrl
      u = torch.Tensor(np.zeros(12))

      s_adv = [s.float().to('cpu')]
      if dstb_cfg.agent.obsrv_list.dstb is not None:
        for i in dstb_cfg.agent.obsrv_list.dstb:
          if i == "ctrl":
            s_adv.append(u)

      if dstb_type == "adversary":
        adv = adv_policy(*s_adv)
      else:
        adv = torch.Tensor(np.zeros(6))

      a = {'ctrl': u.detach().numpy(), 'dstb': adv.detach().numpy()}
      s_, r, done, info = env.step(a, cast_torch=True)
      s = s_

      state_array.append(s)
      action_array.append(a)
      info_array.append(info)
      reward_array.append(r)
      done_array.append(done)

      if done:
        if p.getKeyboardEvents().get(49):
          continue
        else:
          break

    if info['done_type'] != "failure":
      safe_count += 1

    print(f"{i+1}/{no_of_runs}\t{ctrl_type}: {info['done_type']}\tsafe rate: {safe_count/no_of_runs}")

    logger.append({
        "args": args,
        "iteration": i,
        "traj": {
            "state": state_array,
            "action": action_array,
            "info": info_array,
            "done": done_array,
            "reward": reward_array,
        }
    })

  # finish running evaluation
  if log:
    os.makedirs(os.path.join("eval_result", exp_name), exist_ok=True)
    counter = 0
    while True:
      file_path = os.path.join("eval_result", exp_name, log_name + "_{}".format(counter))
      if os.path.exists(file_path + ".pkl"):
        counter += 1
      else:
        save_obj(logger, file_path)
        break
    print(f"Save result to {file_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # general arguments
  parser.add_argument("--gui", help="GUI for Pybullet", action="store_true")
  parser.add_argument("--log", help="log evaluation data", action="store_true")
  parser.add_argument("--log_name", help="name of log file", type=str, default="summary")
  parser.add_argument("--exp_name", help="add experiment name", type=str, default="eval")
  parser.add_argument("--seed", help="random seed", type=int, default=0)

  # argument for ctrl
  parser.add_argument(
      "--ctrl_type", help="Type of control used for the evaluation", type=str, default="safety",
      choices=["performance", "safety", "shield_value", "shield_rollout"]
  )
  # if ctrl_type is "safety", "shield_value" or "shield_rollout", need ctrl_config
  parser.add_argument("--ctrl_config", help="config file path for ctrl", type=str, default=None)
  parser.add_argument("--ctrl_step", help="ctrl policy model step", type=int, default=None)
  parser.add_argument("-eps", "--epsilon", help="Epsilon used for shield_value", type=float, default=0.0)
  parser.add_argument(
      "--imaginary_horizon", help="overwrite the imaginary horizon of rollout-based shielding in the config file",
      type=int, default=None
  )
  parser.add_argument(
      "--gameplay_config", help="Config file for the gameplay solver in case of rollout shielding", type=str,
      default=None
  )

  # argument for dstb
  parser.add_argument(
      "--dstb_type", help="Type of dstb used for the evaluation", type=str, default="adversary",
      choices=["adversary", "bangbang", "uniform"]
  )
  parser.add_argument("--dstb_config", help="config file path for ctrl", type=str, default=None)
  parser.add_argument("--dstb_step", help="dstb policy model step", type=int, default=None)
  parser.add_argument("--force", help="Overwrite force magnitude to be used", default=None, type=int)

  # evaluation conditions
  parser.add_argument("--batch_path", help="file path of batch conditions", type=str, default=None)
  parser.add_argument("--batch_index", help="run index of batch, single run", type=int, default=None)
  parser.add_argument(
      "--index_range", help="range of batch index to run, [min, max]", type=int, default=[0, 49], nargs='*'
  )
  parser.add_argument("--eval_horizon", help="overwrite the eval horizon in the config file", type=int, default=None)
  parser.add_argument(
      "--end_criterion", help="end criterion type", type=str, default="default",
      choices=["default", "failure", "reach-avoid"]
  )
  parser.add_argument(
      "--rollout_end_criterion", help="rollout end criterion type", type=str, default="default",
      choices=["default", "failure", "reach-avoid"]
  )

  args = parser.parse_args()
  main(args)

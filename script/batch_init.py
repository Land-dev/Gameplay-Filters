import sys
import os

sys.path.append(os.getcwd())
from omegaconf import OmegaConf
from utils.functions import load_solver
import argparse
from simulators import SpiritPybulletZeroSumEnv, Go2PybulletZeroSumEnv
from utils.utils import save_obj
import time

timestr = time.strftime("%Y%m%d%H%M%S")


def main(args):
  config_file = args.config_file
  cfg = OmegaConf.load(config_file)

  no_of_runs = args.runs
  suffix = args.suffix
  ctrl_step = args.ctrl_step
  dstb_step = args.dstb_step
  end_criterion = args.end_criterion
  reset_criterion = args.reset_criterion
  rollout_step = args.imaginary_horizon
  state_type = args.type

  if cfg.agent.dyn == "SpiritPybullet":
    env_class = SpiritPybulletZeroSumEnv
  if cfg.agent.dyn == "Go2Pybullet":
    env_class = Go2PybulletZeroSumEnv
  else:
    raise ValueError("Dynamics type not supported!")

  # overwrite the ROLLOUT_END_CRITERION, END_CRITERION, MODE in config
  if end_criterion != "default":
    cfg.solver.eval.end_criterion = end_criterion
    cfg.solver.rollout_end_criterion = end_criterion
    cfg.environment.end_criterion = end_criterion

  # overwrite RESET_CRITERION in config
  if reset_criterion != "default":
    cfg.agent.reset_criterion = reset_criterion

  if rollout_step is None:
    rollout_step = cfg.eval.imaginary_horizon
  else:
    cfg.eval.imaginary_horizon = rollout_step

  env = env_class(cfg.environment, cfg.agent, None)
  solver, cfg = load_solver(config_file, ctrl_step=ctrl_step, dstb_step=dstb_step)

  print("Generating {} initial conditions".format(no_of_runs))

  logger = {
      "terrain_data": [],
      "initial_height": [],
      "initial_rotation": [],
      "initial_joint_value": [],
      "initial_joint_vel": [],
      "initial_linear_vel": [],
      "initial_angular_vel": [],
      "initial_height_reset_type": [],
      "initial_action": []
  }

  if state_type == "stay_safe":
    print(
        "> Finding initial conditions that do not violate safety after {} steps applying safety policy".
        format(rollout_step)
    )
  elif state_type == "stable_stance":
    print("> Find stable stances that satisfy reset_criterion: {}".format(reset_criterion))

  for i in range(no_of_runs):
    # check to see if during the rollout horizon, applying safety policy will result in failure set
    if state_type == "stay_safe":
      while True:
        s = env.reset(cast_torch=True)
        for i in range(rollout_step):
          u = solver.ctrl.net(s.float().to(solver.device))
          s_dstb = [s.float().to(solver.device)]
          if cfg.agent.obsrv_list.dstb is not None:
            for i in cfg.agent.obsrv_list.dstb:
              if i == "ctrl":
                s_dstb.append(u)
          d = solver.dstb.net(*s_dstb)
          a = {'ctrl': u.detach().numpy(), 'dstb': d.detach().numpy()}
          s_, r, done, info = env.step(a, cast_torch=True)
          s = s_

        if not done:
          break
        elif info["done_type"] == "timeout" or info["done_type"] == "success":
          break
    elif state_type == "stable_stance":
      env.reset(cast_torch=True)
    else:
      raise NotImplementedError

    terrain_data = env.agent.dyn.terrain_data
    initial_height = env.agent.dyn.initial_height
    initial_rotation = env.agent.dyn.initial_rotation
    initial_joint_value = env.agent.dyn.initial_joint_value
    initial_joint_vel = env.agent.dyn.initial_joint_vel
    initial_linear_vel = env.agent.dyn.initial_linear_vel
    initial_angular_vel = env.agent.dyn.initial_angular_vel
    initial_height_reset_type = env.agent.dyn.initial_height_reset_type
    initial_action = env.agent.dyn.initial_action

    logger["terrain_data"].append(terrain_data)
    logger["initial_height"].append(initial_height)
    logger["initial_rotation"].append(initial_rotation)
    logger["initial_joint_value"].append(initial_joint_value)
    logger["initial_joint_vel"].append(initial_joint_vel)
    logger["initial_linear_vel"].append(initial_linear_vel)
    logger["initial_angular_vel"].append(initial_angular_vel)
    logger["initial_height_reset_type"].append(initial_height_reset_type)
    logger["initial_action"].append(initial_action)

    print("\r>> Generating initial conditions run #{}".format(i), end="")

  file_name = "batch_{}".format(no_of_runs)
  if suffix is not None:
    file_name = file_name + "_" + suffix

  file_path = os.path.join(os.getcwd(), file_name)
  save_obj(logger, file_path)
  print("\nDone generating initial condition batch, path: {}".format(file_path))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-cf", "--config_file", help="config file path", type=str)
  parser.add_argument("--suffix", help="suffix for .pkl name", type=str)
  parser.add_argument("--runs", help="Number of runs", type=int, default=50)
  parser.add_argument("--ctrl_step", help="ctrl/critic policy model step", type=int, default=None)
  parser.add_argument("--dstb_step", help="dstb policy model step", type=int, default=None)
  parser.add_argument(
      "--end_criterion", help="end criterion type", type=str, default="default",
      choices=["default", "failure", "reach-avoid"]
  )
  parser.add_argument(
      "--reset_criterion", help="reset criterion type", type=str, default="default",
      choices=["default", "failure", "reach-avoid"]
  )
  parser.add_argument(
      "--imaginary_horizon", help="overwrite the imaginary horizon of rollout-based shielding in the config file",
      type=int, default=None
  )
  parser.add_argument(
      "--type", help=
      "what kind of initial states do we want: states that can still be safe after N steps, or just randomized good stances",
      type=str, default="stay_safe", choices=["stay_safe", "stable_stance"]
  )
  args = parser.parse_args()
  main(args)

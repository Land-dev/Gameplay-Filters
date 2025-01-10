import os
from omegaconf import OmegaConf
import pandas as pd

from agent.isaacs import ISAACS
from utils.utils import get_model_index


def load_solver(config_file, ctrl_step=None, dstb_step=None, get_env=False):
  # Loads config.
  cfg = OmegaConf.load(config_file)

  # Constructs solver.
  print("\n== Solver information ==")
  solver = ISAACS(cfg.solver, cfg.arch, cfg.environment.seed)

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

  return solver, cfg


def load_batch(batch_df, batch_index, env):
  if batch_df is not None:
    assert type(batch_df) is pd.core.frame.DataFrame, "Error: batch condition must be DataFrame type"
    if batch_index is not None:
      initial_conditions = batch_df.iloc[batch_index]
      print("\tHas batch_index flag, use batch index {}".format(batch_index))
    else:
      print("Warning: No batch_index is defined, will run the first condition in batch conditions")
      initial_conditions = batch_df.iloc[0]

    terrain_data = initial_conditions.terrain_data
    initial_height = initial_conditions.initial_height
    initial_rotation = initial_conditions.initial_rotation
    initial_joint_value = initial_conditions.initial_joint_value
    initial_linear_vel = initial_conditions.initial_linear_vel
    initial_angular_vel = initial_conditions.initial_angular_vel
    initial_height_reset_type = initial_conditions.initial_height_reset_type
    initial_action = initial_conditions.initial_action
  else:
    assert env is not None, "Error: no batch_df and no env"
    env.reset(cast_torch=True)
    terrain_data = env.agent.dyn.terrain_data
    initial_height = env.agent.dyn.initial_height
    initial_rotation = env.agent.dyn.initial_rotation
    initial_joint_value = env.agent.dyn.initial_joint_value
    initial_linear_vel = env.agent.dyn.initial_linear_vel
    initial_angular_vel = env.agent.dyn.initial_angular_vel
    initial_height_reset_type = env.agent.dyn.initial_height_reset_type
    initial_action = env.agent.dyn.initial_action

  kwargs = {
      "terrain_data": terrain_data,
      "initial_height": initial_height,
      "initial_rotation": initial_rotation,
      "initial_joint_value": initial_joint_value,
      "initial_linear_vel": initial_linear_vel,
      "initial_angular_vel": initial_angular_vel,
      "initial_height_reset_type": initial_height_reset_type,
      "initial_action": initial_action
  }

  return kwargs

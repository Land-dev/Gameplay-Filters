import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Callable, Union, Optional

from ..policy.base_policy import BasePolicy
from .race_car_dstb import RaceCarDstb5DEnv
from .race_car_single import RaceCarSingle5DEnv
from .visualization import plot_traj, get_values


# region: local functions
def get_trajectories_singleEnv(
    env: RaceCarSingle5DEnv,
    vel_list: List[float],
    yaw_list: List[float],
    num_pts: int = 5,
    T_rollout: int = 150,
    end_criterion: str = "failure",
):
  num_traj = len(vel_list) * len(yaw_list) * num_pts
  reset_kwargs_list = []
  for _ in range(num_pts):
    near_failure = True
    cnt = 0
    a_dummy = np.zeros((2, 1))
    while near_failure and (cnt <= 10):
      env.reset()
      state = env.state.copy()
      cons_dict = env.get_constraints(state, a_dummy, state)
      constraint_values = None
      for key, value in cons_dict.items():
        if constraint_values is None:
          num_pts = value.shape[1]
          constraint_values = value
        else:
          assert num_pts == value.shape[1], ("The length of constraint ({}) do not match".format(key))
          constraint_values = np.concatenate((constraint_values, value), axis=0)
      g_x = np.min(constraint_values[:, -1], axis=0)
      near_failure = g_x < 0.1
      cnt += 1
    for vel in vel_list:
      for yaw in yaw_list:
        state[2] = vel
        state[3] = yaw
        reset_kwargs_list.append(dict(state=state.copy()))

  trajectories, results, _ = env.simulate_trajectories(
      num_traj, T_rollout=T_rollout, end_criterion=end_criterion, reset_kwargs_list=reset_kwargs_list
  )
  return trajectories, results


def visualize_singleEnv(
    env: RaceCarSingle5DEnv,
    value_fn: Callable[[np.ndarray, Optional[Union[np.ndarray, torch.Tensor]]], np.ndarray],
    fig_path: str,
    vel_list: List[float],
    yaw_list: List[float],
    num_pts: int = 5,
    T_rollout: int = 150,
    end_criterion: str = "failure",
    subfigsz_x: float = 4.,
    subfigsz_y: float = 4.,
    nx: int = 100,
    ny: int = 100,
    batch_size: int = 512,
    cmap: str = 'seismic_r',
    vmin: float = -.25,
    vmax: float = .25,
    alpha: float = 0.5,
    fontsize: int = 16,
    vel_scatter: bool = False,
    markersz: int = 40,
):
  n_row = len(yaw_list)
  n_col = len(vel_list)
  figsize = (subfigsz_x * n_col, subfigsz_y * n_row)
  fig, axes = plt.subplots(n_row, n_col, figsize=figsize, sharex=True, sharey=True)
  vmin_label = vmin
  vmax_label = vmax
  vmean_label = 0

  xs, ys = env.get_samples(nx, ny)
  trajectories, results = get_trajectories_singleEnv(
      env=env, vel_list=vel_list, yaw_list=yaw_list, num_pts=num_pts, T_rollout=T_rollout, end_criterion=end_criterion
  )

  for i, vel in enumerate(vel_list):
    axes[0][i].set_title(f"Vel: {vel:.2f}", fontsize=fontsize)
    for j, yaw in enumerate(yaw_list):
      ax = axes[j][i]
      if i == 0:
        ax.set_ylabel(f"Yaw: {yaw/np.pi*180:.0f}", fontsize=fontsize)
      # plots value function
      values = get_values(env, value_fn, xs, ys, vel, yaw, batch_size=batch_size, fail_value=vmin)
      im = ax.imshow(
          values.T, interpolation='none', extent=env.visual_extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
          zorder=-1, alpha=alpha
      )

      # Plots trajectories
      for k in range(num_pts):
        idx = int(k * len(yaw_list) * len(vel_list) + i * len(yaw_list) + j)
        trajectory = trajectories[idx]
        result = results[idx]
        plot_traj(ax, trajectory, result, c='g', lw=2., vel_scatter=vel_scatter, zorder=1, s=markersz)

      env.track.plot_track(ax, c='k')
      env.render_obs(ax=ax, c='r')
      ax.axis(env.visual_extent)
      ax.set_xticks(np.around(env.visual_bounds[0], 1))
      ax.set_yticks(np.around(env.visual_bounds[1], 1))
      ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)

  # one color bar
  fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.08, hspace=0.01)
  cbar_ax = fig.add_axes([0.96, 0.05, 0.02, 0.85])
  cbar = fig.colorbar(im, cax=cbar_ax, ax=ax, ticks=[vmin, 0, vmax])
  v_ticklabels = np.around(np.array([vmin_label, vmean_label, vmax_label]), 2)
  cbar.ax.set_yticklabels(labels=v_ticklabels, fontsize=fontsize - 4)
  fig.savefig(fig_path, dpi=400)
  plt.close('all')


# endregion


# region: local functions
def get_trajectories_dstbEnv(
    env: RaceCarDstb5DEnv,
    adversary: BasePolicy,
    vel_list: List[float],
    yaw_list: List[float],
    num_pts: int = 5,
    T_rollout: int = 150,
    end_criterion: str = "failure",
):
  num_traj = len(vel_list) * len(yaw_list) * num_pts
  reset_kwargs_list = []
  for _ in range(num_pts):
    near_failure = True
    cnt = 0
    a_dummy = {'ctrl': np.zeros((2, 1)), 'dstb': np.zeros((5, 1))}
    while near_failure and (cnt <= 10):
      env.reset()
      state = env.state.copy()
      cons_dict = env.get_constraints(state, a_dummy, state)
      constraint_values = None
      for key, value in cons_dict.items():
        if constraint_values is None:
          num_pts = value.shape[1]
          constraint_values = value
        else:
          assert num_pts == value.shape[1], ("The length of constraint ({}) do not match".format(key))
          constraint_values = np.concatenate((constraint_values, value), axis=0)
      g_x = np.min(constraint_values[:, -1], axis=0)
      near_failure = g_x < 0.1
      cnt += 1
    for vel in vel_list:
      for yaw in yaw_list:
        state[2] = vel
        state[3] = yaw
        reset_kwargs_list.append(dict(state=state.copy()))

  trajectories, results, _ = env.simulate_trajectories(
      num_traj, T_rollout=T_rollout, end_criterion=end_criterion, reset_kwargs_list=reset_kwargs_list,
      adversary=adversary
  )
  return trajectories, results


def visualize_dstbEnv(
    env: RaceCarDstb5DEnv,
    value_fn: Callable[[np.ndarray, Optional[Union[np.ndarray, torch.Tensor]]], np.ndarray],
    adversary: BasePolicy,
    fig_path: str,
    vel_list: List[float],
    yaw_list: List[float],
    num_pts: int = 5,
    T_rollout: int = 150,
    end_criterion: str = "failure",
    subfigsz_x: float = 4.,
    subfigsz_y: float = 4.,
    nx: int = 100,
    ny: int = 100,
    batch_size: int = 512,
    cmap: str = 'seismic_r',
    vmin: float = -.25,
    vmax: float = .25,
    alpha: float = 0.5,
    fontsize: int = 16,
    vel_scatter: bool = False,
    markersz: int = 40,
):
  n_row = len(yaw_list)
  n_col = len(vel_list)
  figsize = (subfigsz_x * n_col, subfigsz_y * n_row)
  fig, axes = plt.subplots(n_row, n_col, figsize=figsize, sharex=True, sharey=True)
  vmin_label = vmin
  vmax_label = vmax
  vmean_label = 0

  xs, ys = env.get_samples(nx, ny)
  trajectories, results = get_trajectories_dstbEnv(
      env=env, adversary=adversary, vel_list=vel_list, yaw_list=yaw_list, num_pts=num_pts, T_rollout=T_rollout,
      end_criterion=end_criterion
  )

  for i, vel in enumerate(vel_list):
    axes[0][i].set_title(f"Vel: {vel:.2f}", fontsize=fontsize)
    for j, yaw in enumerate(yaw_list):
      ax = axes[j][i]
      if i == 0:
        ax.set_ylabel(f"Yaw: {yaw/np.pi*180:.0f}", fontsize=fontsize)
      # plots value function
      values = get_values(env, value_fn, xs, ys, vel, yaw, batch_size=batch_size, fail_value=vmin)
      im = ax.imshow(
          values.T, interpolation='none', extent=env.visual_extent, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
          zorder=-1, alpha=alpha
      )

      # Plots trajectories
      for k in range(num_pts):
        idx = int(k * len(yaw_list) * len(vel_list) + i * len(yaw_list) + j)
        trajectory = trajectories[idx]
        result = results[idx]
        plot_traj(ax, trajectory, result, c='g', lw=2., vel_scatter=vel_scatter, zorder=1, s=markersz)

      env.track.plot_track(ax, c='k')
      env.render_obs(ax=ax, c='r')
      ax.axis(env.visual_extent)
      ax.set_xticks(np.around(env.visual_bounds[0], 1))
      ax.set_yticks(np.around(env.visual_bounds[1], 1))
      ax.tick_params(axis='both', which='major', labelsize=fontsize - 4)

  # one color bar
  fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.08, hspace=0.01)
  cbar_ax = fig.add_axes([0.96, 0.05, 0.02, 0.85])
  cbar = fig.colorbar(im, cax=cbar_ax, ax=ax, ticks=[vmin, 0, vmax])
  v_ticklabels = np.around(np.array([vmin_label, vmean_label, vmax_label]), 2)
  cbar.ax.set_yticklabels(labels=v_ticklabels, fontsize=fontsize - 4)
  fig.savefig(fig_path, dpi=400)
  plt.close('all')


# endregion

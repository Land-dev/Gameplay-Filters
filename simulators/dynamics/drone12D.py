# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for 12D Drone pursuit-evasion dynamics.

This file implements a class for 12D drone dynamics in a pursuit-evasion game.
The state is represented by [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z], 
where (p1_x, p1_y, p1_z) is the evader's position, (v1_x, v1_y, v1_z) is the evader's velocity,
(p2_x, p2_y, p2_z) is the pursuer's position, and (v2_x, v2_y, v2_z) is the pursuer's velocity.
The control is [a1_x, a1_y, a1_z] (evader's acceleration), and the disturbance is [a2_x, a2_y, a2_z] (pursuer's acceleration).
"""

from typing import Tuple, Any, Dict, Optional
import numpy as np
from functools import partial
from jax import Array  # modern JAX array type
import jax
from jax import numpy as jnp

from .base_dstb_dynamics import BaseDstbDynamics


class Drone12D(BaseDstbDynamics):

  def __init__(self, cfg: Any, action_space: Dict[str, np.ndarray]) -> None:
    """
    Implements the 12D drone pursuit-evasion dynamics.

    Args:
        cfg (Any): an object specifies configuration.
        action_space (Dict[str, np.ndarray]): action space with 'ctrl' and 'dstb' keys.
    """
    super().__init__(cfg, action_space)
    self.dim_x = 12  # [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z]

    # Load parameters from DronePursuitEvasion12D
    self.input_multiplier: float = getattr(cfg, 'input_multiplier', 16.0)
    self.sideways_multiplier: float = getattr(cfg, 'sideways_multiplier', 2.0)
    self.control_max: float = getattr(cfg, 'control_max', 1.0)
    self.disturbance_max: float = getattr(cfg, 'disturbance_max', 1.0)
    self.k_T: float = getattr(cfg, 'k_T', 0.83)
    self.Gz: float = getattr(cfg, 'Gz', -9.81)
    self.max_v: float = getattr(cfg, 'max_v', 2.0)
    self.capture_radius: float = getattr(cfg, 'goalR', 0.25)  # Use goalR from config
    
    # State bounds from DronePursuitEvasion12D
    self.state_max_x: float = getattr(cfg, 'state_max_x', 4.0)  # Updated to match config
    self.state_max_y: float = getattr(cfg, 'state_max_y', 2.0)
    self.state_max_z: float = getattr(cfg, 'state_max_z', 2.0)
    
    # Box bounds for boundary function
    self.box_bounds = np.array([
        [-4.0, 4.0], [-self.max_v, self.max_v],  # x bounds
        [-2.0, 2.0], [-self.max_v, self.max_v],  # y bounds
        [0.0, 2.0],  [-self.max_v, self.max_v],  # z bounds
    ])
    
    self.dim_u_dstb: int = 3  # dimension of disturbance [a2_x, a2_y, a2_z]
    self.dim_u_ctrl: int = 3  # dimension of control [a1_x, a1_y, a1_z]

  def integrate_forward(
      self, state: np.ndarray, control: np.ndarray,
      noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif',
      adversary: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Override the base method to handle disturbance correctly for Drone12D.
    """
    if adversary is not None:
      assert adversary.shape[0] == self.dim_u_dstb, ("Adversary dim. is incorrect!")
      disturbance = adversary
    elif noise is not None:
      assert noise.shape[0] == self.dim_u_dstb, ("Noise dim. is incorrect!")
      # For Drone12D, disturbance is 3D (pursuer acceleration)
      if noise_type == 'unif':
        rv = (np.random.rand(self.dim_u_dstb) - 0.5) * 2  # Maps to [-1, 1]
      else:
        rv = np.random.normal(size=(self.dim_u_dstb))
      disturbance = noise * rv
    else:
      disturbance = np.zeros(self.dim_u_dstb)
    
    state_nxt, ctrl_clip, dstb_clip = self.integrate_forward_jax(
        jnp.array(state), jnp.array(control), jnp.array(disturbance)
    )
    return np.array(state_nxt), np.array(ctrl_clip), np.array(dstb_clip)

  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: Array, control: Array, disturbance: Array
  ) -> Tuple[Array, Array, Array]:
    """Clips the control and disturbance and computes one-step time evolution
    of the system.

    Args:
        state (Array): [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z].
        control (Array): [a1_x, a1_y, a1_z] (evader's acceleration).
        disturbance (Array): [a2_x, a2_y, a2_z] (pursuer's acceleration).

    Returns:
        Array: next state.
        Array: clipped control.
        Array: clipped disturbance.
    """
    # Clips the controller and disturbance values
    ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])
    dstb_clip = jnp.clip(
        disturbance, self.dstb_space[:, 0], self.dstb_space[:, 1]
    )

    # Compute next state using RK4 integration
    state_nxt = self._integrate_forward(state, ctrl_clip, dstb_clip)
    
    # Clip velocities to maximum speed
    state_nxt = state_nxt.at[1].set(jnp.clip(state_nxt[1], -self.max_v, self.max_v))  # v1_x
    state_nxt = state_nxt.at[3].set(jnp.clip(state_nxt[3], -self.max_v, self.max_v))  # v1_y
    state_nxt = state_nxt.at[5].set(jnp.clip(state_nxt[5], -self.max_v, self.max_v))  # v1_z
    state_nxt = state_nxt.at[7].set(jnp.clip(state_nxt[7], -self.max_v, self.max_v))  # v2_x
    state_nxt = state_nxt.at[9].set(jnp.clip(state_nxt[9], -self.max_v, self.max_v))  # v2_y
    state_nxt = state_nxt.at[11].set(jnp.clip(state_nxt[11], -self.max_v, self.max_v))  # v2_z
    
    return state_nxt, ctrl_clip, dstb_clip

  @partial(jax.jit, static_argnames='self')
  def disc_deriv(
      self, state: Array, control: Array, disturbance: Array
  ) -> Array:
    """Computes the continuous-time derivatives of the drone dynamics.
    
    Args:
        state (Array): [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z].
        control (Array): [a1_x, a1_y, a1_z] (evader's acceleration).
        disturbance (Array): [a2_x, a2_y, a2_z] (pursuer's acceleration).
        
    Returns:
        Array: derivatives [dp1_x/dt, dv1_x/dt, dp1_y/dt, dv1_y/dt, dp1_z/dt, dv1_z/dt, 
                          dp2_x/dt, dv2_x/dt, dp2_y/dt, dv2_y/dt, dp2_z/dt, dv2_z/dt].
    """
    # Extract state components
    p1_x, v1_x, p1_y, v1_y, p1_z, v1_z = state[0], state[1], state[2], state[3], state[4], state[5]
    p2_x, v2_x, p2_y, v2_y, p2_z, v2_z = state[6], state[7], state[8], state[9], state[10], state[11]
    
    # Extract control and disturbance
    a1_x, a1_y, a1_z = control[0], control[1], control[2]  # evader's acceleration
    a2_x, a2_y, a2_z = disturbance[0], disturbance[1], disturbance[2]  # pursuer's acceleration
    
    # Drone dynamics
    deriv = jnp.zeros((self.dim_x,))
    
    # Drone 1 (evader) dynamics
    deriv = deriv.at[1].set(self.sideways_multiplier * a1_x)  # v1_x_dot
    deriv = deriv.at[3].set(self.sideways_multiplier * a1_y)  # v1_y_dot
    deriv = deriv.at[5].set(self.k_T * self.input_multiplier * a1_z + self.Gz)  # v1_z_dot
    
    # Drone 2 (pursuer) dynamics
    deriv = deriv.at[7].set(self.sideways_multiplier * a2_x)  # v2_x_dot
    deriv = deriv.at[9].set(self.sideways_multiplier * a2_y)  # v2_y_dot
    deriv = deriv.at[11].set(self.k_T * self.input_multiplier * a2_z + self.Gz)  # v2_z_dot
    
    # Position dynamics
    deriv = deriv.at[0].set(v1_x)   # p1_x_dot = v1_x
    deriv = deriv.at[2].set(v1_y)   # p1_y_dot = v1_y
    deriv = deriv.at[4].set(v1_z)   # p1_z_dot = v1_z
    deriv = deriv.at[6].set(v2_x)   # p2_x_dot = v2_x
    deriv = deriv.at[8].set(v2_y)   # p2_y_dot = v2_y
    deriv = deriv.at[10].set(v2_z)  # p2_z_dot = v2_z
    
    return deriv

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward(
      self, state: Array, control: Array, disturbance: Array
  ) -> Array:
    """
    Computes one-step time evolution of the system using RK4 integration.
    
    Args:
        state (Array): [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z].
        control (Array): [a1_x, a1_y, a1_z] (evader's acceleration).
        disturbance (Array): [a2_x, a2_y, a2_z] (pursuer's acceleration).

    Returns:
        Array: next state.
    """
    return self._integrate_forward_dt(state, control, disturbance, self.dt)

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward_dt(
      self, state: Array, control: Array, disturbance: Array,
      dt: float
  ) -> Array:
    """RK4 integration for drone dynamics.
    
    Args:
        state (Array): current state.
        control (Array): control input.
        disturbance (Array): disturbance input.
        dt (float): time step.
        
    Returns:
        Array: next state.
    """
    k1 = self.disc_deriv(state, control, disturbance)
    k2 = self.disc_deriv(state + k1*dt/2, control, disturbance)
    k3 = self.disc_deriv(state + k2*dt/2, control, disturbance)
    k4 = self.disc_deriv(state + k3*dt, control, disturbance)
    return state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6

  def boundary_fn(self, state: np.ndarray) -> float:
    """Compute boundary function for collision detection and bounds checking.
    
    This implements the same logic as DronePursuitEvasion12D.boundary_fn but adapted for numpy.
    
    Args:
        state (np.ndarray): [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z].
        
    Returns:
        float: Boundary function value (positive when safe, negative when unsafe).
    """
    # Extract positions
    p1 = np.array([state[0], state[2], state[4]])  # Drone 1 position
    p2 = np.array([state[6], state[8], state[10]])  # Drone 2 position
    
    height = 0.75

    # Smooth SDF for a truncated cone:
    # - Apex at z = 0.5 above pursuer (virtual apex)
    # - Truncated at z = 0.25 above pursuer (top cap)
    # - Base at z = -height
    # Negative inside, positive outside.
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]

    horizontal_dist = np.sqrt(dx**2 + dy**2 + 1e-8)

    # Linear radius shrink for cone (apex at z = 0.5)
    # At z = 0.5: radius = 0 (apex)
    # At z = 0.25: radius = R * 0.25 / (height + 0.25)
    # At z = -height: radius = R * (height + 0.5) / (height + 0.25)
    cone_radius = self.capture_radius * (0.5 - dz) / (height + 0.25)

    # Signed distance to lateral cone surface (negative inside)
    d_lateral = horizontal_dist - cone_radius

    # SDF for top plane (z <= 0.25) - truncation plane
    d_top = dz - 0.25  # positive above truncation plane

    # SDF for bottom plane (z >= -height)
    d_bottom = -(dz + height)  # positive below base

    # Combine using smooth max for outside
    # (soft union: distance = max(d_lateral, d_top, d_bottom))
    sharpness = 16.0
    m = max(max(d_lateral, d_top), d_bottom)
    inter_drone_dist = m + np.log(
        np.exp((d_lateral - m) * sharpness) +
        np.exp((d_top - m) * sharpness) +
        np.exp((d_bottom - m) * sharpness)
    ) / sharpness

    capture_constraint = inter_drone_dist

    # For each dimension, how far from the nearest boundary (positive inside, negative outside)
    px, py, pz = state[0], state[2], state[4]

    # Box bounds
    x_min, x_max = self.box_bounds[0, 0], self.box_bounds[0, 1]
    y_min, y_max = self.box_bounds[2, 0], self.box_bounds[2, 1]
    z_min, z_max = self.box_bounds[4, 0], self.box_bounds[4, 1]

    # Compute per-dimension signed distances to box faces
    dx_min = px - x_min
    dx_max = x_max - px
    dy_min = py - y_min
    dy_max = y_max - py
    dz_min = pz - z_min
    dz_max = z_max - pz

    # Inside: minimum distance to any face (negative inside, zero on surface)
    inside_dist = min(dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)

    # For outside: compute the per-dimension "over" (how far outside the box in each dim)
    over_x = max(0, px - x_max) + max(0, x_min - px)
    over_y = max(0, py - y_max) + max(0, y_min - py)
    over_z = max(0, pz - z_max) + max(0, z_min - pz)
    # Norm of the "over" vector gives Euclidean distance outside
    outside_dist = np.sqrt(over_x**2 + over_y**2 + over_z**2)

    # If all inside (all distances to faces > 0), use inside_dist; else use outside_dist
    is_inside = (dx_min > 0) and (dx_max > 0) and (dy_min > 0) and (dy_max > 0) and (dz_min > 0) and (dz_max > 0)
    inside_constraint = inside_dist if is_inside else -outside_dist
    
    # Safe if outside capture radius AND inside bounds
    return min(capture_constraint, inside_constraint)

  def get_evader_position(self, state: np.ndarray) -> np.ndarray:
    """Get evader's 3D position.
    
    Args:
        state (np.ndarray): [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z].
        
    Returns:
        np.ndarray: [p1_x, p1_y, p1_z] evader's position.
    """
    return np.array([state[0], state[2], state[4]])

  def get_pursuer_position(self, state: np.ndarray) -> np.ndarray:
    """Get pursuer's 3D position.
    
    Args:
        state (np.ndarray): [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z].
        
    Returns:
        np.ndarray: [p2_x, p2_y, p2_z] pursuer's position.
    """
    return np.array([state[6], state[8], state[10]])

  def get_evader_velocity(self, state: np.ndarray) -> np.ndarray:
    """Get evader's 3D velocity.
    
    Args:
        state (np.ndarray): [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z].
        
    Returns:
        np.ndarray: [v1_x, v1_y, v1_z] evader's velocity.
    """
    return np.array([state[1], state[3], state[5]])

  def get_pursuer_velocity(self, state: np.ndarray) -> np.ndarray:
    """Get pursuer's 3D velocity.
    
    Args:
        state (np.ndarray): [p1_x, v1_x, p1_y, v1_y, p1_z, v1_z, p2_x, v2_x, p2_y, v2_y, p2_z, v2_z].
        
    Returns:
        np.ndarray: [v2_x, v2_y, v2_z] pursuer's velocity.
    """
    return np.array([state[7], state[9], state[11]])

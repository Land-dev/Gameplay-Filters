# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for 20D Drone pursuit-evasion dynamics.

This file implements a class for 20D drone dynamics in a pursuit-evasion game.
The state is represented by [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z], 
where (x1, y1, z1) is the evader's position, (v1_x, v1_y, v1_z) is the evader's velocity,
(θ1_x, θ1_y) are the evader's angles, (ω1_x, ω1_y) are the evader's angular velocities,
(x2, y2, z2) is the pursuer's position, (v2_x, v2_y, v2_z) is the pursuer's velocity,
(θ2_x, θ2_y) are the pursuer's angles, and (ω2_x, ω2_y) are the pursuer's angular velocities.
The control is [S1_x, S1_y, T1_z] (evader's torque and thrust), and the disturbance is [S2_x, S2_y, T2_z] (pursuer's torque and thrust).
"""

from typing import Tuple, Any, Dict, Optional
import numpy as np
from functools import partial
from jax import Array  # modern JAX array type
import jax
from jax import numpy as jnp

from .base_dstb_dynamics import BaseDstbDynamics


class Drone20D(BaseDstbDynamics):

  def __init__(self, cfg: Any, action_space: Dict[str, np.ndarray]) -> None:
    """
    Implements the 20D drone pursuit-evasion dynamics.

    Args:
        cfg (Any): an object specifies configuration.
        action_space (Dict[str, np.ndarray]): action space with 'ctrl' and 'dstb' keys.
    """
    super().__init__(cfg, action_space)
    self.dim_x = 20  # [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                      #  x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z]

    # Load parameters from DronePursuitEvasion20D
    self.thrust_max: float = getattr(cfg, 'thrust_max', 16.0)
    self.max_angle: float = getattr(cfg, 'max_angle', 0.3)
    self.max_torque: float = getattr(cfg, 'max_torque', 0.3)
    self.control_max: float = getattr(cfg, 'control_max', 1.0)
    self.disturbance_max: float = getattr(cfg, 'disturbance_max', 1.0)
    self.k_T: float = getattr(cfg, 'k_T', 0.83)
    self.Gz: float = getattr(cfg, 'Gz', -9.81)
    self.max_v: float = getattr(cfg, 'max_v', 2.0)
    self.max_omega: float = getattr(cfg, 'max_omega', 2.0)
    self.capture_radius: float = getattr(cfg, 'goalR', 0.25)
    
    # Drone dynamics parameters
    self.d0: float = getattr(cfg, 'd0', 20.0)
    self.d1: float = getattr(cfg, 'd1', 4.5)
    self.n0: float = getattr(cfg, 'n0', 18.0)
    self.mass: float = getattr(cfg, 'mass', 1.0)
    self.c_x: float = getattr(cfg, 'c_x', 0.3)  # Drag coefficient for x direction
    self.c_y: float = getattr(cfg, 'c_y', 0.3)  # Drag coefficient for y direction
    
    # State bounds from DronePursuitEvasion20D
    self.state_max_x: float = getattr(cfg, 'state_max_x', 4.0)
    self.state_max_y: float = getattr(cfg, 'state_max_y', 2.0)
    self.state_max_z: float = getattr(cfg, 'state_max_z', 2.0)
    
    # Box bounds for boundary function
    self.box_bounds = np.array([
        [-4.0, 4.0], [-self.max_v, self.max_v],  # x bounds
        [-2.0, 2.0], [-self.max_v, self.max_v],  # y bounds
        [0.0, 2.0],  [-self.max_v, self.max_v],  # z bounds
    ])
    
    self.dim_u_dstb: int = 3  # dimension of disturbance [S2_x, S2_y, T2_z]
    self.dim_u_ctrl: int = 3  # dimension of control [S1_x, S1_y, T1_z]

  def integrate_forward(
      self, state: np.ndarray, control: np.ndarray,
      noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif',
      adversary: Optional[np.ndarray] = None, **kwargs
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Override the base method to handle disturbance correctly for Drone20D.
    """
    if adversary is not None:
      assert adversary.shape[0] == self.dim_u_dstb, ("Adversary dim. is incorrect!")
      disturbance = adversary
    elif noise is not None:
      assert noise.shape[0] == self.dim_u_dstb, ("Noise dim. is incorrect!")
      # For Drone20D, disturbance is 3D (pursuer control)
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
        state (Array): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                       x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        control (Array): [S1_x, S1_y, T1_z] (evader's torque and thrust).
        disturbance (Array): [S2_x, S2_y, T2_z] (pursuer's torque and thrust).

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
    state_nxt = state_nxt.at[5].set(jnp.clip(state_nxt[5], -self.max_v, self.max_v))  # v1_y
    state_nxt = state_nxt.at[9].set(jnp.clip(state_nxt[9], -self.max_v, self.max_v))  # v1_z
    state_nxt = state_nxt.at[11].set(jnp.clip(state_nxt[11], -self.max_v, self.max_v))  # v2_x
    state_nxt = state_nxt.at[15].set(jnp.clip(state_nxt[15], -self.max_v, self.max_v))  # v2_y
    state_nxt = state_nxt.at[19].set(jnp.clip(state_nxt[19], -self.max_v, self.max_v))  # v2_z
    
    # Clip angles to maximum angle
    state_nxt = state_nxt.at[2].set(jnp.clip(state_nxt[2], -self.max_angle, self.max_angle))  # θ1_x
    state_nxt = state_nxt.at[6].set(jnp.clip(state_nxt[6], -self.max_angle, self.max_angle))  # θ1_y
    state_nxt = state_nxt.at[12].set(jnp.clip(state_nxt[12], -self.max_angle, self.max_angle))  # θ2_x
    state_nxt = state_nxt.at[16].set(jnp.clip(state_nxt[16], -self.max_angle, self.max_angle))  # θ2_y
    
    # Clip angular velocities to maximum angular velocity
    state_nxt = state_nxt.at[3].set(jnp.clip(state_nxt[3], -self.max_omega, self.max_omega))  # ω1_x
    state_nxt = state_nxt.at[7].set(jnp.clip(state_nxt[7], -self.max_omega, self.max_omega))  # ω1_y
    state_nxt = state_nxt.at[13].set(jnp.clip(state_nxt[13], -self.max_omega, self.max_omega))  # ω2_x
    state_nxt = state_nxt.at[17].set(jnp.clip(state_nxt[17], -self.max_omega, self.max_omega))  # ω2_y
    
    return state_nxt, ctrl_clip, dstb_clip

  @partial(jax.jit, static_argnames='self')
  def disc_deriv(
      self, state: Array, control: Array, disturbance: Array
  ) -> Array:
    """Computes the continuous-time derivatives of the drone dynamics.
    
    Args:
        state (Array): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                       x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        control (Array): [S1_x, S1_y, T1_z] (evader's torque and thrust).
        disturbance (Array): [S2_x, S2_y, T2_z] (pursuer's torque and thrust).
        
    Returns:
        Array: derivatives [dx1/dt, dv1_x/dt, dθ1_x/dt, dω1_x/dt, dy1/dt, dv1_y/dt, dθ1_y/dt, dω1_y/dt, dz1/dt, dv1_z/dt, 
                          dx2/dt, dv2_x/dt, dθ2_x/dt, dω2_x/dt, dy2/dt, dv2_y/dt, dθ2_y/dt, dω2_y/dt, dz2/dt, dv2_z/dt].
    """
    # Extract state components for drone 1 (evader)
    x1, v1_x, theta1_x, omega1_x = state[0], state[1], state[2], state[3]
    y1, v1_y, theta1_y, omega1_y = state[4], state[5], state[6], state[7]
    z1, v1_z = state[8], state[9]
    
    # Extract state components for drone 2 (pursuer)
    x2, v2_x, theta2_x, omega2_x = state[10], state[11], state[12], state[13]
    y2, v2_y, theta2_y, omega2_y = state[14], state[15], state[16], state[17]
    z2, v2_z = state[18], state[19]
    
    # Extract control and disturbance
    S1_x, S1_y, T1_z = control[0], control[1], control[2]  # evader's torque and thrust
    S2_x, S2_y, T2_z = disturbance[0], disturbance[1], disturbance[2]  # pursuer's torque and thrust
    
    # Drone dynamics
    deriv = jnp.zeros((self.dim_x,))
    
    # Drone 1 (evader) dynamics - indices 0-9
    # Position derivatives
    deriv = deriv.at[0].set(v1_x)   # x1_dot = v1_x
    deriv = deriv.at[4].set(v1_y)   # y1_dot = v1_y
    deriv = deriv.at[8].set(v1_z)   # z1_dot = v1_z
    
    # Velocity derivatives (with evader control, disturbance, and drag terms)
    deriv = deriv.at[1].set(-self.Gz * jnp.tan(theta1_x) - self.c_x * v1_x)  # v1_x_dot = -g * tan(θ1_x) - c_x * v1_x
    deriv = deriv.at[5].set(-self.Gz * jnp.tan(theta1_y) - self.c_y * v1_y)  # v1_y_dot = -g * tan(θ1_y) - c_y * v1_y
    deriv = deriv.at[9].set(self.k_T / self.mass * self.thrust_max * T1_z + self.Gz)  # v1_z_dot = T1_z - g
    
    # Angle derivatives
    deriv = deriv.at[2].set(omega1_x - self.d1 * theta1_x)  # θ1_x_dot = ω1_x - d1 * θ1_x
    deriv = deriv.at[6].set(omega1_y - self.d1 * theta1_y)  # θ1_y_dot = ω1_y - d1 * θ1_y
    
    # Angular velocity derivatives
    deriv = deriv.at[3].set(-self.d0 * theta1_x + self.n0 * self.max_torque * S1_x)  # ω1_x_dot
    deriv = deriv.at[7].set(-self.d0 * theta1_y + self.n0 * self.max_torque * S1_y)  # ω1_y_dot
    
    # Drone 2 (pursuer) dynamics - indices 10-19
    # Position derivatives
    deriv = deriv.at[10].set(v2_x)  # x2_dot = v2_x
    deriv = deriv.at[14].set(v2_y)  # y2_dot = v2_y
    deriv = deriv.at[18].set(v2_z)  # z2_dot = v2_z
    
    # Velocity derivatives (with pursuer control, disturbance, and drag terms)
    deriv = deriv.at[11].set(-self.Gz * jnp.tan(theta2_x) - self.c_x * v2_x)  # v2_x_dot = -g * tan(θ2_x) - c_x * v2_x
    deriv = deriv.at[15].set(-self.Gz * jnp.tan(theta2_y) - self.c_y * v2_y)  # v2_y_dot = -g * tan(θ2_y) - c_y * v2_y
    deriv = deriv.at[19].set(self.k_T / self.mass * self.thrust_max * T2_z + self.Gz)  # v2_z_dot = T2_z - g
    
    # Angle derivatives
    deriv = deriv.at[12].set(omega2_x - self.d1 * theta2_x)  # θ2_x_dot = ω2_x - d1 * θ2_x
    deriv = deriv.at[16].set(omega2_y - self.d1 * theta2_y)  # θ2_y_dot = ω2_y - d1 * θ2_y
    
    # Angular velocity derivatives
    deriv = deriv.at[13].set(-self.d0 * theta2_x + self.n0 * self.max_torque * S2_x)  # ω2_x_dot
    deriv = deriv.at[17].set(-self.d0 * theta2_y + self.n0 * self.max_torque * S2_y)  # ω2_y_dot
    
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
    
    This implements the same logic as DronePursuitEvasion20D.boundary_fn but adapted for numpy.
    
    Args:
        state (np.ndarray): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        
    Returns:
        float: Boundary function value (positive when safe, negative when unsafe).
    """
    # Extract positions
    p1 = np.array([state[0], state[4], state[8]])  # Drone 1 position [x1, y1, z1]
    p2 = np.array([state[10], state[14], state[18]])  # Drone 2 position [x2, y2, z2]
    
    height = 0.75

    # Use ellipse capture shape
    horizontal_radius = self.capture_radius  # a
    vertical_radius = height                 # c

    # Relative position
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]

    # Euclidean distance to pursuer (for points above)
    dist_center = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)

    # Approximate ellipsoid SDF (first-order, smooth)
    inv_a2 = 1.0 / (horizontal_radius * horizontal_radius)
    inv_c2 = 1.0 / (vertical_radius * vertical_radius)
    F = (dx * dx) * inv_a2 + (dy * dy) * inv_a2 + (dz * dz) * inv_c2 - 1.0
    G = np.sqrt((dx * inv_a2) ** 2 + (dy * inv_a2) ** 2 + (dz * inv_c2) ** 2 + 1e-8) * 2.0
    d_ellip = F / (G + 1e-8)  # approximate signed distance to ellipsoid

    d_plane = dz - 0.50  # Cut off at z = 0.5 above pursuer
    
    m = max(d_ellip, d_plane)
    sharpness = 8.0
    signed_dist = m + np.log(
        np.exp((d_ellip - m) * sharpness) +
        np.exp((d_plane - m) * sharpness)
    ) / sharpness
    
    above_factor = 1.0 / (1.0 + np.exp(-dz * 10.0))  # sigmoid equivalent
    inter_drone_dist = above_factor * dist_center + (1 - above_factor) * signed_dist

    capture_constraint = inter_drone_dist

    # For each dimension, how far from the nearest boundary (positive inside, negative outside)
    px, py, pz = state[0], state[4], state[8]  # x1, y1, z1

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
        state (np.ndarray): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        
    Returns:
        np.ndarray: [x1, y1, z1] evader's position.
    """
    return np.array([state[0], state[4], state[8]])

  def get_pursuer_position(self, state: np.ndarray) -> np.ndarray:
    """Get pursuer's 3D position.
    
    Args:
        state (np.ndarray): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        
    Returns:
        np.ndarray: [x2, y2, z2] pursuer's position.
    """
    return np.array([state[10], state[14], state[18]])

  def get_evader_velocity(self, state: np.ndarray) -> np.ndarray:
    """Get evader's 3D velocity.
    
    Args:
        state (np.ndarray): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        
    Returns:
        np.ndarray: [v1_x, v1_y, v1_z] evader's velocity.
    """
    return np.array([state[1], state[5], state[9]])

  def get_pursuer_velocity(self, state: np.ndarray) -> np.ndarray:
    """Get pursuer's 3D velocity.
    
    Args:
        state (np.ndarray): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        
    Returns:
        np.ndarray: [v2_x, v2_y, v2_z] pursuer's velocity.
    """
    return np.array([state[11], state[15], state[19]])

  def get_evader_angles(self, state: np.ndarray) -> np.ndarray:
    """Get evader's 2D angles.
    
    Args:
        state (np.ndarray): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        
    Returns:
        np.ndarray: [θ1_x, θ1_y] evader's angles.
    """
    return np.array([state[2], state[6]])

  def get_pursuer_angles(self, state: np.ndarray) -> np.ndarray:
    """Get pursuer's 2D angles.
    
    Args:
        state (np.ndarray): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        
    Returns:
        np.ndarray: [θ2_x, θ2_y] pursuer's angles.
    """
    return np.array([state[12], state[16]])

  def get_evader_angular_velocities(self, state: np.ndarray) -> np.ndarray:
    """Get evader's 2D angular velocities.
    
    Args:
        state (np.ndarray): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        
    Returns:
        np.ndarray: [ω1_x, ω1_y] evader's angular velocities.
    """
    return np.array([state[3], state[7]])

  def get_pursuer_angular_velocities(self, state: np.ndarray) -> np.ndarray:
    """Get pursuer's 2D angular velocities.
    
    Args:
        state (np.ndarray): [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                           x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z].
        
    Returns:
        np.ndarray: [ω2_x, ω2_y] pursuer's angular velocities.
    """
    return np.array([state[13], state[17]])

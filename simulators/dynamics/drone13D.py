# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for 13D Drone dynamics with disturbance.

This file implements a class for 13D drone dynamics with disturbance handling.
The state is represented by [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz], 
where (x, y, z) is the position, (qw, qx, qy, qz) is the quaternion orientation,
(vx, vy, vz) is the velocity, and (wx, wy, wz) is the angular velocity.
The control is [f, wx_cmd, wy_cmd, wz_cmd] (collective thrust and angular velocity commands),
and the disturbance is [fx_dist, fy_dist, fz_dist, tx_dist, ty_dist, tz_dist] (wind forces and torques).
"""

from typing import Tuple, Any, Dict, Optional
import numpy as np
from functools import partial
from jax import Array  # modern JAX array type
import jax
from jax import numpy as jnp

from .base_dstb_dynamics import BaseDstbDynamics


class Drone13D(BaseDstbDynamics):

    def __init__(self, cfg: Any, action_space: Dict[str, np.ndarray]) -> None:
        """
        Implements the 13D drone dynamics with disturbance.

        Args:
            cfg (Any): an object specifies configuration.
            action_space (Dict[str, np.ndarray]): action space with 'ctrl' and 'dstb' keys.
        """
        super().__init__(cfg, action_space)
        self.dim_x = 13  # [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]

        # Load parameters from config
        self.collisionR: float = getattr(cfg, 'collisionR', 0.50)
        self.collective_thrust_max: float = getattr(cfg, 'collective_thrust_max', 20.0)
        self.disturbance_max: float = getattr(cfg, 'disturbance_max', 1.0)
        
        # Physical parameters
        self.m: float = 1.0  # mass
        self.arm_l: float = 0.17  # arm length
        self.CT: float = 1.0  # thrust coefficient
        self.CM: float = 0.016  # moment coefficient
        self.Gz: float = -9.8  # gravity
        
        # Control limits
        self.dwx_max: float = 8.0
        self.dwy_max: float = 8.0
        self.dwz_max: float = 4.0
        
        # Mode for reach-avoid (no weights - equal importance)
        self.set_mode: str = getattr(cfg, 'set_mode', 'avoid')  # 'avoid', 'reach', or 'reach_avoid'
        
        # State bounds
        self.state_max_x: float = getattr(cfg, 'state_max_x', 3.0)
        self.state_max_y: float = getattr(cfg, 'state_max_y', 3.0)
        self.state_max_z: float = getattr(cfg, 'state_max_z', 3.0)
        self.state_max_v: float = getattr(cfg, 'state_max_v', 5.0)
        self.state_max_w: float = getattr(cfg, 'state_max_w', 5.0)
        self.state_max_q: float = getattr(cfg, 'state_max_q', 1.0)
        
        # Box bounds for boundary function
        self.box_bounds = np.array([
            [-self.state_max_x, self.state_max_x],  # x bounds
            [-self.state_max_y, self.state_max_y],  # y bounds
            [-self.state_max_z, self.state_max_z],  # z bounds
        ])
        
        self.dim_u_dstb: int = 6  # dimension of disturbance [fx, fy, fz, tx, ty, tz]
        self.dim_u_ctrl: int = 4  # dimension of control [f, wx_cmd, wy_cmd, wz_cmd]

    def integrate_forward(
        self, state: np.ndarray, control: np.ndarray,
        noise: Optional[np.ndarray] = None, noise_type: Optional[str] = 'unif',
        adversary: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Override the base method to handle disturbance correctly for Drone13D.
        """
        if adversary is not None:
            assert adversary.shape[0] == self.dim_u_dstb, ("Adversary dim. is incorrect!")
            disturbance = adversary
        elif noise is not None:
            assert noise.shape[0] == self.dim_u_dstb, ("Noise dim. is incorrect!")
            # For Drone13D, disturbance is 6D (wind forces and torques)
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
            state (Array): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
            control (Array): [f, wx_cmd, wy_cmd, wz_cmd] (collective thrust and angular velocity commands).
            disturbance (Array): [fx, fy, fz, tx, ty, tz] (wind forces and torques).

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
        
        # Clip velocities and quaternions to bounds
        state_nxt = state_nxt.at[7:10].set(jnp.clip(state_nxt[7:10], -self.state_max_v, self.state_max_v))  # vx, vy, vz
        state_nxt = state_nxt.at[10:13].set(jnp.clip(state_nxt[10:13], -self.state_max_w, self.state_max_w))  # wx, wy, wz
        state_nxt = state_nxt.at[3:7].set(jnp.clip(state_nxt[3:7], -self.state_max_q, self.state_max_q))  # qw, qx, qy, qz
        
        # Normalize quaternion
        q_norm = jnp.linalg.norm(state_nxt[3:7])
        state_nxt = state_nxt.at[3:7].set(state_nxt[3:7] / (q_norm + 1e-8))
        
        return state_nxt, ctrl_clip, dstb_clip

    @partial(jax.jit, static_argnames='self')
    def disc_deriv(
        self, state: Array, control: Array, disturbance: Array
    ) -> Array:
        """Computes the continuous-time derivatives of the drone dynamics.
        
        Args:
            state (Array): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
            control (Array): [f, wx_cmd, wy_cmd, wz_cmd] (collective thrust and angular velocity commands).
            disturbance (Array): [fx, fy, fz, tx, ty, tz] (wind forces and torques).
            
        Returns:
            Array: derivatives [dx/dt, dy/dt, dz/dt, dqw/dt, dqx/dt, dqy/dt, dqz/dt, 
                              dvx/dt, dvy/dt, dvz/dt, dwx/dt, dwy/dt, dwz/dt].
        """
        # Extract state components
        x, y, z = state[0], state[1], state[2]
        qw, qx, qy, qz = state[3], state[4], state[5], state[6]
        vx, vy, vz = state[7], state[8], state[9]
        wx, wy, wz = state[10], state[11], state[12]
        
        # Extract control and disturbance
        f, wx_cmd, wy_cmd, wz_cmd = control[0], control[1], control[2], control[3]  # control inputs
        fx_dist, fy_dist, fz_dist, tx_dist, ty_dist, tz_dist = disturbance[0], disturbance[1], disturbance[2], disturbance[3], disturbance[4], disturbance[5]  # disturbance inputs
        
        # Drone dynamics
        deriv = jnp.zeros((self.dim_x,))
        
        # Position dynamics
        deriv = deriv.at[0].set(vx)   # dx/dt = vx
        deriv = deriv.at[1].set(vy)   # dy/dt = vy
        deriv = deriv.at[2].set(vz)   # dz/dt = vz
        
        # Quaternion dynamics
        deriv = deriv.at[3].set(-(wx * qx + wy * qy + wz * qz) / 2.0)  # dqw/dt
        deriv = deriv.at[4].set((wx * qw + wz * qy - wy * qz) / 2.0)   # dqx/dt
        deriv = deriv.at[5].set((wy * qw - wz * qx + wx * qz) / 2.0)   # dqy/dt
        deriv = deriv.at[6].set((wz * qw + wy * qx - wx * qy) / 2.0)   # dqz/dt
        
        # Velocity dynamics (with thrust and wind forces)
        deriv = deriv.at[7].set(2 * (qw * qy + qx * qz) * self.CT / self.m * f + fx_dist)  # dvx/dt
        deriv = deriv.at[8].set(2 * (-qw * qx + qy * qz) * self.CT / self.m * f + fy_dist)  # dvy/dt
        deriv = deriv.at[9].set(self.Gz + (1 - 2 * jnp.power(qx, 2) - 2 * jnp.power(qy, 2)) * self.CT / self.m * f + fz_dist)  # dvz/dt
        
        # Angular velocity dynamics (with control and wind torques)
        deriv = deriv.at[10].set(wx_cmd - 5 * wy * wz / 9.0 + tx_dist)  # dwx/dt
        deriv = deriv.at[11].set(wy_cmd + 5 * wx * wz / 9.0 + ty_dist)  # dwy/dt
        deriv = deriv.at[12].set(wz_cmd + tz_dist)  # dwz/dt
        
        return deriv

    @partial(jax.jit, static_argnames='self')
    def _integrate_forward(
        self, state: Array, control: Array, disturbance: Array
    ) -> Array:
        """
        Computes one-step time evolution of the system using RK4 integration.
        
        Args:
            state (Array): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
            control (Array): [f, wx_cmd, wy_cmd, wz_cmd] (collective thrust and angular velocity commands).
            disturbance (Array): [fx, fy, fz, tx, ty, tz] (wind forces and torques).

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
        
        Args:
            state (np.ndarray): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
            
        Returns:
            float: Boundary function value (positive when safe, negative when unsafe).
        """
        if self.set_mode == 'avoid':
            # Pure avoidance mode - only check safety constraints
            return self._avoid_fn(state)
        elif self.set_mode == 'reach':
            # Pure reach mode - only check reach function
            return self._reach_fn(state)
        elif self.set_mode == 'reach_avoid':
            # Reach-avoid mode - combine reach and avoid equally
            reach_val = self._reach_fn(state)
            avoid_val = self._avoid_fn(state)
            return max(reach_val, -avoid_val)
        else:
            raise ValueError(f"Unknown set_mode: {self.set_mode}")

    def _reach_fn(self, state: np.ndarray) -> float:
        """Reach function - positive when close to target, negative when far.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            float: Reach function value
        """
        # Extract positions
        x, y = state[0], state[1]
        
        # Target is at origin (0, 0) - distance to target
        dist_to_target = np.sqrt(x**2 + y**2)
        target_radius = 0.3  # Desired proximity to target
        
        # Positive when close to target, negative when far
        return target_radius - dist_to_target

    def _avoid_fn(self, state: np.ndarray) -> float:
        """Avoid function - positive when safe, negative when unsafe.
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            float: Avoid function value (positive = safe, negative = unsafe)
        """
        # Extract positions
        x, y, z = state[0], state[1], state[2]
        
        # Collision detection with cylinders (simplified version)
        # Distance to cylinder at (0, 0.75) and (0, -0.75)
        dist1 = np.sqrt(x**2 + (y - 0.75)**2) - self.collisionR
        dist2 = np.sqrt(x**2 + (y + 0.75)**2) - self.collisionR
        collision_constraint = min(dist1, dist2)
        
        # Box bounds checking
        x_min, x_max = -self.state_max_x, self.state_max_x
        y_min, y_max = -self.state_max_y, self.state_max_y
        z_min, z_max = -self.state_max_z, self.state_max_z
        
        # Compute per-dimension signed distances to box faces
        dx_min = x - x_min
        dx_max = x_max - x
        dy_min = y - y_min
        dy_max = y_max - y
        dz_min = z - z_min
        dz_max = z_max - z
        
        # Inside: minimum distance to any face (negative inside, zero on surface)
        inside_dist = min(dx_min, dx_max, dy_min, dy_max, dz_min, dz_max)
        
        # For outside: compute the per-dimension "over" (how far outside the box in each dim)
        over_x = max(0, x - x_max) + max(0, x_min - x)
        over_y = max(0, y - y_max) + max(0, y_min - y)
        over_z = max(0, z - z_max) + max(0, z_min - z)
        # Norm of the "over" vector gives Euclidean distance outside
        outside_dist = np.sqrt(over_x**2 + over_y**2 + over_z**2)
        
        # If all inside (all distances to faces > 0), use inside_dist; else use outside_dist
        is_inside = (dx_min > 0) and (dx_max > 0) and (dy_min > 0) and (dy_max > 0) and (dz_min > 0) and (dz_max > 0)
        inside_constraint = inside_dist if is_inside else -outside_dist
        
        # Safe if outside collision radius AND inside bounds
        safety_constraint = min(collision_constraint, inside_constraint)
        
        # Return safety constraint without weighting
        return safety_constraint

    def get_position(self, state: np.ndarray) -> np.ndarray:
        """Get drone's 3D position.
        
        Args:
            state (np.ndarray): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
            
        Returns:
            np.ndarray: [x, y, z] drone's position.
        """
        return np.array([state[0], state[1], state[2]])

    def get_velocity(self, state: np.ndarray) -> np.ndarray:
        """Get drone's 3D velocity.
        
        Args:
            state (np.ndarray): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
            
        Returns:
            np.ndarray: [vx, vy, vz] drone's velocity.
        """
        return np.array([state[7], state[8], state[9]])

    def get_angular_velocity(self, state: np.ndarray) -> np.ndarray:
        """Get drone's 3D angular velocity.
        
        Args:
            state (np.ndarray): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
            
        Returns:
            np.ndarray: [wx, wy, wz] drone's angular velocity.
        """
        return np.array([state[10], state[11], state[12]])

    def get_quaternion(self, state: np.ndarray) -> np.ndarray:
        """Get drone's quaternion orientation.
        
        Args:
            state (np.ndarray): [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz].
            
        Returns:
            np.ndarray: [qw, qx, qy, qz] drone's quaternion.
        """
        return np.array([state[3], state[4], state[5], state[6]])

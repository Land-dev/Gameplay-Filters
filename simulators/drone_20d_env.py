# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for 20D Drone pursuit-evasion environment.

This file implements a zero-sum environment for 20D drone pursuit-evasion games.
The evader (controller) tries to avoid collision while staying within bounds,
while the pursuer (disturbance) tries to cause collision.
"""

from typing import Dict, Tuple, Optional, Any, Union
import numpy as np
import torch
from gym import spaces
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .base_zs_env import BaseZeroSumEnv
from .agent import Agent
from .utils import ActionZS


class Drone_20D_PursuitEvasionEnv(BaseZeroSumEnv):
    """
    Implements a zero-sum environment for 20D drone pursuit-evasion games.
    """

    def __init__(self, cfg_env: Any, cfg_agent: Any, cfg_cost: Any) -> None:
        super().__init__(cfg_env, cfg_agent)
        
        # Initialize your dynamics
        if cfg_agent.dyn == "Drone20D":
            self.agent.dyn = self.agent.dyn  # Already set by parent class
        
        # Set up cost function parameters
        self.cost_params = cfg_cost
        
        # Core parameters - now read from agent config
        self.goalR = self.agent.dyn.capture_radius  # collision radius
        self.timeout = getattr(cfg_env, 'timeout', 300)  # max steps per episode
        self.set_mode = getattr(cfg_cost, 'set_mode', 'avoid')  # 'avoid', 'reach', or 'reach_avoid'
        self.obs_type = getattr(cfg_env, 'obs_type', 'perfect')  # observation type
        self.dt = getattr(cfg_env, 'dt', 0.02)  # time step
        
        # Get parameters from dynamics class to ensure consistency
        self.max_v = self.agent.dyn.max_v
        self.max_angle = self.agent.dyn.max_angle
        self.max_omega = self.agent.dyn.max_omega
        self.state_max_x = self.agent.dyn.state_max_x
        self.state_max_y = self.agent.dyn.state_max_y
        self.state_max_z = self.agent.dyn.state_max_z
        
        # Define state dimension
        self.state_dim = 20  # [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                             #  x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z]
        
        # Initialize state space bounds
        self.state_low = np.array([
            -self.state_max_x,  # x1
            -self.max_v,        # v1_x
            -self.max_angle,    # θ1_x
            -self.max_omega,    # ω1_x
            -self.state_max_y,  # y1
            -self.max_v,        # v1_y
            -self.max_angle,    # θ1_y
            -self.max_omega,    # ω1_y
            0.0,                # z1
            -self.max_v,        # v1_z
            -self.state_max_x,  # x2
            -self.max_v,        # v2_x
            -self.max_angle,    # θ2_x
            -self.max_omega,    # ω2_x
            -self.state_max_y,  # y2
            -self.max_v,        # v2_y
            -self.max_angle,    # θ2_y
            -self.max_omega,    # ω2_y
            0.0,                # z2
            -self.max_v         # v2_z
        ])
        self.state_high = np.array([
            self.state_max_x,   # x1
            self.max_v,         # v1_x
            self.max_angle,     # θ1_x
            self.max_omega,     # ω1_x
            self.state_max_y,   # y1
            self.max_v,         # v1_y
            self.max_angle,     # θ1_y
            self.max_omega,     # ω1_y
            self.state_max_z,   # z1
            self.max_v,         # v1_z
            self.state_max_x,   # x2
            self.max_v,         # v2_x
            self.max_angle,     # θ2_x
            self.max_omega,     # ω2_x
            self.state_max_y,   # y2
            self.max_v,         # v2_y
            self.max_angle,     # θ2_y
            self.max_omega,     # ω2_y
            self.state_max_z,   # z2
            self.max_v          # v2_z
        ])
        
        # Set up visualization bounds
        self.visual_bounds = np.array([
            [-self.state_max_x - 0.5, self.state_max_x + 0.5], 
            [-self.state_max_y - 0.5, self.state_max_y + 0.5],
            [0.0, self.state_max_z + 0.5]
        ])
        
        # Initialize observation and reset spaces
        self.build_obs_rst_space(cfg_env, cfg_agent, cfg_cost)
        
        # Override action spaces for continuous actions - matching 20D dynamics
        # Control (evader): [S1_x, S1_y, T1_z] where S are torques and T is thrust
        self.action_space_ctrl = spaces.Box(
            low=np.array([-1.0, -1.0, 0.25]), 
            high=np.array([1.0, 1.0, 1.0]), 
            shape=(3,), 
            dtype=np.float32
        )
        # Disturbance (pursuer): [S2_x, S2_y, T2_z] where S are torques and T is thrust
        self.action_space_dstb = spaces.Box(
            low=np.array([-1.0, -1.0, 0.25]), 
            high=np.array([1.0, 1.0, 1.0]), 
            shape=(3,), 
            dtype=np.float32
        )
        self.action_space = spaces.Dict(
            dict(ctrl=self.action_space_ctrl, dstb=self.action_space_dstb)
        )
        
        self.seed(cfg_env.seed)
        
        # Add track attribute for visualization compatibility
        self.track = self  # Self-reference for track plotting methods
        
        self.reset()

    def get_constraints(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
        """Define safety constraints.
        
        Args:
            state: Current state [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                                 x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z]
            action: Dictionary with 'ctrl' and 'dstb' actions
            state_nxt: Next state
            
        Returns:
            Dictionary of constraint values
        """
        constraints = {}
        
        # Use the boundary function from dynamics for safety constraint
        # This includes both collision detection (ellipse) and bounds checking
        boundary_value = self.agent.dyn.boundary_fn(state)
        constraints['safety'] = np.array([[boundary_value]])
        
        return constraints

    def get_constraints_all(
        self, states: np.ndarray, actions: Union[np.ndarray, dict]
    ) -> Dict[str, np.ndarray]:
        """
        Gets the values of all constraint functions for multiple states.
        
        Args:
            states: Array of states [batch_size, state_dim]
            actions: Array of actions or dict of actions
            
        Returns:
            Dict: each (key, value) pair is the name and values of a constraint
                evaluated at the states and actions input.
        """
        # For simplicity, we'll evaluate constraints for each state individually
        # In a more efficient implementation, you could vectorize this
        batch_size = states.shape[1] if len(states.shape) > 1 else 1
        
        if batch_size == 1:
            # Single state case
            state = states.flatten()
            if isinstance(actions, dict):
                action = actions
            else:
                action = {'ctrl': actions[:3], 'dstb': actions[3:6]}
            
            # Use the existing get_constraints method
            constraints = self.get_constraints(state, action, state)
            
            # Return as is (already in correct format)
            return constraints
        else:
            # Multiple states case - evaluate each one
            constraint_dict = {}
            for i in range(batch_size):
                state = states[:, i]
                if isinstance(actions, dict):
                    action = {k: v[i] if hasattr(v, '__len__') else v for k, v in actions.items()}
                else:
                    action = {'ctrl': actions[i:i+3], 'dstb': actions[i+3:i+6]}
                
                constraints = self.get_constraints(state, action, state)
                
                # Initialize arrays if first iteration
                if i == 0:
                    constraint_dict = {k: np.zeros((1, batch_size)) for k in constraints.keys()}
                
                # Store values
                for k, v in constraints.items():
                    constraint_dict[k][0, i] = v[0, 0]  # Extract scalar value from array
            
            return constraint_dict

    def get_cost(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray, 
                constraints: Optional[Dict] = None) -> float:
        """Define cost function.
        
        Args:
            state: Current state
            action: Dictionary with 'ctrl' and 'dstb' actions
            state_nxt: Next state
            constraints: Constraint values from get_constraints
            
        Returns:
            Cost value (controller wants to minimize, disturbance wants to maximize)
        """
        # Cost function for zero-sum pursuit-evasion game
        # Controller (evader) wants to minimize cost (maximize safety)
        # Disturbance (pursuer) wants to maximize cost (minimize distance to evader)
        if constraints is not None:
            # Use the unified safety constraint from boundary function
            safety_val = constraints['safety'][0, 0] if isinstance(constraints['safety'], np.ndarray) else constraints['safety']
            
            # Return negative safety value as cost
            # This means: negative cost = safe (good for evader), positive cost = unsafe (good for pursuer)
            # Controller minimizes cost (maximizes safety)
            # Disturbance maximizes cost (minimizes distance to evader)
            return -safety_val
        else:
            return 0.0

    def get_target_margin(self, state: np.ndarray, action: ActionZS, state_nxt: np.ndarray) -> Dict:
        """Define target margins for reach-avoid problems.
        
        For pure avoidance problems, targets are not used in the Bellman equation.
        This method returns None to indicate no targets are needed.
        
        Args:
            state: Current state
            action: Dictionary with 'ctrl' and 'dstb' actions
            state_nxt: Next state
            
        Returns:
            Dictionary of target margins (None for pure avoidance)
        """
        # For pure avoidance, no targets are needed
        # The Bellman equation will only use constraints (g_x)
        return None

    def get_done_and_info(self, state: np.ndarray, constraints: Dict, targets: Dict,
                         final_only: bool = True, end_criterion: Optional[str] = None) -> Tuple[bool, Dict]:
        """Define episode termination conditions.
        
        Args:
            state: Current state
            constraints: Constraint values
            targets: Target margin values
            final_only: Whether to only check final state
            end_criterion: End criterion type
            
        Returns:
            Tuple of (done, info)
        """

        done = False
        done_type = "not_raised"
        
        # Check timeout
        if self.cnt >= self.timeout:
            done = True
            done_type = "timeout"
        
        # Get constraint values (handle both scalar and array formats)
        # Use the unified safety constraint from boundary function
        if isinstance(constraints['safety'], np.ndarray):
            g_x = constraints['safety'][0, 0]
        else:
            g_x = constraints['safety']
        
        # Get target values (l_x) - for pure avoidance, no targets are used
        if targets is not None:
            l_x = targets['safety_target']  # Target safety margin
        else:
            l_x = np.inf  # No targets in pure avoidance
        
        # Binary cost (1 if safe, 0 if unsafe)
        binary_cost = 1.0 if g_x > 0 else 0.0
        
        # Check for failure - collision occurs when distance <= goalR (g_x <= 0)
        if g_x <= 0:  # Collision or out of bounds
            done = True
            done_type = "failure"
        
        # Build info dictionary
        info = {
            'g_x': float(g_x),
            'l_x': float(l_x),
            'binary_cost': float(binary_cost),
            'done_type': done_type,
            'termination_reason': done_type
        }
        
        # Add additional info if available
        if targets is not None:
            info['safety_target'] = targets['safety_target']
        
        return done, info

    def get_obsrv(self, state: np.ndarray) -> np.ndarray:
        """Define observation space.
        
        Args:
            state: Current state [x1, v1_x, θ1_x, ω1_x, y1, v1_y, θ1_y, ω1_y, z1, v1_z, 
                                 x2, v2_x, θ2_x, ω2_x, y2, v2_y, θ2_y, ω2_y, z2, v2_z]
            
        Returns:
            Observation array
        """
        if self.obs_type == 'perfect':
            # Perfect observation - return full state
            return state.copy()
        elif self.obs_type == 'relative':
            # Relative observation - return relative positions and velocities
            # Extract positions and velocities for both drones
            x1, v1_x, y1, v1_y, z1, v1_z = state[0], state[1], state[4], state[5], state[8], state[9]
            x2, v2_x, y2, v2_y, z2, v2_z = state[10], state[11], state[14], state[15], state[18], state[19]
            
            # Relative positions and velocities
            rel_x = x2 - x1
            rel_y = y2 - y1
            rel_z = z2 - z1
            rel_vx = v2_x - v1_x
            rel_vy = v2_y - v1_y
            rel_vz = v2_z - v1_z
            
            return np.array([rel_x, rel_vx, rel_y, rel_vy, rel_z, rel_vz])
        else:
            raise ValueError(f"Unknown observation type: {self.obs_type}")

    def build_obs_rst_space(self, cfg_env: Any, cfg_agent: Any, cfg_cost: Any) -> None:
        """Build observation and reset spaces.
        
        Args:
            cfg_env: Environment configuration
            cfg_agent: Agent configuration
            cfg_cost: Cost configuration
        """
        # Observation space
        if self.obs_type == 'perfect':
            obs_dim = self.state_dim
        elif self.obs_type == 'relative':
            obs_dim = 6  # [rel_x, rel_vx, rel_y, rel_vy, rel_z, rel_vz]
        else:
            raise ValueError(f"Unknown observation type: {self.obs_type}")
        
        self.observation_space = spaces.Box(
            low=self.state_low, high=self.state_high, shape=(obs_dim,), dtype=np.float32
        )
        
        # Reset space (same as state space)
        self.reset_space = spaces.Box(
            low=self.state_low, high=self.state_high, dtype=np.float32
        )

    def reset(self, state: Optional[np.ndarray] = None, cast_torch: bool = False, **kwargs) -> Union[np.ndarray, torch.Tensor]:
        """Reset the environment.
        
        Args:
            state: Initial state (optional)
            cast_torch: Whether to cast observation to torch tensor
            **kwargs: Additional arguments
            
        Returns:
            Initial observation
        """
        if state is not None:
            self.state = state.copy()
        else:
            # Sample random initial state
            self.state = self.reset_space.sample()
            
            # Ensure initial positions are within bounds
            self.state[0] = np.clip(self.state[0], -self.state_max_x, self.state_max_x)  # x1
            self.state[4] = np.clip(self.state[4], -self.state_max_y, self.state_max_y)  # y1
            self.state[8] = np.clip(self.state[8], 0.0, self.state_max_z)  # z1
            self.state[10] = np.clip(self.state[10], -self.state_max_x, self.state_max_x)  # x2
            self.state[14] = np.clip(self.state[14], -self.state_max_y, self.state_max_y)  # y2
            self.state[18] = np.clip(self.state[18], 0.0, self.state_max_z)  # z2
            
            # Ensure initial velocities are within limits
            self.state[1] = np.clip(self.state[1], -self.max_v, self.max_v)  # v1_x
            self.state[5] = np.clip(self.state[5], -self.max_v, self.max_v)  # v1_y
            self.state[9] = np.clip(self.state[9], -self.max_v, self.max_v)  # v1_z
            self.state[11] = np.clip(self.state[11], -self.max_v, self.max_v)  # v2_x
            self.state[15] = np.clip(self.state[15], -self.max_v, self.max_v)  # v2_y
            self.state[19] = np.clip(self.state[19], -self.max_v, self.max_v)  # v2_z
            
            # Ensure initial angles are within limits
            self.state[2] = np.clip(self.state[2], -self.max_angle, self.max_angle)  # θ1_x
            self.state[6] = np.clip(self.state[6], -self.max_angle, self.max_angle)  # θ1_y
            self.state[12] = np.clip(self.state[12], -self.max_angle, self.max_angle)  # θ2_x
            self.state[16] = np.clip(self.state[16], -self.max_angle, self.max_angle)  # θ2_y
            
            # Ensure initial angular velocities are within limits
            self.state[3] = np.clip(self.state[3], -self.max_omega, self.max_omega)  # ω1_x
            self.state[7] = np.clip(self.state[7], -self.max_omega, self.max_omega)  # ω1_y
            self.state[13] = np.clip(self.state[13], -self.max_omega, self.max_omega)  # ω2_x
            self.state[17] = np.clip(self.state[17], -self.max_omega, self.max_omega)  # ω2_y
            
            # Ensure initial separation is safe (greater than collision radius × 4)
            p1 = np.array([self.state[0], self.state[4], self.state[8]])  # evader position [x1, y1, z1]
            p2 = np.array([self.state[10], self.state[14], self.state[18]])  # pursuer position [x2, y2, z2]
            initial_dist = np.linalg.norm(p1 - p2)
            
            if initial_dist < self.goalR * 2:
                # Move pursuer away from evader to ensure safe initial separation
                direction = (p2 - p1) / (initial_dist + 1e-8)  # avoid division by zero
                safe_position = p1 + direction * self.goalR * 4
                
                # Update pursuer position while keeping within bounds
                self.state[10] = np.clip(safe_position[0], -self.state_max_x, self.state_max_x)  # x2
                self.state[14] = np.clip(safe_position[1], -self.state_max_y, self.state_max_y)  # y2
                self.state[18] = np.clip(safe_position[2], 0.0, self.state_max_z)  # z2
        
        self.cnt = 0
        obsrv = self.get_obsrv(self.state)
        if cast_torch:
            obsrv = torch.FloatTensor(obsrv)
        return obsrv

    def step(self, action: ActionZS, cast_torch: bool = False) -> Tuple[Union[np.ndarray, torch.Tensor], float, bool, Dict]:
        """Take a step in the environment.
        
        Args:
            action: Dictionary with 'ctrl' and 'dstb' actions
            cast_torch: Whether to cast observation to torch tensor
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Use parent class step method which handles the training pipeline properly
        return super().step(action, cast_torch)

    def render(self):
        """Render the environment (placeholder)."""
        print(f"State: {self.state}")
        print(f"Evader: ({self.state[0]:.2f}, {self.state[4]:.2f}, {self.state[8]:.2f})")
        print(f"Pursuer: ({self.state[10]:.2f}, {self.state[14]:.2f}, {self.state[18]:.2f})")
        print(f"Distance: {np.sqrt((self.state[0]-self.state[10])**2 + (self.state[4]-self.state[14])**2 + (self.state[8]-self.state[18])**2):.2f}")

    def report(self) -> None:
        """Print environment information."""
        print("=== Drone 20D Pursuit-Evasion Environment ===")
        print(f"State dimension: {self.state_dim}")
        print(f"Control dimension: {self.action_space_ctrl.shape[0]}")
        print(f"Disturbance dimension: {self.action_space_dstb.shape[0]}")
        print(f"Collision radius: {self.goalR}")
        print(f"Environment bounds: X=[-{self.state_max_x}, {self.state_max_x}], Y=[-{self.state_max_y}, {self.state_max_y}], Z=[0.0, {self.state_max_z}]")
        print(f"Timeout: {self.timeout} steps")
        print(f"Time step: {self.dt}")
        print(f"Observation type: {self.obs_type}")
        print(f"Set mode: {self.set_mode}")
        print(f"Max angle: {self.max_angle}")
        print(f"Max angular velocity: {self.max_omega}")
        print("=========================================")

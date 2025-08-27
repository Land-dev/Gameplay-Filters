# --------------------------------------------------------
# Copyright (c) 2023 Princeton University
# Email: kaichieh@princeton.edu
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""A class for Dubins pursuit-evasion environment.

This file implements a zero-sum environment for Dubins vehicle pursuit-evasion games.
The evader (controller) tries to avoid collision while staying within bounds,
while the pursuer (disturbance) tries to cause collision.
"""

from typing import Dict, Tuple, Optional, Any, Union
import numpy as np
import torch
from gym import spaces
import matplotlib.pyplot as plt

from .base_zs_env import BaseZeroSumEnv
from .agent import Agent
from .utils import ActionZS


class DubinsPursuitEvasionEnv(BaseZeroSumEnv):
    """
    Implements a zero-sum environment for Dubins vehicle pursuit-evasion games.
    """

    def __init__(self, cfg_env: Any, cfg_agent: Any, cfg_cost: Any) -> None:
        super().__init__(cfg_env, cfg_agent)
        
        # Initialize your dynamics
        if cfg_agent.dyn == "Dubins6D":
            self.agent.dyn = self.agent.dyn  # Already set by parent class
        
        # Set up cost function parameters
        self.cost_params = cfg_cost
        
        # Core parameters - now read from agent config
        self.goalR = getattr(cfg_agent, 'goalR', 0.36)  # collision radius - central parameter
        self.state_max_x = getattr(cfg_agent, 'state_max_x', 3.0)  # environment boundary
        self.state_max_y = getattr(cfg_agent, 'state_max_y', 2.0)  # environment boundary
        self.timeout = getattr(cfg_env, 'timeout', 300)  # max steps per episode
        self.set_mode = getattr(cfg_cost, 'set_mode', 'avoid')  # 'avoid', 'reach', or 'reach_avoid'
        self.obs_type = getattr(cfg_env, 'obs_type', 'perfect')  # observation type
        self.dt = getattr(cfg_env, 'dt', 0.05)  # time step
        
        # Define state dimension
        self.state_dim = 6  # [x_e, y_e, theta_e, x_p, y_p, theta_p]
        
        # Initialize state space bounds
        self.state_low = np.array([
            -self.state_max_x,  # x_e
            -self.state_max_y,  # y_e
            -np.pi,           # theta_e
            -self.state_max_x,  # x_p
            -self.state_max_y,  # y_p
            -np.pi            # theta_p
        ])
        self.state_high = np.array([
            self.state_max_x,   # x_e
            self.state_max_y,   # y_e
            np.pi,            # theta_e
            self.state_max_x,   # x_p
            self.state_max_y,   # y_p
            np.pi             # theta_p
        ])
        
        # Set up visualization bounds
        self.visual_bounds = np.array([[-self.state_max_x - 0.5, self.state_max_x + 0.5], [-self.state_max_y - 0.5, self.state_max_y + 0.5]])
        x_eps = (2 * self.state_max_x) * 0.005
        y_eps = (2 * self.state_max_y) * 0.005
        self.visual_extent = np.array([
            self.visual_bounds[0, 0] - x_eps, self.visual_bounds[0, 1] + x_eps,
            self.visual_bounds[1, 0] - y_eps, self.visual_bounds[1, 1] + y_eps
        ])
        
        
        # Initialize observation and reset spaces
        self.build_obs_rst_space(cfg_env, cfg_agent, cfg_cost)
        
        # Override action spaces for continuous actions
        self.action_space_ctrl = spaces.Box(low=-1.9, high=1.9, shape=(1,), dtype=np.float32)  # Continuous [-1.9, 1.9] for evader
        self.action_space_dstb = spaces.Box(low=-1.9, high=1.9, shape=(1,), dtype=np.float32)  # Continuous [-1.9, 1.9] for pursuer
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
            state: Current state [x_e, y_e, theta_e, x_p, y_p, theta_p]
            action: Dictionary with 'ctrl' and 'dstb' actions
            state_nxt: Next state
            
        Returns:
            Dictionary of constraint values
        """
        constraints = {}
        
        # Extract state components
        xe, ye = state[0], state[1]
        xp, yp = state[3], state[4]
        
        # Collision constraint (positive when safe, negative when colliding)
        collision_dist = np.sqrt((xe - xp)**2 + (ye - yp)**2)
        constraints['collision'] = np.array([[collision_dist - self.goalR]])
        
        # Bounds constraints (positive when within bounds)
        evader_bounds_x = min(xe + self.state_max_x, self.state_max_x - xe)
        evader_bounds_y = min(ye + self.state_max_y, self.state_max_y - ye)
        pursuer_bounds_x = min(xp + self.state_max_x, self.state_max_x - xp)
        pursuer_bounds_y = min(yp + self.state_max_y, self.state_max_y - yp)
        
        constraints['evader_bounds'] = np.array([[min(evader_bounds_x, evader_bounds_y)]])
        constraints['pursuer_bounds'] = np.array([[min(pursuer_bounds_x, pursuer_bounds_y)]])
        
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
                action = {'ctrl': actions[:1], 'dstb': actions[1:2]}
            
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
                    action = {'ctrl': actions[i:i+1], 'dstb': actions[i+1:i+2]}
                
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
            collision_val = constraints['collision'][0, 0] if isinstance(constraints['collision'], np.ndarray) else constraints['collision']
            evader_bounds_val = constraints['evader_bounds'][0, 0] if isinstance(constraints['evader_bounds'], np.ndarray) else constraints['evader_bounds']
            
            # For evader: safety margin (positive = safe, negative = unsafe)
            # For pursuer: negative distance (positive = close, negative = far)
            safety_margin = min(collision_val, evader_bounds_val)
            
            # Return negative safety margin as cost
            # This means: negative cost = safe (good for evader), positive cost = unsafe (good for pursuer)
            # Controller minimizes cost (maximizes safety)
            # Disturbance maximizes cost (minimizes distance to evader)
            return -safety_margin
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
        # Only consider evader constraints for failure condition
        # Pursuer can go out of bounds without causing failure
        if isinstance(constraints['collision'], np.ndarray):
            g_x = min(constraints['collision'][0, 0], constraints['evader_bounds'][0, 0])
        else:
            g_x = min(constraints['collision'], constraints['evader_bounds'])
        
        # Get target values (l_x) - for pure avoidance, no targets are used
        if targets is not None:
            l_x = targets['collision_distance'] - self.goalR  # Positive when safe
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
            info['collision_distance'] = targets['collision_distance']
            info['evader_bounds_distance'] = targets['evader_bounds_distance']
            info['pursuer_bounds_distance'] = targets['pursuer_bounds_distance']
        
        return done, info

    def get_obsrv(self, state: np.ndarray) -> np.ndarray:
        """Define observation space.
        
        Args:
            state: Current state [x_e, y_e, theta_e, x_p, y_p, theta_p]
            
        Returns:
            Observation (can be full state or processed)
        """
        # For now, return the full state
        # You could also return relative coordinates or other processed observations
        return state.copy()

    def step(self, action: ActionZS, cast_torch: bool = False):
        """Override step method to pass through continuous actions.
        
        Args:
            action: Dictionary with 'ctrl' and 'dstb' actions (continuous values)
            cast_torch: Whether to cast observation to torch tensor
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Pass through actions directly (no processing needed)
        return super().step(action, cast_torch)

    def reset(self, state: Optional[np.ndarray] = None, cast_torch: bool = False, **kwargs) -> np.ndarray:
        """Reset the environment.
        
        Args:
            state: Optional initial state
            cast_torch: Whether to cast observation to torch tensor
            **kwargs: Additional arguments
            
        Returns:
            Initial observation
        """
        super().reset()
        
        if state is None:
            # Use reset space if available, otherwise generate random state
            if hasattr(self, 'reset_sample_sapce'):
                self.state = self.reset_sample_sapce.sample()
            else:
                # Generate random initial state
                self.state = np.array([
                    np.random.uniform(-self.state_max_x, self.state_max_x),  # x_e
                    np.random.uniform(-self.state_max_y, self.state_max_y),  # y_e
                    np.random.uniform(-np.pi, np.pi),                    # theta_e
                    np.random.uniform(-self.state_max_x, self.state_max_x),  # x_p
                    np.random.uniform(-self.state_max_y, self.state_max_y),  # y_p
                    np.random.uniform(-np.pi, np.pi)                     # theta_p
                ])
            
            # Ensure initial separation is greater than collision radius
            xe, ye = self.state[0], self.state[1]
            xp, yp = self.state[3], self.state[4]
            initial_dist = np.sqrt((xe - xp)**2 + (ye - yp)**2)
            
            if initial_dist < self.goalR * 4:  # Ensure safe initial separation 
                # Move pursuer away from evader
                angle = np.arctan2(yp - ye, xp - xe)
                self.state[3] = xe + self.goalR * 4 * np.cos(angle)
                self.state[4] = ye + self.goalR * 4 * np.sin(angle)
        else:
            self.state = state.copy()
        
        self.cnt = 0
        obsrv = self.get_obsrv(self.state)
        if cast_torch:
            import torch
            obsrv = torch.FloatTensor(obsrv)
        return obsrv

    def build_obs_rst_space(self, cfg_env, cfg_agent, cfg_cost):
        """Build observation and reset spaces."""
        # Reset Sample Space - same as state space
        self.reset_sample_space = spaces.Box(
            low=np.float32(self.state_low), high=np.float32(self.state_high)
        )

        # Observation space
        if self.obs_type == "perfect":
            # Same as reset space for perfect observations
            self.observation_space = self.reset_sample_space
        else:
            raise ValueError(f"Observation type {self.obs_type} is not supported!")
        
        self.obs_dim = self.observation_space.low.shape[0]

    def seed(self, seed: int = 0):
        """Set random seed."""
        super().seed(seed)
        if hasattr(self, 'reset_sample_space'):
            self.reset_sample_space.seed(seed)

    def get_samples(self, nx: int, ny: int):
        """Get state samples for value function plotting."""
        xs = np.linspace(self.visual_bounds[0, 0], self.visual_bounds[0, 1], nx)
        ys = np.linspace(self.visual_bounds[1, 0], self.visual_bounds[1, 1], ny)
        return xs, ys

    def render(self):
        """Render the environment (placeholder)."""
        print(f"State: {self.state}")
        print(f"Evader: ({self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f})")
        print(f"Pursuer: ({self.state[3]:.2f}, {self.state[4]:.2f}, {self.state[5]:.2f})")
        print(f"Distance: {np.sqrt((self.state[0]-self.state[3])**2 + (self.state[1]-self.state[4])**2):.2f}")

    def render_obs(self, ax=None, c='r'):
        """Render obstacles (placeholder for Dubins environment)."""
        # Dubins environment doesn't have obstacles, so this is a no-op
        pass


    def plot_value_function(self, value_function, ax=None, nx=100, ny=100, 
                           theta_e=0.0, theta_p=0.0, cmap='viridis', 
                           vmin=None, vmax=None, alpha=0.7):
        """Plot value function over state space with pursuer fixed at (0,0).
        
        Args:
            value_function: Function that takes observation and returns value
            ax: Matplotlib axis (if None, creates new figure)
            nx, ny: Grid resolution for x and y
            theta_e: Fixed evader heading angle
            theta_p: Fixed pursuer heading angle (should be 0.0 for fixed pursuer)
            cmap: Colormap for value function
            vmin, vmax: Value range for colormap
            alpha: Transparency of the plot
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create grid
        xs, ys = self.get_samples(nx, ny)
        X, Y = np.meshgrid(xs, ys)
        
        # Initialize value array
        values = np.zeros((ny, nx))
        
        # Evaluate value function at each grid point
        for i in range(ny):
            for j in range(nx):
                # Create state with evader at (x, y) and pursuer at (0, 0)
                state = np.array([
                    xs[j], ys[i], theta_e,  # evader: x, y, theta
                    0.0, 0.0, theta_p       # pursuer: fixed at (0, 0)
                ])
                
                # Get observation
                obs = self.get_obsrv(state)
                
                # Evaluate value function
                try:
                    value = value_function(obs)
                    if isinstance(value, (list, np.ndarray)):
                        value = value[0] if len(value) > 0 else 0.0
                    values[i, j] = float(value)
                except:
                    values[i, j] = 0.0
        
        # Plot value function
        im = ax.imshow(values, extent=self.visual_extent, origin='lower', 
                      cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Value')
        
        # Plot pursuer position (fixed at origin)
        ax.plot(0, 0, 'ro', markersize=10, label='Pursuer (fixed)')
        
        # Plot collision circle around pursuer
        pursuer_circle = plt.Circle((0, 0), self.goalR, color='red', 
                                   alpha=0.3, linestyle='--', fill=False)
        ax.add_patch(pursuer_circle)
        
        # Plot environment boundary
        self.render_obs(ax, c='black')
        
        # Set labels and title
        ax.set_xlabel('Evader X Position')
        ax.set_ylabel('Evader Y Position')
        ax.set_title(f'Value Function (Pursuer fixed at origin, θ_e={theta_e:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax, values

    def report(self):
        """Report environment information."""
        print("=== Dubins Pursuit-Evasion Environment ===")
        print(f"State dimension: {self.state_dim}")
        print(f"Control dimension: {self.action_dim_ctrl}")
        print(f"Disturbance dimension: {self.action_dim_dstb}")
        print(f"Collision radius: {self.goalR}")
        print(f"State bounds: X=[-{self.state_max_x}, {self.state_max_x}], Y=[-{self.state_max_y}, {self.state_max_y}]")
        print(f"Action spaces: Bang-bang discrete (0=-1.9, 1=1.9) for both ctrl and dstb")
        print(f"Set mode: {self.set_mode}")
        print(f"Timeout: {self.timeout}")
        print("==========================================") 
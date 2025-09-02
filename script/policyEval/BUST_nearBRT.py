# ======================= Imports =======================
import os
import sys
import numpy as np

# Force JAX to use CPU to avoid CUDA conflicts and DNN library errors
import os
# Environment variables to completely disable CUDA for JAX
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disable GPU memory preallocation
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use platform allocator (CPU)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'  # Force CPU platform
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'  # Force single CPU device

import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', False)
jax.config.update('jax_platforms', 'cpu')
try:
    jax.config.update('jax_default_device', jax.devices('cpu')[0])
except:
    pass  # Fallback if CPU device configuration fails

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.lines import Line2D
from matplotlib import transforms
from IPython.display import HTML
from tqdm import tqdm
import plotly.graph_objects as go
import hj_reachability as hj
import torch
import pickle
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deepreach.utils.MPC import RobustMPC
from deepreach.utils.modules import SingleBVPNet

# Add Isaacs imports for zero-sum game policy comparison
from omegaconf import OmegaConf
from agent import ISAACS
from simulators import DubinsPursuitEvasionEnv

# ======================= Utility Functions =======================
def angle_wrap(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def find_zero_crossing_time(base_grid, base_all_values, base_times, state, max_iterations=15, tolerance=1e-4):
    """Find the time t where V(state, t) ≈ 0 using bisection method.
    
    Args:
        base_grid: HJ grid
        base_all_values: Value functions at different times
        base_times: Array of time points
        state: Current state [x, y, theta]
        max_iterations: Maximum iterations for bisection
        tolerance: Tolerance for considering V ≈ 0
        
    Returns:
        t_optimal: Time where V ≈ 0
        v_optimal: Value at t_optimal
        success: Whether bisection converged
    """
    def evaluate_value_at_time(t):
        """Helper function to evaluate V(state, t) by interpolating between time points."""
        # Find the closest time index
        t_idx = np.argmin(np.abs(base_times - t))
        V_current = base_all_values[t_idx]
        v_current = base_grid.interpolate(V_current, state)
        return float(v_current)
    
    # Initialize bisection bounds
    t_left = base_times[0]   # Start time (usually 0.0)
    t_right = base_times[-1] # End time (usually -3.0)
    
    # Evaluate at bounds to check if we have a sign change
    v_left = evaluate_value_at_time(t_left)
    v_right = evaluate_value_at_time(t_right)
    
    # If no sign change at bounds, return the endpoint with smaller absolute value
    if v_left * v_right > 0:
        if abs(v_left) < abs(v_right):
            return t_left, v_left, False
        else:
            return t_right, v_right, False
    
    # Bisection method
    for iteration in range(max_iterations):
        t_mid = (t_left + t_right) / 2.0
        v_mid = evaluate_value_at_time(t_mid)
        
        # Check if we're close enough to zero
        if abs(v_mid) < tolerance:
            return t_mid, v_mid, True
        
        # Update bounds based on sign
        if v_mid * v_left > 0:
            # v_mid and v_left have same sign, zero is in right half
            t_left = t_mid
            v_left = v_mid
        else:
            # v_mid and v_left have opposite signs, zero is in left half
            t_right = t_mid
            v_right = v_mid
    
    # If we didn't converge, return the midpoint
    t_optimal = (t_left + t_right) / 2.0
    v_optimal = evaluate_value_at_time(t_optimal)
    return t_optimal, v_optimal, False

def convert_to_rel_state(xA, xB):
    """Convert absolute states to relative state (xB relative to xA)."""
    dx = xB[0] - xA[0]
    dy = xB[1] - xA[1]
    dtheta = angle_wrap(xB[2] - xA[2])
    rot_mat = np.array([
        [np.cos(xA[2]), np.sin(xA[2])],
        [-np.sin(xA[2]), np.cos(xA[2])]
    ])
    rel_xy = rot_mat @ np.array([dx, dy])
    return np.array([rel_xy[0], rel_xy[1], dtheta])

def convert_from_rel_state(xA, xrel):
    """Convert relative state back to absolute pursuer state."""
    xrel = np.array(xrel)
    x_rel, y_rel, theta_rel = xrel[:3]
    rot_mat = np.array([
        [np.cos(xA[2]), -np.sin(xA[2])],
        [np.sin(xA[2]),  np.cos(xA[2])]
    ])
    xy_global = xA[:2] + rot_mat @ np.array([x_rel, y_rel])
    theta_global = angle_wrap(xA[2] + theta_rel)
    return np.array([xy_global[0], xy_global[1], theta_global])

# ======================= Dynamics Classes =======================
class ControlAndDisturbanceAffineDynamics(hj.dynamics.ControlAndDisturbanceAffineDynamics):
    def __call__(self, state, control, disturbance, time):
        return (
            self.open_loop_dynamics(state, time)
            + self.control_jacobian(state, time) @ control
            + self.disturbance_jacobian(state, time) @ disturbance
        )
    def step(self, state, control, disturbance=None, time=0.0, dt=0.05, int_scheme="rk4"):
        if disturbance is None:
            disturbance = jnp.zeros(self.disturbance_space.lo.shape[0])
        if int_scheme == "fe":
            return state + dt * self(state, control, disturbance, time)
        elif int_scheme == "rk4":
            k1 = self(state, control, disturbance, time)
            k2 = self(state + dt / 2.0 * k1, control, disturbance, time + dt / 2.0)
            k3 = self(state + dt / 2.0 * k2, control, disturbance, time + dt / 2.0)
            k4 = self(state + dt * k3, control, disturbance, time + dt)
            return state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

class Air3D(ControlAndDisturbanceAffineDynamics):
    def __init__(self, va=0.6, vb=0.6, max_u=1.9, max_d=1.9,
                 control_mode="max", disturbance_mode="min"):
        control_space = hj.sets.Box(jnp.array([-max_u]), jnp.array([max_u]))
        disturbance_space = hj.sets.Box(jnp.array([-max_d]), jnp.array([max_d]))
        self.va = va
        self.vb = vb
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)
    def open_loop_dynamics(self, state, time):
        x, y, theta = state
        dx = -self.va + self.vb * jnp.cos(theta)
        dy = self.vb * jnp.sin(theta)
        dtheta = 0.0
        return jnp.array([dx, dy, dtheta])
    def control_jacobian(self, state, time):
        x, y, theta = state
        return jnp.array([
            [y],
            [-x],
            [-1.0],
        ])
    def disturbance_jacobian(self, state, time):
        _, _, _ = state
        return jnp.array([
            [0.0],
            [0.0],
            [1.0],
        ])

class Dubins(ControlAndDisturbanceAffineDynamics):
    def __init__(self, speed=0.6, max_turn_rate=1.9, control_mode="max", disturbance_mode="min", control_space=None, disturbance_space=None):
        self.speed = speed
        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-max_turn_rate]), jnp.array([max_turn_rate]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([-0.0]), jnp.array([0.0]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)
    def open_loop_dynamics(self, state, time):
        _, _, psi = state
        v_a = self.speed
        return jnp.array([v_a * jnp.cos(psi), v_a * jnp.sin(psi), 0.])
    def control_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
            [1.],
        ])
    def disturbance_jacobian(self, state, time):
        return jnp.array([
            [0.],
            [0.],
            [0.],
        ])

# ======================= HJ Reachability Setup =======================
def setup_hj_reachability():
    """Setup HJ reachability solver for Air3D (relative coordinates)."""
    base_dynamics = Air3D()
    base_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array([-3., -3., -np.pi]), np.array([3., 3., np.pi])),
        (81, 81, 81),  # Increased from (41, 41, 41) to (81, 81, 81)
        periodic_dims=2
    )
    base_avoid_values = jnp.linalg.norm(base_grid.states[..., :2], axis=-1) - 0.360
    obstacle_brt = lambda obstacle: (lambda t, v: jnp.minimum(v, obstacle))
    value_postprocessor = obstacle_brt(base_avoid_values)

    base_solver_settings = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=value_postprocessor)
    base_start_time = 0.0
    base_final_time = 3.0
    base_times = np.linspace(base_start_time, -base_final_time, 101)  # More time points
    base_initial_values = base_avoid_values
    base_all_values = hj.solve(base_solver_settings, base_dynamics, base_grid, base_times, base_initial_values)

    # Add the obstacle_brt_inversed with jnp.maximum
    obstacle_brt_inversed = lambda obstacle: (lambda t, v: jnp.maximum(v, obstacle))
    value_postprocessor_inversed = obstacle_brt_inversed(base_avoid_values)
    base_solver_settings_inversed = hj.SolverSettings.with_accuracy("very_high", value_postprocessor=value_postprocessor_inversed)
    base_all_values_inversed = hj.solve(base_solver_settings_inversed, base_dynamics, base_grid, base_times, base_initial_values)

    base_target_values = base_all_values[-1]
    
    return base_dynamics, base_grid, base_all_values, base_all_values_inversed, base_target_values

def sample_relative_states_near_brt(base_grid, V_final, num_samples, epsilon=0.02, side='both', rng=None):
    """Sample relative states near the BRT boundary where |V| <= epsilon.

    side: 'both' | 'inside' | 'outside' controls which side of boundary to sample.
    Returns an array of shape (num_samples, 3) with [x_rel, y_rel, theta_rel].
    """
    if rng is None:
        rng = np.random.default_rng()

    grid_states = np.array(base_grid.states).reshape(-1, 3)
    grid_values = np.array(V_final).reshape(-1)

    mask_band = np.abs(grid_values) <= float(epsilon)
    if side == 'inside':
        mask_band = np.logical_and(mask_band, grid_values < 0)
    elif side == 'outside':
        mask_band = np.logical_and(mask_band, grid_values > 0)

    candidate_indices = np.nonzero(mask_band)[0]
    if candidate_indices.size == 0:
        raise ValueError("No grid states found within the requested epsilon band. Increase epsilon.")

    # Sample with replacement if needed
    chosen = rng.choice(candidate_indices, size=num_samples, replace=candidate_indices.size < num_samples)
    return grid_states[chosen]

def sample_initial_states_near_brt(base_grid, V_final, num_pairs, epsilon=0.02, side='both',
                                   x_bounds=(-3.0, 3.0), y_bounds=(-2.0, 2.0), rng=None, max_tries=5000):
    """Create absolute initial states (evader, pursuer) whose relative state is near BRT boundary.

    - Draw relative states near boundary using base_grid/V_final.
    - Place evader uniformly inside given x/y bounds and uniform heading.
    - Convert relative state to absolute pursuer state. Reject if pursuer falls outside bounds.
    Returns a list of (evader_state, pursuer_state) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()

    rel_states = sample_relative_states_near_brt(base_grid, V_final, num_pairs, epsilon=epsilon, side=side, rng=rng)

    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    initial_states = []
    tries = 0
    i = 0
    while i < num_pairs and tries < max_tries:
        tries += 1
        # Sample evader absolute state
        evader_x = rng.uniform(x_min, x_max)
        evader_y = rng.uniform(y_min, y_max)
        evader_theta = rng.uniform(-np.pi, np.pi)
        evader = np.array([evader_x, evader_y, evader_theta])

        # Corresponding relative state target
        xrel = rel_states[i]
        pursuer = convert_from_rel_state(evader, xrel)

        # Keep only if pursuer within bounds
        if (x_min <= pursuer[0] <= x_max) and (y_min <= pursuer[1] <= y_max):
            initial_states.append((evader, pursuer))
            i += 1

    if i < num_pairs:
        raise RuntimeError(f"Failed to sample the requested number of pairs within bounds after {tries} tries.")

    return initial_states

def setup_boundary_brt():
    """Setup BRT for boundary safety (single vehicle in absolute coordinates)."""
    # Define boundary box
    x_min, x_max = -3.0, 3.0
    y_min, y_max = -2.0, 2.0
    
    # Create grid for boundary BRT (3D: x, y, theta)
    boundary_grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        hj.sets.Box(np.array([x_min, y_min, -np.pi]), np.array([x_max, y_max, np.pi])),
        (61, 41, 21),  # Higher resolution for x,y, lower for theta
        periodic_dims=2  # theta is periodic
    )
    
    # Define boundary function (positive inside, negative outside)
    boundary_values = -np.maximum(
        np.maximum(boundary_grid.states[..., 0] - x_max, x_min - boundary_grid.states[..., 0]),
        np.maximum(boundary_grid.states[..., 1] - y_max, y_min - boundary_grid.states[..., 1])
    )
    
    # Setup single vehicle dynamics (Dubins car)
    single_dynamics = Dubins(speed=0.6, max_turn_rate=1.9)
    
    # Solve BRT for boundary safety
    solver_settings = hj.SolverSettings.with_accuracy("high", hamiltonian_postprocessor=hj.solver.backwards_reachable_tube)
    times = np.linspace(0.0, -10.0, 51)  # 5 second horizon
    
    boundary_brt_values = hj.solve(solver_settings, single_dynamics, boundary_grid, times, boundary_values)
    
    return boundary_grid, boundary_brt_values, single_dynamics

def get_safe_control(state, boundary_grid, boundary_brt_values, single_dynamics, threshold=0.05):
    """Get optimally safe control when near boundary."""
    # Interpolate boundary BRT value at current state (3D: x, y, theta)
    boundary_value = boundary_grid.interpolate(boundary_brt_values[-1], state)
    boundary_value = float(boundary_value)
    
    if boundary_value <= threshold:
        # Near boundary, use optimally safe control
        grad_boundary = boundary_grid.interpolate(boundary_grid.grad_values(boundary_brt_values[-1]), state)
        safe_control, _ = single_dynamics.optimal_control_and_disturbance(state, 1.0, grad_boundary)
        return float(safe_control[0])
    else:
        # Not near boundary, return None to use normal control
        return None

# ======================= DeepReach Model Setup =======================
def setup_deepreach_models():
    """Setup DeepReach models for comparison."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from deepreach.dynamics import Air3D as TorchAir3D, Dubins6D
    from deepreach.utils.MPC import RobustMPC
    from deepreach.utils.modules import SingleBVPNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Air3D model
    dynamics = TorchAir3D(set_mode='avoid')
    dynamics.control_init = torch.zeros(dynamics.control_dim).to(device)
    dynamics.disturbance_init = torch.zeros(dynamics.disturbance_dim).to(device)
    dynamics.eps_var = torch.tensor(0.1).to(device)
    
    # Dubins6D model
    dynamicsDubins = Dubins6D(collisionR=0.36, set_mode='avoid')
    dynamicsDubins.control_init = torch.zeros(dynamics.control_dim).to(device)
    dynamicsDubins.disturbance_init = torch.zeros(dynamics.disturbance_dim).to(device)
    dynamicsDubins.eps_var = torch.tensor(0.1).to(device)

    # Load Dubins6D model (MPC)
    model_path2 = "dubins_3sec.pth"
    model2 = SingleBVPNet(
        in_features=9,
        hidden_features=512,
        num_hidden_layers=3,
        out_features=1,
        type='sine',
        periodic_transform_fn=dynamicsDubins.periodic_transform_fn
    )
    checkpoint2 = torch.load(model_path2, map_location=device, weights_only=True)
    model2.load_state_dict(checkpoint2["model"])
    model2.to(device)
    model2.eval()

    class DeepReachVF2:
        def __call__(self, batch):
            coords = batch["coords"]
            coords = coords.clone().detach().requires_grad_(True)
            periodic_coords = dynamicsDubins.periodic_transform_fn(coords)
            value = model2.net(periodic_coords)
            return {"model_in": coords, "model_out": value}

    policy2 = DeepReachVF2()
    
    model_path_2_inversed = "dubins_3sec_inversed.pth"
    model2_inversed = SingleBVPNet(
        in_features=9,
        hidden_features=512,
        num_hidden_layers=3,
        out_features=1,
        type='sine',
        periodic_transform_fn=dynamicsDubins.periodic_transform_fn
    )
    checkpoint2_inversed = torch.load(model_path_2_inversed, map_location=device, weights_only=True)
    model2_inversed.load_state_dict(checkpoint2_inversed["model"])
    model2_inversed.to(device)
    model2_inversed.eval()

    class DeepReachVF2Inversed:
        def __call__(self, batch):
            coords = batch["coords"]
            coords = coords.clone().detach().requires_grad_(True)
            periodic_coords = dynamicsDubins.periodic_transform_fn(coords)
            value = model2_inversed.net(periodic_coords)
            return {"model_in": coords, "model_out": value}

    policy2_inversed = DeepReachVF2Inversed()

    # Load Vanilla DeepReach model
    model_path_vanilla = "dubins_3sec_vanilla.pth"  # Adjust path as needed
    model_vanilla = SingleBVPNet(
        in_features=9,
        hidden_features=512,
        num_hidden_layers=3,
        out_features=1,
        type='sine',
        periodic_transform_fn=dynamicsDubins.periodic_transform_fn
    )
    checkpoint_vanilla = torch.load(model_path_vanilla, map_location=device, weights_only=True)
    model_vanilla.load_state_dict(checkpoint_vanilla["model"])
    model_vanilla.to(device)
    model_vanilla.eval()

    class VanillaDeepReachVF:
        def __call__(self, batch):
            coords = batch["coords"]
            coords = coords.clone().detach().requires_grad_(True)
            periodic_coords = dynamicsDubins.periodic_transform_fn(coords)
            value = model_vanilla.net(periodic_coords)
            return {"model_in": coords, "model_out": value}

    policy_vanilla = VanillaDeepReachVF()

    model_path_vanilla_inversed = "dubins_3sec_vanilla_inversed.pth"
    model_vanilla_inversed = SingleBVPNet(
        in_features=9,
        hidden_features=512,
        num_hidden_layers=3,
        out_features=1,
        type='sine',
        periodic_transform_fn=dynamicsDubins.periodic_transform_fn
    )
    checkpoint_vanilla_inversed = torch.load(model_path_vanilla_inversed, map_location=device, weights_only=True)
    model_vanilla_inversed.load_state_dict(checkpoint_vanilla_inversed["model"])
    model_vanilla_inversed.to(device)
    model_vanilla_inversed.eval()

    class VanillaDeepReachVFInversed:
        def __call__(self, batch):
            coords = batch["coords"]
            coords = coords.clone().detach().requires_grad_(True)
            periodic_coords = dynamicsDubins.periodic_transform_fn(coords)
            value = model_vanilla_inversed.net(periodic_coords)
            return {"model_in": coords, "model_out": value}

    policy_vanilla_inversed = VanillaDeepReachVFInversed()
    
    return dynamics, dynamicsDubins, policy2, policy2_inversed, policy_vanilla, policy_vanilla_inversed, device

# ======================= Isaacs Model Setup =======================
def setup_isaacs_models():
    """Setup trained Isaacs models for comparison.
    
    This function loads the pre-trained Isaacs models for Dubins pursuit-evasion.
    Isaacs is a zero-sum game solver that learns optimal policies for both
    the evader (controller) and pursuer (disturbance) agents.
    
    Returns:
        tuple: (env, solver, evader_policy, pursuer_policy) or (None, None, None, None) if loading fails
    """
    # Configuration paths
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "config", "dubins_isaacs.yaml"))
    model_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "train_result", "isaacs_dubins_2", "model"))
    ctrl_step = 12600000  # Use latest controller checkpoint
    dstb_step = 12600000  # Use latest disturbance checkpoint
    
    # Force JAX to use CPU to avoid CUDA conflicts
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    
    try:
        # Load configuration
        cfg = OmegaConf.load(config_path)
        
        # Override device to CPU for evaluation
        cfg.solver.device = 'cpu'
        cfg.solver.rollout_env_device = 'cpu'
        
        # Construct environment
        print("== Loading Isaacs Environment ==")
        env = DubinsPursuitEvasionEnv(cfg.environment, cfg.agent, None)
        
        # Construct solver and load trained models
        print("== Loading Trained Isaacs Models ==")
        solver = ISAACS(cfg.solver, cfg.arch, cfg.environment.seed)
        
        # Load controller model
        if ctrl_step > 0:
            solver.ctrl.restore(ctrl_step, model_folder, verbose=True)
            print(f"Loaded Isaacs controller from step {ctrl_step}")
        
        # Load disturbance model
        if dstb_step > 0:
            solver.dstb.restore(dstb_step, model_folder, verbose=True)
            print(f"Loaded Isaacs disturbance from step {dstb_step}")
        
        # Create policy wrapper classes for compatibility
        class IsaacsEvaderPolicy:
            def __init__(self, solver, env):
                self.solver = solver
                self.env = env
            
            def __call__(self, state):
                """Get evader action from Isaacs controller."""
                # Convert state format: [x_e, y_e, theta_e, x_p, y_p, theta_p]
                obs = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    ctrl_action, _ = self.solver.ctrl.sample(obs)
                return ctrl_action.cpu().numpy().flatten()
            
            def get_value(self, state):
                """Get value function from Isaacs critic network.
                
                The critic network in Isaacs learns the value function V(s) which represents
                the expected return (cumulative reward) from state s under the optimal policy.
                This is different from the action networks (controller/disturbance) and provides
                the game-theoretic value of the current state.
                
                The ISAACS solver automatically computes optimal actions for both controller
                and disturbance, then evaluates the critic network to get the Q-value.
                """
                # Convert state format: [x_e, y_e, theta_e, x_p, y_p, theta_p]
                obs = np.array(state).reshape(1, -1)  # Add batch dimension
                with torch.no_grad():
                    # Use the ISAACS solver's value method which automatically computes
                    # optimal actions and evaluates the critic network
                    value = self.solver.value(obs)
                    return float(value[0])  # Return first (and only) value
        
        class IsaacsPursuerPolicy:
            def __init__(self, solver, env):
                self.solver = solver
                self.env = env
            
            def __call__(self, state, evader_action=None):
                """Get pursuer action from Isaacs disturbance."""
                # Convert state format: [x_e, y_e, theta_e, x_p, y_p, theta_p]
                obs = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    if evader_action is not None:
                        # Use evader action if provided
                        ctrl_action = torch.FloatTensor(evader_action).unsqueeze(0)
                        dstb_action, _ = self.solver.dstb.sample(obs, agents_action={"ctrl": ctrl_action.cpu().numpy()})
                    else:
                        # Sample without evader action
                        dstb_action, _ = self.solver.dstb.sample(obs)
                return dstb_action.cpu().numpy().flatten()
        
        evader_policy = IsaacsEvaderPolicy(solver, env)
        pursuer_policy = IsaacsPursuerPolicy(solver, env)
        
        return env, solver, evader_policy, pursuer_policy
        
    except Exception as e:
        print(f"Warning: Failed to load Isaacs models: {e}")
        print("Isaacs will not be available as a policy option")
        return None, None, None, None

# ======================= Trajectory Rollout Functions =======================


def rollout_mixed_policy_trajectory(initial_evader_state, initial_pursuer_state, H, dt, 
                                   evader_policy_type="dp", pursuer_policy_type="dp",
                                   base_dynamics=None, base_grid=None, base_all_values=None, base_all_values_inversed=None,
                                   dubins=None, dynamics=None, policy=None, policy_inversed=None, policy_vanilla=None, policy_vanilla_inversed=None, device=None,
                                   boundary_grid=None, boundary_brt_values=None, single_dynamics=None,
                                   isaacs_env=None, isaacs_evader_policy=None, isaacs_pursuer_policy=None):
    """Rollout trajectory with mixed policies for evader and pursuer."""
    
    traj_length = H
    pursuer_states = np.zeros((traj_length + 1, 3))
    evader_states = np.zeros((traj_length + 1, 3))
    relative_states = np.zeros((traj_length + 1, 3))
    evader_angular_vel = np.zeros(traj_length)
    pursuer_angular_vel = np.zeros(traj_length)
    pursuer_strategy = np.zeros(traj_length)
    vfs = np.zeros(traj_length)
    collisions = np.zeros(traj_length + 1, dtype=bool)
    
    pursuer_states[0] = initial_pursuer_state
    evader_states[0] = initial_evader_state
    rel = convert_to_rel_state(evader_states[0], pursuer_states[0])
    relative_states[0] = rel
    
    # Check initial collision
    collisions[0] = np.linalg.norm(relative_states[0][:2]) <= 0.360
    
    for t in tqdm(range(traj_length), desc=f"Mixed Policy Rollout ({evader_policy_type} vs {pursuer_policy_type})"):
        
        # Calculate all policies first
        xe, ye, theta_e = evader_states[t]
        xp, yp, theta_p = pursuer_states[t]
        
        # Calculate DP policy using relative state with line search for zero crossing
        # Find the time where V(relative_state, t) ≈ 0
        base_times = np.linspace(0.0, -3.0, 101)  # Match the times from setup_hj_reachability
        t_optimal, v_optimal, success = find_zero_crossing_time(
            base_grid, base_all_values, base_times, relative_states[t]
        )
        
      
        # Use the value function at the optimal time
        t_idx = np.argmin(np.abs(base_times - t_optimal))
        t_idx = -1
        V_optimal = base_all_values[t_idx]
        v_dp = base_grid.interpolate(V_optimal, relative_states[t])
        v_dp = float(v_dp)
        V_optimal_inversed = base_all_values_inversed[t_idx]
        # Get gradient at the optimal time
        grad_V_all = base_grid.grad_values(V_optimal)
        grad_V_dp = base_grid.interpolate(grad_V_all, relative_states[t])

        grad_V_all_inversed = base_grid.grad_values(V_optimal_inversed)
        grad_V_dp_inversed = base_grid.interpolate(grad_V_all_inversed, relative_states[t])
        
        if np.isnan(grad_V_dp).any():
            print(f"❌ NaN detected in DP gradient at step {t}. Stopping simulation.")
            break
        
        # Use the optimal time for control calculation
        #print(f"t_optimal: {t_optimal}")
        u_dp, d_dp = base_dynamics.optimal_control_and_disturbance(relative_states[t], t_optimal, grad_V_dp)

        u_dp_inversed, d_dp_inversed = base_dynamics.optimal_control_and_disturbance(relative_states[t], t_optimal, grad_V_dp_inversed)

        omega_A_dp = u_dp[0]
        if np.abs(grad_V_dp[2]) < 0.02:
            omega_B_dp = d_dp_inversed[0]
            pursuer_strategy_odp = 1
        else:
            omega_B_dp = d_dp[0]
            pursuer_strategy_odp = 0
        
        # Calculate MPC/Control Loop policy
        t_time = 3.0
        state_vec = torch.tensor(
            [t_time, xe, ye, theta_e, xp, yp, theta_p],
            dtype=torch.float32,
            device=device,
            requires_grad=True
        )
        
        # Add batch dimension
        x = state_vec.unsqueeze(0)
        
        # Evaluate MPC policy
        x.requires_grad_(True)
        policy_output = policy({"coords": dynamics.coord_to_input(x)})
        
        value_tensor = dynamics.io_to_value(policy_output["model_in"], policy_output["model_out"])
        v_mpc = value_tensor.cpu().detach().item()
        
        dvs = dynamics.io_to_dv(policy_output["model_in"], policy_output["model_out"].squeeze(dim=-1))
        grad_V_mpc = dvs[..., 1:4].cpu().detach().numpy()[0]  # Evader gradients (first 3 dimensions)
        grad_V_mpc_pursuer = dvs[..., 4:7].cpu().detach().numpy()[0]  # Pursuer gradients (last 3 dimensions)
        
        # Extract controls from MPC policy
        omega_A_mpc = dynamics.optimal_control(x[..., 1:], dvs[..., 1:])
        omega_A_mpc = omega_A_mpc.item()
        
        omega_B_mpc = dynamics.optimal_disturbance(x[..., 1:], dvs[..., 1:])
        omega_B_mpc = omega_B_mpc.item()

        # Calculate MPC policy inversed
        policy_output_inversed = policy_inversed({"coords": dynamics.coord_to_input(x)})
        value_tensor_inversed = dynamics.io_to_value(policy_output_inversed["model_in"], policy_output_inversed["model_out"])
        v_mpc_inversed = value_tensor_inversed.cpu().detach().item()

        dvs_inversed = dynamics.io_to_dv(policy_output_inversed["model_in"], policy_output_inversed["model_out"].squeeze(dim=-1))
        grad_V_mpc_inversed = dvs_inversed[..., 1:4].cpu().detach().numpy()[0]  # Evader gradients (first 3 dimensions)
        grad_V_mpc_inversed_pursuer = dvs_inversed[..., 4:7].cpu().detach().numpy()[0]  # Pursuer gradients (last 3 dimensions)

        omega_A_mpc_inversed = dynamics.optimal_control(x[..., 1:], dvs_inversed[..., 1:])
        omega_A_mpc_inversed = omega_A_mpc_inversed.item()

        omega_B_mpc_inversed = dynamics.optimal_disturbance(x[..., 1:], dvs_inversed[..., 1:])
        omega_B_mpc_inversed = omega_B_mpc_inversed.item()
        # print(grad_V_mpc_pursuer[2])
        if np.abs(grad_V_mpc_pursuer[2]) < 0.02:
            omega_B_mpc = omega_B_mpc_inversed
            pursuer_strategy_mpc = 1
        else:
            omega_B_mpc = omega_B_mpc
            pursuer_strategy_mpc = 0
        
        # Calculate Vanilla DeepReach policy
        x_vanilla = state_vec.unsqueeze(0)
        x_vanilla.requires_grad_(True)
        policy_output_vanilla = policy_vanilla({"coords": dynamics.coord_to_input(x_vanilla)})
        
        value_tensor_vanilla = dynamics.io_to_value(policy_output_vanilla["model_in"], policy_output_vanilla["model_out"])
        v_vanilla = value_tensor_vanilla.cpu().detach().item()
        
        dvs_vanilla = dynamics.io_to_dv(policy_output_vanilla["model_in"], policy_output_vanilla["model_out"].squeeze(dim=-1))
        grad_V_vanilla = dvs_vanilla[..., 1:4].cpu().detach().numpy()[0]  # Evader gradients (first 3 dimensions)
        grad_V_vanilla_pursuer = dvs_vanilla[..., 4:7].cpu().detach().numpy()[0]  # Pursuer gradients (last 3 dimensions)
        
        # Extract controls from Vanilla policy
        omega_A_vanilla = dynamics.optimal_control(x_vanilla[..., 1:], dvs_vanilla[..., 1:])
        omega_A_vanilla = omega_A_vanilla.item()
        
        omega_B_vanilla = dynamics.optimal_disturbance(x_vanilla[..., 1:], dvs_vanilla[..., 1:])
        omega_B_vanilla = omega_B_vanilla.item()

        # Calculate Vanilla DeepReach policy inversed
        x_vanilla_inversed = state_vec.unsqueeze(0)
        x_vanilla_inversed.requires_grad_(True)
        policy_output_vanilla_inversed = policy_vanilla_inversed({"coords": dynamics.coord_to_input(x_vanilla_inversed)})
        
        value_tensor_vanilla_inversed = dynamics.io_to_value(policy_output_vanilla_inversed["model_in"], policy_output_vanilla_inversed["model_out"])
        v_vanilla_inversed = value_tensor_vanilla_inversed.cpu().detach().item()
        
        dvs_vanilla_inversed = dynamics.io_to_dv(policy_output_vanilla_inversed["model_in"], policy_output_vanilla_inversed["model_out"].squeeze(dim=-1))
        grad_V_vanilla_inversed = dvs_vanilla_inversed[..., 1:4].cpu().detach().numpy()[0]  # Evader gradients (first 3 dimensions)
        grad_V_vanilla_inversed_pursuer = dvs_vanilla_inversed[..., 4:7].cpu().detach().numpy()[0]  # Pursuer gradients (last 3 dimensions)
        
        
        omega_A_vanilla_inversed = dynamics.optimal_control(x_vanilla_inversed[..., 1:], dvs_vanilla_inversed[..., 1:])
        omega_A_vanilla_inversed = omega_A_vanilla_inversed.item()
        
        omega_B_vanilla_inversed = dynamics.optimal_disturbance(x_vanilla_inversed[..., 1:], dvs_vanilla_inversed[..., 1:])
        omega_B_vanilla_inversed = omega_B_vanilla_inversed.item()
        # print(grad_V_vanilla_pursuer[2])

        if np.abs(grad_V_vanilla_pursuer[2]) < 0.02:
            omega_B_vanilla = omega_B_vanilla_inversed
            pursuer_strategy_vanilla = 1
        else:
            omega_B_vanilla = omega_B_vanilla
            pursuer_strategy_vanilla = 0
        
        # Select evader control based on policy type
        if evader_policy_type == "dp":
            # Check if evader needs safe boundary control
            safe_evader_control = get_safe_control(evader_states[t], boundary_grid, boundary_brt_values, single_dynamics, threshold=0.05)
            if safe_evader_control is not None:
                omega_A = safe_evader_control
                #print(f"⚠️ Evader using safe control at step {t} near boundary)")
            else:
                omega_A = omega_A_dp
            vfs[t] = v_dp
        elif evader_policy_type == "mpc":
            #print("Using MPC for evader")
            omega_A = omega_A_mpc
            vfs[t] = v_mpc
        elif evader_policy_type == "vanilla":
            #print("Using Vanilla for evader")
            omega_A = omega_A_vanilla
            vfs[t] = v_vanilla
        elif evader_policy_type == "isaacs":
            #print("Using Isaacs for evader")
            if isaacs_evader_policy is not None:
                # Create full state for Isaacs: [x_e, y_e, theta_e, x_p, y_p, theta_p]
                full_state = np.concatenate([evader_states[t], pursuer_states[t]])
                isaacs_action = isaacs_evader_policy(full_state)
                omega_A = float(isaacs_action[0])
                # Get value function from Isaacs critic network
                try:
                    vfs[t] = isaacs_evader_policy.get_value(full_state)
                except Exception as e:
                    print(f"Warning: Failed to get Isaacs value function: {e}, using placeholder")
                    vfs[t] = 0.0
            else:
                print("Warning: Isaacs evader policy not available, falling back to DP")
                omega_A = omega_A_dp
                vfs[t] = v_dp
        else:
            raise ValueError(f"Invalid evader policy type: {evader_policy_type}")
        
        # Select pursuer control based on policy type
        if pursuer_policy_type == "dp":
            # Check if pursuer needs safe boundary control
            pursuer_strategy[t] = pursuer_strategy_odp
            safe_pursuer_control = get_safe_control(pursuer_states[t], boundary_grid, boundary_brt_values, single_dynamics, threshold=0.05)
            if safe_pursuer_control is not None:
                omega_B = safe_pursuer_control
                #print(f"⚠️ Pursuer using safe control at step {t} near boundary)")
            else:
                omega_B = omega_B_dp
        elif pursuer_policy_type == "mpc":
            #print("Using MPC for pursuer")
            pursuer_strategy[t] = pursuer_strategy_mpc
            omega_B = omega_B_mpc
        elif pursuer_policy_type == "vanilla":
            pursuer_strategy[t] = pursuer_strategy_vanilla
            #print("Using Vanilla for pursuer")
            omega_B = omega_B_vanilla
        elif pursuer_policy_type == "isaacs":
            #print("Using Isaacs for pursuer")
            if isaacs_pursuer_policy is not None:
                # Create full state for Isaacs: [x_e, y_e, theta_e, x_p, y_p, theta_p]
                full_state = np.concatenate([evader_states[t], pursuer_states[t]])
                # Get evader action for coordinated Isaacs pursuer
                if evader_policy_type == "isaacs":
                    evader_action = np.array([omega_A])
                else:
                    evader_action = None
                isaacs_action = isaacs_pursuer_policy(full_state, evader_action)
                omega_B = float(isaacs_action[0])
                pursuer_strategy[t] = 0  # Isaacs pursuer strategy (placeholder)
            else:
                print("Warning: Isaacs pursuer policy not available, falling back to DP")
                omega_B = omega_B_dp
                pursuer_strategy[t] = pursuer_strategy_odp
        else:
            raise ValueError(f"Invalid pursuer policy type: {pursuer_policy_type}")

        evader_angular_vel[t] = float(omega_A)
        pursuer_angular_vel[t] = float(omega_B)
        
        # Step forward
        evader_states[t+1] = dubins.step(evader_states[t], jnp.array([omega_A]), jnp.zeros(1), 0.0, dt=dt)
        pursuer_states[t+1] = dubins.step(pursuer_states[t], jnp.array([omega_B]), jnp.zeros(1), 0.0, dt=dt)
        evader_states[t+1][2] = np.arctan2(np.sin(evader_states[t+1][2]), np.cos(evader_states[t+1][2]))
        pursuer_states[t+1][2] = np.arctan2(np.sin(pursuer_states[t+1][2]), np.cos(pursuer_states[t+1][2]))
        
        rel = convert_to_rel_state(evader_states[t+1], pursuer_states[t+1])
        relative_states[t+1] = rel
        
        # Check collision
        collisions[t+1] = np.linalg.norm(relative_states[t+1][:2]) <= 0.360
        
        if collisions[t+1]:
            print(f"⚠️ Collision detected at step {t+1}")
            break
    
    final_step = t + 1 if collisions[t+1] else t
    
    return {
        'evader_states': evader_states[:final_step+1],
        'pursuer_states': pursuer_states[:final_step+1],
        'relative_states': relative_states[:final_step+1],
        'evader_angular_vel': evader_angular_vel[:final_step],
        'pursuer_angular_vel': pursuer_angular_vel[:final_step],
        'vfs': vfs[:final_step],
        'collisions': collisions[:final_step+1],
        'final_step': final_step,
        'evader_policy': evader_policy_type,
        'pursuer_policy': pursuer_policy_type,
        'pursuer_strategy': pursuer_strategy[:final_step]
    }

# ... existing code ...

def compare_mixed_policies(initial_states, H=20, dt=0.05, save_results=True, plot=True,
                          base_dynamics=None, base_grid=None, base_all_values=None, base_all_values_inversed=None, base_target_values=None,
                          boundary_grid=None, boundary_brt_values=None, single_dynamics=None,
                          dynamics=None, dynamicsDubins=None, policy2=None, policy2_inversed=None, policy_vanilla=None, policy_vanilla_inversed=None, device=None,
                          isaacs_env=None, isaacs_evader_policy=None, isaacs_pursuer_policy=None):
    """Main function to compare different policy combinations."""
    
    # Setup HJ reachability if not provided
    if base_dynamics is None:
        print("Setting up HJ reachability...")
        base_dynamics, base_grid, base_all_values, base_all_values_inversed, base_target_values = setup_hj_reachability()
    
    # Setup boundary BRT if not provided
    if boundary_grid is None:
        print("Setting up boundary BRT...")
        boundary_grid, boundary_brt_values, single_dynamics = setup_boundary_brt()
    
    # Setup DeepReach models if not provided
    if dynamics is None:
        print("Setting up DeepReach models...")
        dynamics, dynamicsDubins, policy2, policy2_inversed, policy_vanilla, policy_vanilla_inversed, device = setup_deepreach_models()
    
    # Setup Isaacs models if not provided
    if 'isaacs_env' not in locals() or isaacs_env is None:
        print("Setting up Isaacs models...")
        isaacs_env, isaacs_solver, isaacs_evader_policy, isaacs_pursuer_policy = setup_isaacs_models()
    
    dubins = Dubins()
    
    all_results = {}
    
    # Define policy combinations to test
    policy_combinations = [
        ("dp", "dp"),      # Both use DP
        ("dp", "mpc"),     # Evader DP, Pursuer MPC
        ("dp", "vanilla"), # Evader DP, Pursuer Vanilla
        ("dp", "isaacs"),  # Evader DP, Pursuer Isaacs
        ("mpc", "dp"),     # Evader MPC, Pursuer DP
        ("mpc", "mpc"),    # Both use MPC
        ("mpc", "vanilla"), # Evader MPC, Pursuer Vanilla
        ("mpc", "isaacs"), # Evader MPC, Pursuer Isaacs
        ("vanilla", "dp"), # Evader Vanilla, Pursuer DP
        ("vanilla", "mpc"), # Evader Vanilla, Pursuer MPC
        ("vanilla", "vanilla"), # Both use Vanilla
        ("vanilla", "isaacs"), # Evader Vanilla, Pursuer Isaacs
        ("isaacs", "dp"),  # Evader Isaacs, Pursuer DP
        ("isaacs", "mpc"), # Evader Isaacs, Pursuer MPC
        ("isaacs", "vanilla"), # Evader Isaacs, Pursuer Vanilla
        ("isaacs", "isaacs"), # Both use Isaacs
    ]
    
    for i, (evader_init, pursuer_init) in enumerate(initial_states):
        print(f"\n=== Trajectory {i+1}/{len(initial_states)} ===")
        print(f"Evader initial state: {evader_init}")
        print(f"Pursuer initial state: {pursuer_init}")
        
        trajectory_results = {}
        
        for evader_policy, pursuer_policy in policy_combinations:
            print(f"\n--- Testing {evader_policy.upper()} vs {pursuer_policy.upper()} ---")
            
            # Rollout mixed policy trajectory
            mixed_results = rollout_mixed_policy_trajectory(
                evader_init, pursuer_init, H, dt,
                evader_policy_type=evader_policy,
                pursuer_policy_type=pursuer_policy,
                base_dynamics=base_dynamics,
                base_grid=base_grid,
                base_all_values=base_all_values,
                base_all_values_inversed=base_all_values_inversed,
                dubins=dubins,
                dynamics=dynamicsDubins,
                policy=policy2,
                policy_inversed=policy2_inversed,
                policy_vanilla=policy_vanilla,
                policy_vanilla_inversed=policy_vanilla_inversed,
                device=device,
                boundary_grid=boundary_grid,
                boundary_brt_values=boundary_brt_values,
                single_dynamics=single_dynamics,
                isaacs_env=isaacs_env,
                isaacs_evader_policy=isaacs_evader_policy,
                isaacs_pursuer_policy=isaacs_pursuer_policy
            )
            
            trajectory_results[f'{evader_policy}_vs_{pursuer_policy}'] = mixed_results
        
        all_results[f'trajectory_{i}'] = {
            'initial_evader': evader_init,
            'initial_pursuer': pursuer_init,
            'policy_results': trajectory_results
        }
        
        # Plot all policy combinations for this trajectory
        if plot:
            plot_mixed_policy_comparison(trajectory_results, i)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mixed_policy_comparison_results_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"\nResults saved to {filename}")
    
    return all_results

def compute_time_to_capture_stats(all_results, dt, save_summary=True):
    """Aggregate time-to-capture statistics per policy across trajectories.

    Returns a dict: { policy_name: { 'capture_rate': float, 'mean_ttc': float, 'num_cases': int, 'false_safe_rate': float, 'mean_min_distance': float } }
    where mean_ttc ignores non-captures (NaNs).
    """
    policy_to_ttcs = {}
    policy_to_false_safe = {}
    policy_to_min_distances = {}
    
    # Track capture outcomes per trajectory for identifying differences
    trajectory_capture_outcomes = {}

    for traj_key, traj_data in all_results.items():
        trajectory_capture_outcomes[traj_key] = {}
        
        for policy_name, results in traj_data['policy_results'].items():
            collisions = results['collisions']
            vfs = results['vfs']
            
            # Find first capture index, if any
            hit_indices = np.where(collisions)[0]
            if hit_indices.size > 0:
                ttc = hit_indices[0] * dt
                captured = True
            else:
                ttc = np.nan
                captured = False
            
            # Check for false safe: positive initial value function but captured
            false_safe = False
            if len(vfs) > 0 and captured:
                initial_vf = vfs[0]  # Value function at first timestep
                false_safe = initial_vf > 0  # Positive value function indicates safety
            
            # Calculate minimum relative distance over the trajectory
            relative_states = results['relative_states']
            distances = np.linalg.norm(relative_states[:, :2], axis=1)  # 2D distances
            min_distance = float(np.min(distances))
            
            policy_to_ttcs.setdefault(policy_name, []).append(ttc)
            policy_to_false_safe.setdefault(policy_name, []).append(false_safe)
            policy_to_min_distances.setdefault(policy_name, []).append(min_distance)
            trajectory_capture_outcomes[traj_key][policy_name] = captured

    summary = {}
    for policy_name, ttcs in policy_to_ttcs.items():
        ttcs_arr = np.array(ttcs, dtype=float)
        false_safe_arr = np.array(policy_to_false_safe[policy_name], dtype=bool)
        min_distances_arr = np.array(policy_to_min_distances[policy_name], dtype=float)
        
        captured_mask = ~np.isnan(ttcs_arr)
        capture_rate = float(np.mean(captured_mask)) if ttcs_arr.size > 0 else 0.0
        
        # Calculate false safe rate: percentage of all cases that had positive initial value function but got captured
        false_safe_rate = float(np.mean(false_safe_arr)) if false_safe_arr.size > 0 else 0.0
        
        # Calculate mean minimum distance across all cases
        mean_min_distance = float(np.mean(min_distances_arr)) if min_distances_arr.size > 0 else np.nan
        
        # Calculate mean TTC for captured cases only, excluding immediate captures (TTC = 0)
        if np.any(captured_mask):
            captured_ttcs = ttcs_arr[captured_mask]
            # Filter out immediate captures (TTC = 0)
            non_immediate_captures = captured_ttcs[captured_ttcs > 0]
            if len(non_immediate_captures) > 0:
                mean_ttc = float(np.mean(non_immediate_captures))
            else:
                # All captures were immediate
                mean_ttc = np.nan
        else:
            mean_ttc = np.nan
            
        summary[policy_name] = {
            'capture_rate': capture_rate,
            'mean_ttc': mean_ttc,
            'num_cases': int(ttcs_arr.size),
            'false_safe_rate': false_safe_rate,
            'mean_min_distance': mean_min_distance,
        }

    # Pretty print
    print("\nTime-to-Capture Statistics (per policy):")
    for policy_name, stats in summary.items():
        cr_pct = 100.0 * stats['capture_rate']
        fs_pct = 100.0 * stats['false_safe_rate']
        mean_ttc_str = f"{stats['mean_ttc']:.3f}s" if not np.isnan(stats['mean_ttc']) else "N/A"
        mean_min_dist_str = f"{stats['mean_min_distance']:.3f}" if not np.isnan(stats['mean_min_distance']) else "N/A"
        print(f"  - {policy_name}: capture_rate={cr_pct:.1f}% over {stats['num_cases']} cases, mean_ttc={mean_ttc_str}, false_safe_rate={fs_pct:.1f}%, mean_min_distance={mean_min_dist_str}")

    # Identify configurations with different capture rates (only DP vs MPC, excluding vanilla)
    print("\nAnalyzing configurations with different capture rates (DP vs MPC only)...")
    different_capture_configs = []
    
    # Define the policy combinations to check (excluding vanilla)
    dp_mpc_combinations = ["dp_vs_dp", "dp_vs_mpc", "mpc_vs_dp", "mpc_vs_mpc"]
    
    for traj_key, traj_data in all_results.items():
        capture_outcomes = trajectory_capture_outcomes[traj_key]
        
        # Check if there are differences in capture outcomes for DP/MPC combinations only
        dp_mpc_outcomes = [capture_outcomes.get(policy, False) for policy in dp_mpc_combinations]
        if len(set(dp_mpc_outcomes)) > 1:  # If there are different outcomes among DP/MPC
            different_capture_configs.append({
                'trajectory_key': traj_key,
                'initial_evader': traj_data['initial_evader'],
                'initial_pursuer': traj_data['initial_pursuer'],
                'capture_outcomes': {k: v for k, v in capture_outcomes.items() if k in dp_mpc_combinations},
                'num_captured': sum(dp_mpc_outcomes),
                'num_total': len(dp_mpc_combinations)
            })
    
    print(f"Found {len(different_capture_configs)} configurations with different capture rates")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save configurations with different capture rates
    if save_summary and len(different_capture_configs) > 0:
        
        # Save as CSV for easy analysis (DP vs MPC only)
        csv_filename = f"different_capture_configs_{timestamp}.csv"
        with open(csv_filename, 'w') as f:
            f.write("Trajectory,Evader_X,Evader_Y,Evader_Theta,Pursuer_X,Pursuer_Y,Pursuer_Theta,")
            f.write("dp_vs_dp,dp_vs_mpc,mpc_vs_dp,mpc_vs_mpc,")
            f.write("Num_Captured,Num_Total,Capture_Rate\n")
            
            for config in different_capture_configs:
                evader = config['initial_evader']
                pursuer = config['initial_pursuer']
                outcomes = config['capture_outcomes']
                
                f.write(f"{config['trajectory_key']},{evader[0]:.6f},{evader[1]:.6f},{evader[2]:.6f},")
                f.write(f"{pursuer[0]:.6f},{pursuer[1]:.6f},{pursuer[2]:.6f},")
                
                # Write capture outcomes for DP/MPC policy combinations only
                for policy in dp_mpc_combinations:
                    captured = outcomes.get(policy, False)
                    f.write(f"{1 if captured else 0},")
                
                capture_rate = config['num_captured'] / config['num_total']
                f.write(f"{config['num_captured']},{config['num_total']},{capture_rate:.3f}\n")
        
        print(f"Different capture configurations also saved as CSV: {csv_filename}")
        
        # Print summary of different configurations (DP vs MPC only)
        print(f"\nSummary of configurations with different capture rates (DP vs MPC only):")
        for config in different_capture_configs:
            print(f"  {config['trajectory_key']}: {config['num_captured']}/{config['num_total']} policies captured")
            print(f"    Evader: [{config['initial_evader'][0]:.3f}, {config['initial_evader'][1]:.3f}, {config['initial_evader'][2]:.3f}]")
            print(f"    Pursuer: [{config['initial_pursuer'][0]:.3f}, {config['initial_pursuer'][1]:.3f}, {config['initial_pursuer'][2]:.3f}]")

    # Save summary to file for easy table creation
    if save_summary:
        
        # Also save as CSV for easy table creation
        csv_filename = f"bust_capture_statistics_summary_{timestamp}.csv"
        with open(csv_filename, 'w') as f:
            f.write("Policy,Capture_Rate_Percent,Mean_Time_to_Capture_Seconds,Num_Cases,False_Safe_Rate_Percent,Mean_Min_Distance\n")
            for policy_name, stats in summary.items():
                cr_pct = 100.0 * stats['capture_rate']
                fs_pct = 100.0 * stats['false_safe_rate']
                mean_ttc = stats['mean_ttc'] if not np.isnan(stats['mean_ttc']) else "N/A"
                mean_min_dist = stats['mean_min_distance'] if not np.isnan(stats['mean_min_distance']) else "N/A"
                f.write(f"{policy_name},{cr_pct:.1f},{mean_ttc},{stats['num_cases']},{fs_pct:.1f},{mean_min_dist}\n")
        print(f"Statistics summary also saved as CSV: {csv_filename}")

    return summary

def plot_mixed_policy_comparison(trajectory_results, trajectory_idx):
    """Plot comparison between different policy combinations."""
    
    policy_combinations = list(trajectory_results.keys())
    num_policies = len(policy_combinations)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Mixed Policy Comparison {trajectory_idx + 1}', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']
    
    for idx, (policy_name, results) in enumerate(trajectory_results.items()):
        color = colors[idx % len(colors)]
        
        # Create descriptive legend labels
        evader_policy = results['evader_policy'].upper()
        pursuer_policy = results['pursuer_policy'].upper()
        legend_label = f"{evader_policy} vs {pursuer_policy}"
        
        # Plot trajectories
        results['pursuer_strategy'] = np.array(results['pursuer_strategy'])
        where_pursuer_strategy_1 = np.where(results['pursuer_strategy'] == 1)[0]
        where_pursuer_strategy_0 = np.where(results['pursuer_strategy'] == 0)[0]
        # offset = idx * 0.01
        offset = 0
        # Helper function to find consecutive groups
        def find_consecutive_groups(indices):
            if len(indices) == 0:
                return []
            groups = []
            current_group = [indices[0]]
            for i in range(1, len(indices)):
                if indices[i] == indices[i-1] + 1:
                    current_group.append(indices[i])
                else:
                    current_group.append(current_group[-1] + 1)
                    groups.append(current_group)
                    current_group = [indices[i]]
            groups.append(current_group)
            return groups
        
        # Plot evader trajectory
        axes[0, 0].plot(results['evader_states'][:, 0] + offset, results['evader_states'][:, 1] + offset, 
                       color=color, linestyle='-.', linewidth=2, 
                       label=f'Evader ({legend_label})')
        
        # Plot pursuer trajectory with different styles for different strategies
        strategy_1_groups = find_consecutive_groups(where_pursuer_strategy_1)
        strategy_0_groups = find_consecutive_groups(where_pursuer_strategy_0)
        
        # Plot strategy 1 segments
        # Plot strategy 0 segments
        for i, group in enumerate(strategy_0_groups):
            label = f'Pursuer ({legend_label})' if i == 0 else None
            axes[0, 0].plot(results['pursuer_states'][group, 0] + offset, results['pursuer_states'][group, 1] + offset, 
                           color=color, linestyle='-', linewidth=2, label=label)
        for i, group in enumerate(strategy_1_groups):
            axes[0, 0].plot(results['pursuer_states'][group, 0] + offset, results['pursuer_states'][group, 1] + offset, 
                           color=color, linestyle='--', linewidth=2, label=None)
        
        
        # Plot circle of radius 0.360 around final pursuer position
        # final_pursuer_pos = results['pursuer_states'][-1]
        # circle = plt.Circle((final_pursuer_pos[0], final_pursuer_pos[1]), 0.360, 
                        #    color=color, fill=False, linestyle=':', alpha=0.7, linewidth=2)
        # axes[0, 0].add_patch(circle)
        
        # Mark initial positions with squares
        initial_evader_pos = results['evader_states'][0]
        initial_pursuer_pos = results['pursuer_states'][0]
        
        # Evader initial position (filled square)
        axes[0, 0].plot(initial_evader_pos[0] + offset, initial_evader_pos[1] + offset, 's', 
                       color=color, markersize=8, markeredgewidth=2, 
                       markerfacecolor=color, markeredgecolor='black')
        
        # Pursuer initial position (hollow square)
        axes[0, 0].plot(initial_pursuer_pos[0] + offset, initial_pursuer_pos[1] + offset, 's', 
                       color=color, markersize=8, markeredgewidth=2, 
                       markerfacecolor='none', markeredgecolor=color)
        
        # Plot value function
        if len(results['vfs']) > 0:
            for i, group in enumerate(strategy_0_groups):
                label = f'({legend_label})' if i == 0 else None
                axes[0, 1].plot(group, results['vfs'][group], 
                               color=color, linewidth=2, label=label)
            for i, group in enumerate(strategy_1_groups):
                axes[0, 1].plot(group, results['vfs'][group], 
                               color=color, linewidth=2, linestyle='--', label=None)
        
        # Plot relative distance
        distances = np.linalg.norm(results['relative_states'][:, :2], axis=1)
        for i, group in enumerate(strategy_0_groups):
            label = f'({legend_label})' if i == 0 else None
            axes[1, 0].plot(group, distances[group], 
                           color=color, linewidth=2, label=label)
        for i, group in enumerate(strategy_1_groups):
            axes[1, 0].plot(group, distances[group], 
                           color=color, linewidth=2, linestyle='--', label=None)
        
        # Plot controls
        if len(results['evader_angular_vel']) > 0:
            axes[1, 1].plot(range(len(results['evader_angular_vel'])), 
                           results['evader_angular_vel'], 
                           color=color, linestyle='-', linewidth=2, 
                           label=f'Evader Control ({legend_label})')
    
    # Add collision threshold lines
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Collision Threshold')
    axes[1, 0].axhline(y=0.360, color='red', linestyle='--', alpha=0.7, label='Collision Radius')
    
    # Set labels and titles
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_title('Agent Trajectories')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')
    
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Value Function')
    axes[0, 1].set_title('Value Function Evolution (Using Evader Value )')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Relative Distance')
    axes[1, 0].set_title('Relative Distance Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Angular Velocity')
    axes[1, 1].set_title('Evader Control Inputs')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'bust_{trajectory_idx + 1}.png', dpi=300, bbox_inches='tight')
    plt.show()

    
# ======================= Example Usage =======================
if __name__ == "__main__":
    # Option to load existing results instead of running new simulation
    load_existing_results = False  # Set to True to load from file
    results_file_path = "mixed_policy_comparison_results_20250822_181532.pkl"  # Adjust path as needed
    
    if load_existing_results:
        print(f"Loading existing results from {results_file_path}...")
        try:
            with open(results_file_path, 'rb') as f:
                results = pickle.load(f)
            print("Results loaded successfully!")
        except FileNotFoundError:
            print(f"Error: Results file {results_file_path} not found. Running new simulation...")
            load_existing_results = False
        except Exception as e:
            print(f"Error loading results: {e}. Running new simulation...")
            load_existing_results = False
    
    if not load_existing_results:
        # Build initial states near the BRT boundary (relative Air3D BRT)
        print("Setting up HJ reachability...")
        base_dynamics, base_grid, base_all_values, base_all_values_inversed, base_target_values = setup_hj_reachability()
        V_final = base_all_values[-1]

        # Configure sampling
        num_pairs = 2  # streamlined test size
        epsilon = 0.03  # band around boundary |V| <= epsilon
        side = 'both'   # 'inside' or 'outside' or 'both'
        x_bounds = (-3.0, 3.0)
        y_bounds = (-2.0, 2.0)

        print("sampling initial states...")

        # Set random seed for reproducible sampling
        rng = np.random.default_rng(seed=0)  # You can change this seed value

        initial_states = sample_initial_states_near_brt(
            base_grid, V_final, num_pairs, epsilon=epsilon, side=side,
            x_bounds=x_bounds, y_bounds=y_bounds, rng=rng
        )

        # initial_states = [
        #     (np.array([-0.3, -0.2, np.pi/4]), np.array([0.3, 0.2, 3*np.pi/2])),  # Different angles
        # ]

        # Setup Isaacs models
        print("Setting up Isaacs models...")
        isaacs_env, isaacs_solver, isaacs_evader_policy, isaacs_pursuer_policy = setup_isaacs_models()

        # Run comparison with pre-computed setups
        results = compare_mixed_policies(initial_states, H=60, dt=0.05, save_results=False, plot=True,
                                       base_dynamics=base_dynamics, base_grid=base_grid, base_all_values=base_all_values,
                                       base_all_values_inversed=base_all_values_inversed,
                                       base_target_values=base_target_values,
                                       isaacs_env=isaacs_env, isaacs_evader_policy=isaacs_evader_policy, isaacs_pursuer_policy=isaacs_pursuer_policy)

    # Compute and print time-to-capture statistics across policies
    compute_time_to_capture_stats(results, dt=0.05)
    
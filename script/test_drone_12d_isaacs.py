# --------------------------------------------------------
# Test script for ISAACS 12D Drone pursuit-evasion model
# --------------------------------------------------------

import os
import sys
import copy
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from omegaconf import OmegaConf

# Force JAX to use CPU to avoid CUDA conflicts
import jax
jax.config.update('jax_platform_name', 'cpu')

from agent import ISAACS
from simulators import Drone_12D_PursuitEvasionEnv
from utils.eval import evaluate_zero_sum


def load_trained_model(config_path, model_folder, ctrl_step, dstb_step):
    """Load trained ISAACS model."""
    # Load configuration
    cfg = OmegaConf.load(config_path)
    
    # Override device to CPU for evaluation
    cfg.solver.device = 'cpu'
    cfg.solver.rollout_env_device = 'cpu'
    
    # Construct environment
    print("== Loading Environment ==")
    env = Drone_12D_PursuitEvasionEnv(cfg.environment, cfg.agent, None)
    env.report()
    
    # Construct solver and load trained models
    print("== Loading Trained Models ==")
    solver = ISAACS(cfg.solver, cfg.arch, cfg.environment.seed)
    
    # Load controller model
    if ctrl_step > 0:
        solver.ctrl.restore(ctrl_step, model_folder, verbose=True)
        print(f"Loaded controller from step {ctrl_step}")
    
    # Load disturbance model
    if dstb_step > 0:
        solver.dstb.restore(dstb_step, model_folder, verbose=True)
        print(f"Loaded disturbance from step {dstb_step}")
    
    return env, solver, cfg


def test_single_trajectory(env, solver, max_steps=300, initial_state=None, render=True):
    """Test a single trajectory with the trained model."""
    print(f"\n== Testing Single Trajectory (max {max_steps} steps) ==")
    
    # Reset environment
    if initial_state is not None:
        obs = env.reset(state=initial_state)
    else:
        obs = env.reset()
    
    trajectory = []
    total_reward = 0
    step = 0
    
    while step < max_steps:
        # Get actions from trained models
        with torch.no_grad():
            ctrl_action, _ = solver.ctrl.sample(torch.FloatTensor(obs).unsqueeze(0))
            dstb_action, _ = solver.dstb.sample(
                torch.FloatTensor(obs).unsqueeze(0), 
                agents_action={"ctrl": ctrl_action.cpu().numpy()}
            )
        
        action = {
            'ctrl': ctrl_action.cpu().numpy().flatten(),
            'dstb': dstb_action.cpu().numpy().flatten()
        }
        
        # Take step
        obs, reward, done, info = env.step(action)
        
        # Record trajectory
        trajectory.append({
            'step': step,
            'state': env.state.copy(),
            'action': action,
            'reward': reward,
            'done': done,
            'info': info
        })
        
        total_reward += reward
        step += 1
        
        if render:
            # Calculate distance between drones
            evader_pos = np.array([env.state[0], env.state[2], env.state[4]])
            pursuer_pos = np.array([env.state[6], env.state[8], env.state[10]])
            distance = np.linalg.norm(evader_pos - pursuer_pos)
            
            print(f"Step {step}: Evader=({env.state[0]:.2f}, {env.state[2]:.2f}, {env.state[4]:.2f}) "
                  f"Pursuer=({env.state[6]:.2f}, {env.state[8]:.2f}, {env.state[10]:.2f}) "
                  f"Distance={distance:.2f} "
                  f"Reward={reward:.3f}")
        
        if done:
            break
    
    print(f"Trajectory completed in {step} steps")
    print(f"Total reward: {total_reward:.3f}")
    
    # Calculate final distance
    evader_pos = np.array([env.state[0], env.state[2], env.state[4]])
    pursuer_pos = np.array([env.state[6], env.state[8], env.state[10]])
    final_distance = np.linalg.norm(evader_pos - pursuer_pos)
    print(f"Final distance: {final_distance:.3f}")
    print(f"Termination reason: {info.get('done_type', 'unknown')}")
    
    return trajectory, total_reward, step, info


def test_multiple_trajectories(env, solver, num_trajectories=100, max_steps=300):
    """Test multiple trajectories and compute statistics."""
    print(f"\n== Testing {num_trajectories} Trajectories ==")
    
    results = []
    safety_violations = 0
    successful_evasions = 0
    
    for i in range(num_trajectories):
        print(f"Running trajectory {i+1}/{num_trajectories}")
        
        trajectory, reward, steps, info = test_single_trajectory(
            env, solver, max_steps, render=False
        )
        
        # Calculate final distance
        evader_pos = np.array([env.state[0], env.state[2], env.state[4]])
        pursuer_pos = np.array([env.state[6], env.state[8], env.state[10]])
        final_distance = np.linalg.norm(evader_pos - pursuer_pos)
        
        results.append({
            'trajectory_id': i,
            'reward': reward,
            'steps': steps,
            'final_distance': final_distance,
            'done_type': info.get('done_type', 'unknown'),
            'safety_violation': info.get('g_x', 0) <= 0
        })
        
        if info.get('g_x', 0) <= 0:
            safety_violations += 1
        else:
            successful_evasions += 1
    
    # Compute statistics
    safety_rate = successful_evasions / num_trajectories
    avg_reward = np.mean([r['reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    avg_distance = np.mean([r['final_distance'] for r in results])
    
    print(f"\n== Results Summary ==")
    print(f"Safety rate: {safety_rate:.3f} ({successful_evasions}/{num_trajectories})")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average final distance: {avg_distance:.3f}")
    print(f"Safety violations: {safety_violations}")
    
    return results


def visualize_trajectory(trajectory, env, save_path=None):
    """Visualize a single trajectory in 3D, XY projection, and Z/distance over time."""
    fig = plt.figure(figsize=(20, 12))
    
    # Extract positions
    evader_x = [t['state'][0] for t in trajectory]
    evader_y = [t['state'][2] for t in trajectory]
    evader_z = [t['state'][4] for t in trajectory]
    pursuer_x = [t['state'][6] for t in trajectory]
    pursuer_y = [t['state'][8] for t in trajectory]
    pursuer_z = [t['state'][10] for t in trajectory]
    
    # Calculate relative distances over time
    distances = []
    for i in range(len(trajectory)):
        evader_pos = np.array([evader_x[i], evader_y[i], evader_z[i]])
        pursuer_pos = np.array([pursuer_x[i], pursuer_y[i], pursuer_z[i]])
        distance = np.linalg.norm(evader_pos - pursuer_pos)
        distances.append(distance)
    
    # Calculate time steps
    time_steps = list(range(len(trajectory)))
    
    # Determine evader end color based on termination condition
    final_info = trajectory[-1]['info']
    done_type = final_info.get('done_type', 'unknown')
    
    if done_type == 'failure':
        # Evader captured or out of bounds - use black
        evader_end_color = 'black'
        evader_end_label = 'Evader End (Captured/Out of Bounds)'
    elif done_type == 'timeout':
        # Timeout - use orange
        evader_end_color = 'orange'
        evader_end_label = 'Evader End (Timeout)'
    else:
        # Normal end - use blue
        evader_end_color = 'blue'
        evader_end_label = 'Evader End'
    
    # 3D Plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot trajectories
    ax1.plot(evader_x, evader_y, evader_z, 'b-', linewidth=2, label='Evader')
    ax1.plot(pursuer_x, pursuer_y, pursuer_z, 'r-', linewidth=2, label='Pursuer')
    
    # Plot start and end points
    ax1.plot(evader_x[0], evader_y[0], evader_z[0], 'bo', markersize=10, label='Evader Start')
    ax1.plot(pursuer_x[0], pursuer_y[0], pursuer_z[0], 'ro', markersize=10, label='Pursuer Start')
    ax1.plot(pursuer_x[-1], pursuer_y[-1], pursuer_z[-1], 'r*', markersize=15, label='Pursuer End')
    ax1.plot(evader_x[-1], evader_y[-1], evader_z[-1], '*', color=evader_end_color, markersize=15, label=evader_end_label)
    
    # Plot collision ellipsoid around final pursuer position (truncated cone shape)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    U, V = np.meshgrid(u, v)
    
    horizontal_radius = env.goalR
    height = 0.75  # Same height as in boundary function
    
    X_ellipse = pursuer_x[-1] + horizontal_radius * np.cos(U) * np.sin(V)
    Y_ellipse = pursuer_y[-1] + horizontal_radius * np.sin(U) * np.sin(V)
    Z_ellipse = pursuer_z[-1] + height * np.cos(V)
    
    # Only show the bottom half (z <= 0.5 above pursuer) to match truncated cone
    mask = Z_ellipse <= pursuer_z[-1] + 0.5
    X_ellipse = np.where(mask, X_ellipse, np.nan)
    Y_ellipse = np.where(mask, Y_ellipse, np.nan)
    Z_ellipse = np.where(mask, Z_ellipse, np.nan)
    
    ax1.plot_surface(X_ellipse, Y_ellipse, Z_ellipse, alpha=0.3, color='red')
    
    # Set 3D plot limits and labels
    ax1.set_xlim(env.visual_bounds[0])
    ax1.set_ylim(env.visual_bounds[1])
    ax1.set_zlim(env.visual_bounds[2])
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('3D Drone Pursuit-Evasion Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # XY Projection Plot
    ax2 = fig.add_subplot(222)
    
    # Plot XY trajectories
    ax2.plot(evader_x, evader_y, 'b-', linewidth=2, label='Evader')
    ax2.plot(pursuer_x, pursuer_y, 'r-', linewidth=2, label='Pursuer')
    
    # Plot start and end points
    ax2.plot(evader_x[0], evader_y[0], 'bo', markersize=10, label='Evader Start')
    ax2.plot(pursuer_x[0], pursuer_y[0], 'ro', markersize=10, label='Pursuer Start')
    ax2.plot(pursuer_x[-1], pursuer_y[-1], 'r*', markersize=15, label='Pursuer End')
    ax2.plot(evader_x[-1], evader_y[-1], '*', color=evader_end_color, markersize=15, label=evader_end_label)
    
    # Plot collision circle around final pursuer position (XY projection)
    circle = plt.Circle((pursuer_x[-1], pursuer_y[-1]), env.goalR, 
                       color='red', alpha=0.3, linestyle='--', fill=False)
    ax2.add_patch(circle)
    
    # Set XY plot limits and labels
    ax2.set_xlim(env.visual_bounds[0])
    ax2.set_ylim(env.visual_bounds[1])
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('XY Projection of Drone Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Combined Z Position and Distance Over Time Plot
    ax3 = fig.add_subplot(223)
    
    # Create twin axes for different y-scales
    ax3_twin = ax3.twinx()
    
    # Plot Z trajectories on primary y-axis
    line1 = ax3.plot(time_steps, evader_z, 'b-', linewidth=2, label='Evader Z')
    line2 = ax3.plot(time_steps, pursuer_z, 'r-', linewidth=2, label='Pursuer Z')
    
    # Set labels and titles
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Z Position (Altitude)', color='black')
    ax3_twin.set_ylabel('Distance / Boundary Value', color='black')
    ax3.set_title('Z Position and Distance Over Time')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper right')
    
    ax3.grid(True, alpha=0.3)
    
    # Relative Distance Over Time Plot
    ax4 = fig.add_subplot(224)
    
    # Plot distance over time
    ax4.plot(time_steps, distances, 'g-', linewidth=2, label='Relative Distance')    
    boundary_values = [env.agent.dyn.boundary_fn(t['state']) for t in trajectory]
    ax4.plot(time_steps, boundary_values, 'm-', linewidth=2, label='Boundary Function')
   
    # Set distance plot limits and labels
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Distance / Boundary Value')
    ax4.set_title('Relative Distance and Boundary Function Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trajectory visualization saved to {save_path}")
    
    plt.show()


def test_different_scenarios(env, solver):
    """Test different initial scenarios."""
    print("\n== Testing Different Scenarios ==")
    
    scenarios = [
        {
            'name': 'Close Pursuit',
            'state': np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0])  # Pursuer close behind
        },
        {
            'name': 'Side Pursuit',
            'state': np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0])  # Pursuer from side
        },
        {
            'name': 'Corner Escape',
            'state': np.array([3.0, 0.0, 1.5, 0.0, 1.0, 0.0, 2.5, 0.0, 1.0, 0.0, 1.0, 0.0])  # Near corner
        },
        {
            'name': 'Center Start',
            'state': np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 1.0, 0.0])  # Center area
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- Testing {scenario['name']} ---")
        trajectory, reward, steps, info = test_single_trajectory(
            env, solver, max_steps=200, 
            initial_state=scenario['state'], render=False
        )
        
        # Calculate final distance
        evader_pos = np.array([env.state[0], env.state[2], env.state[4]])
        pursuer_pos = np.array([env.state[6], env.state[8], env.state[10]])
        final_distance = np.linalg.norm(evader_pos - pursuer_pos)
        
        print(f"Result: {info.get('done_type', 'unknown')}, "
              f"Final distance: {final_distance:.3f}, "
              f"Steps: {steps}, Reward: {reward:.3f}")


def main():
    # Hardcoded parameters
    config_path = '../config/isaacs_12dDrones.yaml'
    model_folder = '../train_result/isaacs_drone_12d/model'
    ctrl_step = 15000001  # 0 for latest ctrl-15000001
    dstb_step = 15000001  # 0 for latest dstb-15000001
    num_trajectories = 2
    max_steps = 300
    visualize = True
    save_plots = 'test_results'
    

    # Create save directory if it doesn't exist
    if save_plots and not os.path.exists(save_plots):
        os.makedirs(save_plots)
    
    # Load model
    env, solver, cfg = load_trained_model(
        config_path, model_folder, ctrl_step, dstb_step
    )
    
    # Test single trajectory
    print("\n" + "="*50)
    trajectory, reward, steps, info = test_single_trajectory(env, solver, max_steps)
    
    if visualize:
        save_path = os.path.join(save_plots, 'drone_trajectory.png') if save_plots else None
        visualize_trajectory(trajectory, env, save_path)
    
    # Test multiple trajectories
    print("\n" + "="*50)
    results = test_multiple_trajectories(env, solver, num_trajectories, max_steps)
    
    # Test different scenarios
    print("\n" + "="*50)
    test_different_scenarios(env, solver)
    
    print("\n== Testing Complete ==")


if __name__ == "__main__":
    main()

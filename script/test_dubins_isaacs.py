# --------------------------------------------------------
# Test script for ISAACS Dubins pursuit-evasion model
# --------------------------------------------------------

import os
import sys
import copy
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Force JAX to use CPU to avoid CUDA conflicts
import jax
jax.config.update('jax_platform_name', 'cpu')

from agent import ISAACS
from simulators import DubinsPursuitEvasionEnv


def load_trained_model(config_path, model_folder, ctrl_step, dstb_step):
    """Load trained ISAACS model."""
    # Load configuration
    cfg = OmegaConf.load(config_path)
    
    # Override device to CPU for evaluation
    cfg.solver.device = 'cpu'
    cfg.solver.rollout_env_device = 'cpu'
    
    # Construct environment
    print("== Loading Environment ==")
    env = DubinsPursuitEvasionEnv(cfg.environment, cfg.agent, None)
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
            print(f"Step {step}: Evader=({env.state[0]:.2f}, {env.state[1]:.2f}, {env.state[2]:.2f}) "
                  f"Pursuer=({env.state[3]:.2f}, {env.state[4]:.2f}, {env.state[5]:.2f}) "
                  f"Distance={np.sqrt((env.state[0]-env.state[3])**2 + (env.state[1]-env.state[4])**2):.2f} "
                  f"Reward={reward:.3f}")
        
        if done:
            break
    
    print(f"Trajectory completed in {step} steps")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final distance: {np.sqrt((env.state[0]-env.state[3])**2 + (env.state[1]-env.state[4])**2):.3f}")
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
        
        results.append({
            'trajectory_id': i,
            'reward': reward,
            'steps': steps,
            'final_distance': np.sqrt((env.state[0]-env.state[3])**2 + (env.state[1]-env.state[4])**2),
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
    """Visualize a single trajectory."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Extract positions
    evader_x = [t['state'][0] for t in trajectory]
    evader_y = [t['state'][1] for t in trajectory]
    pursuer_x = [t['state'][3] for t in trajectory]
    pursuer_y = [t['state'][4] for t in trajectory]
    
    # Plot trajectories
    ax.plot(evader_x, evader_y, 'b-', linewidth=2, label='Evader')
    ax.plot(pursuer_x, pursuer_y, 'r-', linewidth=2, label='Pursuer')
    
    # Plot start and end points
    ax.plot(evader_x[0], evader_y[0], 'bo', markersize=10, label='Evader Start')
    ax.plot(pursuer_x[0], pursuer_y[0], 'ro', markersize=10, label='Pursuer Start')
    ax.plot(pursuer_x[-1], pursuer_y[-1], 'r*', markersize=15, label='Pursuer End')
    
    # Determine evader end color based on termination condition
    final_info = trajectory[-1]['info']
    done_type = final_info.get('done_type', 'unknown')
    
    if done_type == 'failure':
        # Evader captured or out of bounds - use red
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
    
    # Plot evader end with appropriate color
    ax.plot(evader_x[-1], evader_y[-1], '*', color=evader_end_color, markersize=15, label=evader_end_label)
    
    # Plot collision radius around final pursuer position
    circle = plt.Circle((pursuer_x[-1], pursuer_y[-1]), env.goalR, 
                       color='red', alpha=0.3, linestyle='--', fill=False)
    ax.add_patch(circle)
    
    # Set plot limits and labels
    ax.set_xlim(env.visual_bounds[0])
    ax.set_ylim(env.visual_bounds[1])
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Dubins Pursuit-Evasion Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
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
            'state': np.array([0.0, 0.0, 0.0, 1.0, 0.0, np.pi])  # Pursuer close behind
        },
        {
            'name': 'Side Pursuit',
            'state': np.array([0.0, 0.0, 0.0, 0.0, 1.0, -np.pi/2])  # Pursuer from side
        },
        {
            'name': 'Corner Escape',
            'state': np.array([2.5, 1.5, np.pi/4, 2.0, 1.0, 0.0])  # Near corner
        },
        {
            'name': 'Center Start',
            'state': np.array([0.0, 0.0, 0.0, 0.5, 0.5, np.pi/4])  # Center area
        }
    ]
    
    for scenario in scenarios:
        print(f"\n--- Testing {scenario['name']} ---")
        trajectory, reward, steps, info = test_single_trajectory(
            env, solver, max_steps=200, 
            initial_state=scenario['state'], render=False
        )
        
        final_distance = np.sqrt((env.state[0]-env.state[3])**2 + (env.state[1]-env.state[4])**2)
        print(f"Result: {info.get('done_type', 'unknown')}, "
              f"Final distance: {final_distance:.3f}, "
              f"Steps: {steps}, Reward: {reward:.3f}")


def main():
    # Hardcoded parameters
    config_path = '../config/dubins_isaacs.yaml'
    model_folder = '../train_result/isaacs_dubins_2/model'
    ctrl_step = 12600000  # 0 for latest ctrl-12600000
    dstb_step = 12600000  # 0 for latest dstb-12600000
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
        save_path = os.path.join(save_plots, 'trajectory.png') if save_plots else None
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

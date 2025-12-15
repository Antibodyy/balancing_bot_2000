"""Test velocity tracking without distance limits.

This script runs a simple velocity tracking scenario where the robot
continuously tracks a target velocity without any distance constraints.

Usage:
    python3 scripts/debug/test_velocity_tracking.py
    python3 scripts/debug/test_velocity_tracking.py --velocity 1.5 --duration 10.0
    python3 scripts/debug/test_velocity_tracking.py --velocity 2.0 --viewer
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from simulation.mpc_simulation import SimulationConfig, MPCSimulation
from mpc import ReferenceCommand, ReferenceMode, MPCConfig
from robot_dynamics.parameters import (
    PITCH_INDEX, PITCH_RATE_INDEX,
    POSITION_INDEX, VELOCITY_INDEX,
    YAW_INDEX, YAW_RATE_INDEX
)

# Parse arguments
parser = argparse.ArgumentParser(description='Test velocity tracking')
parser.add_argument('--velocity', type=float, default=2.0,
                    help='Target velocity in m/s (default: 2.0)')
parser.add_argument('--duration', type=float, default=10.0,
                    help='Simulation duration in seconds (default: 10.0, ignored in viewer mode)')
parser.add_argument('--initial-pitch', type=float, default=0.0,
                    help='Initial pitch perturbation in degrees (default: 0.0)')
parser.add_argument('--viewer', action='store_true',
                    help='Run with interactive MuJoCo viewer (runs until closed or robot falls)')
if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = parser.parse_args([])

# Convert initial pitch to radians
initial_pitch_rad = np.deg2rad(args.initial_pitch)

print("\n" + "="*80)
print("VELOCITY TRACKING TEST")
print("="*80)
print(f"Configuration:")
print(f"  Target velocity: {args.velocity} m/s")
print(f"  Duration: {args.duration} s" + (" (ignored in viewer mode)" if args.viewer else ""))
print(f"  Initial pitch: {args.initial_pitch} deg ({initial_pitch_rad:.4f} rad)")
print(f"  Viewer mode: {'ENABLED' if args.viewer else 'DISABLED'}")
print("="*80)

# Create simulation configuration
config = SimulationConfig(
    model_path='mujoco_sim/robot_model.xml',
    robot_params_path='config/simulation/robot_params.yaml',
    mpc_params_path='config/simulation/mpc_params.yaml',
    estimator_params_path='config/simulation/estimator_params.yaml',
    duration_s=args.duration,
)

# Create simulation
simulation = MPCSimulation(config)

# Static velocity reference command
reference_command = ReferenceCommand(
    mode=ReferenceMode.VELOCITY,
    velocity_mps=args.velocity,
    yaw_rate_radps=0.0,
)

# Run simulation
if args.viewer:
    print("\nLaunching viewer...")
    print("Close the viewer window to end simulation and see results.")
    result = simulation.run_with_viewer(
        initial_pitch_rad=initial_pitch_rad,
        reference_command=reference_command,
    )
else:
    print("\nRunning simulation...")
    result = simulation.run(
        initial_pitch_rad=initial_pitch_rad,
        reference_command=reference_command,
    )

# Print results
print("\n" + "="*80)
print("SIMULATION RESULTS")
print("="*80)
print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
print(f"Duration: {result.total_duration_s:.2f}s")
print(f"Mean solve time: {result.mean_solve_time_ms:.2f}ms")
print(f"Max solve time: {result.max_solve_time_ms:.2f}ms")
print(f"Deadline violations: {result.deadline_violations}")

# Final state values
final_state = result.state_history[-1]
print(f"\nFinal State:")
print(f"  Position: {final_state[POSITION_INDEX]:.2f} m")
print(f"  Velocity: {final_state[VELOCITY_INDEX]:.4f} m/s (target: {args.velocity} m/s)")
print(f"  Pitch: {np.rad2deg(final_state[PITCH_INDEX]):.4f} deg")
print(f"  Pitch rate: {np.rad2deg(final_state[PITCH_RATE_INDEX]):.4f} deg/s")
print(f"  Yaw: {np.rad2deg(final_state[YAW_INDEX]):.4f} deg")

# Compute tracking error statistics (if we have data)
if len(result.state_history) > 0:
    velocity_error = result.state_history[:, VELOCITY_INDEX] - args.velocity
    velocity_rmse = np.sqrt(np.mean(velocity_error**2))
    velocity_max_error = np.max(np.abs(velocity_error))

    print(f"\nVelocity Tracking Performance:")
    print(f"  RMSE: {velocity_rmse:.4f} m/s")
    print(f"  Max error: {velocity_max_error:.4f} m/s")
    print(f"  Mean error: {np.mean(velocity_error):.4f} m/s")
    print(f"  Std dev: {np.std(velocity_error):.4f} m/s")

    # Pitch statistics
    max_pitch = np.max(np.abs(result.state_history[:, PITCH_INDEX]))
    mean_pitch = np.mean(np.abs(result.state_history[:, PITCH_INDEX]))

    print(f"\nPitch Statistics:")
    print(f"  Max absolute pitch: {np.rad2deg(max_pitch):.4f} deg")
    print(f"  Mean absolute pitch: {np.rad2deg(mean_pitch):.4f} deg")

    # Control effort
    control_rms = np.sqrt(np.mean(result.control_history**2))
    max_control = np.max(np.abs(result.control_history))

    print(f"\nControl Effort:")
    print(f"  RMS torque: {control_rms:.4f} Nm")
    print(f"  Max torque: {max_control:.4f} Nm")

    # Generate plots
    save_dir = f"test_and_debug_output/velocity_tracking_{args.velocity}mps"
    if args.viewer:
        save_dir += "_viewer"
    else:
        save_dir += f"_{args.duration}s"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nGenerating plots...")

    # Create figure with 6 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

    time_s = result.time_s

    # 1. Velocity tracking
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time_s, result.state_history[:, VELOCITY_INDEX],
             'b-', linewidth=2, label='Actual')
    ax1.axhline(args.velocity, color='r', linestyle='--', linewidth=2, label='Target')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title('Velocity Tracking', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Velocity error
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time_s, velocity_error, 'r-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax2.axhline(velocity_rmse, color='orange', linestyle='--',
                alpha=0.5, label=f'RMSE: {velocity_rmse:.4f} m/s')
    ax2.axhline(-velocity_rmse, color='orange', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity Error (m/s)')
    ax2.set_title('Velocity Tracking Error', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Position
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time_s, result.state_history[:, POSITION_INDEX], 'b-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Position (No Limit)', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Pitch angle
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time_s, np.rad2deg(result.state_history[:, PITCH_INDEX]),
             'b-', linewidth=2)
    ax4.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Pitch (deg)')
    ax4.set_title('Pitch Angle', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Pitch rate
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(time_s, np.rad2deg(result.state_history[:, PITCH_RATE_INDEX]),
             'b-', linewidth=2)
    ax5.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Pitch Rate (deg/s)')
    ax5.set_title('Pitch Rate', fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Control torques
    ax6 = fig.add_subplot(gs[2, 1])
    n_controls = len(result.control_history)
    time_controls = time_s[:n_controls]
    ax6.plot(time_controls, result.control_history[:, 0],
             'b-', linewidth=1.5, label='Left wheel')
    ax6.plot(time_controls, result.control_history[:, 1],
             'r--', linewidth=1.5, label='Right wheel')
    ax6.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Torque (Nm)')
    ax6.set_title('Control Torques', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    mode_str = f'{args.velocity} m/s (viewer mode)' if args.viewer else f'{args.velocity} m/s for {args.duration} s'
    fig.suptitle(f'Velocity Tracking: {mode_str}',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(f"{save_dir}/velocity_tracking.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/velocity_tracking.png")

    # Create state space plot
    fig2, ax = plt.subplots(figsize=(10, 8))

    # Plot trajectory in pitch-pitch_rate space
    ax.plot(np.rad2deg(result.state_history[:, PITCH_INDEX]),
            np.rad2deg(result.state_history[:, PITCH_RATE_INDEX]),
            'b-', linewidth=1.5, alpha=0.7)
    ax.plot(np.rad2deg(result.state_history[0, PITCH_INDEX]),
            np.rad2deg(result.state_history[0, PITCH_RATE_INDEX]),
            'go', markersize=10, label='Start')
    ax.plot(np.rad2deg(result.state_history[-1, PITCH_INDEX]),
            np.rad2deg(result.state_history[-1, PITCH_RATE_INDEX]),
            'ro', markersize=10, label='End')
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Pitch (deg)')
    ax.set_ylabel('Pitch Rate (deg/s)')
    ax.set_title('Pitch State Space Trajectory', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(f"{save_dir}/state_space.png", dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_dir}/state_space.png")

    plt.close('all')

    print(f"\nPlots saved to: {save_dir}/")
else:
    print("\nNo data collected (simulation ended immediately)")

print("\n" + "="*80)
print("Test complete!")
print("="*80)

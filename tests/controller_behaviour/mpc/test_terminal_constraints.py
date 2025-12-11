"""Test terminal constraints: compare MPC behavior with constraints ON vs OFF.

This script runs the same scenario twice to compare the effect of terminal constraints:
1. With terminal constraints: pitch and pitch_rate forced to 0 at terminal state
2. Without terminal constraints: only stage constraints apply

Usage:
    python3 scripts/debug/test_terminal_constraints.py
    python3 scripts/debug/test_terminal_constraints.py --distance 5 --velocity 1.5
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
from robot_dynamics.parameters import PITCH_INDEX, PITCH_RATE_INDEX, POSITION_INDEX, VELOCITY_INDEX

# Parse arguments
parser = argparse.ArgumentParser(description='Test terminal constraints')
parser.add_argument('--velocity', type=float, default=4,
                    help='Forward velocity in m/s (default: 2.2)')
parser.add_argument('--distance', type=float, default=10.0,
                    help='Target distance in m (default: 10.0)')
parser.add_argument('--stop-time', type=float, default=3.0,
                    help='Time to hold balance after stopping (default: 3.0s)')
args = parser.parse_args()

# Calculate timing
drive_time = args.distance / args.velocity if args.velocity > 0 else 5.0
total_duration = drive_time + args.stop_time

print("\n" + "="*80)
print("TERMINAL CONSTRAINTS COMPARISON TEST")
print("="*80)
print(f"Scenario:")
print(f"  Velocity: {args.velocity} m/s")
print(f"  Distance: {args.distance} m")
print(f"  Drive time: {drive_time:.2f}s")
print(f"  Stop time: {args.stop_time:.2f}s")
print(f"  Total duration: {total_duration:.2f}s")
print("="*80)

# Reference callback
def reference_callback(time_s: float) -> ReferenceCommand:
    """Switch from velocity mode to balance mode."""
    if time_s < drive_time:
        return ReferenceCommand(
            mode=ReferenceMode.VELOCITY,
            velocity_mps=args.velocity,
            yaw_rate_radps=0.0,
        )
    else:
        return ReferenceCommand(mode=ReferenceMode.BALANCE)


def run_simulation(terminal_constraints_enabled: bool):
    """Run simulation with or without terminal constraints."""

    # Load config
    config = SimulationConfig(
        model_path='robot_model.xml',
        robot_params_path='config/simulation/robot_params.yaml',
        mpc_params_path='config/simulation/mpc_params.yaml',
        estimator_params_path='config/simulation/estimator_params.yaml',
        duration_s=total_duration,
    )

    # Modify MPC config to enable/disable terminal constraints
    mpc_config = MPCConfig.from_yaml(config.mpc_params_path)

    # Create new config with modified terminal constraints
    from dataclasses import replace
    if not terminal_constraints_enabled:
        mpc_config = replace(
            mpc_config,
            terminal_pitch_limit_rad=None,
            terminal_pitch_rate_limit_radps=None,
            terminal_velocity_limit_mps=None
        )
        label = "WITHOUT terminal constraints"
    else:
        label = "WITH terminal constraints"

    print(f"\n{label}:")
    print(f"  terminal_pitch_limit_rad = {mpc_config.terminal_pitch_limit_rad}")
    print(f"  terminal_pitch_rate_limit_radps = {mpc_config.terminal_pitch_rate_limit_radps}")
    print(f"  terminal_velocity_limit_mps = {mpc_config.terminal_velocity_limit_mps}")

    # Create simulation with modified config
    # We need to temporarily save modified config
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        # Convert config to dict for YAML
        config_dict = {
            'prediction_horizon_steps': mpc_config.prediction_horizon_steps,
            'sampling_period_s': mpc_config.sampling_period_s,
            'state_cost_diagonal': list(mpc_config.state_cost_diagonal),
            'control_cost_diagonal': list(mpc_config.control_cost_diagonal),
            'pitch_limit_rad': mpc_config.pitch_limit_rad,
            'pitch_rate_limit_radps': mpc_config.pitch_rate_limit_radps,
            'control_limit_nm': mpc_config.control_limit_nm,
            'use_terminal_cost_dare': mpc_config.use_terminal_cost_dare,
            'terminal_cost_scale': mpc_config.terminal_cost_scale,
            'solver_name': mpc_config.solver_name,
            'warm_start_enabled': mpc_config.warm_start_enabled,
            'terminal_pitch_limit_rad': mpc_config.terminal_pitch_limit_rad,
            'terminal_pitch_rate_limit_radps': mpc_config.terminal_pitch_rate_limit_radps,
            'terminal_velocity_limit_mps': mpc_config.terminal_velocity_limit_mps,
        }
        yaml.dump(config_dict, f)
        temp_config_path = f.name

    # Create simulation with temporary config
    temp_config = SimulationConfig(
        model_path=config.model_path,
        robot_params_path=config.robot_params_path,
        mpc_params_path=temp_config_path,
        estimator_params_path=config.estimator_params_path,
        duration_s=total_duration,
    )

    simulation = MPCSimulation(temp_config)

    # Run simulation
    print(f"  Running simulation...")
    result = simulation.run(
        initial_pitch_rad=0.0,
        reference_command_callback=reference_callback,
    )

    # Clean up temp file
    os.remove(temp_config_path)

    print(f"  Status: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"  Mean solve time: {result.mean_solve_time_ms:.2f}ms")
    print(f"  Max solve time: {result.max_solve_time_ms:.2f}ms")

    return result


# Run both simulations
print("\n" + "="*80)
print("Running simulations...")
print("="*80)

result_with = run_simulation(terminal_constraints_enabled=True)
result_without = run_simulation(terminal_constraints_enabled=False)

# Compare results
print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

print("\nFinal State Values:")
print(f"{'Metric':<30} {'WITH Constraints':<20} {'WITHOUT Constraints':<20} {'Difference':<15}")
print("-" * 85)

# Position
pos_with = result_with.state_history[-1, POSITION_INDEX]
pos_without = result_without.state_history[-1, POSITION_INDEX]
print(f"{'Final position (m)':<30} {pos_with:>19.4f} {pos_without:>19.4f} {abs(pos_with-pos_without):>14.4f}")

# Velocity
vel_with = result_with.state_history[-1, VELOCITY_INDEX]
vel_without = result_without.state_history[-1, VELOCITY_INDEX]
print(f"{'Final velocity (m/s)':<30} {vel_with:>19.4f} {vel_without:>19.4f} {abs(vel_with-vel_without):>14.4f}")

# Pitch
pitch_with = np.rad2deg(result_with.state_history[-1, PITCH_INDEX])
pitch_without = np.rad2deg(result_without.state_history[-1, PITCH_INDEX])
print(f"{'Final pitch (deg)':<30} {pitch_with:>19.4f} {pitch_without:>19.4f} {abs(pitch_with-pitch_without):>14.4f}")

# Pitch rate
pitch_rate_with = np.rad2deg(result_with.state_history[-1, PITCH_RATE_INDEX])
pitch_rate_without = np.rad2deg(result_without.state_history[-1, PITCH_RATE_INDEX])
print(f"{'Final pitch rate (deg/s)':<30} {pitch_rate_with:>19.4f} {pitch_rate_without:>19.4f} {abs(pitch_rate_with-pitch_rate_without):>14.4f}")

# Max pitch
max_pitch_with = np.rad2deg(np.max(np.abs(result_with.state_history[:, PITCH_INDEX])))
max_pitch_without = np.rad2deg(np.max(np.abs(result_without.state_history[:, PITCH_INDEX])))
print(f"{'Max pitch (deg)':<30} {max_pitch_with:>19.4f} {max_pitch_without:>19.4f} {abs(max_pitch_with-max_pitch_without):>14.4f}")

# Control effort (RMS)
control_rms_with = np.sqrt(np.mean(result_with.control_history**2))
control_rms_without = np.sqrt(np.mean(result_without.control_history**2))
print(f"{'RMS control effort (Nm)':<30} {control_rms_with:>19.4f} {control_rms_without:>19.4f} {abs(control_rms_with-control_rms_without):>14.4f}")

# Generate comparison plots
save_dir = f"debug_output/terminal_constraints_test_{args.velocity}mps_{args.distance}m"
os.makedirs(save_dir, exist_ok=True)

print(f"\nGenerating comparison plots...")

# Create figure with 5 subplots
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

time_with = result_with.time_s
time_without = result_without.time_s

# 1. Pitch comparison
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(time_with, np.rad2deg(result_with.state_history[:, PITCH_INDEX]),
         'b-', linewidth=2, label='WITH constraints')
ax1.plot(time_without, np.rad2deg(result_without.state_history[:, PITCH_INDEX]),
         'r--', linewidth=2, label='WITHOUT constraints')
ax1.axvline(drive_time, color='g', linestyle=':', alpha=0.5, label='Stop transition')
ax1.axhline(0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Pitch (deg)')
ax1.set_title('Pitch Angle Comparison', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Pitch rate comparison
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(time_with, np.rad2deg(result_with.state_history[:, PITCH_RATE_INDEX]),
         'b-', linewidth=2, label='WITH constraints')
ax2.plot(time_without, np.rad2deg(result_without.state_history[:, PITCH_RATE_INDEX]),
         'r--', linewidth=2, label='WITHOUT constraints')
ax2.axvline(drive_time, color='g', linestyle=':', alpha=0.5, label='Stop transition')
ax2.axhline(0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Pitch Rate (deg/s)')
ax2.set_title('Pitch Rate Comparison', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Position comparison
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(time_with, result_with.state_history[:, POSITION_INDEX],
         'b-', linewidth=2, label='WITH constraints')
ax3.plot(time_without, result_without.state_history[:, POSITION_INDEX],
         'r--', linewidth=2, label='WITHOUT constraints')
ax3.axhline(args.distance, color='g', linestyle=':', alpha=0.5, label='Target')
ax3.axvline(drive_time, color='g', linestyle=':', alpha=0.5)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Position (m)')
ax3.set_title('Position Tracking Comparison', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Velocity comparison
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(time_with, result_with.state_history[:, VELOCITY_INDEX],
         'b-', linewidth=2, label='WITH constraints')
ax4.plot(time_without, result_without.state_history[:, VELOCITY_INDEX],
         'r--', linewidth=2, label='WITHOUT constraints')
ax4.axhline(0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
ax4.axvline(drive_time, color='g', linestyle=':', alpha=0.5, label='Stop transition')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Velocity (m/s)')
ax4.set_title('Velocity Profile Comparison', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Control effort comparison
ax5 = fig.add_subplot(gs[2, :])
# Match time array length to control history
n_controls_with = len(result_with.control_history)
n_controls_without = len(result_without.control_history)
time_controls_with = time_with[:n_controls_with]
time_controls_without = time_without[:n_controls_without]

ax5.plot(time_controls_with, result_with.control_history[:, 0],
         'b-', linewidth=1.5, alpha=0.7, label='WITH constraints (Left)')
ax5.plot(time_controls_with, result_with.control_history[:, 1],
         'b--', linewidth=1.5, alpha=0.7, label='WITH constraints (Right)')
ax5.plot(time_controls_without, result_without.control_history[:, 0],
         'r-', linewidth=1.5, alpha=0.7, label='WITHOUT constraints (Left)')
ax5.plot(time_controls_without, result_without.control_history[:, 1],
         'r--', linewidth=1.5, alpha=0.7, label='WITHOUT constraints (Right)')
ax5.axvline(drive_time, color='g', linestyle=':', alpha=0.5, label='Stop transition')
ax5.axhline(0, color='k', linestyle='-', alpha=0.2, linewidth=0.5)
ax5.set_xlabel('Time (s)')
ax5.set_ylabel('Control Torque (Nm)')
ax5.set_title('Control Effort Comparison', fontweight='bold')
ax5.legend(ncol=3, fontsize=9)
ax5.grid(True, alpha=0.3)

fig.suptitle(f'Terminal Constraints Comparison: {args.velocity} m/s, {args.distance} m',
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig(f"{save_dir}/comparison.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {save_dir}/comparison.png")

# Create zoomed-in plot of stopping phase
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

# Find indices for stopping phase (last 2 seconds)
stop_start_time = max(0, total_duration - 2.0)
stop_idx_with = np.searchsorted(time_with, stop_start_time)
stop_idx_without = np.searchsorted(time_without, stop_start_time)

# Pitch during stopping
axes[0, 0].plot(time_with[stop_idx_with:], np.rad2deg(result_with.state_history[stop_idx_with:, PITCH_INDEX]),
                'b-', linewidth=2, label='WITH constraints', marker='o', markersize=2)
axes[0, 0].plot(time_without[stop_idx_without:], np.rad2deg(result_without.state_history[stop_idx_without:, PITCH_INDEX]),
                'r--', linewidth=2, label='WITHOUT constraints', marker='s', markersize=2)
axes[0, 0].axhline(0, color='k', linestyle='-', alpha=0.2)
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Pitch (deg)')
axes[0, 0].set_title('Pitch During Stop Phase (Zoomed)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Pitch rate during stopping
axes[0, 1].plot(time_with[stop_idx_with:], np.rad2deg(result_with.state_history[stop_idx_with:, PITCH_RATE_INDEX]),
                'b-', linewidth=2, label='WITH constraints', marker='o', markersize=2)
axes[0, 1].plot(time_without[stop_idx_without:], np.rad2deg(result_without.state_history[stop_idx_without:, PITCH_RATE_INDEX]),
                'r--', linewidth=2, label='WITHOUT constraints', marker='s', markersize=2)
axes[0, 1].axhline(0, color='k', linestyle='-', alpha=0.2)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Pitch Rate (deg/s)')
axes[0, 1].set_title('Pitch Rate During Stop Phase (Zoomed)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Velocity during stopping
axes[1, 0].plot(time_with[stop_idx_with:], result_with.state_history[stop_idx_with:, VELOCITY_INDEX],
                'b-', linewidth=2, label='WITH constraints', marker='o', markersize=2)
axes[1, 0].plot(time_without[stop_idx_without:], result_without.state_history[stop_idx_without:, VELOCITY_INDEX],
                'r--', linewidth=2, label='WITHOUT constraints', marker='s', markersize=2)
axes[1, 0].axhline(0, color='k', linestyle='-', alpha=0.2)
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Velocity (m/s)')
axes[1, 0].set_title('Velocity During Stop Phase (Zoomed)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Position during stopping
axes[1, 1].plot(time_with[stop_idx_with:], result_with.state_history[stop_idx_with:, POSITION_INDEX],
                'b-', linewidth=2, label='WITH constraints', marker='o', markersize=2)
axes[1, 1].plot(time_without[stop_idx_without:], result_without.state_history[stop_idx_without:, POSITION_INDEX],
                'r--', linewidth=2, label='WITHOUT constraints', marker='s', markersize=2)
axes[1, 1].axhline(args.distance, color='g', linestyle=':', alpha=0.5, label='Target')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].set_ylabel('Position (m)')
axes[1, 1].set_title('Position During Stop Phase (Zoomed)', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

fig2.suptitle('Stop Phase Detail (Final 2 seconds)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{save_dir}/stop_phase_detail.png", dpi=150, bbox_inches='tight')
print(f"  Saved: {save_dir}/stop_phase_detail.png")

plt.close('all')

print(f"\nPlots saved to: {save_dir}/")
print("\n" + "="*80)
print("Test complete!")
print("="*80)

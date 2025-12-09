"""Debug script for external force disturbance analysis."""

import numpy as np
import matplotlib.pyplot as plt
from simulation import SimulationConfig, MPCSimulation
from debug import MPCDiagnostics
from mpc import ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import PITCH_INDEX, VELOCITY_INDEX

config = SimulationConfig(
    model_path='robot_model.xml',
    robot_params_path='config/robot_params.yaml',
    mpc_params_path='config/mpc_params.yaml',
    estimator_params_path='config/estimator_params.yaml',
)

# Create simulation and diagnostics
sim = MPCSimulation(config)
diag = MPCDiagnostics(config)

# Test different force magnitudes
force_magnitudes = [1.0, 5.0, 10.0]  # Newtons

print("="*80)
print("EXTERNAL FORCE DISTURBANCE RECOVERY ANALYSIS")
print("="*80)

for force_N in force_magnitudes:
    print(f"\n{'='*80}")
    print(f"Testing {force_N:.1f}N force disturbance at t=1.0s")
    print('='*80)

    # Wrapper to apply disturbance at t=1.0s
    def timed_disturbance(time_s, data):
        if 1.0 <= time_s <= 1.1:
            data.xfrc_applied[1, 0] = force_N  # Forward force on body
        else:
            data.xfrc_applied[1, :] = 0

    # Run simulation
    result = sim.run(
        duration_s=3.0,
        initial_pitch_rad=0.0,  # Start at equilibrium
        disturbance_callback=timed_disturbance,
    )

    # Analyze using diagnostics plotting
    save_dir = f"debug_output/force_{force_N:.0f}N"

    from debug.plotting import (
        plot_state_comparison,
        plot_control_analysis,
    )

    # Generate plots
    plot_state_comparison(
        result.time_s,
        result.state_history,
        result.state_estimate_history,
        title=f"State Response to {force_N:.1f}N Disturbance",
        save_path=f"{save_dir}/state_response.png",
    )

    plot_control_analysis(
        result.time_s,
        result.control_history,
        control_limit=0.25,
        title=f"Control Response to {force_N:.1f}N Disturbance",
        save_path=f"{save_dir}/control_response.png",
    )

    # Custom disturbance analysis plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"Disturbance Recovery: {force_N:.1f}N Force at t=1.0s", fontsize=14)

    # Pitch
    ax = axes[0]
    ax.plot(result.time_s, np.rad2deg(result.state_history[:, PITCH_INDEX]), 'b-', linewidth=2)
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Disturbance')
    ax.axvline(x=1.1, color='r', linestyle='--', alpha=0.5)
    ax.axhspan(-30, 30, alpha=0.1, color='green', label='Safe region')
    ax.set_ylabel('Pitch (deg)')
    ax.set_title('Pitch Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[1]
    ax.plot(result.time_s, result.state_history[:, VELOCITY_INDEX], 'b-', label='True', linewidth=2)
    ax.plot(result.time_s, result.state_estimate_history[:, VELOCITY_INDEX], 'r--', label='Estimated', linewidth=2)
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
    ax.axvline(x=1.1, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity (Shows Estimation Error)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Control
    ax = axes[2]
    total_torque = result.control_history[:, 0] + result.control_history[:, 1]
    ax.plot(result.time_s, total_torque, 'b-', linewidth=2)
    ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Disturbance')
    ax.axvline(x=1.1, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Total Torque (N⋅m)')
    ax.set_title('Control Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/disturbance_timeline.png", dpi=150)
    plt.close()

    # Results
    if result.success:
        print(f"✓ Recovered from {force_N:.1f}N disturbance")
        max_pitch_after = np.max(np.abs(result.state_history[50:, PITCH_INDEX]))  # After t=1s
        print(f"  Max pitch excursion: {np.rad2deg(max_pitch_after):.2f}°")
    else:
        print(f"✗ Failed to recover from {force_N:.1f}N disturbance")
        print(f"  Fell at t={result.time_s[-1]:.3f}s")

    print(f"  Plots saved to {save_dir}/")

print("\n" + "="*80)
print("Disturbance analysis complete!")
print("="*80)

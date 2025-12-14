#!/usr/bin/env python3
"""Standalone script to compare online vs. fixed linearization strategies.

This script runs MPC simulations with two different linearization approaches:
1. Fixed linearization around 0 degrees (equilibrium)
2. Online linearization at the current state (successive linearization)

Usage:
    python3 scripts/compare_linearization_strategies.py --angle 15.0 --duration 5.0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE
from mpc import ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import PITCH_INDEX, PITCH_RATE_INDEX
from robot_dynamics.linearization import build_jacobian_functions


def run_comparison(initial_pitch_deg: float, duration_s: float):
    """Run comparison between online and fixed linearization.

    Args:
        initial_pitch_deg: Initial pitch perturbation in degrees
        duration_s: Simulation duration in seconds
    """
    if not MUJOCO_AVAILABLE:
        print("ERROR: MuJoCo is not installed. Cannot run simulation.")
        print("Install with: pip install mujoco")
        return

    initial_pitch_rad = np.deg2rad(initial_pitch_deg)

    print(f"\n{'='*80}")
    print(f"MPC Linearization Strategy Comparison")
    print(f"{'='*80}")
    print(f"Initial pitch perturbation: {initial_pitch_deg}°")
    print(f"Simulation duration: {duration_s}s")
    print(f"{'='*80}\n")

    # Simulation 1: Fixed linearization around 0 degrees
    print("1. Running FIXED linearization (around 0°)...")
    config_fixed = SimulationConfig(
        model_path='Mujoco sim/robot_model.xml',
        robot_params_path='config/simulation/robot_params.yaml',
        mpc_params_path='config/simulation/mpc_params.yaml',
        estimator_params_path='config/simulation/estimator_params.yaml',
        duration_s=duration_s,
    )
    sim_fixed = MPCSimulation(config_fixed)
    sim_fixed.controller._online_linearization_enabled = False

    result_fixed = sim_fixed.run(
        duration_s=duration_s,
        initial_pitch_rad=initial_pitch_rad,
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    if result_fixed.success:
        final_pitch_fixed = np.rad2deg(result_fixed.state_history[-1, PITCH_INDEX])
        max_pitch_fixed = np.rad2deg(np.max(np.abs(result_fixed.state_history[:, PITCH_INDEX])))
        print(f"   ✓ Success!")
        print(f"   - Final pitch: {final_pitch_fixed:.3f}°")
        print(f"   - Max pitch: {max_pitch_fixed:.3f}°")
        print(f"   - Mean solve time: {result_fixed.mean_solve_time_ms:.2f}ms")
        print(f"   - Max solve time: {result_fixed.max_solve_time_ms:.2f}ms")
    else:
        print(f"   ✗ Robot fell!")
        final_pitch_fixed = np.nan
        max_pitch_fixed = np.nan

    # Simulation 2: Online linearization
    print("\n2. Running ONLINE linearization (at current state)...")
    config_online = SimulationConfig(
        model_path='Mujoco sim/robot_model.xml',
        robot_params_path='config/simulation/robot_params.yaml',
        mpc_params_path='config/simulation/mpc_params.yaml',
        estimator_params_path='config/simulation/estimator_params.yaml',
        duration_s=duration_s,
    )
    sim_online = MPCSimulation(config_online)
    sim_online.controller._online_linearization_enabled = True
    sim_online.controller._jacobian_functions = build_jacobian_functions(
        sim_online.controller._robot_params
    )

    result_online = sim_online.run(
        duration_s=duration_s,
        initial_pitch_rad=initial_pitch_rad,
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    if result_online.success:
        final_pitch_online = np.rad2deg(result_online.state_history[-1, PITCH_INDEX])
        max_pitch_online = np.rad2deg(np.max(np.abs(result_online.state_history[:, PITCH_INDEX])))
        print(f"   ✓ Success!")
        print(f"   - Final pitch: {final_pitch_online:.3f}°")
        print(f"   - Max pitch: {max_pitch_online:.3f}°")
        print(f"   - Mean solve time: {result_online.mean_solve_time_ms:.2f}ms")
        print(f"   - Max solve time: {result_online.max_solve_time_ms:.2f}ms")
    else:
        print(f"   ✗ Robot fell!")
        final_pitch_online = np.nan
        max_pitch_online = np.nan

    # Print comparison summary
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    print(f"Initial pitch perturbation: {initial_pitch_deg}°\n")

    print("Fixed linearization (0°):")
    print(f"  - Final pitch:     {final_pitch_fixed:7.3f}°")
    print(f"  - Max pitch:       {max_pitch_fixed:7.3f}°")
    if result_fixed.success:
        print(f"  - Mean solve time: {result_fixed.mean_solve_time_ms:7.2f}ms")
        print(f"  - Status:          SUCCESS")
    else:
        print(f"  - Status:          FAILED (robot fell)")

    print(f"\nOnline linearization (current state):")
    print(f"  - Final pitch:     {final_pitch_online:7.3f}°")
    print(f"  - Max pitch:       {max_pitch_online:7.3f}°")
    if result_online.success:
        print(f"  - Mean solve time: {result_online.mean_solve_time_ms:7.2f}ms")
        print(f"  - Status:          SUCCESS")
    else:
        print(f"  - Status:          FAILED (robot fell)")

    if result_fixed.success and result_online.success:
        print(f"\nDifference:")
        print(f"  - Final pitch diff: {abs(final_pitch_fixed - final_pitch_online):7.3f}°")
        print(f"  - Max pitch diff:   {abs(max_pitch_fixed - max_pitch_online):7.3f}°")
        print(f"  - Solve time diff:  {abs(result_fixed.mean_solve_time_ms - result_online.mean_solve_time_ms):7.2f}ms")

    print(f"{'='*80}\n")

    # Generate plots
    if result_fixed.success or result_online.success:
        print("Generating comparison plots...")
        plot_comparison(
            result_fixed,
            result_online,
            initial_pitch_deg,
        )


def plot_comparison(result_fixed, result_online, initial_pitch_deg: float):
    """Generate comparison plots."""
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Actual pitch evolution comparison
    ax1 = plt.subplot(2, 3, 1)
    if result_fixed.success:
        ax1.plot(
            result_fixed.time_s,
            np.rad2deg(result_fixed.state_history[:, PITCH_INDEX]),
            'b-',
            linewidth=2.5,
            label='Fixed linearization (0°)',
        )
    if result_online.success:
        ax1.plot(
            result_online.time_s,
            np.rad2deg(result_online.state_history[:, PITCH_INDEX]),
            'r-',
            linewidth=2.5,
            label='Online linearization',
            alpha=0.8,
        )
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('Pitch angle [deg]', fontsize=11)
    ax1.set_title(f'Actual Pitch Evolution\n(Initial: {initial_pitch_deg}°)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)

    # Subplot 2: Fixed linearization - predicted horizons
    if result_fixed.success:
        ax2 = plt.subplot(2, 3, 2)
        plot_predicted_horizons(ax2, result_fixed, 'Fixed linearization (0°)', 'Blues')

    # Subplot 3: Online linearization - predicted horizons
    if result_online.success:
        ax3 = plt.subplot(2, 3, 3)
        plot_predicted_horizons(ax3, result_online, 'Online linearization', 'Reds')

    # Subplot 4: Pitch rate comparison
    ax4 = plt.subplot(2, 3, 4)
    if result_fixed.success:
        ax4.plot(
            result_fixed.time_s,
            np.rad2deg(result_fixed.state_history[:, PITCH_RATE_INDEX]),
            'b-',
            linewidth=2,
            label='Fixed linearization',
        )
    if result_online.success:
        ax4.plot(
            result_online.time_s,
            np.rad2deg(result_online.state_history[:, PITCH_RATE_INDEX]),
            'r-',
            linewidth=2,
            label='Online linearization',
            alpha=0.8,
        )
    ax4.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Time [s]', fontsize=11)
    ax4.set_ylabel('Pitch rate [deg/s]', fontsize=11)
    ax4.set_title('Pitch Rate Evolution', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)

    # Subplot 5: First prediction comparison
    if result_fixed.success and result_online.success:
        ax5 = plt.subplot(2, 3, 5)
        plot_first_prediction_comparison(ax5, result_fixed, result_online)

    # Subplot 6: Control effort comparison
    ax6 = plt.subplot(2, 3, 6)
    if result_fixed.success:
        control_effort_fixed = np.sum(np.abs(result_fixed.control_history), axis=1)
        ax6.plot(
            result_fixed.time_s,
            control_effort_fixed,
            'b-',
            linewidth=2,
            label='Fixed linearization',
        )
    if result_online.success:
        control_effort_online = np.sum(np.abs(result_online.control_history), axis=1)
        ax6.plot(
            result_online.time_s,
            control_effort_online,
            'r-',
            linewidth=2,
            label='Online linearization',
            alpha=0.8,
        )
    ax6.grid(True, alpha=0.3)
    ax6.set_xlabel('Time [s]', fontsize=11)
    ax6.set_ylabel('Control effort [N·m]', fontsize=11)
    ax6.set_title('Total Control Effort (|τ_L| + |τ_R|)', fontsize=12, fontweight='bold')
    ax6.legend(loc='best', fontsize=10)

    plt.tight_layout()

    # Save plot
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'linearization_comparison_{initial_pitch_deg}deg.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}\n")

    # Show plot
    plt.show()


def plot_predicted_horizons(ax, result, title: str, colormap: str):
    """Plot predicted pitch horizons over time."""
    # Extract sampling period
    if len(result.time_s) > 1:
        dt = result.time_s[1] - result.time_s[0]
    else:
        dt = 0.065

    # Plot every Nth horizon to avoid overcrowding
    n_horizons = len(result.reference_history)
    plot_every = max(1, n_horizons // 15)

    cmap = plt.get_cmap(colormap)

    for i in range(0, n_horizons, plot_every):
        predicted_traj = result.reference_history[i]
        horizon_length = predicted_traj.shape[0]

        t_start = result.time_s[i]
        t_horizon = t_start + np.arange(horizon_length) * dt

        predicted_pitch_deg = np.rad2deg(predicted_traj[:, PITCH_INDEX])

        color = cmap(0.3 + 0.6 * (i / n_horizons))

        ax.plot(
            t_horizon,
            predicted_pitch_deg,
            color=color,
            linewidth=0.8,
            alpha=0.5,
        )

    # Overlay actual trajectory
    ax.plot(
        result.time_s,
        np.rad2deg(result.state_history[:, PITCH_INDEX]),
        'k-',
        linewidth=2.5,
        label='Actual trajectory',
        zorder=100,
    )

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Pitch angle [deg]', fontsize=11)
    ax.set_title(f'Predicted Horizons\n{title}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)


def plot_first_prediction_comparison(ax, result_fixed, result_online):
    """Plot comparison of first prediction from each method."""
    pred_fixed = result_fixed.reference_history[0]
    pred_online = result_online.reference_history[0]

    if len(result_fixed.time_s) > 1:
        dt = result_fixed.time_s[1] - result_fixed.time_s[0]
    else:
        dt = 0.065

    horizon_length = min(pred_fixed.shape[0], pred_online.shape[0])
    t_horizon = np.arange(horizon_length) * dt

    pitch_fixed_deg = np.rad2deg(pred_fixed[:horizon_length, PITCH_INDEX])
    pitch_online_deg = np.rad2deg(pred_online[:horizon_length, PITCH_INDEX])

    ax.plot(
        t_horizon,
        pitch_fixed_deg,
        'b-',
        linewidth=2,
        marker='o',
        markersize=4,
        label='Fixed linearization prediction',
    )
    ax.plot(
        t_horizon,
        pitch_online_deg,
        'r-',
        linewidth=2,
        marker='s',
        markersize=4,
        label='Online linearization prediction',
    )

    # Plot actual trajectory
    n_actual = min(horizon_length, len(result_fixed.time_s))
    ax.plot(
        result_fixed.time_s[:n_actual],
        np.rad2deg(result_fixed.state_history[:n_actual, PITCH_INDEX]),
        'k--',
        linewidth=2,
        marker='x',
        markersize=6,
        label='Actual trajectory',
    )

    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time from first solve [s]', fontsize=11)
    ax.set_ylabel('Pitch angle [deg]', fontsize=11)
    ax.set_title('First Predicted Horizon Comparison', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare MPC with online vs. fixed linearization strategies'
    )
    parser.add_argument(
        '--angle',
        type=float,
        default=15.0,
        help='Initial pitch perturbation angle in degrees (default: 15.0)',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=5.0,
        help='Simulation duration in seconds (default: 5.0)',
    )

    args = parser.parse_args()

    run_comparison(args.angle, args.duration)

"""Debug script to test MPC performance with varying horizon lengths.

Tests the velocity drive and stop scenario with different prediction horizons,
from CFTOCP-like (very long horizon) to short-sighted (very short horizon).
Generates comparison plots to analyze the effect of horizon length.
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
import yaml
import os
from typing import List, Dict, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from simulation import SimulationConfig
from debug import MPCDiagnostics
from mpc import ReferenceCommand, ReferenceMode


class TestType(Enum):
    """Available test scenarios."""
    DRIVE_STOP = "drive-stop"
    VELOCITY_TRACKING = "velocity-tracking"
    BALANCE = "balance"

# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Test MPC with varying horizon lengths across different scenarios'
)
parser.add_argument('--test', type=str, default='drive-stop',
                    choices=['drive-stop', 'velocity-tracking', 'balance'],
                    help='Test scenario type (default: drive-stop)')
parser.add_argument('--velocity', type=float, default=None,
                    help='Forward velocity in m/s (default: test-dependent)')
parser.add_argument('--distance', type=float, default=None,
                    help='Target distance in m (default: test-dependent)')
parser.add_argument('--duration', type=float, default=None,
                    help='Test duration in seconds (default: test-dependent)')
parser.add_argument('--stop-time', type=float, default=None,
                    help='Time to hold balance after stopping (default: 3.0s for drive-stop)')
parser.add_argument('--horizons', type=int, nargs='+',
                    default=[10, 20, 40, 80, 160],
                    help='Horizon lengths to test (default: 10 20 40 80 160)')
parser.add_argument('--challenge', action='store_true',
                    help='Use challenging parameters (test-dependent)')
parser.add_argument('--no-terminal-cost', action='store_true',
                    help='Disable terminal cost to show raw horizon effects')
args = parser.parse_args()


@dataclass
class TestConfig:
    """Configuration for a specific test scenario."""
    name: str
    description: str
    velocity: float
    duration: float
    reference_callback: Callable[[float], ReferenceCommand]

    # Optional parameters
    distance: float = 0.0
    stop_time: float = 0.0


def create_test_config(test_type: str, args) -> TestConfig:
    """Create test configuration based on test type and arguments."""

    if test_type == 'drive-stop':
        # Drive and stop test
        velocity = args.velocity if args.velocity is not None else 0.1
        distance = args.distance if args.distance is not None else 0.5
        stop_time = args.stop_time if args.stop_time is not None else 3.0

        # Apply challenge mode
        if args.challenge:
            velocity = 0.3
            distance = 1.5
            stop_time = 2.0

        drive_time = distance / velocity if velocity > 0 else 5.0
        duration = drive_time + stop_time

        def reference_callback(time_s: float) -> ReferenceCommand:
            if time_s < drive_time:
                return ReferenceCommand(
                    mode=ReferenceMode.VELOCITY,
                    velocity_mps=velocity,
                    yaw_rate_radps=0.0,
                )
            else:
                return ReferenceCommand(mode=ReferenceMode.BALANCE)

        return TestConfig(
            name="Drive and Stop",
            description=f"Drive {distance}m at {velocity}m/s, then stop and balance for {stop_time}s",
            velocity=velocity,
            duration=duration,
            distance=distance,
            stop_time=stop_time,
            reference_callback=reference_callback,
        )

    elif test_type == 'velocity-tracking':
        # Constant velocity tracking
        velocity = args.velocity if args.velocity is not None else 0.15
        duration = args.duration if args.duration is not None else 8

        # Apply challenge mode
        if args.challenge:
            velocity = 0.4
            duration = 10.0

        def reference_callback(time_s: float) -> ReferenceCommand:
            return ReferenceCommand(
                mode=ReferenceMode.VELOCITY,
                velocity_mps=velocity,
                yaw_rate_radps=0.0,
            )

        return TestConfig(
            name="Velocity Tracking",
            description=f"Maintain constant velocity {velocity}m/s for {duration}s",
            velocity=velocity,
            duration=duration,
            reference_callback=reference_callback,
        )

    elif test_type == 'balance':
        # Pure balance test
        duration = args.duration if args.duration is not None else 5.0

        # Apply challenge mode (start with larger initial perturbation)
        if args.challenge:
            duration = 8.0

        def reference_callback(time_s: float) -> ReferenceCommand:
            return ReferenceCommand(mode=ReferenceMode.BALANCE)

        return TestConfig(
            name="Balance",
            description=f"Maintain balance at zero velocity for {duration}s",
            velocity=0.0,
            duration=duration,
            reference_callback=reference_callback,
        )

    else:
        raise ValueError(f"Unknown test type: {test_type}")


@dataclass
class HorizonTestResult:
    """Results from testing a single horizon configuration."""
    horizon_steps: int
    horizon_seconds: float
    success: bool
    time_s: np.ndarray
    state_history: np.ndarray
    control_history: np.ndarray
    reference_history: np.ndarray

    # Performance metrics
    final_position: float
    position_error: float
    final_velocity: float
    max_pitch_deg: float
    final_pitch_deg: float
    mean_solve_time_ms: float
    max_solve_time_ms: float
    deadline_violations: int

    # Tracking metrics
    mean_drive_velocity: float = 0.0
    velocity_tracking_error: float = 0.0


def run_horizon_test(horizon_steps: int,
                     test_config: TestConfig,
                     disable_terminal_cost: bool = False) -> HorizonTestResult:
    """Run test with specified horizon and test configuration."""

    print(f"\n{'='*60}")
    print(f"Testing Horizon: {horizon_steps} steps")
    print(f"{'='*60}")

    # Create temporary MPC config with modified horizon
    mpc_config_path = 'config/simulation/mpc_params.yaml'
    with open(mpc_config_path, 'r') as f:
        mpc_params = yaml.safe_load(f)

    # Modify horizon
    original_horizon = mpc_params['prediction_horizon_steps']
    mpc_params['prediction_horizon_steps'] = horizon_steps
    sampling_period = mpc_params['sampling_period_s']
    horizon_seconds = horizon_steps * sampling_period

    # Disable terminal cost if requested
    if disable_terminal_cost:
        mpc_params['use_terminal_cost_dare'] = False
        mpc_params['terminal_cost_scale'] = 0.001  # Very small, effectively disabled

    # Save temporary config
    temp_config_path = f'/tmp/mpc_params_h{horizon_steps}.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(mpc_params, f)

    # Configuration
    config = SimulationConfig(
        model_path='robot_model.xml',
        robot_params_path='config/simulation/robot_params.yaml',
        mpc_params_path=temp_config_path,
        estimator_params_path='config/simulation/estimator_params.yaml',
        duration_s=15.0,
    )

    # Create diagnostics and run simulation
    print(f"  Horizon: {horizon_steps} steps ({horizon_seconds:.3f}s)")
    diag = MPCDiagnostics(config)

    # VERIFY the horizon was actually loaded
    actual_horizon = diag.mpc_config.prediction_horizon_steps
    print(f"  Verified loaded horizon: {actual_horizon} steps")
    if actual_horizon != horizon_steps:
        print(f"  ⚠️  WARNING: Expected {horizon_steps} but loaded {actual_horizon}!")

    result, summary = diag.run_with_diagnostics(
        duration_s=test_config.duration,
        initial_pitch_rad=0.0,
        reference_command_callback=test_config.reference_callback,
        verbose=False,
    )

    # Extract metrics
    if len(result.state_history) > 0:
        final_position = result.state_history[-1, 0]
        final_velocity = result.state_history[-1, 3]
        max_pitch_deg = np.rad2deg(np.max(np.abs(result.state_history[:, 1])))
        final_pitch_deg = np.rad2deg(result.state_history[-1, 1])

        # Calculate test-specific metrics
        if test_config.distance > 0:
            # Drive-stop test: measure position error
            position_error = abs(final_position - test_config.distance)
        else:
            # Balance or tracking test: measure position drift
            position_error = abs(final_position)

        # Velocity tracking error
        if test_config.velocity > 0:
            # For velocity tracking, calculate mean velocity error over steady-state
            velocities = result.state_history[:, 3]
            # Skip first 1 second for transient
            start_idx = min(50, len(velocities) // 4)
            end_idx = len(velocities)

            # For drive-stop, only measure during drive phase
            if test_config.stop_time > 0:
                drive_time = test_config.duration - test_config.stop_time
                end_idx = int(drive_time / (test_config.duration / len(velocities)))

            if start_idx < end_idx:
                mean_velocity = np.mean(velocities[start_idx:end_idx])
                velocity_tracking_error = abs(mean_velocity - test_config.velocity)
                mean_drive_velocity = mean_velocity
            else:
                velocity_tracking_error = 0.0
                mean_drive_velocity = 0.0
        else:
            # Balance test: measure velocity stability
            velocities = result.state_history[:, 3]
            velocity_tracking_error = np.std(velocities)  # Use std as measure of stability
            mean_drive_velocity = 0.0
    else:
        final_position = 0.0
        position_error = test_config.distance if test_config.distance > 0 else 0.0
        final_velocity = 0.0
        max_pitch_deg = 0.0
        final_pitch_deg = 0.0
        mean_drive_velocity = 0.0
        velocity_tracking_error = 0.0

    print(f"  Final position: {final_position:.3f} m (error: {position_error:.3f} m)")
    print(f"  Final velocity: {final_velocity:.4f} m/s")
    print(f"  Max pitch: {max_pitch_deg:.2f}°")
    print(f"  Success: {result.success}")
    print(f"  Mean solve time: {result.mean_solve_time_ms:.2f} ms")

    # Clean up temp file
    os.remove(temp_config_path)

    return HorizonTestResult(
        horizon_steps=horizon_steps,
        horizon_seconds=horizon_seconds,
        success=result.success,
        time_s=result.time_s,
        state_history=result.state_history,
        control_history=result.control_history,
        reference_history=result.reference_history,
        final_position=final_position,
        position_error=position_error,
        final_velocity=final_velocity,
        max_pitch_deg=max_pitch_deg,
        final_pitch_deg=final_pitch_deg,
        mean_solve_time_ms=result.mean_solve_time_ms,
        max_solve_time_ms=result.max_solve_time_ms,
        deadline_violations=result.deadline_violations,
        mean_drive_velocity=mean_drive_velocity,
        velocity_tracking_error=velocity_tracking_error,
    )


def plot_comparison(results: List[HorizonTestResult],
                   test_config: TestConfig,
                   save_dir: str):
    """Generate comparison plots for different horizons."""

    os.makedirs(save_dir, exist_ok=True)

    # Sort results by horizon length
    results = sorted(results, key=lambda r: r.horizon_steps)

    # Calculate phase transition time (for drive-stop test)
    if test_config.stop_time > 0:
        phase_transition = test_config.duration - test_config.stop_time
    else:
        phase_transition = None

    # Define color map for different horizons
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    # ========================================================================
    # Plot 1: State Trajectories Comparison
    # ========================================================================
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig)

    # Position
    ax_pos = fig.add_subplot(gs[0, :])
    for i, result in enumerate(results):
        if result.success and len(result.state_history) > 0:
            pos = result.state_history[:, 0]
            label = f'N={result.horizon_steps} ({result.horizon_seconds:.2f}s)'
            ax_pos.plot(result.time_s, pos, linewidth=2, color=colors[i],
                       label=label, alpha=0.8)

    if test_config.distance > 0:
        ax_pos.axhline(test_config.distance, color='k', linestyle='--', linewidth=2,
                       label='Target', alpha=0.5)
    if phase_transition is not None:
        ax_pos.axvline(phase_transition, color='red', linestyle=':', linewidth=2,
                       alpha=0.5, label='Mode switch')
    ax_pos.set_xlabel('Time (s)', fontsize=11)
    ax_pos.set_ylabel('Position (m)', fontsize=11)
    ax_pos.set_title('Position Tracking vs Horizon Length',
                     fontsize=13, fontweight='bold')
    ax_pos.grid(True, alpha=0.3)
    ax_pos.legend(fontsize=9, ncol=3)

    # Velocity
    ax_vel = fig.add_subplot(gs[1, :])
    for i, result in enumerate(results):
        if result.success and len(result.state_history) > 0:
            vel = result.state_history[:, 3]
            label = f'N={result.horizon_steps}'
            ax_vel.plot(result.time_s, vel, linewidth=2, color=colors[i],
                       label=label, alpha=0.8)

    # Reference velocity
    if len(results) > 0 and results[0].success and test_config.velocity > 0:
        time_ref = results[0].time_s
        vel_ref = np.full_like(time_ref, test_config.velocity)

        # For drive-stop, set to 0 after transition
        if phase_transition is not None:
            stop_steps = np.where(time_ref >= phase_transition)[0]
            vel_ref[stop_steps] = 0.0

        ax_vel.plot(time_ref, vel_ref, 'k--', linewidth=2,
                   label='Reference', alpha=0.5)

    if phase_transition is not None:
        ax_vel.axvline(phase_transition, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax_vel.set_xlabel('Time (s)', fontsize=11)
    ax_vel.set_ylabel('Velocity (m/s)', fontsize=11)
    ax_vel.set_title('Velocity Tracking vs Horizon Length',
                     fontsize=13, fontweight='bold')
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend(fontsize=9, ncol=3)

    # Pitch
    ax_pitch = fig.add_subplot(gs[2, 0])
    for i, result in enumerate(results):
        if result.success and len(result.state_history) > 0:
            pitch = np.rad2deg(result.state_history[:, 1])
            label = f'N={result.horizon_steps}'
            ax_pitch.plot(result.time_s, pitch, linewidth=2, color=colors[i],
                         label=label, alpha=0.8)

    ax_pitch.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.3)
    if phase_transition is not None:
        ax_pitch.axvline(phase_transition, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax_pitch.set_xlabel('Time (s)', fontsize=11)
    ax_pitch.set_ylabel('Pitch (deg)', fontsize=11)
    ax_pitch.set_title('Pitch Angle vs Horizon Length', fontsize=12, fontweight='bold')
    ax_pitch.grid(True, alpha=0.3)

    # Control effort
    ax_ctrl = fig.add_subplot(gs[2, 1])
    for i, result in enumerate(results):
        if result.success and len(result.control_history) > 0:
            # Average torque magnitude
            ctrl_mag = np.mean(np.abs(result.control_history), axis=1)
            label = f'N={result.horizon_steps}'
            ax_ctrl.plot(result.time_s[:len(ctrl_mag)], ctrl_mag, linewidth=2,
                        color=colors[i], label=label, alpha=0.8)

    if phase_transition is not None:
        ax_ctrl.axvline(phase_transition, color='red', linestyle=':', linewidth=2, alpha=0.5)
    ax_ctrl.set_xlabel('Time (s)', fontsize=11)
    ax_ctrl.set_ylabel('Avg Torque Magnitude (Nm)', fontsize=11)
    ax_ctrl.set_title('Control Effort vs Horizon Length', fontsize=12, fontweight='bold')
    ax_ctrl.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/trajectory_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - trajectory_comparison.png")

    # ========================================================================
    # Plot 2: Performance Metrics vs Horizon
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    horizon_steps = [r.horizon_steps for r in results]
    horizon_seconds = [r.horizon_seconds for r in results]

    # Position error
    ax = axes[0, 0]
    pos_errors = [r.position_error for r in results]
    ax.plot(horizon_seconds, pos_errors, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Horizon Length (s)', fontsize=11)
    ax.set_ylabel('Position Error (m)', fontsize=11)
    ax.set_title('Position Accuracy vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add secondary x-axis for steps
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(horizon_seconds)
    ax2.set_xticklabels([f'{int(h)}' for h in horizon_steps], fontsize=9)
    ax2.set_xlabel('Horizon Length (steps)', fontsize=10)

    # Velocity tracking error
    ax = axes[0, 1]
    vel_errors = [r.velocity_tracking_error for r in results]
    ax.plot(horizon_seconds, vel_errors, 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Horizon Length (s)', fontsize=11)
    ax.set_ylabel('Velocity Tracking Error (m/s)', fontsize=11)
    ax.set_title('Velocity Tracking vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(horizon_seconds)
    ax2.set_xticklabels([f'{int(h)}' for h in horizon_steps], fontsize=9)
    ax2.set_xlabel('Horizon Length (steps)', fontsize=10)

    # Max pitch
    ax = axes[0, 2]
    max_pitches = [r.max_pitch_deg for r in results]
    ax.plot(horizon_seconds, max_pitches, 'o-', linewidth=2, markersize=8, color='red')
    ax.set_xlabel('Horizon Length (s)', fontsize=11)
    ax.set_ylabel('Max Pitch (deg)', fontsize=11)
    ax.set_title('Balance Performance vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(horizon_seconds)
    ax2.set_xticklabels([f'{int(h)}' for h in horizon_steps], fontsize=9)
    ax2.set_xlabel('Horizon Length (steps)', fontsize=10)

    # Final velocity (stopping performance)
    ax = axes[1, 0]
    final_vels = [abs(r.final_velocity) for r in results]
    ax.plot(horizon_seconds, final_vels, 'o-', linewidth=2, markersize=8, color='purple')
    ax.axhline(0.05, color='orange', linestyle='--', linewidth=2, alpha=0.5,
              label='Success threshold')
    ax.set_xlabel('Horizon Length (s)', fontsize=11)
    ax.set_ylabel('Final Velocity (m/s)', fontsize=11)
    ax.set_title('Stopping Performance vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(horizon_seconds)
    ax2.set_xticklabels([f'{int(h)}' for h in horizon_steps], fontsize=9)
    ax2.set_xlabel('Horizon Length (steps)', fontsize=10)

    # Solve time
    ax = axes[1, 1]
    solve_times = [r.mean_solve_time_ms for r in results]
    ax.plot(horizon_seconds, solve_times, 'o-', linewidth=2, markersize=8, color='brown')
    ax.set_xlabel('Horizon Length (s)', fontsize=11)
    ax.set_ylabel('Mean Solve Time (ms)', fontsize=11)
    ax.set_title('Computational Cost vs Horizon', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(horizon_seconds)
    ax2.set_xticklabels([f'{int(h)}' for h in horizon_steps], fontsize=9)
    ax2.set_xlabel('Horizon Length (steps)', fontsize=10)

    # Success rate
    ax = axes[1, 2]
    successes = [1 if r.success else 0 for r in results]
    ax.bar(horizon_seconds, successes, width=0.1, color='green', alpha=0.7,
          edgecolor='black', linewidth=2)
    ax.set_xlabel('Horizon Length (s)', fontsize=11)
    ax.set_ylabel('Success', fontsize=11)
    ax.set_title('Simulation Success vs Horizon', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Failed', 'Success'])
    ax.grid(True, alpha=0.3, axis='x')

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(horizon_seconds)
    ax2.set_xticklabels([f'{int(h)}' for h in horizon_steps], fontsize=9)
    ax2.set_xlabel('Horizon Length (steps)', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/performance_metrics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - performance_metrics.png")

    # ========================================================================
    # Plot 3: Phase-specific Analysis (only for drive-stop test)
    # ========================================================================
    if phase_transition is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Drive phase: Position vs velocity zoomed
        ax = axes[0, 0]
        for i, result in enumerate(results):
            if result.success and len(result.state_history) > 0:
                # Only plot drive phase
                drive_idx = np.where(result.time_s <= phase_transition)[0]
                if len(drive_idx) > 0:
                    pos = result.state_history[drive_idx, 0]
                    label = f'N={result.horizon_steps}'
                    ax.plot(result.time_s[drive_idx], pos, linewidth=2,
                           color=colors[i], label=label, alpha=0.8)

        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position (m)', fontsize=11)
        ax.set_title('Drive Phase - Position', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Drive phase: Velocity
        ax = axes[0, 1]
        for i, result in enumerate(results):
            if result.success and len(result.state_history) > 0:
                drive_idx = np.where(result.time_s <= phase_transition)[0]
                if len(drive_idx) > 0:
                    vel = result.state_history[drive_idx, 3]
                    label = f'N={result.horizon_steps}'
                    ax.plot(result.time_s[drive_idx], vel, linewidth=2,
                           color=colors[i], label=label, alpha=0.8)

        if test_config.velocity > 0:
            ax.axhline(test_config.velocity, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Target')
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Velocity (m/s)', fontsize=11)
        ax.set_title('Drive Phase - Velocity', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Stop phase: Position error
        ax = axes[1, 0]
        for i, result in enumerate(results):
            if result.success and len(result.state_history) > 0:
                stop_idx = np.where(result.time_s >= phase_transition)[0]
                if len(stop_idx) > 0:
                    pos = result.state_history[stop_idx, 0]
                    pos_error = pos - test_config.distance
                    label = f'N={result.horizon_steps}'
                    ax.plot(result.time_s[stop_idx], pos_error, linewidth=2,
                           color=colors[i], label=label, alpha=0.8)

        ax.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Position Error (m)', fontsize=11)
        ax.set_title('Stop Phase - Position Error', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        # Stop phase: Velocity decay
        ax = axes[1, 1]
        for i, result in enumerate(results):
            if result.success and len(result.state_history) > 0:
                stop_idx = np.where(result.time_s >= phase_transition)[0]
                if len(stop_idx) > 0:
                    vel = result.state_history[stop_idx, 3]
                    label = f'N={result.horizon_steps}'
                    ax.plot(result.time_s[stop_idx], vel, linewidth=2,
                           color=colors[i], label=label, alpha=0.8)

        ax.axhline(0, color='k', linestyle='-', linewidth=1, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Velocity (m/s)', fontsize=11)
        ax.set_title('Stop Phase - Velocity Decay', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/phase_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  - phase_analysis.png")


def plot_velocity_only(results: List[HorizonTestResult],
                       test_config: TestConfig,
                       save_dir: str):
    """Generate a single plot showing all velocity profiles together."""

    os.makedirs(save_dir, exist_ok=True)

    # Sort results by horizon length
    results = sorted(results, key=lambda r: r.horizon_steps)

    # Calculate phase transition time (for drive-stop test)
    if test_config.stop_time > 0:
        phase_transition = test_config.duration - test_config.stop_time
    else:
        phase_transition = None

    # Define color map for different horizons
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(results)))

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot velocity for each horizon
    for i, result in enumerate(results):
        if len(result.state_history) > 0:
            vel = result.state_history[:, 3]

            # Different styling for failed vs successful runs
            if result.success:
                linestyle = '-'
                alpha = 0.85
                linewidth = 2.5
                status = ''
            else:
                linestyle = '--'
                alpha = 0.6
                linewidth = 2.0
                status = ' [FAILED]'

            label = f'N={result.horizon_steps:3d} ({result.horizon_seconds:.2f}s, solve: {result.mean_solve_time_ms:.1f}ms){status}'
            ax.plot(result.time_s, vel, linewidth=linewidth, color=colors[i],
                   label=label, alpha=alpha, linestyle=linestyle)

    # Reference velocity (use longest successful run for time reference)
    successful_results = [r for r in results if r.success and len(r.state_history) > 0]
    if len(successful_results) > 0 and test_config.velocity > 0:
        # Use the result with the longest time series
        ref_result = max(successful_results, key=lambda r: len(r.time_s))
        time_ref = ref_result.time_s
        vel_ref = np.full_like(time_ref, test_config.velocity)

        # For drive-stop, set reference to 0 after transition
        if phase_transition is not None:
            stop_steps = np.where(time_ref >= phase_transition)[0]
            vel_ref[stop_steps] = 0.0

        ax.plot(time_ref, vel_ref, 'k--', linewidth=3,
               label='Reference', alpha=0.7, zorder=10)
    elif len(results) > 0 and test_config.velocity > 0:
        # If no successful runs, use any result for time reference
        ref_result = max(results, key=lambda r: len(r.time_s) if len(r.state_history) > 0 else 0)
        if len(ref_result.time_s) > 0:
            time_ref = ref_result.time_s
            vel_ref = np.full_like(time_ref, test_config.velocity)

            if phase_transition is not None:
                stop_steps = np.where(time_ref >= phase_transition)[0]
                vel_ref[stop_steps] = 0.0

            ax.plot(time_ref, vel_ref, 'k--', linewidth=3,
                   label='Reference', alpha=0.7, zorder=10)

    # Add phase transition line (for drive-stop test)
    if phase_transition is not None:
        ax.axvline(phase_transition, color='red', linestyle=':', linewidth=2.5, alpha=0.6,
                    label='Mode switch (drive → stop)', zorder=5)

    # Add tolerance bands
    if test_config.velocity > 0 and len(results) > 0:
        # Find max time across all results (successful or not)
        valid_results = [r for r in results if len(r.state_history) > 0]
        if len(valid_results) > 0:
            max_time = max(r.time_s[-1] for r in valid_results)

            if phase_transition is not None:
                # Drive phase tolerance
                ax.fill_between([0, phase_transition],
                              test_config.velocity - 0.02, test_config.velocity + 0.02,
                              color='green', alpha=0.15, label='±0.02 m/s tolerance (drive)')
                # Stop phase tolerance
                ax.fill_between([phase_transition, max_time], -0.02, 0.02,
                               color='blue', alpha=0.15, label='±0.02 m/s tolerance (stop)')
            else:
                # Constant velocity tracking tolerance
                ax.fill_between([0, max_time],
                              test_config.velocity - 0.02, test_config.velocity + 0.02,
                              color='green', alpha=0.15, label='±0.02 m/s tolerance')

    # Labels and formatting
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Velocity (m/s)', fontsize=14, fontweight='bold')
    ax.set_title(f'Velocity Tracking: {test_config.name} - MPC Horizon Comparison',
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black')

    # Add test-specific annotations
    if phase_transition is not None and len(results) > 0:
        valid_results = [r for r in results if len(r.state_history) > 0]
        if len(valid_results) > 0:
            # Drive-stop test: show both phases
            ax.text(phase_transition/2, ax.get_ylim()[1]*0.95, 'DRIVE PHASE',
                     ha='center', fontsize=13, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            max_time = max(r.time_s[-1] for r in valid_results)
            ax.text(phase_transition + (max_time-phase_transition)/2, ax.get_ylim()[1]*0.95, 'STOP PHASE',
                     ha='center', fontsize=13, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f"{save_dir}/velocity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - velocity_comparison.png")


def print_summary_table(results: List[HorizonTestResult],
                       velocity: float,
                       distance: float):
    """Print a summary table of all results."""

    print("\n" + "="*100)
    print("HORIZON COMPARISON SUMMARY")
    print("="*100)

    # Header
    header = (f"{'Horizon':<15} {'Lookahead':<12} {'Pos Error':<12} {'Vel Error':<12} "
             f"{'Max Pitch':<12} {'Final Vel':<12} {'Solve Time':<12} {'Success':<10}")
    print(header)
    print("-"*100)

    # Sort by horizon
    results = sorted(results, key=lambda r: r.horizon_steps)

    # Rows
    for r in results:
        row = (f"{r.horizon_steps:>6} steps   "
               f"{r.horizon_seconds:>7.3f}s    "
               f"{r.position_error:>7.4f} m   "
               f"{r.velocity_tracking_error:>7.4f} m/s  "
               f"{r.max_pitch_deg:>7.2f}°    "
               f"{abs(r.final_velocity):>8.5f} m/s "
               f"{r.mean_solve_time_ms:>7.2f} ms   "
               f"{'✓' if r.success else '✗':^10}")
        print(row)

    print("="*100)

    # Analysis
    print("\nKEY OBSERVATIONS:")

    # Best position accuracy
    best_pos = min(results, key=lambda r: r.position_error)
    print(f"  Best position accuracy: N={best_pos.horizon_steps} "
          f"(error: {best_pos.position_error:.4f} m)")

    # Best velocity tracking
    best_vel = min(results, key=lambda r: r.velocity_tracking_error)
    print(f"  Best velocity tracking: N={best_vel.horizon_steps} "
          f"(error: {best_vel.velocity_tracking_error:.4f} m/s)")

    # Best stopping
    best_stop = min(results, key=lambda r: abs(r.final_velocity))
    print(f"  Best stopping performance: N={best_stop.horizon_steps} "
          f"(final vel: {abs(best_stop.final_velocity):.5f} m/s)")

    # Fastest solve
    fastest = min(results, key=lambda r: r.mean_solve_time_ms)
    print(f"  Fastest solver: N={fastest.horizon_steps} "
          f"({fastest.mean_solve_time_ms:.2f} ms)")

    # Slowest solve
    slowest = max(results, key=lambda r: r.mean_solve_time_ms)
    print(f"  Slowest solver: N={slowest.horizon_steps} "
          f"({slowest.mean_solve_time_ms:.2f} ms)")

    print("="*100)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Create test configuration
    test_config = create_test_config(args.test, args)

    # Print test information
    print("\n" + "="*80)
    print("MPC HORIZON COMPARISON TEST")
    print("="*80)
    print(f"  Test: {test_config.name}")
    print(f"  Description: {test_config.description}")
    print(f"  Horizons to test: {args.horizons}")
    if args.challenge:
        print(f"  Mode: ⚡ CHALLENGE ⚡")
    if args.no_terminal_cost:
        print(f"  Terminal cost: DISABLED (testing raw horizon effects)")
    else:
        print(f"  Terminal cost: ENABLED (DARE)")
    print("="*80)

    # Run tests for each horizon
    results = []
    for horizon in args.horizons:
        try:
            result = run_horizon_test(
                horizon_steps=horizon,
                test_config=test_config,
                disable_terminal_cost=args.no_terminal_cost
            )
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Error testing horizon {horizon}: {e}")
            import traceback
            traceback.print_exc()

    if len(results) == 0:
        print("\n❌ No successful tests. Exiting.")
        sys.exit(1)

    # Print summary table
    print_summary_table(results, test_config.velocity, test_config.distance)

    # Generate comparison plots
    save_dir = f"debug_output/horizon_comparison_{args.test.replace('-', '_')}"
    print(f"\nGenerating comparison plots...")
    plot_velocity_only(results, test_config, save_dir)
    plot_comparison(results, test_config, save_dir)

    print(f"\n✓ All plots saved to: {save_dir}/")
    print(f"  - velocity_comparison.png (MAIN PLOT)")
    print(f"  - trajectory_comparison.png")
    print(f"  - performance_metrics.png")
    print(f"  - phase_analysis.png")

    print("\n" + "="*80)
    print("HORIZON COMPARISON COMPLETE!")
    print("="*80)

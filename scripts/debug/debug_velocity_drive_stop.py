"""Debug script for velocity mode: drive from A to B and stop."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
from simulation import SimulationConfig, MUJOCO_AVAILABLE
from debug import MPCDiagnostics
from mpc import ReferenceCommand, ReferenceMode

# Parse command line arguments
parser = argparse.ArgumentParser(description='Debug velocity drive and stop')
parser.add_argument('--viewer', action='store_true',
                    help='Show MuJoCo viewer (disables batch plotting)')
parser.add_argument('--velocity', type=float, default=0.1,
                    help='Forward velocity in m/s (default: 0.1)')
parser.add_argument('--distance', type=float, default=0.5,
                    help='Target distance in m (default: 0.5)')
parser.add_argument('--stop-time', type=float, default=3.0,
                    help='Time to hold balance after stopping (default: 3.0s)')
args = parser.parse_args()

if args.viewer and not MUJOCO_AVAILABLE:
    print("MuJoCo is not installed. Install with: pip install mujoco")
    exit(1)

# Configuration
config = SimulationConfig(
    model_path='mujoco_sim/robot_model.xml',
    robot_params_path='config/simulation/robot_params.yaml',
    mpc_params_path='config/simulation/mpc_params.yaml',
    estimator_params_path='config/simulation/estimator_params.yaml',
    duration_s=15.0,  # Default duration
)

# Create diagnostics
print("Creating MPC Diagnostics...")
diag = MPCDiagnostics(config)

# Calculate drive time based on velocity and distance
drive_time = args.distance / args.velocity if args.velocity > 0 else 5.0
total_duration = drive_time + args.stop_time

# Create time-varying reference callback
def reference_callback(time_s: float) -> ReferenceCommand:
    """Switch from velocity mode to balance mode at target position."""
    if time_s < drive_time:
        # Phase 1: Drive forward at constant velocity
        return ReferenceCommand(
            mode=ReferenceMode.VELOCITY,
            velocity_mps=args.velocity,
            yaw_rate_radps=0.0,
        )
    else:
        # Phase 2: Stop and balance
        return ReferenceCommand(mode=ReferenceMode.BALANCE)

print("\n" + "="*80)
print("VELOCITY MODE: DRIVE AND STOP TEST")
print("="*80)
print(f"  Velocity: {args.velocity} m/s")
print(f"  Target distance: {args.distance} m")
print(f"  Drive time: {drive_time:.1f}s")
print(f"  Stop time: {args.stop_time:.1f}s")
print(f"  Total duration: {total_duration:.1f}s")
print("="*80)

if args.viewer:
    # Interactive viewer mode
    print("\nMode: MuJoCo Viewer (interactive)")
    print("\nStarting MuJoCo viewer... (close window to stop)")
    print("Controls: Space=pause, Backspace=reset, Mouse=rotate, Scroll=zoom\n")

    # Run with viewer
    result = diag.simulation.run_with_viewer(
        initial_pitch_rad=0.0,
        reference_command_callback=reference_callback,
    )

    # Print results
    print("\n" + "=" * 50)
    print("Simulation Results")
    print("=" * 50)
    print(f"  Success (no fall): {result.success}")
    print(f"  Duration: {result.time_s[-1]:.1f}s" if len(result.time_s) > 0 else "  Duration: 0s")
    print(f"  Mean solve time: {result.mean_solve_time_ms:.1f}ms")
    print(f"  Max solve time: {result.max_solve_time_ms:.1f}ms")
    print(f"  Deadline violations: {result.deadline_violations}")

    if len(result.state_history) > 0:
        # Position analysis
        final_position = result.state_history[-1, 0]
        position_error = abs(final_position - args.distance)
        print(f"\n  Position:")
        print(f"    Target: {args.distance:.3f} m")
        print(f"    Actual: {final_position:.3f} m")
        print(f"    Error: {position_error:.3f} m ({position_error/args.distance*100:.1f}%)")

        # Velocity analysis
        final_velocity = result.state_history[-1, 3]
        print(f"\n  Final velocity: {final_velocity:.4f} m/s")
        if abs(final_velocity) < 0.05:
            print(f"    ✓ Stopped successfully")
        else:
            print(f"    ⚠ Still moving")

        # Pitch analysis
        max_pitch = np.rad2deg(np.max(np.abs(result.state_history[:, 1])))
        final_pitch = np.rad2deg(result.state_history[-1, 1])
        print(f"\n  Pitch:")
        print(f"    Max: {max_pitch:.1f}°")
        print(f"    Final: {final_pitch:.1f}°")

else:
    # Batch analysis mode with plots
    print("\nMode: Batch analysis with plots")
    print("\nRunning simulation...")

    # Run simulation with diagnostics
    result, summary = diag.run_with_diagnostics(
        duration_s=total_duration,
        initial_pitch_rad=0.0,
        reference_command_callback=reference_callback,
        verbose=True,
    )

    # Print summary statistics
    summary.print_summary()

    # Analysis
    print("\n" + "="*80)
    print("TRAJECTORY ANALYSIS")
    print("="*80)

    if len(result.state_history) > 0:
        # Position tracking
        final_position = result.state_history[-1, 0]
        position_error = abs(final_position - args.distance)
        print(f"\nPosition Tracking:")
        print(f"  Target distance: {args.distance:.3f} m")
        print(f"  Final position: {final_position:.3f} m")
        print(f"  Error: {position_error:.3f} m ({position_error/args.distance*100:.1f}%)")

        if position_error < 0.1:
            print(f"  ✓ Position accuracy: PASS (< 0.1m)")
        else:
            print(f"  ✗ Position accuracy: FAIL (> 0.1m)")

        # Velocity analysis
        velocities = result.state_history[:, 3]
        final_velocity = velocities[-1]

        # Find drive phase (approximate)
        drive_steps = int(drive_time / (total_duration / len(result.state_history)))
        if drive_steps > 0 and drive_steps < len(velocities):
            mean_drive_velocity = np.mean(velocities[10:drive_steps])  # Skip first 10 steps
            velocity_error = abs(mean_drive_velocity - args.velocity)

            print(f"\nVelocity Tracking (Drive Phase):")
            print(f"  Target: {args.velocity:.3f} m/s")
            print(f"  Mean actual: {mean_drive_velocity:.3f} m/s")
            print(f"  Error: {velocity_error:.3f} m/s ({velocity_error/args.velocity*100:.1f}%)")

        print(f"\nStopping Performance:")
        print(f"  Final velocity: {final_velocity:.4f} m/s")
        if abs(final_velocity) < 0.05:
            print(f"  ✓ Stopped successfully (< 0.05 m/s)")
        else:
            print(f"  ✗ Still moving (> 0.05 m/s)")

        # Pitch stability
        pitch_history = result.state_history[:, 1]
        max_pitch = np.rad2deg(np.max(np.abs(pitch_history)))
        final_pitch = np.rad2deg(pitch_history[-1])

        print(f"\nBalance Maintenance:")
        print(f"  Max pitch: {max_pitch:.2f}°")
        print(f"  Final pitch: {final_pitch:.2f}°")
        if max_pitch < 10.0:
            print(f"  ✓ Balance maintained (< 10°)")
        else:
            print(f"  ✗ Balance lost (> 10°)")

    # Generate plots
    save_dir = f"test_and_debug_output/velocity_drive_stop_{args.velocity}mps_{args.distance}m"
    print(f"\n\nGenerating diagnostic plots...")
    diag.plot_all(result, save_dir=save_dir, show=False)
    print(f"Plots saved to: {save_dir}/")
    print(f"  - state_comparison.png")
    print(f"  - control_analysis.png")
    print(f"  - prediction_accuracy.png")
    print(f"  - closed_loop_diagnosis.png")

    # Generate custom velocity plots
    print(f"\nGenerating velocity-specific plots...")
    import matplotlib.pyplot as plt
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Extract state data
    time = result.time_s
    pos = result.state_history[:, 0]
    pitch = result.state_history[:, 1]
    vel = result.state_history[:, 3]

    # Find transition time (drive to stop)
    drive_steps = int(drive_time / (total_duration / len(time)))

    # Plot 1: Position Tracking
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, pos, 'b-', linewidth=2, label='Actual position')
    ax.axhline(args.distance, color='r', linestyle='--', linewidth=2, label='Target position')
    ax.fill_between(time, args.distance - 0.1, args.distance + 0.1,
                      color='r', alpha=0.2, label='±0.1m tolerance')
    ax.axvline(drive_time, color='g', linestyle=':', linewidth=2, alpha=0.5,
                label='Switch to balance mode')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Position (m)', fontsize=12)
    ax.set_title('Position Tracking - Drive and Stop', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/position_tracking.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - position_tracking.png")

    # Plot 2: Velocity Profile
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, vel, 'b-', linewidth=2, label='Actual velocity')

    # Reference velocity
    vel_ref = np.zeros_like(time)
    vel_ref[:drive_steps] = args.velocity
    ax.plot(time, vel_ref, 'r--', linewidth=2, label='Target velocity')

    # Tolerance bands for drive phase
    ax.fill_between(time[:drive_steps], args.velocity - 0.05, args.velocity + 0.05,
                      color='r', alpha=0.2, label='±0.05 m/s tolerance')

    # Stop phase tolerance
    ax.fill_between(time[drive_steps:], -0.05, 0.05,
                      color='g', alpha=0.2, label='Stop tolerance')

    ax.axvline(drive_time, color='g', linestyle=':', linewidth=2, alpha=0.5,
                label='Mode switch')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Velocity (m/s)', fontsize=12)
    ax.set_title('Velocity Profile - Drive and Stop', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/velocity_profile.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - velocity_profile.png")

    # Plot 3: Phase Analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Drive phase
    if drive_steps > 0:
        drive_time_arr = time[:drive_steps]
        drive_pos = pos[:drive_steps]
        drive_vel = vel[:drive_steps]

        ax1.plot(drive_time_arr, drive_pos, 'b-', linewidth=2, label='Position')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(drive_time_arr, drive_vel, 'g-', linewidth=2, label='Velocity')

        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Position (m)', fontsize=12, color='b')
        ax1_twin.set_ylabel('Velocity (m/s)', fontsize=12, color='g')
        ax1.set_title('Drive Phase Analysis', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='y', labelcolor='b')
        ax1_twin.tick_params(axis='y', labelcolor='g')

    # Stop phase
    if drive_steps < len(time):
        stop_time_arr = time[drive_steps:]
        stop_pos = pos[drive_steps:]
        stop_vel = vel[drive_steps:]

        ax2.plot(stop_time_arr, stop_pos, 'b-', linewidth=2, label='Position')
        ax2.axhline(args.distance, color='r', linestyle='--', linewidth=1, alpha=0.5,
                     label='Target position')

        ax2_twin = ax2.twinx()
        ax2_twin.plot(stop_time_arr, stop_vel, 'g-', linewidth=2, label='Velocity')
        ax2_twin.axhline(0, color='k', linestyle=':', linewidth=1, alpha=0.5)

        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Position (m)', fontsize=12, color='b')
        ax2_twin.set_ylabel('Velocity (m/s)', fontsize=12, color='g')
        ax2.set_title('Stop Phase Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='g')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/phase_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - phase_analysis.png")

    # Plot 4: Balance (Pitch) Maintenance
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, np.rad2deg(pitch), 'b-', linewidth=2, label='Pitch angle')
    ax.axhline(10, color='r', linestyle='--', linewidth=2, alpha=0.5, label='±10° limit')
    ax.axhline(-10, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.fill_between(time, -10, 10, color='g', alpha=0.1, label='Safe zone')
    ax.axvline(drive_time, color='orange', linestyle=':', linewidth=2, alpha=0.5,
                label='Mode switch')

    # Annotate phases
    ax.text(drive_time/2, ax.get_ylim()[1]*0.9, 'DRIVE PHASE',
             ha='center', fontsize=12, fontweight='bold', color='blue', alpha=0.7)
    ax.text(drive_time + (total_duration-drive_time)/2, ax.get_ylim()[1]*0.9, 'STOP PHASE',
             ha='center', fontsize=12, fontweight='bold', color='green', alpha=0.7)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Pitch (degrees)', fontsize=12)
    ax.set_title('Balance Maintenance - Drive and Stop', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/pitch_stability.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - pitch_stability.png")

    # Overall result
    if not result.success:
        print(f"\n⚠️  FAILED: Robot fell after {result.time_s[-1]:.3f}s")
    else:
        print(f"\n✓  SUCCESS: Robot completed drive and stop maneuver")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

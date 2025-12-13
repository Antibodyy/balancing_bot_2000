"""Debug script for velocity mode: circular motion at constant speed."""

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
parser = argparse.ArgumentParser(description='Debug velocity circular motion')
parser.add_argument('--viewer', action='store_true',
                    help='Show MuJoCo viewer (disables batch plotting)')
parser.add_argument('--velocity', type=float, default=0.2,
                    help='Forward velocity in m/s (default: 0.2)')
parser.add_argument('--radius', type=float, default=0.5,
                    help='Circle radius in m (default: 0.5)')
parser.add_argument('--circles', type=float, default=1.0,
                    help='Number of circles to complete (default: 1.0)')
args = parser.parse_args()

# Calculate duration for specified number of circles
# Circle circumference = 2πr, time = distance/speed
circle_circumference = 2 * np.pi * args.radius
duration_per_circle = circle_circumference / args.velocity if args.velocity > 0 else 10.0
args.duration = duration_per_circle * args.circles

if args.viewer and not MUJOCO_AVAILABLE:
    print("MuJoCo is not installed. Install with: pip install mujoco")
    exit(1)

# Configuration
config = SimulationConfig(
    model_path='mujoco_sim/robot_model.xml',
    robot_params_path='config/simulation/robot_params.yaml',
    mpc_params_path='config/simulation/mpc_params.yaml',
    estimator_params_path='config/simulation/estimator_params.yaml',
    duration_s=args.duration,
)

# Create diagnostics
print("Creating MPC Diagnostics...")
diag = MPCDiagnostics(config)

# Calculate yaw rate for circular motion: ω = v / r
yaw_rate = args.velocity / args.radius if args.radius > 0 else 0.0

# Calculate expected heading change
expected_heading_change = yaw_rate * args.duration
circle_fraction = expected_heading_change / (2 * np.pi)

# Create constant circular motion reference
def reference_callback(time_s: float) -> ReferenceCommand:
    """Constant velocity and yaw rate for circular motion."""
    return ReferenceCommand(
        mode=ReferenceMode.VELOCITY,
        velocity_mps=args.velocity,
        yaw_rate_radps=yaw_rate,
    )

print("\n" + "="*80)
print("VELOCITY MODE: CIRCULAR MOTION TEST")
print("="*80)
print(f"  Forward velocity: {args.velocity} m/s")
print(f"  Circle radius: {args.radius} m")
print(f"  Circle circumference: {circle_circumference:.3f} m")
print(f"  Yaw rate: {yaw_rate:.3f} rad/s ({np.rad2deg(yaw_rate):.1f}°/s)")
print(f"  Duration: {args.duration:.2f}s ({args.circles:.1f} circles)")
print(f"\n  Expected heading change: {expected_heading_change:.3f} rad ({np.rad2deg(expected_heading_change):.1f}°)")
print(f"  Circle fraction: {circle_fraction:.2f} ({circle_fraction*100:.1f}%)")
if circle_fraction >= 1.0:
    print(f"  → Will complete {circle_fraction:.1f} full circles")
else:
    print(f"  → Will complete partial circle arc")
print("="*80)

if args.viewer:
    # Interactive viewer mode
    print("\nMode: MuJoCo Viewer (interactive)")
    print("\nStarting MuJoCo viewer... (close window to stop)")
    print("Controls: Space=pause, Backspace=reset, Mouse=rotate, Scroll=zoom")
    print("\nTIP: Use top-down view to see circular motion clearly\n")

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
        # Velocity analysis
        velocities = result.state_history[:, 3]
        mean_velocity = np.mean(velocities[10:])  # Skip first 10 steps
        velocity_error = abs(mean_velocity - args.velocity)
        print(f"\n  Velocity tracking:")
        print(f"    Target: {args.velocity:.3f} m/s")
        print(f"    Mean actual: {mean_velocity:.3f} m/s")
        print(f"    Error: {velocity_error:.4f} m/s ({velocity_error/args.velocity*100:.1f}%)")

        # Yaw rate analysis
        yaw_rates = result.state_history[:, 5]
        mean_yaw_rate = np.mean(yaw_rates[10:])
        yaw_rate_error = abs(mean_yaw_rate - yaw_rate)
        print(f"\n  Yaw rate tracking:")
        print(f"    Target: {yaw_rate:.3f} rad/s ({np.rad2deg(yaw_rate):.1f}°/s)")
        print(f"    Mean actual: {mean_yaw_rate:.3f} rad/s ({np.rad2deg(mean_yaw_rate):.1f}°/s)")
        print(f"    Error: {yaw_rate_error:.4f} rad/s ({yaw_rate_error/yaw_rate*100:.1f}%)")

        # Heading analysis
        headings = result.state_history[:, 2]
        actual_heading_change = headings[-1] - headings[0]
        heading_error = abs(actual_heading_change - expected_heading_change)
        print(f"\n  Heading change:")
        print(f"    Expected: {expected_heading_change:.3f} rad ({np.rad2deg(expected_heading_change):.1f}°)")
        print(f"    Actual: {actual_heading_change:.3f} rad ({np.rad2deg(actual_heading_change):.1f}°)")
        print(f"    Error: {heading_error:.3f} rad ({np.rad2deg(heading_error):.1f}°)")

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
        duration_s=args.duration,
        initial_pitch_rad=0.0,
        reference_command_callback=reference_callback,
        verbose=True,
    )

    # Print summary statistics
    summary.print_summary()

    # Analysis
    print("\n" + "="*80)
    print("CIRCULAR MOTION ANALYSIS")
    print("="*80)

    if len(result.state_history) > 0:
        # Velocity tracking (skip settling time)
        velocities = result.state_history[:, 3]
        settling_steps = max(10, int(2.0 / (args.duration / len(result.state_history))))

        if settling_steps < len(velocities):
            mean_velocity = np.mean(velocities[settling_steps:])
            velocity_error = abs(mean_velocity - args.velocity)

            print(f"\nVelocity Tracking (after {settling_steps} steps settling):")
            print(f"  Target: {args.velocity:.3f} m/s")
            print(f"  Mean actual: {mean_velocity:.3f} m/s")
            print(f"  Error: {velocity_error:.4f} m/s ({velocity_error/args.velocity*100:.1f}%)")

            if velocity_error < 0.05:
                print(f"  ✓ Velocity tracking: PASS (< 0.05 m/s)")
            else:
                print(f"  ✗ Velocity tracking: FAIL (> 0.05 m/s)")

        # Yaw rate tracking
        yaw_rates = result.state_history[:, 5]
        if settling_steps < len(yaw_rates):
            mean_yaw_rate = np.mean(yaw_rates[settling_steps:])
            yaw_rate_error = abs(mean_yaw_rate - yaw_rate)

            print(f"\nYaw Rate Tracking:")
            print(f"  Target: {yaw_rate:.3f} rad/s ({np.rad2deg(yaw_rate):.1f}°/s)")
            print(f"  Mean actual: {mean_yaw_rate:.3f} rad/s ({np.rad2deg(mean_yaw_rate):.1f}°/s)")
            print(f"  Error: {yaw_rate_error:.4f} rad/s ({yaw_rate_error/yaw_rate*100:.1f}%)")

            if yaw_rate_error < 0.1:
                print(f"  ✓ Yaw rate tracking: PASS (< 0.1 rad/s)")
            else:
                print(f"  ✗ Yaw rate tracking: FAIL (> 0.1 rad/s)")

        # Heading change analysis
        headings = result.state_history[:, 2]
        actual_heading_change = headings[-1] - headings[0]
        heading_error = abs(actual_heading_change - expected_heading_change)

        print(f"\nHeading Change:")
        print(f"  Expected: {expected_heading_change:.3f} rad ({np.rad2deg(expected_heading_change):.1f}°)")
        print(f"  Actual: {actual_heading_change:.3f} rad ({np.rad2deg(actual_heading_change):.1f}°)")
        print(f"  Error: {heading_error:.3f} rad ({np.rad2deg(heading_error):.1f}°)")

        if heading_error < 0.3:
            print(f"  ✓ Heading accuracy: PASS (< 0.3 rad)")
        else:
            print(f"  ✗ Heading accuracy: FAIL (> 0.3 rad)")

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

        # Circular trajectory analysis
        positions_x = result.state_history[:, 0]
        arc_length = abs(positions_x[-1] - positions_x[0])  # Approximate
        expected_arc = args.velocity * args.duration

        print(f"\nTrajectory:")
        print(f"  Expected arc length: {expected_arc:.3f} m")
        print(f"  Approximate traveled: {arc_length:.3f} m")

    # Generate plots
    save_dir = f"test_and_debug_output/velocity_circle_{args.velocity}mps_{args.radius}m_{args.circles}circles"
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
    pos_x = result.state_history[:, 0]
    pitch = result.state_history[:, 1]
    yaw = result.state_history[:, 2]
    vel = result.state_history[:, 3]
    yaw_rate_actual = result.state_history[:, 5]

    # Plot 1: Velocity and Yaw Rate Tracking
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Velocity tracking
    ax1.plot(time, vel, 'b-', linewidth=2, label='Actual velocity')
    ax1.axhline(args.velocity, color='r', linestyle='--', linewidth=2, label='Target velocity')
    ax1.fill_between(time, args.velocity - 0.05, args.velocity + 0.05,
                      color='r', alpha=0.2, label='±0.05 m/s tolerance')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Velocity (m/s)', fontsize=12)
    ax1.set_title('Forward Velocity Tracking', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Yaw rate tracking
    ax2.plot(time, yaw_rate_actual, 'b-', linewidth=2, label='Actual yaw rate')
    ax2.axhline(yaw_rate, color='r', linestyle='--', linewidth=2, label='Target yaw rate')
    ax2.fill_between(time, yaw_rate - 0.1, yaw_rate + 0.1,
                      color='r', alpha=0.2, label='±0.1 rad/s tolerance')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Yaw rate (rad/s)', fontsize=12)
    ax2.set_title('Yaw Rate Tracking', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/velocity_tracking.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - velocity_tracking.png")

    # Plot 2: 2D Trajectory (X-Y path)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Calculate Y positions from heading (approximate for visualization)
    # For circular motion: x(t) = r*sin(θ), y(t) = r*(1-cos(θ))
    headings = yaw
    pos_y = np.zeros_like(pos_x)
    for i in range(1, len(pos_x)):
        # Integrate velocity in x and y directions based on heading
        dt = time[i] - time[i-1]
        pos_y[i] = pos_y[i-1] + vel[i] * np.sin(headings[i]) * dt
        # Adjust pos_x to account for heading
        if i > 1:
            pos_x[i] = pos_x[i-1] + vel[i] * np.cos(headings[i]) * dt

    # Plot trajectory
    ax.plot(pos_x, pos_y, 'b-', linewidth=2, label='Actual path')
    ax.plot(pos_x[0], pos_y[0], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(pos_x[-1], pos_y[-1], 'ro', markersize=12, label='End', zorder=5)

    # Plot theoretical circle
    if args.radius > 0:
        theta_theory = np.linspace(0, expected_heading_change, 100)
        x_theory = args.radius * np.sin(theta_theory)
        y_theory = args.radius * (1 - np.cos(theta_theory))
        ax.plot(x_theory, y_theory, 'r--', linewidth=2, alpha=0.5, label=f'Ideal circle (r={args.radius}m)')

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('2D Trajectory - Circular Motion', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/trajectory_2d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - trajectory_2d.png")

    # Plot 3: Heading over time
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, np.rad2deg(yaw), 'b-', linewidth=2, label='Actual heading')
    # Expected heading (using target yaw_rate, not actual)
    expected_yaw = yaw_rate * time
    ax.plot(time, np.rad2deg(expected_yaw), 'r--', linewidth=2, label='Expected heading')
    ax.fill_between(time, np.rad2deg(expected_yaw - 0.3), np.rad2deg(expected_yaw + 0.3),
                      color='r', alpha=0.2, label='±0.3 rad tolerance')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Heading (degrees)', fontsize=12)
    ax.set_title('Heading Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/heading_tracking.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - heading_tracking.png")

    # Plot 4: Balance (Pitch) during motion
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time, np.rad2deg(pitch), 'b-', linewidth=2, label='Pitch angle')
    ax.axhline(10, color='r', linestyle='--', linewidth=2, alpha=0.5, label='±10° limit')
    ax.axhline(-10, color='r', linestyle='--', linewidth=2, alpha=0.5)
    ax.fill_between(time, -10, 10, color='g', alpha=0.1, label='Safe zone')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Pitch (degrees)', fontsize=12)
    ax.set_title('Balance Maintenance During Circular Motion', fontsize=14, fontweight='bold')
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
        print(f"\n✓  SUCCESS: Robot completed circular motion")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

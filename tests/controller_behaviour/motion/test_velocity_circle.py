"""Debug script for velocity mode: circular motion at constant speed."""

import sys
from pathlib import Path

# Add project root and debug to path so mpc_diagnostics is importable when run directly
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "debug"))

import argparse
import numpy as np
from simulation import SimulationConfig, MUJOCO_AVAILABLE
from debug import mpc_diagnostics
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
diag = mpc_diagnostics.MPCDiagnostics(config)

# Calculate yaw rate for circular motion: omega = v / r
yaw_rate = args.velocity / args.radius if args.radius > 0 else 0.0
expected_heading_change = yaw_rate * args.duration
circle_fraction = expected_heading_change / (2 * np.pi)


def reference_callback(time_s: float) -> ReferenceCommand:
    """Constant velocity and yaw rate for circular motion."""
    return ReferenceCommand(
        mode=ReferenceMode.VELOCITY,
        velocity_mps=args.velocity,
        yaw_rate_radps=yaw_rate,
    )


def heading_tracking(result):
    headings = result.state_history[:, 2]
    actual_heading_change = headings[-1] - headings[0]
    heading_error = abs(actual_heading_change - expected_heading_change)
    return headings, actual_heading_change, heading_error


def print_results(result):
    print("\n" + "=" * 50)
    print("Simulation Results")
    print("=" * 50)
    print(f"  Success (no fall): {result.success}")
    print(f"  Duration: {result.time_s[-1]:.1f}s" if len(result.time_s) > 0 else "  Duration: 0s")
    print(f"  Mean solve time: {result.mean_solve_time_ms:.1f}ms")
    print(f"  Max solve time: {result.max_solve_time_ms:.1f}ms")
    print(f"  Deadline violations: {result.deadline_violations}")

    if len(result.state_history) > 0:
        velocities = result.state_history[:, 3]
        mean_velocity = np.mean(velocities[10:]) if len(velocities) > 10 else np.mean(velocities)
        velocity_error = abs(mean_velocity - args.velocity)
        print(f"\n  Velocity tracking:")
        print(f"    Target: {args.velocity:.3f} m/s")
        print(f"    Mean actual: {mean_velocity:.3f} m/s")
        print(f"    Error: {velocity_error:.4f} m/s ({velocity_error/args.velocity*100:.1f}%)")

        yaw_rates = result.state_history[:, 5]
        mean_yaw_rate = np.mean(yaw_rates[10:]) if len(yaw_rates) > 10 else np.mean(yaw_rates)
        yaw_rate_error = abs(mean_yaw_rate - yaw_rate)
        print(f"\n  Yaw rate tracking:")
        print(f"    Target: {yaw_rate:.3f} rad/s ({np.rad2deg(yaw_rate):.1f} deg/s)")
        print(f"    Mean actual: {mean_yaw_rate:.3f} rad/s ({np.rad2deg(mean_yaw_rate):.1f} deg/s)")
        print(f"    Error: {yaw_rate_error:.4f} rad/s ({(yaw_rate_error / yaw_rate * 100) if yaw_rate != 0 else 0:.1f}%)")

        headings, actual_heading_change, heading_error = heading_tracking(result)
        print(f"\n  Heading change:")
        print(f"    Expected: {expected_heading_change:.3f} rad ({np.rad2deg(expected_heading_change):.1f} deg)")
        print(f"    Actual: {actual_heading_change:.3f} rad ({np.rad2deg(actual_heading_change):.1f} deg)")
        print(f"    Error: {heading_error:.3f} rad ({np.rad2deg(heading_error):.1f} deg)")

        max_pitch = np.rad2deg(np.max(np.abs(result.state_history[:, 1])))
        final_pitch = np.rad2deg(result.state_history[-1, 1])
        print(f"\n  Pitch:")
        print(f"    Max: {max_pitch:.1f} deg")
        print(f"    Final: {final_pitch:.1f} deg")


print("\n" + "="*80)
print("VELOCITY MODE: CIRCULAR MOTION TEST")
print("="*80)
print(f"  Forward velocity: {args.velocity} m/s")
print(f"  Circle radius: {args.radius} m")
print(f"  Circle circumference: {circle_circumference:.3f} m")
print(f"  Yaw rate: {yaw_rate:.3f} rad/s ({np.rad2deg(yaw_rate):.1f} deg/s)")
print(f"  Duration: {args.duration:.2f}s ({args.circles:.1f} circles)")
print(f"\n  Expected heading change: {expected_heading_change:.3f} rad ({np.rad2deg(expected_heading_change):.1f} deg)")
print(f"  Circle fraction: {circle_fraction:.2f} ({circle_fraction*100:.1f}%)")
print("="*80)

if args.viewer:
    print("\nMode: MuJoCo Viewer (interactive)")
    print("\nStarting MuJoCo viewer... (close window to stop)")
    print("Controls: Space=pause, Backspace=reset, Mouse=rotate, Scroll=zoom")
    print("\nTIP: Use top-down view to see circular motion clearly\n")

    result = diag.simulation.run_with_viewer(
        initial_pitch_rad=0.0,
        reference_command_callback=reference_callback,
    )
    print_results(result)
else:
    print("\nMode: Batch analysis with plots")
    print("\nRunning simulation...")
    result, summary = diag.run_with_diagnostics(
        duration_s=args.duration,
        initial_pitch_rad=0.0,
        reference_command_callback=reference_callback,
        verbose=True,
    )
    summary.print_summary()
    headings, actual_heading_change, heading_error = heading_tracking(result)

    print("\n" + "="*80)
    print("CIRCULAR MOTION ANALYSIS")
    print("="*80)
    # Print heading and velocity errors (already in summary)
    print_results(result)

    save_dir = f"test_and_debug_output/velocity_circle_{args.velocity}mps_{args.radius}m_{args.circles}circles"
    print(f"\nGenerating diagnostic plots...")
    diag.plot_all(result, save_dir=save_dir, show=False)
    print(f"Plots saved to: {save_dir}/")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)

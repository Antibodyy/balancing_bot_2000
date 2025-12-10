#!/usr/bin/env python3
"""Run MPC simulation with MuJoCo viewer.

Usage:
    python3 run_simulation.py              # Run with default settings
    python3 run_simulation.py --perturb 5  # Start with 5 degree tilt
    python3 run_simulation.py --velocity   # Track velocity reference
"""

import argparse
import numpy as np

from simulation import MPCSimulation, MUJOCO_AVAILABLE
from mpc import ReferenceCommand, ReferenceMode


def main():
    parser = argparse.ArgumentParser(description='Run MPC simulation with visualization')
    parser.add_argument('--perturb', type=float, default=0.0,
                        help='Initial pitch perturbation in degrees')
    parser.add_argument('--velocity', action='store_true',
                        help='Use velocity tracking mode')
    parser.add_argument('--vel-mps', type=float, default=0.1,
                        help='Target velocity in m/s (with --velocity)')
    parser.add_argument('--circular', action='store_true',
                        help='Use circular path tracking mode')
    parser.add_argument('--radius', type=float, default=1.0,
                        help='Circle radius in meters (with --circular)')
    parser.add_argument('--clockwise', action='store_true',
                        help='Rotate clockwise (default: counter-clockwise)')
    args = parser.parse_args()

    if not MUJOCO_AVAILABLE:
        print("MuJoCo is not installed. Install with: pip install mujoco")
        return 1

    # Create simulation
    sim = MPCSimulation()

    # Set up reference command
    if args.circular:
        yaw_rate = args.vel_mps / args.radius
        command = ReferenceCommand(
            mode=ReferenceMode.CIRCULAR,
            radius_m=args.radius,
            target_velocity_mps=args.vel_mps,
            center_x_m=args.radius,  # Start robot on circle
            center_y_m=0.0,
            clockwise=args.clockwise,
        )
        direction = "clockwise" if args.clockwise else "counter-clockwise"
        print(f"Mode: Circular path (r={args.radius}m, v={args.vel_mps}m/s, "
              f"ω={yaw_rate:.3f}rad/s, {direction})")
    elif args.velocity:
        command = ReferenceCommand(
            mode=ReferenceMode.VELOCITY,
            velocity_mps=args.vel_mps,
            yaw_rate_radps=0.0,
        )
        print(f"Mode: Velocity tracking ({args.vel_mps} m/s)")
    else:
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)
        print("Mode: Balance")

    initial_pitch = np.deg2rad(args.perturb)
    print(f"Initial perturbation: {args.perturb:.1f} degrees")
    print("\nStarting MuJoCo viewer... (close window to stop)")
    print("Controls: Space=pause, Backspace=reset, Mouse=rotate, Scroll=zoom\n")

    # Run with viewer
    result = sim.run_with_viewer(
        initial_pitch_rad=initial_pitch,
        reference_command=command,
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
        max_pitch = np.rad2deg(np.max(np.abs(result.state_history[:, 1])))
        print(f"  Max pitch: {max_pitch:.1f} degrees")

    return 0


if __name__ == '__main__':
    exit(main())

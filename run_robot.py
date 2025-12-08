#!/usr/bin/env python3
"""Main entry point to launch the MPC balancing robot.

This script provides a simple command-line interface to run the robot in
different modes:
- balance: Stationary balancing (default)
- velocity: Track constant forward velocity and yaw rate
- waypoint: Follow waypoint trajectory (future feature)

Examples:
    # Balance mode (stationary)
    python run_robot.py --mode balance

    # Forward motion at 0.1 m/s
    python run_robot.py --mode velocity --velocity 0.1

    # Turn in place at 0.5 rad/s
    python run_robot.py --mode velocity --yaw-rate 0.5

    # Drive in circle
    python run_robot.py --mode velocity --velocity 0.1 --yaw-rate 0.5

    # Specify custom I2C address
    python run_robot.py --address 0x20

    # Run for 60 seconds
    python run_robot.py --mode balance --duration 60
"""

import argparse
import sys

from hardware import (
    load_hardware_mpc,
    BalboaI2CInterface,
    HardwareBalanceController
)
from mpc import ReferenceCommand, ReferenceMode


def parse_args():
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='MPC Balancing Robot Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Control mode
    parser.add_argument(
        '--mode',
        choices=['balance', 'velocity', 'waypoint'],
        default='balance',
        help='Control mode (default: balance)'
    )

    # Velocity mode parameters
    parser.add_argument(
        '--velocity',
        type=float,
        default=0.0,
        help='Forward velocity in m/s (for velocity mode, default: 0.0)'
    )
    parser.add_argument(
        '--yaw-rate',
        type=float,
        default=0.0,
        help='Yaw rate in rad/s (for velocity mode, default: 0.0)'
    )

    # Waypoint mode parameters
    parser.add_argument(
        '--waypoints',
        type=str,
        default=None,
        help='Path to waypoints file (for waypoint mode)'
    )

    # Hardware parameters
    parser.add_argument(
        '--bus',
        type=int,
        default=1,
        help='I2C bus number (default: 1)'
    )
    parser.add_argument(
        '--address',
        type=lambda x: int(x, 0),  # Support 0x20 hex notation
        default=0x20,
        help='I2C slave address (default: 0x20)'
    )

    # Control loop parameters
    parser.add_argument(
        '--frequency',
        type=float,
        default=50.0,
        help='Control loop frequency in Hz (default: 50.0)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        help='Run duration in seconds (default: run until Ctrl+C)'
    )

    # Configuration paths
    parser.add_argument(
        '--robot-params',
        type=str,
        default='config/hardware/robot_params.yaml',
        help='Path to robot parameters YAML'
    )
    parser.add_argument(
        '--mpc-params',
        type=str,
        default='config/hardware/mpc_params.yaml',
        help='Path to MPC parameters YAML'
    )
    parser.add_argument(
        '--estimator-params',
        type=str,
        default='config/hardware/estimator_params.yaml',
        help='Path to estimator parameters YAML'
    )

    # Logging
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Disable progress logging'
    )

    return parser.parse_args()


def create_reference_command(args) -> ReferenceCommand:
    """Create reference command from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        ReferenceCommand for MPC controller
    """
    if args.mode == 'balance':
        return ReferenceCommand(mode=ReferenceMode.BALANCE)

    elif args.mode == 'velocity':
        return ReferenceCommand(
            mode=ReferenceMode.VELOCITY,
            velocity_mps=args.velocity,
            yaw_rate_radps=args.yaw_rate
        )

    elif args.mode == 'waypoint':
        if args.waypoints is None:
            print("Error: --waypoints file required for waypoint mode")
            sys.exit(1)
        # TODO: Implement waypoint loading from file
        print("Waypoint mode not yet implemented")
        sys.exit(1)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("MPC BALANCING ROBOT - Hardware Controller")
    print("=" * 60)
    print(f"Mode:           {args.mode}")
    if args.mode == 'velocity':
        print(f"Velocity:       {args.velocity:.3f} m/s")
        print(f"Yaw rate:       {args.yaw_rate:.3f} rad/s")
    print(f"I2C bus:        {args.bus}")
    print(f"I2C address:    0x{args.address:02X}")
    print(f"Frequency:      {args.frequency:.1f} Hz")
    print(f"Duration:       {args.duration if args.duration else 'unlimited'}")
    print("=" * 60)
    print()

    # Load MPC controller configuration
    print("Loading MPC controller...")
    try:
        balance_controller = load_hardware_mpc(
            robot_params_path=args.robot_params,
            mpc_params_path=args.mpc_params,
            estimator_params_path=args.estimator_params
        )
        print("✓ MPC controller loaded")
    except Exception as e:
        print(f"✗ Failed to load MPC controller: {e}")
        sys.exit(1)

    # Connect to Arduino via I2C
    print(f"Connecting to Arduino on I2C bus {args.bus} address 0x{args.address:02X}...")
    try:
        i2c_interface = BalboaI2CInterface(
            bus=args.bus,
            address=args.address
        )
        print("✓ I2C connection established")
    except Exception as e:
        print(f"✗ Failed to connect to Arduino: {e}")
        print("\nTroubleshooting:")
        print("  - Check that Raspberry Pi is mounted on Balboa")
        print("  - Verify Balboa is powered on (green LED blink on startup)")
        print("  - Check I2C is enabled: sudo raspi-config > Interface Options > I2C")
        print("  - Verify firmware uploaded with correct I2C address (0x20)")
        print("  - Test I2C detection: sudo i2cdetect -y 1")
        sys.exit(1)

    # Create hardware controller
    print("Initializing hardware controller...")
    hw_controller = HardwareBalanceController(
        i2c_interface=i2c_interface,
        balance_controller=balance_controller,
        target_frequency_hz=args.frequency,
        enable_logging=not args.quiet
    )
    print("✓ Hardware controller ready")
    print()

    # Create reference command
    reference = create_reference_command(args)

    # Run control loop
    try:
        stats = hw_controller.run_control_loop(
            reference_command=reference,
            duration_s=args.duration
        )
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nControl loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nClosing I2C connection...")
        i2c_interface.close()
        print("✓ Done")


if __name__ == '__main__':
    main()

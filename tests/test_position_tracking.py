#!/usr/bin/env python3
"""Position control example using serial communication.

Demonstrates using POSITION mode to navigate to specific locations
via serial interface (USB connection to Arduino).

Usage:
    python scripts/example_position_control.py
    
    # Or specify custom serial port:
    python scripts/example_position_control.py --port /dev/ttyACM0
"""

import argparse
from hardware import (
    load_hardware_mpc,
    HardwareBalanceController
)
from hardware.serial_interface import BalboaSerialInterface  # Import serial interface
from mpc import ReferenceCommand, ReferenceMode


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Position Control via Serial Communication'
    )
    
    parser.add_argument(
        '--port',
        type=str,
        default='/dev/ttyACM0',
        help='Serial port (default: /dev/ttyACM0, Windows: COM3)'
    )
    
    parser.add_argument(
        '--baudrate',
        type=int,
        default=115200,
        help='Serial baud rate (default: 115200)'
    )
    
    parser.add_argument(
        '--target-position',
        type=float,
        default=1.5,
        help='Target position in meters (default: 1.5)'
    )
    
    parser.add_argument(
        '--target-heading',
        type=float,
        default=0.0,
        help='Target heading in radians (default: 0.0)'
    )
    
    parser.add_argument(
        '--frequency',
        type=float,
        default=50.0,
        help='Control loop frequency in Hz (default: 50.0)'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=30.0,
        help='Run duration in seconds (default: 30.0)'
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
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("POSITION CONTROL TEST - Serial Communication")
    print("=" * 60)
    print(f"Serial port:        {args.port}")
    print(f"Baud rate:          {args.baudrate}")
    print(f"Target position:    {args.target_position:.3f} m")
    print(f"Target heading:     {args.target_heading:.3f} rad")
    print(f"Control frequency:  {args.frequency:.1f} Hz")
    print(f"Duration:           {args.duration:.1f} s")
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
        return
    
    # Connect to Arduino via Serial
    print(f"Connecting to Arduino on {args.port}...")
    try:
        serial_interface = BalboaSerialInterface(
            port=args.port,
            baudrate=args.baudrate,
            timeout=0.1
        )
        print("✓ Serial connection established")
        print("  (Arduino should have reset - wait 2 seconds...)")
    except Exception as e:
        print(f"✗ Failed to connect to Arduino: {e}")
        print("\nTroubleshooting:")
        print("  - Check USB cable connection")
        print("  - Verify Arduino is powered on")
        print("  - Check port name (Linux: /dev/ttyACM0, Windows: COM3)")
        print("  - Ensure no other program is using the port")
        print("  - Check permissions: sudo usermod -a -G dialout $USER")
        return
    
    # Create hardware controller with serial interface
    print("Initializing hardware controller...")
    hw_controller = HardwareBalanceController(
        i2c_interface=serial_interface,  # Pass serial interface here
        balance_controller=balance_controller,
        target_frequency_hz=args.frequency,
        enable_logging=True
    )
    print("✓ Hardware controller ready")
    print()
    
    # Create position reference command
    reference = ReferenceCommand(
        mode=ReferenceMode.POSITION,
        target_position_m=args.target_position,
        target_heading_rad=args.target_heading
    )
    
    print(f"Starting position control to {args.target_position}m...")
    print("Press Ctrl+C to stop\n")
    
    # Run control loop
    try:
        stats = hw_controller.run_control_loop(
            reference_command=reference,
            duration_s=args.duration
        )
        
        print("\n" + "=" * 60)
        print("Control Loop Statistics:")
        print("=" * 60)
        print(f"Total time:         {stats.total_time_s:.2f} s")
        print(f"Total cycles:       {stats.total_cycles}")
        print(f"Average frequency:  {stats.average_frequency_hz:.2f} Hz")
        print(f"Min frequency:      {stats.min_frequency_hz:.2f} Hz")
        print(f"Max frequency:      {stats.max_frequency_hz:.2f} Hz")
        
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\n\nControl loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup - send zero torque and close serial
        print("\nClosing serial connection...")
        serial_interface.close()
        print("✓ Done")


if __name__ == '__main__':
    main()

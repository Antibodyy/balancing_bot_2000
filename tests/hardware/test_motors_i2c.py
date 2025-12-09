#!/usr/bin/env python3
"""Test Motors via I2C

Simple motor test to verify motors respond to torque commands.
⚠️ Robot wheels must be elevated or on test stand!

Usage:
    python tests/hardware/test_motors_i2c.py --bus 1 --torque 0.05 --duration 2
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hardware import BalboaI2CInterface


def test_motors(bus: int, torque_nm: float, duration_s: float):
    """Test motor response via I2C.

    Args:
        bus: I2C bus number
        torque_nm: Torque command in N·m
        duration_s: Test duration in seconds
    """
    print("=" * 60)
    print("TEST: Motor Response (I2C)")
    print("=" * 60)
    print(f"I2C Bus:  {bus}")
    print(f"Torque:   {torque_nm:.3f} N·m")
    print(f"Duration: {duration_s}s")
    print("\n⚠️  WARNING: Wheels must be elevated!")
    print("=" * 60)

    # Confirm test
    response = input("\nAre wheels elevated? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Test cancelled")
        return False

    # Connect via I2C
    print("\nConnecting to I2C devices...")
    try:
        i2c = BalboaI2CInterface(bus=bus)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    try:
        print(f"\nSending torque command: {torque_nm:.3f} N·m to both motors")
        print(f"Motors will run for {duration_s}s...")
        print("Press Ctrl+C to stop early\n")

        start_time = time.time()
        sample_count = 0

        while time.time() - start_time < duration_s:
            # Send motor commands
            success = i2c.send_motor_command(torque_nm, torque_nm)
            if not success:
                print("✗ Failed to send motor command!")
                return False

            # Read sensor data to get encoder feedback
            sensor_data = i2c.read_sensors()
            if sensor_data is not None and sample_count % 10 == 0:  # Print every 10th sample
                print(f"[{time.time() - start_time:.1f}s] "
                      f"Encoders: L={sensor_data.encoder_left_rad:.3f} rad, "
                      f"R={sensor_data.encoder_right_rad:.3f} rad")
                sample_count += 1

            time.sleep(0.01)  # 100 Hz

        print("\n✓ Motor test complete")
        print(f"  Sent commands for {duration_s}s")

        # Stop motors
        print("\nStopping motors...")
        i2c.emergency_stop()
        print("✓ Motors stopped")

        return True

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        print("Stopping motors...")
        i2c.emergency_stop()
        return False
    finally:
        i2c.close()


def main():
    parser = argparse.ArgumentParser(description='Test motors via I2C')
    parser.add_argument('--bus', type=int, default=1,
                       help='I2C bus number (default: 1)')
    parser.add_argument('--torque', type=float, default=0.05,
                       help='Torque command in N·m (default: 0.05)')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Test duration in seconds (default: 2.0)')
    args = parser.parse_args()

    success = test_motors(args.bus, args.torque, args.duration)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

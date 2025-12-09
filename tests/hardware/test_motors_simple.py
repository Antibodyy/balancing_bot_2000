#!/usr/bin/env python3
"""Simple Motor Test - Quick verification that motors respond to commands.

This is a minimal test to quickly check if motors are working.
More comprehensive testing is available in test_motor_open_loop.py

⚠️  WARNING: Motors will spin! Elevate wheels or use test stand.

Usage:
    # Default: Small torque for 2 seconds
    python tests/hardware/test_motors_simple.py --port /dev/ttyACM0

    # Custom torque and duration
    python tests/hardware/test_motors_simple.py --torque 0.1 --duration 3
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hardware import BalboaSerialInterface


def test_motors_simple(port: str, torque_nm: float, duration_s: float):
    """Simple motor test - spin both wheels forward.

    Args:
        port: Serial port
        torque_nm: Torque command in N⋅m
        duration_s: How long to spin motors
    """
    print("=" * 70)
    print("SIMPLE MOTOR TEST")
    print("=" * 70)
    print(f"Port:     {port}")
    print(f"Torque:   {torque_nm:.3f} N⋅m")
    print(f"Duration: {duration_s:.1f} seconds")
    print()
    print("⚠️  WARNING: Motors will spin!")
    print("   - Ensure wheels are elevated or on test stand")
    print("   - Keep hands clear of wheels")
    print("-" * 70)

    input("\nPress Enter when ready to start motors...")

    # Connect to Arduino
    print("\nConnecting to Arduino...")
    try:
        serial = BalboaSerialInterface(port=port)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    try:
        print(f"\nSpinning motors at {torque_nm:.3f} N⋅m for {duration_s:.1f}s...")
        print("(Press Ctrl+C to emergency stop)")

        start_time = time.time()

        while time.time() - start_time < duration_s:
            # Send motor commands
            serial.send_motor_command(torque_nm, torque_nm)

            # Read sensors (to keep communication active)
            serial.read_sensors()

            # Small delay
            time.sleep(0.01)  # 100 Hz

        print("\n✓ Test complete!")

        # Stop motors
        print("Stopping motors...")
        serial.emergency_stop()
        time.sleep(0.2)

        print("\n" + "=" * 70)
        print("SUCCESS - Motors responded to commands")
        print("=" * 70)
        print()
        print("Next steps:")
        print("  1. Verify both wheels spun in the same direction")
        print("  2. Run full motor test: python tests/hardware/test_motor_open_loop.py")
        print("  3. Test encoders: python tests/hardware/test_encoders.py")

        return True

    except KeyboardInterrupt:
        print("\n\nEmergency stop!")
        serial.emergency_stop()
        return False
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        serial.emergency_stop()
        return False
    finally:
        serial.close()


def main():
    parser = argparse.ArgumentParser(
        description='Simple motor test - quick verification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                       help='Serial port (default: /dev/ttyACM0)')
    parser.add_argument('--torque', type=float, default=0.05,
                       help='Torque command in N⋅m (default: 0.05)')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Duration in seconds (default: 2.0)')
    args = parser.parse_args()

    success = test_motors_simple(args.port, args.torque, args.duration)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

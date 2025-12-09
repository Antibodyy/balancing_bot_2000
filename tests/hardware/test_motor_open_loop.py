#!/usr/bin/env python3
"""Test 4: Motor Open-Loop Control

Verifies that motors respond correctly to torque commands.
Tests both wheels in both directions.

⚠️  WARNING: Robot wheels will spin! Keep robot elevated or on test stand.

Usage:
    python tests/hardware/test_motor_open_loop.py --port /dev/ttyACM0
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hardware import BalboaSerialInterface


def test_motor_command(serial: BalboaSerialInterface,
                       left_torque: float,
                       right_torque: float,
                       duration_s: float,
                       description: str):
    """Test a single motor command.

    Args:
        serial: Serial interface
        left_torque: Left wheel torque (Nm)
        right_torque: Right wheel torque (Nm)
        duration_s: Command duration
        description: Test description
    """
    print(f"\n{description}")
    print(f"  Command: Left={left_torque:+.3f} Nm, Right={right_torque:+.3f} Nm")
    print(f"  Duration: {duration_s:.1f}s")

    # Record initial encoder positions
    sensor_data = serial.read_sensors()
    while sensor_data is None:
        sensor_data = serial.read_sensors()
        time.sleep(0.001)

    initial_left = sensor_data.encoder_left_rad
    initial_right = -sensor_data.encoder_right_rad

    # Apply command
    start_time = time.time()
    encoder_samples_left = []
    encoder_samples_right = []

    while time.time() - start_time < duration_s:
        # Send motor command
        serial.send_motor_command(left_torque, right_torque)

        # Read sensors
        sensor_data = serial.read_sensors()
        if sensor_data is not None:
            encoder_samples_left.append(sensor_data.encoder_left_rad)
            encoder_samples_right.append(-sensor_data.encoder_right_rad)

        time.sleep(0.005)  # ~200 Hz

    # Stop motors
    serial.emergency_stop()
    time.sleep(0.2)

    # Analyze results
    final_left = encoder_samples_left[-1] if encoder_samples_left else initial_left
    final_right = encoder_samples_right[-1] if encoder_samples_right else initial_right

    delta_left = final_left - initial_left
    delta_right = final_right - initial_right

    print(f"  Left wheel:  {delta_left:+.4f} rad ({delta_left/(2*np.pi):+.3f} rev)")
    print(f"  Right wheel: {delta_right:+.4f} rad ({delta_right/(2*np.pi):+.3f} rev)")

    return delta_left, delta_right


def test_motors(port: str):
    """Test motor open-loop control.

    Args:
        port: Serial port
    """
    print("=" * 60)
    print("TEST 4: Motor Open-Loop Control")
    print("=" * 60)
    print(f"Port: {port}")
    print("\n⚠️  WARNING: Motors will spin!")
    print("   - Ensure robot is on test stand or wheels are elevated")
    print("   - Keep hands clear of wheels")
    print("-" * 60)

    input("\nPress Enter when robot is safely positioned...")

    # Connect to Arduino
    print("\nConnecting to Arduino...")
    try:
        serial = BalboaSerialInterface(port=port)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    try:
        # Test parameters
        test_torque = 0.05  # Small torque for safety (Nm)
        test_duration = 1.0  # seconds

        results = {}

        print("\n" + "=" * 60)
        print("MOTOR TESTS")
        print("=" * 60)

        # Test 1: Left wheel forward
        print("\n[1/4] Left wheel forward")
        delta_l, delta_r = test_motor_command(
            serial, test_torque, 0.0, test_duration,
            "Test 1: Left wheel forward"
        )
        results['left_forward'] = (delta_l, delta_r)
        time.sleep(0.5)

        # Test 2: Right wheel forward
        print("\n[2/4] Right wheel forward")
        delta_l, delta_r = test_motor_command(
            serial, 0.0, test_torque, test_duration,
            "Test 2: Right wheel forward"
        )
        results['right_forward'] = (delta_l, delta_r)
        time.sleep(0.5)

        # Test 3: Both wheels forward
        print("\n[3/4] Both wheels forward")
        delta_l, delta_r = test_motor_command(
            serial, test_torque, test_torque, test_duration,
            "Test 3: Both wheels forward"
        )
        results['both_forward'] = (delta_l, delta_r)
        time.sleep(0.5)

        # Test 4: Differential (turn in place)
        print("\n[4/4] Differential (left forward, right backward)")
        delta_l, delta_r = test_motor_command(
            serial, test_torque, -test_torque, test_duration,
            "Test 4: Differential steering"
        )
        results['differential'] = (delta_l, delta_r)

        # Validation
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)

        checks_passed = 0
        checks_total = 6

        # Check 1: Left wheel moves forward with positive torque
        left_fwd_ok = results['left_forward'][0] > 0.01
        print(f"{'✓' if left_fwd_ok else '✗'} Left wheel moves forward with positive torque "
              f"({'PASS' if left_fwd_ok else 'FAIL'})")
        if left_fwd_ok:
            checks_passed += 1

        # Check 2: Right wheel stays still when not commanded
        left_test_right_still = abs(results['left_forward'][1]) < 0.05
        print(f"{'✓' if left_test_right_still else '✗'} Right wheel stays still during left test "
              f"({'PASS' if left_test_right_still else 'FAIL'})")
        if left_test_right_still:
            checks_passed += 1

        # Check 3: Right wheel moves forward with positive torque
        right_fwd_ok = results['right_forward'][1] > 0.01
        print(f"{'✓' if right_fwd_ok else '✗'} Right wheel moves forward with positive torque "
              f"({'PASS' if right_fwd_ok else 'FAIL'})")
        if right_fwd_ok:
            checks_passed += 1

        # Check 4: Both wheels move forward together
        both_fwd_ok = results['both_forward'][0] > 0.01 and results['both_forward'][1] > 0.01
        print(f"{'✓' if both_fwd_ok else '✗'} Both wheels move forward together "
              f"({'PASS' if both_fwd_ok else 'FAIL'})")
        if both_fwd_ok:
            checks_passed += 1

        # Check 5: Wheels approximately balanced
        ratio = abs(results['both_forward'][0] / (results['both_forward'][1] + 1e-9))
        balanced_ok = 0.5 < ratio < 2.0  # Within factor of 2
        print(f"{'✓' if balanced_ok else '✗'} Wheels approximately balanced (ratio: {ratio:.2f}) "
              f"({'PASS' if balanced_ok else 'FAIL'})")
        if balanced_ok:
            checks_passed += 1

        # Check 6: Differential works (opposite directions)
        diff_ok = results['differential'][0] > 0.01 and results['differential'][1] < -0.01
        print(f"{'✓' if diff_ok else '✗'} Differential steering works "
              f"({'PASS' if diff_ok else 'FAIL'})")
        if diff_ok:
            checks_passed += 1

        print("\n" + "=" * 60)
        success = checks_passed == checks_total
        if success:
            print(f"✓ TEST PASSED ({checks_passed}/{checks_total} checks)")
        else:
            print(f"✗ TEST FAILED ({checks_passed}/{checks_total} checks passed)")
            print("\nPossible issues:")
            print("  - Motor connections reversed (swap motor wires)")
            print("  - Motor driver not working properly")
            print("  - Torque-to-PWM conversion incorrect in firmware")
            print("  - Insufficient torque command (try increasing test_torque)")

        return success

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        serial.emergency_stop()
        return False
    finally:
        serial.close()


def main():
    parser = argparse.ArgumentParser(description='Test motor open-loop control')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                       help='Serial port (default: /dev/ttyACM0)')
    args = parser.parse_args()

    success = test_motors(args.port)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

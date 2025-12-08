#!/usr/bin/env python3
"""Test 3: Encoder Position Tracking

Verifies that encoder positions update correctly when wheels are rotated manually.

Usage:
    python tests/hardware/test_encoders.py --port /dev/ttyACM0
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


def test_encoders(port: str):
    """Test encoder position tracking.

    Args:
        port: Serial port
    """
    print("=" * 60)
    print("TEST 3: Encoder Position Tracking")
    print("=" * 60)
    print(f"Port: {port}")
    print("\n⚠️  INTERACTIVE TEST:")
    print("   1. Keep robot stationary for baseline")
    print("   2. Manually rotate LEFT wheel forward")
    print("   3. Manually rotate RIGHT wheel forward")
    print("-" * 60)

    # Connect to Arduino
    print("\nConnecting to Arduino...")
    try:
        serial = BalboaSerialInterface(port=port)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    try:
        # Phase 1: Baseline (stationary)
        print("\n" + "=" * 60)
        print("PHASE 1: Baseline (keep robot still)")
        print("=" * 60)
        input("Press Enter when ready...")

        baseline_samples = []
        for _ in range(100):  # ~0.5s at 200Hz
            sensor_data = serial.read_sensors()
            if sensor_data is not None:
                baseline_samples.append([
                    sensor_data.encoder_left_rad,
                    sensor_data.encoder_right_rad
                ])
            time.sleep(0.005)

        baseline = np.array(baseline_samples)
        baseline_left = baseline[:, 0].mean()
        baseline_right = baseline[:, 1].mean()
        noise_left = baseline[:, 0].std()
        noise_right = baseline[:, 1].std()

        print(f"Left encoder:  {baseline_left:.4f} rad (noise: {noise_left:.4f} rad)")
        print(f"Right encoder: {baseline_right:.4f} rad (noise: {noise_right:.4f} rad)")

        # Phase 2: Left wheel rotation
        print("\n" + "=" * 60)
        print("PHASE 2: Rotate LEFT wheel FORWARD slowly")
        print("=" * 60)
        input("Press Enter when ready, then rotate wheel...")

        left_samples = []
        for _ in range(200):  # ~1s
            sensor_data = serial.read_sensors()
            if sensor_data is not None:
                left_samples.append([
                    sensor_data.encoder_left_rad,
                    sensor_data.encoder_right_rad
                ])
            time.sleep(0.005)

        left_data = np.array(left_samples)
        left_final = left_data[-10:, 0].mean()  # Average last 10 samples
        right_final = left_data[-10:, 1].mean()

        left_delta = left_final - baseline_left
        right_delta = right_final - baseline_right

        print(f"Left wheel change:  {left_delta:.4f} rad ({left_delta/(2*np.pi):.2f} revolutions)")
        print(f"Right wheel change: {right_delta:.4f} rad ({right_delta/(2*np.pi):.2f} revolutions)")

        # Phase 3: Right wheel rotation
        print("\n" + "=" * 60)
        print("PHASE 3: Rotate RIGHT wheel FORWARD slowly")
        print("=" * 60)
        input("Press Enter when ready, then rotate wheel...")

        # Reset baseline to current position
        baseline_left = left_final
        baseline_right = right_final

        right_samples = []
        for _ in range(200):  # ~1s
            sensor_data = serial.read_sensors()
            if sensor_data is not None:
                right_samples.append([
                    sensor_data.encoder_left_rad,
                    sensor_data.encoder_right_rad
                ])
            time.sleep(0.005)

        right_data = np.array(right_samples)
        left_final2 = right_data[-10:, 0].mean()
        right_final2 = right_data[-10:, 1].mean()

        left_delta2 = left_final2 - baseline_left
        right_delta2 = right_final2 - baseline_right

        print(f"Left wheel change:  {left_delta2:.4f} rad ({left_delta2/(2*np.pi):.2f} revolutions)")
        print(f"Right wheel change: {right_delta2:.4f} rad ({right_delta2/(2*np.pi):.2f} revolutions)")

        # Validation
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)

        checks_passed = 0
        checks_total = 4

        # Check 1: Baseline noise low
        noise_ok = noise_left < 0.01 and noise_right < 0.01
        print(f"{'✓' if noise_ok else '✗'} Baseline noise < 0.01 rad "
              f"({'PASS' if noise_ok else 'FAIL'})")
        if noise_ok:
            checks_passed += 1

        # Check 2: Left wheel moved forward (positive)
        left_moved = left_delta > 0.1  # At least ~0.016 revolutions
        print(f"{'✓' if left_moved else '✗'} Left wheel moved forward "
              f"({'PASS' if left_moved else 'FAIL - expected positive change'})")
        if left_moved:
            checks_passed += 1

        # Check 3: Right wheel didn't move much during left wheel test
        right_still = abs(right_delta) < 0.1
        print(f"{'✓' if right_still else '✗'} Right wheel stayed still during left test "
              f"({'PASS' if right_still else 'FAIL'})")
        if right_still:
            checks_passed += 1

        # Check 4: Right wheel moved forward (positive)
        right_moved = right_delta2 > 0.1
        print(f"{'✓' if right_moved else '✗'} Right wheel moved forward "
              f"({'PASS' if right_moved else 'FAIL - expected positive change'})")
        if right_moved:
            checks_passed += 1

        print("\n" + "=" * 60)
        success = checks_passed == checks_total
        if success:
            print(f"✓ TEST PASSED ({checks_passed}/{checks_total} checks)")
        else:
            print(f"✗ TEST FAILED ({checks_passed}/{checks_total} checks passed)")
            print("\nPossible issues:")
            print("  - Encoders not connected properly")
            print("  - Wheel rotated wrong direction (backward instead of forward)")
            print("  - Encoder counts per revolution incorrect in firmware")

        return success

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return False
    finally:
        serial.close()


def main():
    parser = argparse.ArgumentParser(description='Test encoder position tracking')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                       help='Serial port (default: /dev/ttyACM0)')
    args = parser.parse_args()

    success = test_encoders(args.port)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

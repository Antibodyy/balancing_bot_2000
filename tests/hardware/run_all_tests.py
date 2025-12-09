#!/usr/bin/env python3
"""Run all hardware validation tests in sequence.

This script runs all Phase 1 hardware tests:
1. Serial communication
2. IMU data validation
3. Encoder tracking
4. Motor open-loop control

Usage:
    python tests/hardware/run_all_tests.py --port /dev/ttyACM0
"""

import argparse
import sys
import subprocess


def run_test(script_name: str, args: list) -> bool:
    """Run a test script.

    Args:
        script_name: Test script filename
        args: Additional arguments to pass

    Returns:
        True if test passed, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"Running: {script_name}")
    print("=" * 70)

    cmd = ['python3', f'tests/hardware/{script_name}'] + args
    result = subprocess.run(cmd)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description='Run all hardware validation tests'
    )
    parser.add_argument(
        '--port',
        type=str,
        default='/dev/ttyACM0',
        help='Serial port (default: /dev/ttyACM0)'
    )
    parser.add_argument(
        '--skip-interactive',
        action='store_true',
        help='Skip interactive tests (encoders and motors)'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("HARDWARE VALIDATION TEST SUITE")
    print("=" * 70)
    print(f"Port: {args.port}")
    print()

    results = {}
    port_arg = ['--port', args.port]

    # Test 1: Serial Communication (automated)
    results['Serial Communication'] = run_test(
        'test_serial_communication.py',
        port_arg + ['--duration', '5']
    )

    # Test 2: IMU Data (automated)
    results['IMU Data'] = run_test(
        'test_imu_data.py',
        port_arg + ['--duration', '5']
    )

    if not args.skip_interactive:
        # Test 3: Encoders (interactive)
        results['Encoder Tracking'] = run_test(
            'test_encoders.py',
            port_arg
        )

        # Test 4: Motors (interactive)
        results['Motor Control'] = run_test(
            'test_motor_open_loop.py',
            port_arg
        )
    else:
        print("\n⚠️  Skipping interactive tests (--skip-interactive)")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ ALL TESTS PASSED - Hardware is ready for Phase 2!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED - Review failures above")
        print("\nTroubleshooting:")
        print("  1. Check Arduino firmware is uploaded")
        print("  2. Verify serial port is correct")
        print("  3. Check hardware connections (IMU, encoders, motors)")
        print("  4. Review individual test output for specific issues")
        return 1


if __name__ == '__main__':
    sys.exit(main())

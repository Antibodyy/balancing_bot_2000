#!/usr/bin/env python3
"""Test 2: IMU Data Validation

Verifies that IMU readings are reasonable when robot is stationary:
- Accelerometer should read ~9.8 m/s² in Z-axis (gravity)
- Gyroscope should read near zero (no rotation)

Usage:
    python tests/hardware/test_imu_data.py --port /dev/ttyACM0 --duration 5
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


def test_imu_data(port: str, duration_s: float):
    """Test IMU data quality.

    Args:
        port: Serial port
        duration_s: Test duration in seconds
    """
    print("=" * 60)
    print("TEST 2: IMU Data Validation")
    print("=" * 60)
    print(f"Port:     {port}")
    print(f"Duration: {duration_s}s")
    print("\n⚠️  IMPORTANT: Keep robot STATIONARY during this test!")
    print("-" * 60)

    # Connect to Arduino
    print("\nConnecting to Arduino...")
    try:
        serial = BalboaSerialInterface(port=port)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    # Collect IMU data
    print(f"\nCollecting IMU data for {duration_s}s (keep robot still)...")
    accel_samples = []
    gyro_samples = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration_s:
            sensor_data = serial.read_sensors()
            if sensor_data is not None:
                accel_samples.append(sensor_data.acceleration_mps2)
                gyro_samples.append(sensor_data.angular_velocity_radps)
            time.sleep(0.001)

        # Convert to numpy arrays
        accel_data = np.array(accel_samples)  # Shape: (N, 3)
        gyro_data = np.array(gyro_samples)    # Shape: (N, 3)

        # Compute statistics
        print("\n" + "=" * 60)
        print("ACCELEROMETER RESULTS (m/s²)")
        print("=" * 60)
        print("           X-axis    Y-axis    Z-axis")
        print(f"Mean:      {accel_data[:, 0].mean():6.3f}    {accel_data[:, 1].mean():6.3f}    {accel_data[:, 2].mean():6.3f}")
        print(f"Std Dev:   {accel_data[:, 0].std():6.3f}    {accel_data[:, 1].std():6.3f}    {accel_data[:, 2].std():6.3f}")
        print(f"Min:       {accel_data[:, 0].min():6.3f}    {accel_data[:, 1].min():6.3f}    {accel_data[:, 2].min():6.3f}")
        print(f"Max:       {accel_data[:, 0].max():6.3f}    {accel_data[:, 1].max():6.3f}    {accel_data[:, 2].max():6.3f}")

        # Check magnitude (should be ~9.8 m/s²)
        accel_magnitude = np.linalg.norm(accel_data, axis=1)
        mean_magnitude = accel_magnitude.mean()
        print(f"\nAcceleration magnitude:")
        print(f"  Mean:     {mean_magnitude:.3f} m/s²")
        print(f"  Expected: ~9.81 m/s² (gravity)")
        print(f"  Std Dev:  {accel_magnitude.std():.3f} m/s²")

        print("\n" + "=" * 60)
        print("GYROSCOPE RESULTS (rad/s)")
        print("=" * 60)
        print("           X-axis    Y-axis    Z-axis")
        print(f"Mean:      {gyro_data[:, 0].mean():6.4f}    {gyro_data[:, 1].mean():6.4f}    {gyro_data[:, 2].mean():6.4f}")
        print(f"Std Dev:   {gyro_data[:, 0].std():6.4f}    {gyro_data[:, 1].std():6.4f}    {gyro_data[:, 2].std():6.4f}")
        print(f"Min:       {gyro_data[:, 0].min():6.4f}    {gyro_data[:, 1].min():6.4f}    {gyro_data[:, 2].min():6.4f}")
        print(f"Max:       {gyro_data[:, 0].max():6.4f}    {gyro_data[:, 1].max():6.4f}    {gyro_data[:, 2].max():6.4f}")

        # Check gyro bias (should be near zero when stationary)
        gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
        mean_gyro_mag = gyro_magnitude.mean()
        print(f"\nGyroscope magnitude:")
        print(f"  Mean:     {mean_gyro_mag:.4f} rad/s")
        print(f"  Expected: ~0.0 rad/s (stationary)")
        print(f"  Std Dev:  {gyro_magnitude.std():.4f} rad/s")

        # Validation checks
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)

        checks_passed = 0
        checks_total = 3

        # Check 1: Accelerometer magnitude near gravity
        accel_ok = 9.0 <= mean_magnitude <= 10.5
        print(f"{'✓' if accel_ok else '✗'} Accel magnitude: {mean_magnitude:.2f} m/s² "
              f"({'PASS' if accel_ok else 'FAIL - expected 9.0-10.5'})")
        if accel_ok:
            checks_passed += 1

        # Check 2: Accelerometer noise reasonable
        accel_noise_ok = accel_magnitude.std() < 0.5
        print(f"{'✓' if accel_noise_ok else '✗'} Accel noise: {accel_magnitude.std():.3f} m/s² "
              f"({'PASS' if accel_noise_ok else 'FAIL - expected <0.5'})")
        if accel_noise_ok:
            checks_passed += 1

        # Check 3: Gyro bias small
        gyro_ok = mean_gyro_mag < 0.1
        print(f"{'✓' if gyro_ok else '✗'} Gyro bias: {mean_gyro_mag:.4f} rad/s "
              f"({'PASS' if gyro_ok else 'FAIL - expected <0.1'})")
        if gyro_ok:
            checks_passed += 1

        print("\n" + "=" * 60)
        success = checks_passed == checks_total
        if success:
            print(f"✓ TEST PASSED ({checks_passed}/{checks_total} checks)")
        else:
            print(f"✗ TEST FAILED ({checks_passed}/{checks_total} checks passed)")

        return success

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return False
    finally:
        serial.close()


def main():
    parser = argparse.ArgumentParser(description='Test IMU data quality')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                       help='Serial port (default: /dev/ttyACM0)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Test duration in seconds (default: 5.0)')
    args = parser.parse_args()

    success = test_imu_data(args.port, args.duration)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

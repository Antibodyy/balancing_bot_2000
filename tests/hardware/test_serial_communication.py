#!/usr/bin/env python3
"""Test 1: Serial Communication Validation

Verifies that sensor packets arrive from Arduino at expected rate (200 Hz).

Usage:
    python tests/hardware/test_serial_communication.py --port /dev/ttyACM0 --duration 5
"""

import argparse
import sys
import time
from pathlib import Path
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hardware import BalboaSerialInterface


def test_serial_communication(port: str, baudrate: int, duration_s: float):
    """Test serial communication with Arduino.

    Args:
        port: Serial port
        baudrate: Baud rate
        duration_s: Test duration in seconds
    """
    print("=" * 60)
    print("TEST 1: Serial Communication")
    print("=" * 60)
    print(f"Port:     {port}")
    print(f"Baudrate: {baudrate}")
    print(f"Duration: {duration_s}s")
    print("-" * 60)

    # Connect to Arduino
    print("Connecting to Arduino...")
    try:
        serial = BalboaSerialInterface(port=port, baudrate=baudrate)
        print("✓ Connected")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return False

    # Collect packets
    print(f"\nCollecting packets for {duration_s}s...")
    packets_received = 0
    packets_failed_checksum = 0
    start_time = time.time()
    timestamps = []

    try:
        while time.time() - start_time < duration_s:
            sensor_data = serial.read_sensors()
            if sensor_data is not None:
                packets_received += 1
                timestamps.append(sensor_data.timestamp_s)
            time.sleep(0.001)  # Small sleep to prevent busy-wait

        elapsed = time.time() - start_time

        # Analyze results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Packets received:     {packets_received}")
        print(f"Duration:             {elapsed:.2f}s")
        print(f"Average rate:         {packets_received / elapsed:.1f} Hz")
        print(f"Expected rate:        200 Hz")

        if len(timestamps) > 1:
            # Compute inter-packet intervals
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            avg_rate = 1.0 / avg_interval if avg_interval > 0 else 0

            print(f"\nArduino timestamp analysis:")
            print(f"  Average interval:   {avg_interval*1000:.2f} ms")
            print(f"  Measured rate:      {avg_rate:.1f} Hz")
            print(f"  Min interval:       {min(intervals)*1000:.2f} ms")
            print(f"  Max interval:       {max(intervals)*1000:.2f} ms")

        # Check if rate is acceptable
        measured_rate = packets_received / elapsed
        target_rate = 200.0
        success = 180.0 <= measured_rate <= 220.0  # ±10% tolerance

        print("\n" + "=" * 60)
        if success:
            print("✓ TEST PASSED - Packet rate is acceptable")
        else:
            print("✗ TEST FAILED - Packet rate is outside acceptable range")
            print(f"  Expected: {target_rate:.0f} ± 10% Hz")
            print(f"  Measured: {measured_rate:.1f} Hz")

        return success

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return False
    finally:
        serial.close()


def main():
    parser = argparse.ArgumentParser(description='Test serial communication with Arduino')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                       help='Serial port (default: /dev/ttyACM0)')
    parser.add_argument('--baudrate', type=int, default=115200,
                       help='Baud rate (default: 115200)')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Test duration in seconds (default: 5.0)')
    args = parser.parse_args()

    success = test_serial_communication(args.port, args.baudrate, args.duration)
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

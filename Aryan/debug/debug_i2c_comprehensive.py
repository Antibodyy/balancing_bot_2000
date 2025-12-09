#!/usr/bin/env python3
"""Comprehensive I2C diagnostic - track when and why it fails."""

import struct
import time
from smbus2 import SMBus, i2c_msg

BALBOA_ADDRESS = 0x20
BUFFER_SIZE = 16

def main():
    print("=" * 60)
    print("Comprehensive I2C Diagnostic")
    print("=" * 60)

    bus = SMBus(1)

    success_count = 0
    failure_count = 0
    last_successful_data = None

    print("\nReading continuously. Press Ctrl+C to stop.\n")
    print("Sample | Status | Timestamp | Accel Z | Gyro Magnitude | Error")
    print("-" * 80)

    for i in range(200):  # Read 200 samples
        try:
            # Direct I2C read
            msg = i2c_msg.read(BALBOA_ADDRESS, BUFFER_SIZE)
            bus.i2c_rdwr(msg)
            data = list(msg)

            # Parse data
            timestamp_us = struct.unpack('<I', bytes(data[0:4]))[0]
            accel_raw = struct.unpack('<hhh', bytes(data[4:10]))
            gyro_raw = struct.unpack('<hhh', bytes(data[10:16]))

            # Check if data is valid
            all_zeros = all(b == 0 for b in data)
            all_0xff = all(b == 0xFF for b in data)

            if all_zeros:
                status = "ZERO"
                failure_count += 1
                error_msg = "All zeros"
            elif all_0xff:
                status = "0xFF"
                failure_count += 1
                error_msg = "All 0xFF (no response)"
            elif timestamp_us == 0:
                status = "BAD"
                failure_count += 1
                error_msg = "Zero timestamp"
            else:
                status = "OK"
                success_count += 1
                last_successful_data = data
                error_msg = "-"

            gyro_mag = (gyro_raw[0]**2 + gyro_raw[1]**2 + gyro_raw[2]**2)**0.5

            print(f"{i+1:4d}   | {status:4s}   | {timestamp_us:10d} | {accel_raw[2]:6d} | "
                  f"{gyro_mag:6.0f}         | {error_msg}")

            time.sleep(0.05)  # 20 Hz

        except IOError as e:
            print(f"{i+1:4d}   | ERR    | I2C Error: {e}")
            failure_count += 1
            time.sleep(0.05)

    print("\n" + "=" * 60)
    print(f"Results: {success_count} successful, {failure_count} failed")
    print(f"Success rate: {100*success_count/(success_count+failure_count):.1f}%")

    if last_successful_data:
        print(f"\nLast successful read:")
        hex_str = ' '.join([f'{b:02X}' for b in last_successful_data])
        print(f"  {hex_str}")

    bus.close()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user")

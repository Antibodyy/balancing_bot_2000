#!/usr/bin/env python3
"""Simple I2C read test without register offset."""

import struct
import time

try:
    from smbus2 import SMBus, i2c_msg
except ImportError:
    print("ERROR: smbus2 not installed. Run: pip install smbus2")
    exit(1)

BALBOA_ADDRESS = 0x20
BUFFER_SIZE = 16

def main():
    print("=" * 60)
    print("Simple I2C Read Test (No Register Offset)")
    print("=" * 60)

    try:
        bus = SMBus(1)
        print("✓ I2C bus opened")

        print("\nTrying different read methods...")
        print("-" * 60)

        # Method 1: read_i2c_block_data (what we're currently using)
        print("\nMethod 1: read_i2c_block_data(addr, 0, 16)")
        try:
            data = bus.read_i2c_block_data(BALBOA_ADDRESS, 0, BUFFER_SIZE)
            hex_str = ' '.join([f'{b:02X}' for b in data])
            print(f"  Result: {hex_str}")
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(0.1)

        # Method 2: Direct I2C read (no register write)
        print("\nMethod 2: i2c_msg.read (direct read, no offset)")
        try:
            msg = i2c_msg.read(BALBOA_ADDRESS, BUFFER_SIZE)
            bus.i2c_rdwr(msg)
            data = list(msg)
            hex_str = ' '.join([f'{b:02X}' for b in data])
            print(f"  Result: {hex_str}")

            if not all(b == 0xFF for b in data[1:]):
                print("  ✓ SUCCESS! Got non-0xFF values")
                # Decode it
                timestamp_us = struct.unpack('<I', bytes(data[0:4]))[0]
                accel_raw = struct.unpack('<hhh', bytes(data[4:10]))
                gyro_raw = struct.unpack('<hhh', bytes(data[10:16]))
                print(f"  Timestamp: {timestamp_us} us")
                print(f"  Accel: {accel_raw}")
                print(f"  Gyro: {gyro_raw}")
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(0.1)

        # Method 3: Write offset then read (separate transactions)
        print("\nMethod 3: write_byte(0) then read_i2c_block_data")
        try:
            # First, write the offset
            bus.write_byte(BALBOA_ADDRESS, 0)
            time.sleep(0.001)  # Small delay
            # Then read
            # Note: This might not work as expected
            data = bus.read_i2c_block_data(BALBOA_ADDRESS, 0, BUFFER_SIZE)
            hex_str = ' '.join([f'{b:02X}' for b in data])
            print(f"  Result: {hex_str}")
        except Exception as e:
            print(f"  Error: {e}")

        bus.close()
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Debug script to read raw I2C data from Balboa."""

import struct
import time

try:
    from smbus2 import SMBus
except ImportError:
    print("ERROR: smbus2 not installed. Run: pip install smbus2")
    exit(1)

BALBOA_ADDRESS = 0x20
BUFFER_SIZE = 16

def main():
    print("=" * 60)
    print("I2C Raw Read Debug")
    print("=" * 60)
    print(f"Balboa Address: 0x{BALBOA_ADDRESS:02X}")
    print(f"Buffer Size: {BUFFER_SIZE} bytes")
    print("-" * 60)

    try:
        bus = SMBus(1)
        print("✓ I2C bus opened")

        # Read 10 samples
        print("\nReading 10 samples from Balboa...")
        print("-" * 60)

        for i in range(10):
            try:
                # Read 16 bytes starting at offset 0
                data = bus.read_i2c_block_data(BALBOA_ADDRESS, 0, BUFFER_SIZE)

                # Print raw bytes
                hex_str = ' '.join([f'{b:02X}' for b in data])
                print(f"\nSample {i+1}:")
                print(f"  Raw bytes: {hex_str}")

                # Check if all zeros
                if all(b == 0 for b in data):
                    print(f"  ⚠️  WARNING: All bytes are zero!")
                else:
                    # Unpack and display
                    timestamp_us = struct.unpack('<I', bytes(data[0:4]))[0]
                    accel_raw = struct.unpack('<hhh', bytes(data[4:10]))
                    gyro_raw = struct.unpack('<hhh', bytes(data[10:16]))

                    print(f"  Timestamp: {timestamp_us} us")
                    print(f"  Accel raw: X={accel_raw[0]:6d} Y={accel_raw[1]:6d} Z={accel_raw[2]:6d}")
                    print(f"  Gyro raw:  X={gyro_raw[0]:6d} Y={gyro_raw[1]:6d} Z={gyro_raw[2]:6d}")

                    # Check for expected accel Z value (should be ~8000 for 1g at ±4g scale)
                    if abs(accel_raw[2]) > 6000:
                        print(f"  ✓ Accel Z looks reasonable (~1g)")
                    else:
                        print(f"  ⚠️  Accel Z seems low (expected ~8000)")

                time.sleep(0.1)

            except IOError as e:
                print(f"  ✗ I2C read error: {e}")

        bus.close()
        print("\n" + "=" * 60)
        print("Debug complete")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

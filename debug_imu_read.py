#!/usr/bin/env python3
"""Debug IMU reading to diagnose zero values issue."""

import time
from smbus2 import SMBus

GYRO_ADDRESS = 0x6B
LSM6_OUTX_L_XL = 0x28

bus = SMBus(1)

print("Reading IMU data directly...")
print("=" * 60)

# Try reading individual bytes
print("\nReading individual accelerometer bytes:")
for addr in range(0x28, 0x2E):
    try:
        value = bus.read_byte_data(GYRO_ADDRESS, addr)
        print(f"  Register 0x{addr:02X}: 0x{value:02X} ({value})")
    except Exception as e:
        print(f"  Register 0x{addr:02X}: Error - {e}")

# Try reading block with 0x80 bit (auto-increment)
print("\nReading block with auto-increment (|0x80):")
try:
    data = bus.read_i2c_block_data(GYRO_ADDRESS, LSM6_OUTX_L_XL | 0x80, 6)
    print(f"  Data: {[f'0x{b:02X}' for b in data]}")
    print(f"  Decimal: {data}")
except Exception as e:
    print(f"  Error: {e}")

# Try reading block without 0x80 bit
print("\nReading block without auto-increment:")
try:
    data = bus.read_i2c_block_data(GYRO_ADDRESS, LSM6_OUTX_L_XL, 6)
    print(f"  Data: {[f'0x{b:02X}' for b in data]}")
    print(f"  Decimal: {data}")

    # Decode as accelerometer values
    ax = int.from_bytes([data[0], data[1]], byteorder='little', signed=True)
    ay = int.from_bytes([data[2], data[3]], byteorder='little', signed=True)
    az = int.from_bytes([data[4], data[5]], byteorder='little', signed=True)
    print(f"  Decoded: ax={ax}, ay={ay}, az={az}")

    # Convert to m/s²
    accel_scale = 0.122e-3 * 9.81
    print(f"  In m/s²: ax={ax*accel_scale:.3f}, ay={ay*accel_scale:.3f}, az={az*accel_scale:.3f}")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "=" * 60)
bus.close()

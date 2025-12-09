#!/usr/bin/env python3
"""Debug Balboa I2C reading."""

import struct
import time
from smbus2 import SMBus

BALBOA_ADDRESS = 0x20
BALBOA_DATA_SIZE = 12

bus = SMBus(1)

print("Testing Balboa I2C reads...")
print("=" * 60)

# Try different read methods
print("\n1. Standard block read (register 0, 12 bytes):")
try:
    data = bus.read_i2c_block_data(BALBOA_ADDRESS, 0, BALBOA_DATA_SIZE)
    print(f"   Raw data: {[f'0x{b:02X}' for b in data]}")
    print(f"   Decimal:  {data}")

    # Try to unpack
    values = struct.unpack('<Iff', bytes(data))
    print(f"   Unpacked: timestamp={values[0]}, left={values[1]:.4f}, right={values[2]:.4f}")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Direct I2C read (no register, 12 bytes):")
try:
    # Some I2C slaves don't use register addressing
    data = bus.read_i2c_block_data(BALBOA_ADDRESS, 0x00, BALBOA_DATA_SIZE)
    print(f"   Raw data: {[f'0x{b:02X}' for b in data]}")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Read individual bytes:")
try:
    for i in range(BALBOA_DATA_SIZE):
        byte_val = bus.read_byte_data(BALBOA_ADDRESS, i)
        print(f"   Byte {i}: 0x{byte_val:02X} ({byte_val})")
except Exception as e:
    print(f"   Error: {e}")

print("\n4. Check if PololuRPiSlave uses special protocol:")
print("   PololuRPiSlave may require reading from specific offsets")
print("   Let's try reading 20 bytes (full buffer including motor commands):")
try:
    data = bus.read_i2c_block_data(BALBOA_ADDRESS, 0, 20)
    print(f"   Raw data (20 bytes): {[f'0x{b:02X}' for b in data]}")

    # Unpack sensor data (first 12 bytes)
    sensor_values = struct.unpack('<Iff', bytes(data[:12]))
    print(f"   Sensor data: timestamp={sensor_values[0]}, left={sensor_values[1]:.4f}, right={sensor_values[2]:.4f}")

    # Unpack motor commands (next 8 bytes)
    motor_values = struct.unpack('<ff', bytes(data[12:20]))
    print(f"   Motor commands: left={motor_values[0]:.4f}, right={motor_values[1]:.4f}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
bus.close()

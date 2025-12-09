#!/usr/bin/env python3
"""Test PololuRPiSlave protocol directly."""

import struct
import time
from smbus2 import SMBus

BALBOA_ADDRESS = 0x20

bus = SMBus(1)

print("Testing PololuRPiSlave read protocol...")
print("=" * 60)

# Method 1: Pololu protocol (write offset, wait, read individual bytes)
print("\n1. Pololu protocol (write offset, wait, read bytes individually):")
try:
    bus.write_byte(BALBOA_ADDRESS, 0)  # Write offset 0
    time.sleep(0.0001)  # 100µs delay

    byte_list = []
    for i in range(12):
        byte_val = bus.read_byte(BALBOA_ADDRESS)
        byte_list.append(byte_val)

    print(f"   Raw bytes: {[f'0x{b:02X}' for b in byte_list]}")

    # Unpack
    values = struct.unpack('<Iff', bytes(byte_list))
    print(f"   Unpacked: timestamp={values[0]}, left={values[1]:.4f}, right={values[2]:.4f}")
except Exception as e:
    print(f"   Error: {e}")

# Method 2: Standard block read (for comparison)
print("\n2. Standard block read (for comparison):")
try:
    data = bus.read_i2c_block_data(BALBOA_ADDRESS, 0, 12)
    print(f"   Raw bytes: {[f'0x{b:02X}' for b in data]}")

    values = struct.unpack('<Iff', bytes(data))
    print(f"   Unpacked: timestamp={values[0]}, left={values[1]:.4f}, right={values[2]:.4f}")
except Exception as e:
    print(f"   Error: {e}")

# Method 3: Try reading from different offsets
print("\n3. Try reading timestamp only (first 4 bytes):")
try:
    bus.write_byte(BALBOA_ADDRESS, 0)
    time.sleep(0.0001)

    byte_list = [bus.read_byte(BALBOA_ADDRESS) for _ in range(4)]
    timestamp = struct.unpack('<I', bytes(byte_list))[0]
    print(f"   Timestamp: {timestamp} us")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
bus.close()

#!/usr/bin/env python3
"""Debug BalboaI2CInterface to see raw values."""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hardware import BalboaI2CInterface

print("Creating BalboaI2CInterface...")
i2c = BalboaI2CInterface(bus=1)

print("\nReading sensors 5 times:")
for i in range(5):
    print(f"\n--- Read {i+1} ---")

    # Read accelerometer
    accel = i2c._read_accel_raw()
    print(f"Accel raw: {accel}")

    # Read gyroscope
    gyro_raw = i2c._read_gyro_raw()
    print(f"Gyro raw (before offset): {gyro_raw}")
    if gyro_raw is not None:
        gyro = gyro_raw - i2c._gyro_offset
        print(f"Gyro (after offset): {gyro}")

    # Read encoders
    encoder_data = i2c._read_encoders()
    print(f"Encoders: {encoder_data}")

    # Read full sensor data
    sensor_data = i2c.read_sensors()
    if sensor_data:
        print(f"Full SensorData:")
        print(f"  accel: {sensor_data.acceleration_mps2}")
        print(f"  gyro: {sensor_data.angular_velocity_radps}")
        print(f"  encoders: L={sensor_data.encoder_left_rad:.4f}, R={sensor_data.encoder_right_rad:.4f}")
        print(f"  timestamp: {sensor_data.timestamp_s:.3f}")
    else:
        print("Full SensorData: None")

i2c.close()
print("\nDone")

"""I2C communication interface for Pololu Balboa 32U4.

This module provides low-level I2C communication with:
- Balboa 32U4 (0x20): Encoders and motor commands
- IMU sensors (0x1e, 0x6b): Accelerometer and gyroscope

The Balboa acts as an I2C slave, while the Raspberry Pi reads
all sensors as I2C master.
"""

import struct
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    from smbus2 import SMBus
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False
    SMBus = None

from control_pipeline import SensorData


# I2C Device Addresses
BALBOA_ADDRESS = 0x20        # Balboa 32U4 (encoders + motors)
ACCEL_ADDRESS = 0x1E         # Accelerometer
GYRO_ADDRESS = 0x6B          # Gyroscope (LSM6DS33)
DEFAULT_I2C_BUS = 1          # Raspberry Pi I2C bus 1

# LSM6DS33 Register Addresses
LSM6_CTRL1_XL = 0x10    # Accelerometer control
LSM6_CTRL2_G = 0x11     # Gyroscope control
LSM6_CTRL3_C = 0x12     # Common control
LSM6_OUTX_L_G = 0x22    # Gyro data start
LSM6_OUTX_L_XL = 0x28   # Accel data start


class BalboaI2CInterface:
    """I2C communication interface for Balboa robot with separate IMU reading.

    Handles I2C-based communication with:
    - Balboa 32U4 (0x20): Provides encoder data, receives motor commands
    - IMU sensors (0x1e, 0x6b): Accelerometer and gyroscope

    Example:
        >>> i2c = BalboaI2CInterface(bus=1)
        >>> sensor_data = i2c.read_sensors()
        >>> i2c.send_motor_command(0.1, 0.1)
        >>> i2c.close()
    """

    # Balboa data structure offsets
    BALBOA_DATA_SIZE = 12       # timestamp(4) + 2×encoder(8)
    OFFSET_TIMESTAMP = 0        # uint32 (4 bytes)
    OFFSET_ENCODER_L = 4        # float32 (4 bytes)
    OFFSET_ENCODER_R = 8        # float32 (4 bytes)

    # Motor command offsets
    MOTOR_DATA_SIZE = 8
    OFFSET_TORQUE_L = 12        # float32 (4 bytes) - after sensor data
    OFFSET_TORQUE_R = 16        # float32 (4 bytes)

    def __init__(self, bus: int = DEFAULT_I2C_BUS, address: int = BALBOA_ADDRESS):
        """Initialize I2C connection.

        Args:
            bus: I2C bus number (default 1 for Raspberry Pi)
            address: Balboa I2C slave address (default 0x20)

        Raises:
            ImportError: If smbus2 is not installed
            IOError: If I2C bus cannot be opened
        """
        if not SMBUS_AVAILABLE:
            raise ImportError(
                "smbus2 not installed. Install with: pip install smbus2"
            )

        self._bus_num = bus
        self._balboa_addr = address
        self._bus = SMBus(bus)

        # Initialize IMU
        self._init_imu()

        # Gyro calibration offsets
        self._gyro_offset = np.zeros(3)
        self._calibrate_gyro()

        time.sleep(0.1)

        print(f"Balboa I2C Interface initialized")
        print(f"  Bus: {bus}")
        print(f"  Balboa: 0x{address:02X}")
        print(f"  Accel:  0x{ACCEL_ADDRESS:02X}")
        print(f"  Gyro:   0x{GYRO_ADDRESS:02X}")

    def _init_imu(self):
        """Initialize IMU sensors via I2C."""
        try:
            # Soft reset the IMU to ensure clean state
            self._bus.write_byte_data(GYRO_ADDRESS, LSM6_CTRL3_C, 0b00000001)  # SW_RESET bit
            time.sleep(0.05)  # Wait for reset to complete

            # Configure accelerometer: ODR=208Hz, FS=±4g
            self._bus.write_byte_data(GYRO_ADDRESS, LSM6_CTRL1_XL, 0b01011000)
            time.sleep(0.01)

            # Configure gyroscope: ODR=208Hz, FS=±500dps
            self._bus.write_byte_data(GYRO_ADDRESS, LSM6_CTRL2_G, 0b01010100)
            time.sleep(0.01)

            # Enable block data update and auto-increment
            self._bus.write_byte_data(GYRO_ADDRESS, LSM6_CTRL3_C, 0b01000100)

            # Wait for IMU to stabilize and start producing data (at least one sample period)
            # At 208Hz ODR, one sample takes ~5ms, wait 100ms to be safe
            time.sleep(0.1)

            print("IMU initialized successfully")
        except IOError as e:
            print(f"Warning: IMU initialization failed: {e}")
            print("Continuing without IMU...")

    def _calibrate_gyro(self, num_samples: int = 100):
        """Calibrate gyroscope zero offsets.

        Args:
            num_samples: Number of samples to average (default 100)
        """
        print("Calibrating gyroscope (keep robot stationary)...")
        samples = []

        for _ in range(num_samples):
            try:
                gyro_data = self._read_gyro_raw()
                if gyro_data is not None:
                    samples.append(gyro_data)
                time.sleep(0.01)
            except IOError:
                continue

        if samples:
            self._gyro_offset = np.mean(samples, axis=0)
            print(f"Gyro calibration complete. Offsets: {self._gyro_offset}")
        else:
            print("Warning: Gyro calibration failed, using zero offsets")
            self._gyro_offset = np.zeros(3)

    def _read_accel_raw(self) -> Optional[np.ndarray]:
        """Read raw accelerometer data from IMU.

        Returns:
            3D numpy array [ax, ay, az] in m/s², or None on error
        """
        try:
            # Read 6 bytes starting at OUTX_L_XL
            data = self._bus.read_i2c_block_data(GYRO_ADDRESS, LSM6_OUTX_L_XL | 0x80, 6)

            # Convert to signed 16-bit values
            ax = np.int16((data[1] << 8) | data[0])
            ay = np.int16((data[3] << 8) | data[2])
            az = np.int16((data[5] << 8) | data[4])

            # Convert to m/s² (±4g mode: 1 LSB = 0.122 mg)
            accel_scale = 0.122e-3 * 9.81
            return np.array([ax, ay, az]) * accel_scale

        except IOError as e:
            print(f"Accel read error: {e}")
            return None

    def _read_gyro_raw(self) -> Optional[np.ndarray]:
        """Read raw gyroscope data from IMU.

        Returns:
            3D numpy array [gx, gy, gz] in rad/s (before offset subtraction), or None on error
        """
        try:
            # Read 6 bytes starting at OUTX_L_G
            data = self._bus.read_i2c_block_data(GYRO_ADDRESS, LSM6_OUTX_L_G | 0x80, 6)

            # Convert to signed 16-bit values
            gx = np.int16((data[1] << 8) | data[0])
            gy = np.int16((data[3] << 8) | data[2])
            gz = np.int16((data[5] << 8) | data[4])

            # Convert to rad/s (±500 dps mode: 1 LSB = 17.50 mdps)
            gyro_scale = 17.50e-3 * (np.pi / 180.0)
            return np.array([gx, gy, gz]) * gyro_scale

        except IOError as e:
            print(f"Gyro read error: {e}")
            return None

    def _read_encoders(self) -> Optional[tuple]:
        """Read encoder data from Balboa.

        Returns:
            Tuple of (timestamp_us, encoder_left_rad, encoder_right_rad), or None on error
        """
        try:
            # Read 12 bytes from Balboa (timestamp + 2 encoders)
            data = self._bus.read_i2c_block_data(
                self._balboa_addr, 0, self.BALBOA_DATA_SIZE
            )

            # Unpack: uint32 timestamp, 2×float32 encoders
            values = struct.unpack('<Iff', bytes(data))
            return values[0], values[1], values[2]

        except IOError as e:
            print(f"Balboa read error: {e}")
            return None

    def read_sensors(self) -> Optional[SensorData]:
        """Read all sensor data (IMU + encoders).

        Returns:
            SensorData object compatible with BalanceController, or None if error
        """
        try:
            # Read accelerometer
            accel = self._read_accel_raw()
            if accel is None:
                return None

            # Read gyroscope
            gyro_raw = self._read_gyro_raw()
            if gyro_raw is None:
                return None
            gyro = gyro_raw - self._gyro_offset

            # Read encoders from Balboa
            encoder_data = self._read_encoders()
            if encoder_data is None:
                return None

            timestamp_us, encoder_left_rad, encoder_right_rad = encoder_data
            timestamp_s = timestamp_us / 1e6

            # Pack into SensorData format
            return SensorData(
                acceleration_mps2=accel,
                angular_velocity_radps=gyro,
                encoder_left_rad=encoder_left_rad,
                encoder_right_rad=encoder_right_rad,
                timestamp_s=timestamp_s
            )

        except Exception as e:
            print(f"Sensor read error: {e}")
            return None

    def send_motor_command(
        self, torque_left_nm: float, torque_right_nm: float
    ) -> bool:
        """Send torque commands to motors via I2C.

        Args:
            torque_left_nm: Left wheel torque (N⋅m), positive = forward
            torque_right_nm: Right wheel torque (N⋅m), positive = forward

        Returns:
            True if command sent successfully, False otherwise
        """
        try:
            # Pack motor commands as two float32 values
            data = struct.pack('<ff', torque_left_nm, torque_right_nm)

            # Write to motor command offset (byte 12-19)
            self._bus.write_i2c_block_data(
                self._balboa_addr,
                self.OFFSET_TORQUE_L,
                list(data)
            )

            return True

        except IOError as e:
            print(f"Motor command error: {e}")
            return False

    def emergency_stop(self) -> bool:
        """Send zero torque command to stop motors immediately."""
        return self.send_motor_command(0.0, 0.0)

    def close(self):
        """Close I2C connection and stop motors."""
        self.emergency_stop()
        time.sleep(0.1)
        self._bus.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_bus'):
            try:
                self.close()
            except:
                pass

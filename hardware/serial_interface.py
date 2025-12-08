"""Serial communication interface for Pololu Balboa 32U4.

This module provides low-level serial communication with the Arduino,
handling binary packet encoding/decoding for efficient data transfer.
"""

import struct
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import serial
    PYSERIAL_AVAILABLE = True
except ImportError:
    PYSERIAL_AVAILABLE = False
    serial = None

from control_pipeline import SensorData


# Protocol constants
CONTROL_PACKET_HEADER = 0xAA
SENSOR_PACKET_HEADER = 0xBB
CONTROL_PACKET_SIZE = 10  # header(1) + 2*float32(8) + checksum(1) = 10
SENSOR_PACKET_SIZE = 38   # header(1) + timestamp(4) + 8*float32(32) + checksum(1) = 38


@dataclass
class RawSensorPacket:
    """Raw sensor data packet from Arduino.

    Attributes:
        timestamp_us: Micros since Arduino boot
        accel_x_mps2: X acceleration (m/s²)
        accel_y_mps2: Y acceleration (m/s²)
        accel_z_mps2: Z acceleration (m/s²)
        gyro_x_radps: X angular velocity (rad/s)
        gyro_y_radps: Y angular velocity (rad/s)
        gyro_z_radps: Z angular velocity (rad/s)
        encoder_left_rad: Left wheel position (radians)
        encoder_right_rad: Right wheel position (radians)
    """
    timestamp_us: int
    accel_x_mps2: float
    accel_y_mps2: float
    accel_z_mps2: float
    gyro_x_radps: float
    gyro_y_radps: float
    gyro_z_radps: float
    encoder_left_rad: float
    encoder_right_rad: float


class BalboaSerialInterface:
    """Serial communication interface to Balboa 32U4.

    Handles binary packet-based communication:
    - Sends torque commands to Arduino
    - Receives sensor data from Arduino
    - Converts to SensorData format for controller

    Example:
        >>> serial = BalboaSerialInterface(port='/dev/ttyACM0')
        >>> sensor_data = serial.read_sensors()
        >>> serial.send_motor_command(0.1, 0.1)  # Send 0.1 Nm to each wheel
        >>> serial.close()
    """

    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 115200,
                 timeout: float = 0.1):
        """Initialize serial connection to Arduino.

        Args:
            port: Serial port device (e.g., /dev/ttyACM0 on Linux, COM3 on Windows)
            baudrate: Communication baud rate (default 115200)
            timeout: Read timeout in seconds

        Raises:
            ImportError: If pyserial is not installed
            serial.SerialException: If port cannot be opened
        """
        if not PYSERIAL_AVAILABLE:
            raise ImportError("pyserial not installed. Install with: pip install pyserial")

        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._serial = serial.Serial(port, baudrate, timeout=timeout)

        # Give Arduino time to reset after serial connection
        time.sleep(2.0)

        # Flush any startup garbage
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

    def read_sensors(self) -> Optional[SensorData]:
        """Read and parse sensor packet from Arduino.

        Returns:
            SensorData object compatible with BalanceController, or None if:
            - No packet available
            - Checksum fails
            - Parse error

        Note:
            This is a non-blocking read. Returns None immediately if no data available.
        """
        # Try to read a complete packet
        if self._serial.in_waiting < SENSOR_PACKET_SIZE:
            return None

        # Search for packet header
        while self._serial.in_waiting > 0:
            byte = self._serial.read(1)
            if len(byte) == 0:
                return None
            if byte[0] == SENSOR_PACKET_HEADER:
                break
        else:
            return None

        # Read rest of packet (header already read)
        packet_data = byte + self._serial.read(SENSOR_PACKET_SIZE - 1)

        if len(packet_data) != SENSOR_PACKET_SIZE:
            return None  # Incomplete packet

        # Verify checksum
        checksum_received = packet_data[-1]
        checksum_computed = self._compute_checksum(packet_data[:-1])
        if checksum_received != checksum_computed:
            print(f"Checksum mismatch: expected {checksum_computed}, got {checksum_received}")
            return None

        # Parse packet
        try:
            raw_packet = self._parse_sensor_packet(packet_data)
        except struct.error as e:
            print(f"Packet parse error: {e}")
            return None

        # Convert to SensorData format
        return self._raw_to_sensor_data(raw_packet)

    def _parse_sensor_packet(self, packet_data: bytes) -> RawSensorPacket:
        """Parse binary sensor packet.

        Packet format (45 bytes):
        - uint8: header (0xBB)
        - uint32: timestamp (microseconds)
        - float32[3]: accelerometer XYZ (m/s²)
        - float32[3]: gyroscope XYZ (rad/s)
        - float32[2]: encoder L/R (radians)
        - uint8: checksum
        """
        # Unpack using little-endian format
        # B = unsigned char (1 byte)
        # I = unsigned int (4 bytes)
        # f = float (4 bytes)
        values = struct.unpack('<BIffffffffB', packet_data)

        return RawSensorPacket(
            timestamp_us=values[1],
            accel_x_mps2=values[2],
            accel_y_mps2=values[3],
            accel_z_mps2=values[4],
            gyro_x_radps=values[5],
            gyro_y_radps=values[6],
            gyro_z_radps=values[7],
            encoder_left_rad=values[8],
            encoder_right_rad=values[9]
        )

    def _raw_to_sensor_data(self, raw: RawSensorPacket) -> SensorData:
        """Convert raw Arduino packet to SensorData format.

        Args:
            raw: Raw sensor packet from Arduino

        Returns:
            SensorData object compatible with BalanceController

        Note:
            The controller computes encoder velocities internally from positions,
            so we only need to provide the raw position measurements here.
        """
        # Convert timestamp to seconds
        timestamp_s = raw.timestamp_us / 1e6

        # Pack into SensorData format (controller will compute velocities)
        return SensorData(
            acceleration_mps2=np.array([raw.accel_x_mps2, raw.accel_y_mps2, raw.accel_z_mps2]),
            angular_velocity_radps=np.array([raw.gyro_x_radps, raw.gyro_y_radps, raw.gyro_z_radps]),
            encoder_left_rad=raw.encoder_left_rad,
            encoder_right_rad=raw.encoder_right_rad,
            timestamp_s=timestamp_s
        )

    def send_motor_command(self, torque_left_nm: float, torque_right_nm: float) -> bool:
        """Send torque commands to motors.

        Args:
            torque_left_nm: Left wheel torque (N⋅m), positive = forward
            torque_right_nm: Right wheel torque (N⋅m), positive = forward

        Returns:
            True if command sent successfully, False otherwise

        Note:
            The Arduino will convert torques to PWM values based on
            motor characterization. Torques are typically in range [-0.25, +0.25] Nm.
        """
        # Build control packet
        # Format: header (1) + left torque (4) + right torque (4) + checksum (1) = 10 bytes
        packet = struct.pack('<Bff', CONTROL_PACKET_HEADER, torque_left_nm, torque_right_nm)
        checksum = self._compute_checksum(packet)
        packet += struct.pack('<B', checksum)

        try:
            bytes_written = self._serial.write(packet)
            return bytes_written == len(packet)
        except serial.SerialException as e:
            print(f"Serial write error: {e}")
            return False

    def _compute_checksum(self, data: bytes) -> int:
        """Compute XOR checksum of data.

        Args:
            data: Bytes to checksum

        Returns:
            Checksum byte (0-255)
        """
        checksum = 0
        for byte in data:
            checksum ^= byte
        return checksum & 0xFF

    def emergency_stop(self) -> bool:
        """Send zero torque command to stop motors immediately.

        Returns:
            True if stop command sent successfully
        """
        return self.send_motor_command(0.0, 0.0)

    def close(self):
        """Close serial connection and stop motors."""
        self.emergency_stop()
        time.sleep(0.1)  # Give time for stop command to arrive
        self._serial.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, '_serial') and self._serial.is_open:
            self.close()

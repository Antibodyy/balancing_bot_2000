"""Complementary filter for pitch angle estimation.

Fuses accelerometer and gyroscope data to estimate the robot's pitch angle.
The accelerometer provides a noisy but drift-free estimate, while the
gyroscope provides a smooth but drifting estimate. The complementary filter
combines them optimally.

Filter equation:
    theta_est = alpha * theta_gyro + (1 - alpha) * theta_accel

where:
    theta_gyro = previous_theta + omega_y * dt  (integrated gyroscope)
    theta_accel = atan2(a_x, a_z)               (accelerometer-based)
    alpha = tau / (tau + dt)                     (filter coefficient)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from state_estimation._internal.imu_fusion import (
    pitch_from_accelerometer,
    integrate_gyroscope,
    compute_complementary_filter_alpha,
)


@dataclass
class IMUReading:
    """Raw IMU sensor reading.

    Attributes:
        acceleration_mps2: Accelerometer readings [a_x, a_y, a_z] in m/s^2
        angular_velocity_radps: Gyroscope readings [omega_x, omega_y, omega_z] in rad/s
    """

    acceleration_mps2: np.ndarray  # Shape (3,)
    angular_velocity_radps: np.ndarray  # Shape (3,)

    def __post_init__(self) -> None:
        """Validate array shapes."""
        self.acceleration_mps2 = np.asarray(self.acceleration_mps2)
        self.angular_velocity_radps = np.asarray(self.angular_velocity_radps)

        if self.acceleration_mps2.shape != (3,):
            raise ValueError(
                f"acceleration_mps2 must have shape (3,), "
                f"got {self.acceleration_mps2.shape}"
            )
        if self.angular_velocity_radps.shape != (3,):
            raise ValueError(
                f"angular_velocity_radps must have shape (3,), "
                f"got {self.angular_velocity_radps.shape}"
            )


class ComplementaryFilter:
    """First-order complementary filter for pitch estimation.

    Fuses accelerometer (low-frequency trust) with gyroscope (high-frequency trust)
    to produce a stable pitch angle estimate.

    The filter has one tunable parameter: the time constant tau.
    - Larger tau: trusts gyroscope more, smoother but may drift
    - Smaller tau: trusts accelerometer more, noisier but no drift

    Typical values: tau = 0.05 to 0.5 seconds

    Attributes:
        time_constant_s: Filter time constant
        sampling_period_s: Expected time between updates
        pitch_estimate_rad: Current pitch angle estimate
    """

    def __init__(
        self,
        time_constant_s: float,
        sampling_period_s: float,
        initial_pitch_rad: float = 0.0,
    ) -> None:
        """Initialize the complementary filter.

        Args:
            time_constant_s: Filter time constant in seconds
            sampling_period_s: Expected time between updates in seconds
            initial_pitch_rad: Initial pitch angle estimate in radians
        """
        self._time_constant_s = time_constant_s
        self._sampling_period_s = sampling_period_s
        self._pitch_estimate_rad = initial_pitch_rad

        # Precompute filter coefficient
        self._alpha = compute_complementary_filter_alpha(
            time_constant_s, sampling_period_s
        )

        # Track if filter has been initialized with first measurement
        self._initialized = False

    def update(
        self,
        imu_reading: IMUReading,
        timestep_s: Optional[float] = None,
    ) -> float:
        """Update pitch estimate with new IMU reading.

        Args:
            imu_reading: Raw IMU sensor data
            timestep_s: Actual time since last update. If None, uses
                sampling_period_s from initialization.

        Returns:
            Updated pitch angle estimate in radians
        """
        if timestep_s is None:
            timestep_s = self._sampling_period_s

        # Extract relevant readings
        accel_x = imu_reading.acceleration_mps2[0]
        accel_z = imu_reading.acceleration_mps2[2]
        gyro_y = imu_reading.angular_velocity_radps[1]  # Pitch rate

        # Accelerometer-based pitch estimate
        pitch_accel = pitch_from_accelerometer(accel_x, accel_z)

        if not self._initialized:
            # First measurement: initialize from accelerometer
            self._pitch_estimate_rad = pitch_accel
            self._initialized = True
            return self._pitch_estimate_rad

        # Gyroscope-based pitch estimate (integrated)
        pitch_gyro = integrate_gyroscope(
            self._pitch_estimate_rad, gyro_y, timestep_s
        )

        # Recompute alpha if timestep differs from nominal
        if timestep_s != self._sampling_period_s:
            alpha = compute_complementary_filter_alpha(
                self._time_constant_s, timestep_s
            )
        else:
            alpha = self._alpha

        # Complementary filter fusion
        self._pitch_estimate_rad = (
            alpha * pitch_gyro + (1 - alpha) * pitch_accel
        )

        return self._pitch_estimate_rad

    def reset(self, initial_pitch_rad: float = 0.0) -> None:
        """Reset the filter to a known state.

        Args:
            initial_pitch_rad: New initial pitch angle estimate
        """
        self._pitch_estimate_rad = initial_pitch_rad
        self._initialized = False

    @property
    def pitch_estimate_rad(self) -> float:
        """Current pitch angle estimate in radians."""
        return self._pitch_estimate_rad

    @property
    def time_constant_s(self) -> float:
        """Filter time constant in seconds."""
        return self._time_constant_s

    @property
    def alpha(self) -> float:
        """Filter coefficient (gyroscope trust factor)."""
        return self._alpha

    @property
    def is_initialized(self) -> bool:
        """Whether filter has received at least one measurement."""
        return self._initialized

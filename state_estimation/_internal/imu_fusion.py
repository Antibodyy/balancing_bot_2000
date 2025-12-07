"""IMU sensor fusion mathematics.

Provides low-level functions for computing pitch angle from IMU data.
The robot's IMU coordinate frame is assumed to be:
    - X-axis: forward (direction of travel)
    - Y-axis: left (perpendicular to travel direction)
    - Z-axis: up (opposite to gravity when upright)

Pitch angle theta is defined as rotation about the Y-axis:
    - theta = 0: robot is upright (unstable equilibrium)
    - theta > 0: robot tilted forward
    - theta < 0: robot tilted backward
"""

import numpy as np


def pitch_from_accelerometer(
    acceleration_x_mps2: float,
    acceleration_z_mps2: float,
) -> float:
    """Compute pitch angle from accelerometer readings.

    Uses the relationship between gravity components when the robot is
    quasi-static (not accelerating significantly).

    When upright (theta=0): a_x = 0, a_z = g
    When tilted by theta: a_x = g*sin(theta), a_z = g*cos(theta)

    Therefore: theta = atan2(a_x, a_z)

    Args:
        acceleration_x_mps2: Accelerometer X reading (forward direction)
        acceleration_z_mps2: Accelerometer Z reading (up direction)

    Returns:
        Estimated pitch angle in radians

    Note:
        This estimate is noisy but has no drift. It is inaccurate
        during dynamic motion when linear acceleration is significant.
    """
    return np.arctan2(acceleration_x_mps2, acceleration_z_mps2)


def integrate_gyroscope(
    previous_pitch_rad: float,
    pitch_rate_radps: float,
    timestep_s: float,
) -> float:
    """Integrate gyroscope reading to update pitch estimate.

    Simple Euler integration: theta_new = theta_old + omega_y * dt

    Args:
        previous_pitch_rad: Previous pitch angle estimate in radians
        pitch_rate_radps: Gyroscope Y-axis reading (pitch rate) in rad/s
        timestep_s: Time step for integration in seconds

    Returns:
        Updated pitch angle estimate in radians

    Note:
        This estimate is smooth but drifts over time due to gyroscope bias.
    """
    return previous_pitch_rad + pitch_rate_radps * timestep_s


def compute_complementary_filter_alpha(
    time_constant_s: float,
    sampling_period_s: float,
) -> float:
    """Compute complementary filter coefficient alpha.

    The filter is: theta = alpha * theta_gyro + (1 - alpha) * theta_accel

    Alpha determines the crossover frequency between trusting the gyroscope
    (high frequency) and the accelerometer (low frequency).

    alpha = tau / (tau + T_s)

    where:
        tau = filter time constant (larger = trust gyro more)
        T_s = sampling period

    Args:
        time_constant_s: Filter time constant in seconds
        sampling_period_s: Sampling period in seconds

    Returns:
        Filter coefficient alpha in range (0, 1)

    Raises:
        ValueError: If time_constant_s or sampling_period_s is not positive
    """
    if time_constant_s <= 0:
        raise ValueError(
            f"time_constant_s must be positive, got {time_constant_s}"
        )
    if sampling_period_s <= 0:
        raise ValueError(
            f"sampling_period_s must be positive, got {sampling_period_s}"
        )

    alpha = time_constant_s / (time_constant_s + sampling_period_s)
    return alpha

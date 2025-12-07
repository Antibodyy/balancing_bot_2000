"""State estimation module for self-balancing robot.

This module provides sensor fusion algorithms for estimating the robot's
state from IMU and encoder data.

Public API:
    - EstimatorConfig: Configuration dataclass for estimator parameters
    - ComplementaryFilter: First-order complementary filter for pitch estimation
    - IMUReading: Dataclass for raw IMU sensor data
"""

from state_estimation.config import EstimatorConfig
from state_estimation.complementary_filter import (
    ComplementaryFilter,
    IMUReading,
)

__all__ = [
    'EstimatorConfig',
    'ComplementaryFilter',
    'IMUReading',
]

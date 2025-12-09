"""Control pipeline module for self-balancing robot.

This module provides the main control loop orchestration, coordinating
state estimation, reference generation, and MPC optimization.

Public API:
    - BalanceController: Main controller class
    - SensorData: Raw sensor measurements dataclass
    - ControlOutput: Control output dataclass
    - ControlLoopTimer: Timing utilities
    - IterationTiming: Timing breakdown dataclass
    - TimingStatistics: Timing statistics dataclass
"""

from control_pipeline.controller import (
    BalanceController,
    SensorData,
    ControlOutput,
)
from control_pipeline.timing import (
    ControlLoopTimer,
    IterationTiming,
    TimingStatistics,
)

__all__ = [
    'BalanceController',
    'SensorData',
    'ControlOutput',
    'ControlLoopTimer',
    'IterationTiming',
    'TimingStatistics',
]

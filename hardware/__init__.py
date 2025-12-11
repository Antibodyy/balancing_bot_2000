"""Hardware interface module for Pololu Balboa 32U4 robot.

This module provides the hardware abstraction layer for deploying the MPC
controller on the physical robot, including:
- Serial communication with Arduino
- Hardware controller wrapper
- Configuration loading
"""

from .config_loader import load_hardware_mpc, load_hardware_mpc_with_custom_params
from .serial_interface import BalboaSerialInterface, RawSensorPacket
from .hardware_controller import HardwareBalanceController, ControlLoopStats

__all__ = [
    'load_hardware_mpc',
    'load_hardware_mpc_with_custom_params',
    'BalboaSerialInterface',
    'RawSensorPacket',
    'HardwareBalanceController',
    'ControlLoopStats',
]

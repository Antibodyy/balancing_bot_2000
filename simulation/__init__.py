"""Simulation module for MPC validation using MuJoCo.

This module provides a simulation environment for testing and validating
the MPC controller before deployment to hardware.

Note: MuJoCo is an optional dependency. Check MUJOCO_AVAILABLE before
running simulations.

Public API:
    - MPCSimulation: Main simulation class
    - SimulationResult: Result dataclass from simulation runs
    - SimulationConfig: Configuration dataclass for simulation
    - MUJOCO_AVAILABLE: Boolean indicating if MuJoCo is installed
"""

from simulation.mpc_simulation import (
    MPCSimulation,
    SimulationResult,
    SimulationConfig,
    MUJOCO_AVAILABLE,
)

__all__ = [
    'MPCSimulation',
    'SimulationResult',
    'SimulationConfig',
    'MUJOCO_AVAILABLE',
]

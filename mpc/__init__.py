"""MPC module for self-balancing robot control.

This module provides a CasADi-based linear MPC controller with
IPOPT backend for real-time optimization.

Public API:
    - MPCConfig: Configuration dataclass for MPC parameters
    - LinearMPCSolver: Main MPC solver class
    - MPCSolution: Solution dataclass from MPC solver
    - StateConstraints: State bound constraints
    - InputConstraints: Control bound constraints
    - ReferenceGenerator: Reference trajectory generator
    - ReferenceCommand: Command for reference generation
    - ReferenceMode: Reference generation mode enum
    - build_state_cost_matrix: Build Q matrix from diagonal
    - build_control_cost_matrix: Build R matrix from diagonal
    - compute_terminal_cost_dare: Compute P via DARE
"""

from mpc.config import MPCConfig
from mpc.cost_matrices import (
    build_state_cost_matrix,
    build_control_cost_matrix,
    compute_terminal_cost_dare,
)
from mpc.constraints import (
    StateConstraints,
    InputConstraints,
    create_constraints_from_config,
)
from mpc.linear_mpc_solver import LinearMPCSolver, MPCSolution
from mpc.reference_generator import (
    ReferenceGenerator,
    ReferenceCommand,
    ReferenceMode,
)

__all__ = [
    'MPCConfig',
    'LinearMPCSolver',
    'MPCSolution',
    'StateConstraints',
    'InputConstraints',
    'create_constraints_from_config',
    'ReferenceGenerator',
    'ReferenceCommand',
    'ReferenceMode',
    'build_state_cost_matrix',
    'build_control_cost_matrix',
    'compute_terminal_cost_dare',
]

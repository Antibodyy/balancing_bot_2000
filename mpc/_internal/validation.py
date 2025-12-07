"""Validation utilities for MPC module.

Provides input validation for MPC configuration and solver inputs.
"""

import numpy as np

from robot_dynamics.parameters import STATE_DIMENSION, CONTROL_DIMENSION


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is strictly positive.

    Args:
        value: The value to validate
        name: Parameter name for error messages

    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative.

    Args:
        value: The value to validate
        name: Parameter name for error messages

    Raises:
        ValueError: If value is negative
    """
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def validate_positive_integer(value: int, name: str) -> None:
    """Validate that a value is a positive integer.

    Args:
        value: The value to validate
        name: Parameter name for error messages

    Raises:
        ValueError: If value is not a positive integer
    """
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")


def validate_state_cost_diagonal(diagonal: np.ndarray) -> None:
    """Validate state cost diagonal elements.

    Args:
        diagonal: Array of diagonal elements

    Raises:
        ValueError: If diagonal has wrong shape or non-positive elements
    """
    if diagonal.shape != (STATE_DIMENSION,):
        raise ValueError(
            f"state_cost_diagonal must have shape ({STATE_DIMENSION},), "
            f"got {diagonal.shape}"
        )
    if not np.all(diagonal > 0):
        raise ValueError("All state cost diagonal elements must be positive")
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("All state cost diagonal elements must be finite")


def validate_control_cost_diagonal(diagonal: np.ndarray) -> None:
    """Validate control cost diagonal elements.

    Args:
        diagonal: Array of diagonal elements

    Raises:
        ValueError: If diagonal has wrong shape or non-positive elements
    """
    if diagonal.shape != (CONTROL_DIMENSION,):
        raise ValueError(
            f"control_cost_diagonal must have shape ({CONTROL_DIMENSION},), "
            f"got {diagonal.shape}"
        )
    if not np.all(diagonal > 0):
        raise ValueError("All control cost diagonal elements must be positive")
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("All control cost diagonal elements must be finite")


def validate_solver_name(solver_name: str) -> None:
    """Validate solver name is supported.

    Note: CasADi Opti interface uses IPOPT internally for all solver names.
    The solver_name is kept for configuration compatibility but IPOPT
    is always used via the Opti interface.

    Args:
        solver_name: Name of the QP solver (osqp, qpoases, or ipopt)

    Raises:
        ValueError: If solver name is not supported
    """
    supported_solvers = {'osqp', 'qpoases', 'ipopt'}
    if solver_name not in supported_solvers:
        raise ValueError(
            f"solver_name must be one of {supported_solvers}, got '{solver_name}'"
        )

"""Runtime contract validation utilities.

Internal module for parameter and input validation.
Enforces defensive programming per style_guide.md section 4.3.
"""

import numpy as np


def validate_positive(value: float, name: str) -> None:
    """Validate that a value is strictly positive.

    Args:
        value: Value to validate
        name: Parameter name for error message

    Raises:
        ValueError: If value <= 0
    """
    if value <= 0:
        raise ValueError(
            f"{name} must be positive, got {value}"
        )


def validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative.

    Args:
        value: Value to validate
        name: Parameter name for error message

    Raises:
        ValueError: If value < 0
    """
    if value < 0:
        raise ValueError(
            f"{name} must be non-negative, got {value}"
        )


def validate_state_vector(state: np.ndarray) -> None:
    """Validate state vector has correct shape and finite values.

    Args:
        state: State vector [x, theta, psi, dx, dtheta, dpsi]

    Raises:
        ValueError: If shape incorrect or contains non-finite values
    """
    if state.shape != (6,):
        raise ValueError(
            f"State vector must have shape (6,), got {state.shape}"
        )

    if not np.all(np.isfinite(state)):
        raise ValueError(
            f"State vector contains non-finite values: {state}"
        )


def validate_control_vector(control: np.ndarray) -> None:
    """Validate control vector has correct shape and finite values.

    Args:
        control: Control vector [tau_L, tau_R]

    Raises:
        ValueError: If shape incorrect or contains non-finite values
    """
    if control.shape != (2,):
        raise ValueError(
            f"Control vector must have shape (2,), got {control.shape}"
        )

    if not np.all(np.isfinite(control)):
        raise ValueError(
            f"Control vector contains non-finite values: {control}"
        )

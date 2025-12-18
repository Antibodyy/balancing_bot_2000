"""Constraint definitions for MPC.

Provides dataclasses for state and input constraints used in the
MPC optimization problem.

State vector: [x, theta, psi, dx, dtheta, dpsi]
Control vector: [tau_L, tau_R]
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from robot_dynamics.parameters import (
    STATE_DIMENSION,
    CONTROL_DIMENSION,
    PITCH_INDEX,
    PITCH_RATE_INDEX,
    VELOCITY_INDEX,
)


@dataclass(frozen=True)
class StateConstraints:
    """State box constraints for MPC.

    Encodes: x_min <= x_k <= x_max for each prediction step.

    Only pitch and pitch rate are constrained. Position, heading, and
    their rates are left unconstrained (infinite bounds) to allow
    free movement while maintaining balance.

    Attributes:
        pitch_limit_rad: Maximum absolute pitch angle |theta| <= pitch_limit_rad
        pitch_rate_limit_radps: Maximum absolute pitch rate |dtheta| <= limit
        velocity_limit_mps: Optional forward velocity bound |dx|
    """

    pitch_limit_rad: float
    pitch_rate_limit_radps: float
    velocity_limit_mps: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate constraint values are positive."""
        if self.pitch_limit_rad <= 0:
            raise ValueError(
                f"pitch_limit_rad must be positive, got {self.pitch_limit_rad}"
            )
        if self.pitch_rate_limit_radps <= 0:
            raise ValueError(
                f"pitch_rate_limit_radps must be positive, "
                f"got {self.pitch_rate_limit_radps}"
            )
        if self.velocity_limit_mps is not None and self.velocity_limit_mps <= 0:
            raise ValueError(
                f"velocity_limit_mps must be positive, got {self.velocity_limit_mps}"
            )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bound arrays for state constraints.

        Returns:
            Tuple of (lower_bound, upper_bound), each shape (STATE_DIMENSION,)
            Unconstrained states have -inf/+inf bounds.
        """
        lower_bound = np.full(STATE_DIMENSION, -np.inf)
        upper_bound = np.full(STATE_DIMENSION, np.inf)

        # Pitch constraints
        lower_bound[PITCH_INDEX] = -self.pitch_limit_rad
        upper_bound[PITCH_INDEX] = self.pitch_limit_rad

        # Pitch rate constraints
        lower_bound[PITCH_RATE_INDEX] = -self.pitch_rate_limit_radps
        upper_bound[PITCH_RATE_INDEX] = self.pitch_rate_limit_radps

        # Optional velocity constraint
        if self.velocity_limit_mps is not None:
            lower_bound[VELOCITY_INDEX] = -self.velocity_limit_mps
            upper_bound[VELOCITY_INDEX] = self.velocity_limit_mps

        return lower_bound, upper_bound


@dataclass(frozen=True)
class InputConstraints:
    """Control input box constraints for MPC.

    Encodes: u_min <= u_k <= u_max

    Both wheel torques have symmetric bounds.

    Attributes:
        control_limit_nm: Maximum absolute torque |tau| <= control_limit_nm
    """

    control_limit_nm: float

    def __post_init__(self) -> None:
        """Validate constraint value is positive."""
        if self.control_limit_nm <= 0:
            raise ValueError(
                f"control_limit_nm must be positive, got {self.control_limit_nm}"
            )

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get lower and upper bound arrays for control constraints.

        Returns:
            Tuple of (lower_bound, upper_bound), each shape (CONTROL_DIMENSION,)
        """
        lower_bound = np.full(CONTROL_DIMENSION, -self.control_limit_nm)
        upper_bound = np.full(CONTROL_DIMENSION, self.control_limit_nm)

        return lower_bound, upper_bound


def create_constraints_from_config(
    pitch_limit_rad: float,
    pitch_rate_limit_radps: float,
    control_limit_nm: float,
    velocity_limit_mps: Optional[float] = None,
) -> Tuple[StateConstraints, InputConstraints]:
    """Create constraint objects from configuration values.

    Convenience function to create both constraint types from
    scalar configuration values.

    Args:
        pitch_limit_rad: Maximum absolute pitch angle
        pitch_rate_limit_radps: Maximum absolute pitch rate
        control_limit_nm: Maximum absolute control torque
        velocity_limit_mps: Optional forward velocity bound

    Returns:
        Tuple of (StateConstraints, InputConstraints)
    """
    state_constraints = StateConstraints(
        pitch_limit_rad=pitch_limit_rad,
        pitch_rate_limit_radps=pitch_rate_limit_radps,
        velocity_limit_mps=velocity_limit_mps,
    )
    input_constraints = InputConstraints(
        control_limit_nm=control_limit_nm,
    )
    return state_constraints, input_constraints

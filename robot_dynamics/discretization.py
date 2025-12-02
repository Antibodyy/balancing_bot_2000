"""Discretization of continuous-time dynamics.

Converts continuous linear systems to discrete-time for MPC implementation.
Uses zero-order hold (ZOH) assumption via matrix exponential.
"""

import numpy as np
import scipy.linalg
from dataclasses import dataclass
from robot_dynamics._internal.validation import validate_positive


@dataclass(frozen=True)
class DiscreteDynamics:
    """Discrete-time linear system matrices.

    Represents: x[k+1] = A_d·x[k] + B_d·u[k]

    Attributes:
        state_matrix_discrete: A_d matrix (6, 6)
        control_matrix_discrete: B_d matrix (6, 2)
        sampling_period_s: Discretization time step
    """
    state_matrix_discrete: np.ndarray
    control_matrix_discrete: np.ndarray
    sampling_period_s: float

    def __post_init__(self):
        """Validate matrix dimensions and sampling period."""
        if self.state_matrix_discrete.shape != (6, 6):
            raise ValueError(
                f"Discrete state matrix must be (6, 6), "
                f"got {self.state_matrix_discrete.shape}"
            )
        if self.control_matrix_discrete.shape != (6, 2):
            raise ValueError(
                f"Discrete control matrix must be (6, 2), "
                f"got {self.control_matrix_discrete.shape}"
            )
        validate_positive(self.sampling_period_s, 'sampling_period_s')


def discretize_linear_dynamics(
    state_matrix: np.ndarray,
    control_matrix: np.ndarray,
    sampling_period_s: float
) -> DiscreteDynamics:
    """Convert continuous linear dynamics to discrete-time using ZOH.

    Given continuous system: x_dot = A·x + B·u
    Computes discrete system: x[k+1] = A_d·x[k] + B_d·u[k]

    Uses exact discretization via matrix exponential:
    A_d = exp(A·T_s)
    B_d = ∫[0 to T_s] exp(A·τ) dτ · B

    Args:
        state_matrix: Continuous state matrix A (n, n)
        control_matrix: Continuous control matrix B (n, m)
        sampling_period_s: Sampling period T_s in seconds

    Returns:
        DiscreteDynamics object with A_d, B_d matrices

    Raises:
        ValueError: If matrices have incompatible shapes or T_s <= 0

    References:
        Franklin, Powell, Workman - Digital Control of Dynamic Systems
    """
    validate_positive(sampling_period_s, 'sampling_period_s')

    state_dimension, _ = state_matrix.shape
    _, control_dimension = control_matrix.shape

    # Validate matrix dimensions
    if state_matrix.shape != (state_dimension, state_dimension):
        raise ValueError(
            f"State matrix must be square, got {state_matrix.shape}"
        )
    if control_matrix.shape != (state_dimension, control_dimension):
        raise ValueError(
            f"Control matrix shape {control_matrix.shape} incompatible "
            f"with state matrix shape {state_matrix.shape}"
        )

    # Build augmented matrix for simultaneous discretization
    # M = [[A,  B],
    #      [0,  0]]
    augmented_dimension = state_dimension + control_dimension
    augmented_matrix = np.zeros((augmented_dimension, augmented_dimension))
    augmented_matrix[:state_dimension, :state_dimension] = state_matrix
    augmented_matrix[:state_dimension, state_dimension:] = control_matrix

    # Compute matrix exponential: exp(M·T_s)
    augmented_exponential = scipy.linalg.expm(
        augmented_matrix * sampling_period_s
    )

    # Extract discrete matrices
    # exp(M·T_s) = [[A_d, B_d],
    #               [ 0,   I ]]
    state_matrix_discrete = augmented_exponential[
        :state_dimension, :state_dimension
    ]
    control_matrix_discrete = augmented_exponential[
        :state_dimension, state_dimension:
    ]

    return DiscreteDynamics(
        state_matrix_discrete=state_matrix_discrete,
        control_matrix_discrete=control_matrix_discrete,
        sampling_period_s=sampling_period_s
    )

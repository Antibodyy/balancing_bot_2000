"""Cost matrix construction for MPC.

Provides functions to build Q, R, P matrices for the MPC cost function:
    J = sum_{k=0}^{N-1} [(x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k]
        + (x_N - x_ref)^T P (x_N - x_ref)
"""

import numpy as np
import scipy.linalg

from robot_dynamics.parameters import STATE_DIMENSION, CONTROL_DIMENSION


def build_state_cost_matrix(diagonal: np.ndarray) -> np.ndarray:
    """Build state cost matrix Q from diagonal elements.

    Args:
        diagonal: Array of diagonal elements (STATE_DIMENSION,)

    Returns:
        Diagonal state cost matrix Q (STATE_DIMENSION, STATE_DIMENSION)

    Raises:
        ValueError: If diagonal has wrong shape
    """
    diagonal = np.asarray(diagonal)
    if diagonal.shape != (STATE_DIMENSION,):
        raise ValueError(
            f"diagonal must have shape ({STATE_DIMENSION},), got {diagonal.shape}"
        )
    return np.diag(diagonal)


def build_control_cost_matrix(diagonal: np.ndarray) -> np.ndarray:
    """Build control cost matrix R from diagonal elements.

    Args:
        diagonal: Array of diagonal elements (CONTROL_DIMENSION,)

    Returns:
        Diagonal control cost matrix R (CONTROL_DIMENSION, CONTROL_DIMENSION)

    Raises:
        ValueError: If diagonal has wrong shape
    """
    diagonal = np.asarray(diagonal)
    if diagonal.shape != (CONTROL_DIMENSION,):
        raise ValueError(
            f"diagonal must have shape ({CONTROL_DIMENSION},), got {diagonal.shape}"
        )
    return np.diag(diagonal)


def compute_terminal_cost_dare(
    state_matrix_discrete: np.ndarray,
    control_matrix_discrete: np.ndarray,
    state_cost: np.ndarray,
    control_cost: np.ndarray,
) -> np.ndarray:
    """Compute terminal cost matrix P via discrete algebraic Riccati equation.

    Solves the DARE to find P that satisfies:
        P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q

    This terminal cost provides theoretical closed-loop stability guarantees
    for the MPC controller.

    Args:
        state_matrix_discrete: Discrete state transition matrix A_d (n, n)
        control_matrix_discrete: Discrete control matrix B_d (n, m)
        state_cost: State cost matrix Q (n, n)
        control_cost: Control cost matrix R (m, m)

    Returns:
        Terminal cost matrix P (n, n), symmetric positive semi-definite

    Raises:
        ValueError: If matrix dimensions are incompatible
        numpy.linalg.LinAlgError: If DARE has no solution (system not stabilizable)

    Note:
        If DARE fails, consider using compute_terminal_cost_scaled as fallback.
    """
    # Validate dimensions
    n_states = state_matrix_discrete.shape[0]
    n_controls = control_matrix_discrete.shape[1]

    if state_matrix_discrete.shape != (n_states, n_states):
        raise ValueError(
            f"state_matrix_discrete must be square, got shape "
            f"{state_matrix_discrete.shape}"
        )
    if control_matrix_discrete.shape[0] != n_states:
        raise ValueError(
            f"control_matrix_discrete row count {control_matrix_discrete.shape[0]} "
            f"must match state dimension {n_states}"
        )
    if state_cost.shape != (n_states, n_states):
        raise ValueError(
            f"state_cost shape {state_cost.shape} must match "
            f"({n_states}, {n_states})"
        )
    if control_cost.shape != (n_controls, n_controls):
        raise ValueError(
            f"control_cost shape {control_cost.shape} must match "
            f"({n_controls}, {n_controls})"
        )

    # Solve discrete algebraic Riccati equation (DARE)
    terminal_cost = scipy.linalg.solve_discrete_are(
        state_matrix_discrete,
        control_matrix_discrete,
        state_cost,
        control_cost,
    )

    return terminal_cost


# Fallback option: scaled state cost matrix
# Uncomment and use if DARE fails or simpler tuning is desired
#
# def compute_terminal_cost_scaled(
#     state_cost: np.ndarray,
#     terminal_cost_scale: float = 10.0,
# ) -> np.ndarray:
#     """Compute terminal cost as scaled state cost.
#
#     Simple alternative to DARE when:
#     - System is not stabilizable
#     - Simpler tuning is preferred
#     - Faster computation is needed
#
#     Args:
#         state_cost: State cost matrix Q (n, n)
#         terminal_cost_scale: Scaling factor k (typically 10-100)
#
#     Returns:
#         Terminal cost matrix P = k * Q
#     """
#     return terminal_cost_scale * state_cost

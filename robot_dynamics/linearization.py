"""Linearization of robot dynamics around equilibrium.

Provides A and B matrices for linear control design (LQR, MPC).
Uses CasADi automatic differentiation for exact Jacobians.
"""

import casadi as ca
import numpy as np
from dataclasses import dataclass
from typing import Optional
from robot_dynamics.parameters import RobotParameters
from robot_dynamics.continuous_dynamics import (
    build_dynamics_model,
    compute_state_derivative
)


@dataclass(frozen=True)
class LinearizedDynamics:
    """Linearized system matrices around an equilibrium point.

    Represents: x_dot = A·x + B·u (continuous time)

    Attributes:
        state_matrix: A matrix (6, 6) - state Jacobian
        control_matrix: B matrix (6, 2) - control Jacobian
        equilibrium_state: Equilibrium state (6,)
        equilibrium_control: Equilibrium control (2,)
        parameters: Robot parameters used for linearization
    """
    state_matrix: np.ndarray
    control_matrix: np.ndarray
    equilibrium_state: np.ndarray
    equilibrium_control: np.ndarray
    parameters: RobotParameters

    def __post_init__(self):
        """Validate matrix dimensions."""
        if self.state_matrix.shape != (6, 6):
            raise ValueError(
                f"State matrix must be (6, 6), got {self.state_matrix.shape}"
            )
        if self.control_matrix.shape != (6, 2):
            raise ValueError(
                f"Control matrix must be (6, 2), got {self.control_matrix.shape}"
            )


def build_jacobian_functions(params: RobotParameters) -> tuple:
    """Build reusable CasADi Jacobian functions for efficient online linearization.

    Builds symbolic Jacobian functions ONCE, which can then be evaluated
    at different states efficiently without rebuilding symbolic expressions.
    This is significantly faster than rebuilding symbolic expressions at
    each linearization point.

    Args:
        params: Robot parameters

    Returns:
        Tuple of (jacobian_state_fn, jacobian_control_fn) - CasADi Functions
        that can be evaluated at any (state, control) point
    """
    # Create symbolic variables
    state_sym = ca.SX.sym('state', 6)
    control_sym = ca.SX.sym('control', 2)

    # Build symbolic dynamics
    dynamics_function = build_dynamics_model(params)
    state_derivative_sym = dynamics_function(state_sym, control_sym)

    # Compute Jacobians symbolically
    state_jacobian_sym = ca.jacobian(state_derivative_sym, state_sym)
    control_jacobian_sym = ca.jacobian(state_derivative_sym, control_sym)

    # Create reusable Functions
    jacobian_state_fn = ca.Function(
        'jacobian_state', [state_sym, control_sym], [state_jacobian_sym]
    )
    jacobian_control_fn = ca.Function(
        'jacobian_control', [state_sym, control_sym], [control_jacobian_sym]
    )

    return (jacobian_state_fn, jacobian_control_fn)


def linearize_at_state(
    params: RobotParameters,
    state: np.ndarray,
    control: np.ndarray,
    jacobian_functions: Optional[tuple] = None,
) -> LinearizedDynamics:
    """Compute linearized dynamics A, B matrices around any state.

    Unlike linearize_at_equilibrium(), this does NOT validate equilibrium.
    Use for successive/online linearization in MPC where we linearize
    around the current (non-equilibrium) state.

    Args:
        params: Robot parameters
        state: State around which to linearize (6,)
        control: Control input at linearization point (2,)
        jacobian_functions: Optional pre-built (jacobian_state_fn, jacobian_control_fn)
                           from build_jacobian_functions(). If None, builds them.
                           Providing cached functions is ~10x faster.

    Returns:
        LinearizedDynamics object containing A, B matrices
    """
    # Use cached Jacobian functions if provided, otherwise build them
    if jacobian_functions is not None:
        jacobian_state_fn, jacobian_control_fn = jacobian_functions
    else:
        jacobian_state_fn, jacobian_control_fn = build_jacobian_functions(params)

    # Evaluate at specified state (FAST - just numerical evaluation)
    state_matrix = np.array(jacobian_state_fn(state, control))
    control_matrix = np.array(jacobian_control_fn(state, control))

    return LinearizedDynamics(
        state_matrix=state_matrix,
        control_matrix=control_matrix,
        equilibrium_state=state,  # Not truly equilibrium, just linearization point
        equilibrium_control=control,
        parameters=params
    )


def linearize_at_equilibrium(
    params: RobotParameters,
    equilibrium_state: np.ndarray,
    equilibrium_control: np.ndarray
) -> LinearizedDynamics:
    """Compute linearized dynamics A, B matrices around equilibrium.

    Uses CasADi automatic differentiation for exact Jacobian computation.

    Args:
        params: Robot parameters
        equilibrium_state: Equilibrium state (6,)
        equilibrium_control: Equilibrium control (2,)

    Returns:
        LinearizedDynamics object containing A, B matrices

    Raises:
        ValueError: If equilibrium point is not valid
    """
    # Verify this is actually an equilibrium (f(x_eq, u_eq) ≈ 0)
    derivative_at_equilibrium = compute_state_derivative(
        equilibrium_state, equilibrium_control, params
    )

    # Velocity components should match state (first 3 elements)
    # Acceleration components should be near zero (last 3 elements)
    max_acceleration = np.max(np.abs(derivative_at_equilibrium[3:]))
    if max_acceleration > 1e-3:
        raise ValueError(
            f"Not a valid equilibrium: max acceleration = {max_acceleration:.6f}. "
            f"Expected near-zero accelerations at equilibrium."
        )

    # Use linearize_at_state with equilibrium validation passed
    return linearize_at_state(params, equilibrium_state, equilibrium_control)

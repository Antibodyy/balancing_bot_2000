"""Robot dynamics module for MPC-based self-balancing control.

This module provides CasADi-based symbolic dynamics computation with
NumPy array outputs for cvxopt compatibility.

Public API:
    - RobotParameters: Physical parameter dataclass
    - compute_state_derivative: Evaluate nonlinear dynamics
    - compute_equilibrium_state: Compute equilibrium point
    - LinearizedDynamics: Linearized system dataclass
    - linearize_at_equilibrium: Compute A, B matrices
    - DiscreteDynamics: Discrete system dataclass
    - discretize_linear_dynamics: Convert continuous to discrete
"""

from robot_dynamics.parameters import RobotParameters
from robot_dynamics.continuous_dynamics import (
    compute_state_derivative,
    compute_equilibrium_state,
    build_dynamics_model
)
from robot_dynamics.linearization import (
    LinearizedDynamics,
    linearize_at_equilibrium
)
from robot_dynamics.discretization import (
    DiscreteDynamics,
    discretize_linear_dynamics
)

__all__ = [
    'RobotParameters',
    'compute_state_derivative',
    'compute_equilibrium_state',
    'build_dynamics_model',
    'LinearizedDynamics',
    'linearize_at_equilibrium',
    'DiscreteDynamics',
    'discretize_linear_dynamics',
]

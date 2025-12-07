"""Unit tests for MPC cost matrices."""

import numpy as np
import pytest
import scipy.linalg

from robot_dynamics import (
    RobotParameters,
    linearize_at_equilibrium,
    discretize_linear_dynamics,
    compute_equilibrium_state,
)
from robot_dynamics.parameters import STATE_DIMENSION, CONTROL_DIMENSION
from mpc.cost_matrices import (
    build_state_cost_matrix,
    build_control_cost_matrix,
    compute_terminal_cost_dare,
)


@pytest.fixture
def robot_params():
    """Load robot parameters."""
    return RobotParameters.from_yaml('config/robot_params.yaml')


@pytest.fixture
def discrete_dynamics(robot_params):
    """Get discretized dynamics."""
    eq_state, eq_control = compute_equilibrium_state(robot_params)
    linearized = linearize_at_equilibrium(robot_params, eq_state, eq_control)
    return discretize_linear_dynamics(
        linearized.state_matrix,
        linearized.control_matrix,
        sampling_period_s=0.02,
    )


class TestBuildStateCostMatrix:
    """Tests for build_state_cost_matrix."""

    def test_creates_diagonal_matrix(self):
        """Test that result is a diagonal matrix."""
        diagonal = np.array([1.0, 100.0, 10.0, 1.0, 10.0, 1.0])
        Q = build_state_cost_matrix(diagonal)

        assert Q.shape == (STATE_DIMENSION, STATE_DIMENSION)
        assert np.allclose(np.diag(Q), diagonal)
        # Off-diagonal elements should be zero
        assert np.allclose(Q - np.diag(np.diag(Q)), 0)

    def test_wrong_shape_raises(self):
        """Test that wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="must have shape"):
            build_state_cost_matrix(np.array([1.0, 2.0, 3.0]))

    def test_positive_definite(self):
        """Test that result is positive definite."""
        diagonal = np.array([1.0, 100.0, 10.0, 1.0, 10.0, 1.0])
        Q = build_state_cost_matrix(diagonal)

        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues > 0)


class TestBuildControlCostMatrix:
    """Tests for build_control_cost_matrix."""

    def test_creates_diagonal_matrix(self):
        """Test that result is a diagonal matrix."""
        diagonal = np.array([0.1, 0.1])
        R = build_control_cost_matrix(diagonal)

        assert R.shape == (CONTROL_DIMENSION, CONTROL_DIMENSION)
        assert np.allclose(np.diag(R), diagonal)

    def test_wrong_shape_raises(self):
        """Test that wrong shape raises ValueError."""
        with pytest.raises(ValueError, match="must have shape"):
            build_control_cost_matrix(np.array([1.0, 2.0, 3.0]))


class TestComputeTerminalCostDare:
    """Tests for compute_terminal_cost_dare."""

    def test_returns_symmetric_matrix(self, discrete_dynamics):
        """Test that DARE solution is symmetric."""
        Q = build_state_cost_matrix(np.array([1.0, 100.0, 10.0, 1.0, 10.0, 1.0]))
        R = build_control_cost_matrix(np.array([0.1, 0.1]))

        P = compute_terminal_cost_dare(
            discrete_dynamics.state_matrix_discrete,
            discrete_dynamics.control_matrix_discrete,
            Q, R,
        )

        assert P.shape == (STATE_DIMENSION, STATE_DIMENSION)
        assert np.allclose(P, P.T), "P should be symmetric"

    def test_returns_positive_semidefinite(self, discrete_dynamics):
        """Test that DARE solution is positive semi-definite."""
        Q = build_state_cost_matrix(np.array([1.0, 100.0, 10.0, 1.0, 10.0, 1.0]))
        R = build_control_cost_matrix(np.array([0.1, 0.1]))

        P = compute_terminal_cost_dare(
            discrete_dynamics.state_matrix_discrete,
            discrete_dynamics.control_matrix_discrete,
            Q, R,
        )

        eigenvalues = np.linalg.eigvalsh(P)
        assert np.all(eigenvalues >= -1e-10), "P should be positive semi-definite"

    def test_satisfies_dare_equation(self, discrete_dynamics):
        """Test that P satisfies the discrete ARE."""
        A = discrete_dynamics.state_matrix_discrete
        B = discrete_dynamics.control_matrix_discrete
        Q = build_state_cost_matrix(np.array([1.0, 100.0, 10.0, 1.0, 10.0, 1.0]))
        R = build_control_cost_matrix(np.array([0.1, 0.1]))

        P = compute_terminal_cost_dare(A, B, Q, R)

        # DARE: P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P_check = A.T @ P @ A - A.T @ P @ B @ K + Q

        assert np.allclose(P, P_check, rtol=1e-5), "P should satisfy DARE"

    def test_matches_scipy_reference(self, discrete_dynamics):
        """Test that result matches scipy's solve_discrete_are."""
        A = discrete_dynamics.state_matrix_discrete
        B = discrete_dynamics.control_matrix_discrete
        Q = build_state_cost_matrix(np.array([1.0, 100.0, 10.0, 1.0, 10.0, 1.0]))
        R = build_control_cost_matrix(np.array([0.1, 0.1]))

        P = compute_terminal_cost_dare(A, B, Q, R)
        P_scipy = scipy.linalg.solve_discrete_are(A, B, Q, R)

        assert np.allclose(P, P_scipy)

    def test_incompatible_dimensions_raises(self):
        """Test that incompatible dimensions raise ValueError."""
        A = np.eye(4)  # Wrong size
        B = np.ones((6, 2))
        Q = np.eye(6)
        R = np.eye(2)

        with pytest.raises(ValueError):
            compute_terminal_cost_dare(A, B, Q, R)

"""Unit tests for Linear MPC solver."""

import numpy as np
import pytest

from robot_dynamics import (
    RobotParameters,
    linearize_at_equilibrium,
    discretize_linear_dynamics,
    compute_equilibrium_state,
)
from robot_dynamics.parameters import (
    STATE_DIMENSION,
    CONTROL_DIMENSION,
    PITCH_INDEX,
    PITCH_RATE_INDEX,
)
from mpc.config import MPCConfig
from mpc.cost_matrices import (
    build_state_cost_matrix,
    build_control_cost_matrix,
    compute_terminal_cost_dare,
)
from mpc.constraints import create_constraints_from_config
from mpc.linear_mpc_solver import LinearMPCSolver, MPCSolution


@pytest.fixture
def robot_params():
    """Load robot parameters."""
    return RobotParameters.from_yaml('config/simulation/robot_params.yaml')


@pytest.fixture
def mpc_config():
    """Load MPC configuration."""
    return MPCConfig.from_yaml('config/simulation/mpc_params.yaml')


@pytest.fixture
def discrete_dynamics(robot_params, mpc_config):
    """Get discretized dynamics."""
    eq_state, eq_control = compute_equilibrium_state(robot_params)
    linearized = linearize_at_equilibrium(robot_params, eq_state, eq_control)
    return discretize_linear_dynamics(
        linearized.state_matrix,
        linearized.control_matrix,
        mpc_config.sampling_period_s,
    )


@pytest.fixture
def mpc_solver(mpc_config, discrete_dynamics):
    """Create configured MPC solver."""
    Q = mpc_config.state_cost_matrix
    R = mpc_config.control_cost_matrix
    P = compute_terminal_cost_dare(
        discrete_dynamics.state_matrix_discrete,
        discrete_dynamics.control_matrix_discrete,
        Q, R,
    )
    state_constraints, input_constraints = create_constraints_from_config(
        mpc_config.pitch_limit_rad,
        mpc_config.pitch_rate_limit_radps,
        mpc_config.control_limit_nm,
    )

    return LinearMPCSolver(
        prediction_horizon_steps=mpc_config.prediction_horizon_steps,
        discrete_dynamics=discrete_dynamics,
        state_cost=Q,
        control_cost=R,
        terminal_cost=P,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
    )


class TestMPCSolverCreation:
    """Tests for MPC solver creation."""

    def test_creates_successfully(self, mpc_solver):
        """Test that solver creates without error."""
        assert mpc_solver is not None
        assert mpc_solver.prediction_horizon_steps == 20


class TestMPCSolverSolve:
    """Tests for MPC solver solve method."""

    def test_returns_mpc_solution(self, mpc_solver):
        """Test that solve returns MPCSolution."""
        current_state = np.zeros(STATE_DIMENSION)
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        assert isinstance(solution, MPCSolution)

    def test_optimal_control_shape(self, mpc_solver):
        """Test that optimal control has correct shape."""
        current_state = np.zeros(STATE_DIMENSION)
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        assert solution.optimal_control.shape == (CONTROL_DIMENSION,)

    def test_predicted_trajectory_shape(self, mpc_solver, mpc_config):
        """Test that predicted trajectory has correct shape."""
        current_state = np.zeros(STATE_DIMENSION)
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        expected_shape = (mpc_config.prediction_horizon_steps + 1, STATE_DIMENSION)
        assert solution.predicted_trajectory.shape == expected_shape

    def test_control_sequence_shape(self, mpc_solver, mpc_config):
        """Test that control sequence has correct shape."""
        current_state = np.zeros(STATE_DIMENSION)
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        expected_shape = (mpc_config.prediction_horizon_steps, CONTROL_DIMENSION)
        assert solution.control_sequence.shape == expected_shape

    def test_at_equilibrium_returns_zero_control(self, mpc_solver):
        """Test that at equilibrium, optimal control is zero."""
        current_state = np.zeros(STATE_DIMENSION)
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        # At equilibrium with zero reference, control should be near zero
        assert np.allclose(solution.optimal_control, 0, atol=1e-4)

    def test_positive_pitch_produces_positive_torque(self, mpc_solver):
        """Test that positive pitch produces positive (corrective) torque."""
        current_state = np.zeros(STATE_DIMENSION)
        current_state[1] = 0.1  # Positive pitch (tilted forward)
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        # Both torques should be positive to push robot backward
        assert solution.optimal_control[0] > 0
        assert solution.optimal_control[1] > 0

    def test_control_respects_bounds(self, mpc_solver, mpc_config):
        """Test that control respects input constraints."""
        current_state = np.zeros(STATE_DIMENSION)
        current_state[1] = 0.5  # Large pitch perturbation
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        control_limit = mpc_config.control_limit_nm
        assert np.all(solution.optimal_control >= -control_limit - 1e-6)
        assert np.all(solution.optimal_control <= control_limit + 1e-6)

    def test_solver_status_optimal(self, mpc_solver):
        """Test that solver returns optimal status for feasible problem."""
        current_state = np.zeros(STATE_DIMENSION)
        current_state[1] = 0.1
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        assert solution.solver_status == 'optimal'

    def test_solve_time_recorded(self, mpc_solver):
        """Test that solve time is recorded."""
        current_state = np.zeros(STATE_DIMENSION)
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        assert solution.solve_time_s > 0

    def test_cost_is_finite(self, mpc_solver):
        """Test that cost is finite for feasible problem."""
        current_state = np.zeros(STATE_DIMENSION)
        current_state[1] = 0.1
        reference = np.zeros(STATE_DIMENSION)

        solution = mpc_solver.solve(current_state, reference)

        assert np.isfinite(solution.cost)
        assert solution.cost >= 0


class TestMPCSolverWarmStart:
    """Tests for MPC solver warm-starting."""

    def test_warm_start_reduces_solve_time(self, mpc_solver):
        """Test that warm-starting reduces solve time."""
        current_state = np.zeros(STATE_DIMENSION)
        current_state[1] = 0.1
        reference = np.zeros(STATE_DIMENSION)

        # First solve (cold start)
        solution1 = mpc_solver.solve(current_state, reference)

        # Second solve (warm start)
        current_state[1] = 0.09  # Slightly different
        solution2 = mpc_solver.solve(current_state, reference)

        # Warm-started solve should be faster
        # Note: First solve includes IPOPT initialization overhead
        assert solution2.solve_time_s < solution1.solve_time_s


class TestMPCSolverReferenceHandling:
    """Tests for reference trajectory handling."""

    def test_constant_reference(self, mpc_solver):
        """Test that constant reference works."""
        current_state = np.zeros(STATE_DIMENSION)
        reference = np.zeros(STATE_DIMENSION)  # Shape (6,)

        solution = mpc_solver.solve(current_state, reference)

        assert solution.solver_status == 'optimal'

    def test_trajectory_reference(self, mpc_solver, mpc_config):
        """Test that time-varying reference works."""
        current_state = np.zeros(STATE_DIMENSION)
        horizon = mpc_config.prediction_horizon_steps

        # Time-varying reference
        reference = np.zeros((horizon + 1, STATE_DIMENSION))
        reference[:, 3] = 0.1  # Constant velocity reference

        solution = mpc_solver.solve(current_state, reference)

        assert solution.solver_status == 'optimal'

    def test_wrong_reference_shape_raises(self, mpc_solver):
        """Test that wrong reference shape raises ValueError."""
        current_state = np.zeros(STATE_DIMENSION)
        reference = np.zeros((5, STATE_DIMENSION))  # Wrong horizon

        with pytest.raises(ValueError, match="reference_trajectory must have shape"):
            mpc_solver.solve(current_state, reference)


class TestMPCSolverInputValidation:
    """Tests for input validation."""

    def test_wrong_state_shape_raises(self, mpc_solver):
        """Test that wrong state shape raises ValueError."""
        current_state = np.zeros(4)  # Wrong shape
        reference = np.zeros(STATE_DIMENSION)

        with pytest.raises(ValueError, match="current_state must have shape"):
            mpc_solver.solve(current_state, reference)


class TestMPCSolverTerminalConstraints:
    """Tests for MPC solver terminal constraints."""

    def test_terminal_constraints_enabled(self, mpc_config, discrete_dynamics):
        """Test that terminal constraints are applied when specified."""
        Q = mpc_config.state_cost_matrix
        R = mpc_config.control_cost_matrix
        P = compute_terminal_cost_dare(
            discrete_dynamics.state_matrix_discrete,
            discrete_dynamics.control_matrix_discrete,
            Q, R,
        )
        state_constraints, input_constraints = create_constraints_from_config(
            mpc_config.pitch_limit_rad,
            mpc_config.pitch_rate_limit_radps,
            mpc_config.control_limit_nm,
        )

        # Create solver with terminal constraints
        solver = LinearMPCSolver(
            prediction_horizon_steps=mpc_config.prediction_horizon_steps,
            discrete_dynamics=discrete_dynamics,
            state_cost=Q,
            control_cost=R,
            terminal_cost=P,
            state_constraints=state_constraints,
            input_constraints=input_constraints,
            terminal_pitch_limit_rad=0.0,
            terminal_pitch_rate_limit_radps=0.0,
        )

        # Solve from perturbed state
        current_state = np.zeros(STATE_DIMENSION)
        current_state[PITCH_INDEX] = 0.1  # 0.1 rad pitch
        current_state[PITCH_RATE_INDEX] = 0.5  # 0.5 rad/s pitch rate
        reference = np.zeros(STATE_DIMENSION)

        solution = solver.solve(current_state, reference)

        # Check that terminal state has pitch near zero
        terminal_pitch = solution.predicted_trajectory[-1, PITCH_INDEX]
        terminal_pitch_rate = solution.predicted_trajectory[-1, PITCH_RATE_INDEX]

        assert abs(terminal_pitch) < 1e-3, f"Terminal pitch should be near 0, got {terminal_pitch}"
        assert abs(terminal_pitch_rate) < 1e-3, f"Terminal pitch rate should be near 0, got {terminal_pitch_rate}"

    def test_terminal_constraints_none(self, mpc_config, discrete_dynamics):
        """Test backward compatibility when terminal constraints are None."""
        Q = mpc_config.state_cost_matrix
        R = mpc_config.control_cost_matrix
        P = compute_terminal_cost_dare(
            discrete_dynamics.state_matrix_discrete,
            discrete_dynamics.control_matrix_discrete,
            Q, R,
        )
        state_constraints, input_constraints = create_constraints_from_config(
            mpc_config.pitch_limit_rad,
            mpc_config.pitch_rate_limit_radps,
            mpc_config.control_limit_nm,
        )

        # Create solver without terminal constraints
        solver = LinearMPCSolver(
            prediction_horizon_steps=mpc_config.prediction_horizon_steps,
            discrete_dynamics=discrete_dynamics,
            state_cost=Q,
            control_cost=R,
            terminal_cost=P,
            state_constraints=state_constraints,
            input_constraints=input_constraints,
            terminal_pitch_limit_rad=None,
            terminal_pitch_rate_limit_radps=None,
        )

        # Should solve without error
        current_state = np.zeros(STATE_DIMENSION)
        reference = np.zeros(STATE_DIMENSION)

        solution = solver.solve(current_state, reference)
        assert solution.solver_status == 'optimal'

    def test_terminal_pitch_rate_nonzero(self, mpc_config, discrete_dynamics):
        """Test terminal constraints with non-zero pitch rate limit."""
        Q = mpc_config.state_cost_matrix
        R = mpc_config.control_cost_matrix
        P = compute_terminal_cost_dare(
            discrete_dynamics.state_matrix_discrete,
            discrete_dynamics.control_matrix_discrete,
            Q, R,
        )
        state_constraints, input_constraints = create_constraints_from_config(
            mpc_config.pitch_limit_rad,
            mpc_config.pitch_rate_limit_radps,
            mpc_config.control_limit_nm,
        )

        # Create solver with terminal pitch=0 but pitch_rate allowed up to 1.0
        solver = LinearMPCSolver(
            prediction_horizon_steps=mpc_config.prediction_horizon_steps,
            discrete_dynamics=discrete_dynamics,
            state_cost=Q,
            control_cost=R,
            terminal_cost=P,
            state_constraints=state_constraints,
            input_constraints=input_constraints,
            terminal_pitch_limit_rad=0.0,
            terminal_pitch_rate_limit_radps=1.0,
        )

        # Solve from perturbed state
        current_state = np.zeros(STATE_DIMENSION)
        current_state[PITCH_INDEX] = 0.1
        reference = np.zeros(STATE_DIMENSION)

        solution = solver.solve(current_state, reference)

        # Terminal pitch should be near zero
        terminal_pitch = solution.predicted_trajectory[-1, PITCH_INDEX]
        assert abs(terminal_pitch) < 1e-3

        # Terminal pitch rate should respect the limit
        terminal_pitch_rate = solution.predicted_trajectory[-1, PITCH_RATE_INDEX]
        assert abs(terminal_pitch_rate) <= 1.0 + 1e-6

    def test_terminal_constraints_from_config(self):
        """Test that terminal constraints can be loaded from config."""
        config = MPCConfig.from_yaml('config/simulation/mpc_params.yaml')

        # Check that terminal constraints are loaded
        assert config.terminal_pitch_limit_rad is not None
        assert config.terminal_pitch_rate_limit_radps is not None
        assert config.terminal_pitch_limit_rad == 0.0
        assert config.terminal_pitch_rate_limit_radps == 0.0

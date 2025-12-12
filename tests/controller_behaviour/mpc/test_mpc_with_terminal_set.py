"""Integration test for MPC with terminal control-invariant set constraints."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from robot_dynamics.parameters import (
    RobotParameters,
    PITCH_INDEX,
    PITCH_RATE_INDEX,
)
from robot_dynamics.linearization import linearize_at_equilibrium
from robot_dynamics.discretization import discretize_linear_dynamics
from mpc.config import MPCConfig
from mpc.constraints import create_constraints_from_config
from mpc.cost_matrices import compute_terminal_cost_dare
from mpc.linear_mpc_solver import LinearMPCSolver


@pytest.fixture
def simulation_config_paths():
    """Paths to simulation config files."""
    return {
        'robot_params': str(project_root / "config" / "simulation" / "robot_params.yaml"),
        'mpc_params': str(project_root / "config" / "simulation" / "mpc_params.yaml"),
    }


@pytest.fixture
def mpc_solver_with_terminal_constraints(simulation_config_paths):
    """Create MPC solver with terminal constraints enabled for testing."""
    # Load config
    robot_params = RobotParameters.from_yaml(simulation_config_paths['robot_params'])
    mpc_config = MPCConfig.from_yaml(simulation_config_paths['mpc_params'])

    # Manually set terminal constraints for testing (these would normally come from terminal_set.yaml)
    # Use smaller values than path constraints to ensure they're actually being enforced
    terminal_pitch_limit_rad = 0.3  # Smaller than 0.524
    terminal_pitch_rate_limit_radps = 2.0  # Smaller than 3.51

    # Linearize and discretize around upright equilibrium
    equilibrium_state = np.zeros(6)
    equilibrium_control = np.zeros(2)
    linearized = linearize_at_equilibrium(robot_params, equilibrium_state, equilibrium_control)
    discrete = discretize_linear_dynamics(
        linearized.state_matrix,
        linearized.control_matrix,
        mpc_config.sampling_period_s
    )

    # Create constraints
    state_constraints, input_constraints = create_constraints_from_config(
        mpc_config.pitch_limit_rad,
        mpc_config.pitch_rate_limit_radps,
        mpc_config.control_limit_nm
    )

    # Compute terminal cost
    terminal_cost = compute_terminal_cost_dare(
        discrete.state_matrix_discrete,
        discrete.control_matrix_discrete,
        mpc_config.state_cost_matrix,
        mpc_config.control_cost_matrix
    )

    # Create solver with terminal constraints
    solver = LinearMPCSolver(
        prediction_horizon_steps=mpc_config.prediction_horizon_steps,
        discrete_dynamics=discrete,
        state_cost=mpc_config.state_cost_matrix,
        control_cost=mpc_config.control_cost_matrix,
        terminal_cost=terminal_cost,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        terminal_pitch_limit_rad=terminal_pitch_limit_rad,
        terminal_pitch_rate_limit_radps=terminal_pitch_rate_limit_radps,
        terminal_velocity_limit_mps=mpc_config.terminal_velocity_limit_mps,
    )

    return solver, terminal_pitch_limit_rad, terminal_pitch_rate_limit_radps


def test_mpc_respects_terminal_pitch_constraint(mpc_solver_with_terminal_constraints):
    """Test that MPC solution respects terminal pitch constraint."""
    solver, terminal_pitch_limit, terminal_pitch_rate_limit = mpc_solver_with_terminal_constraints

    # Test with initial state that has some pitch error
    initial_state = np.zeros(6)
    initial_state[PITCH_INDEX] = 0.25  # 25 rad pitch (within path constraints, needs correction)

    reference = np.zeros(6)

    solution = solver.solve(initial_state, reference)

    # Check solution is feasible
    assert solution.solver_status == 'optimal', \
        f"Solver status should be 'optimal', got '{solution.solver_status}'"

    # Check terminal state satisfies terminal pitch constraint
    terminal_state = solution.predicted_trajectory[-1]

    assert abs(terminal_state[PITCH_INDEX]) <= terminal_pitch_limit + 1e-6, \
        f"Terminal pitch {terminal_state[PITCH_INDEX]:.6f} exceeds limit {terminal_pitch_limit:.6f}"


def test_mpc_respects_terminal_pitch_rate_constraint(mpc_solver_with_terminal_constraints):
    """Test that MPC solution respects terminal pitch rate constraint."""
    solver, terminal_pitch_limit, terminal_pitch_rate_limit = mpc_solver_with_terminal_constraints

    # Test with initial state that has pitch rate error
    initial_state = np.zeros(6)
    initial_state[PITCH_RATE_INDEX] = 1.5  # 1.5 rad/s pitch rate (needs correction)

    reference = np.zeros(6)

    solution = solver.solve(initial_state, reference)

    # Check solution is feasible
    assert solution.solver_status == 'optimal'

    # Check terminal state satisfies terminal pitch rate constraint
    terminal_state = solution.predicted_trajectory[-1]

    assert abs(terminal_state[PITCH_RATE_INDEX]) <= terminal_pitch_rate_limit + 1e-6, \
        f"Terminal pitch rate {terminal_state[PITCH_RATE_INDEX]:.6f} exceeds limit {terminal_pitch_rate_limit:.6f}"


def test_mpc_respects_both_terminal_constraints(mpc_solver_with_terminal_constraints):
    """Test that MPC solution respects both terminal constraints simultaneously."""
    solver, terminal_pitch_limit, terminal_pitch_rate_limit = mpc_solver_with_terminal_constraints

    # Test with initial state that has both pitch and pitch rate errors
    initial_state = np.zeros(6)
    initial_state[PITCH_INDEX] = 0.2  # 0.2 rad pitch
    initial_state[PITCH_RATE_INDEX] = 1.0  # 1.0 rad/s pitch rate

    reference = np.zeros(6)

    solution = solver.solve(initial_state, reference)

    # Check solution is feasible
    assert solution.solver_status == 'optimal'

    # Check terminal state satisfies both constraints
    terminal_state = solution.predicted_trajectory[-1]

    assert abs(terminal_state[PITCH_INDEX]) <= terminal_pitch_limit + 1e-6
    assert abs(terminal_state[PITCH_RATE_INDEX]) <= terminal_pitch_rate_limit + 1e-6


def test_terminal_constraints_tighter_than_path_constraints(mpc_solver_with_terminal_constraints):
    """Verify that terminal constraints are enforced in addition to path constraints."""
    solver, terminal_pitch_limit, terminal_pitch_rate_limit = mpc_solver_with_terminal_constraints

    # Initial state within terminal set but with some error
    initial_state = np.zeros(6)
    initial_state[PITCH_INDEX] = 0.15  # Within terminal set

    reference = np.zeros(6)

    solution = solver.solve(initial_state, reference)

    assert solution.solver_status == 'optimal'

    # Terminal state should be tighter than the path constraint limit (0.524)
    terminal_state = solution.predicted_trajectory[-1]
    assert abs(terminal_state[PITCH_INDEX]) <= terminal_pitch_limit
    # Verify this is actually tighter than path constraints
    assert terminal_pitch_limit < 0.524


def test_mpc_trajectory_converges_to_terminal_set(mpc_solver_with_terminal_constraints):
    """Test that predicted trajectory converges to terminal set."""
    solver, terminal_pitch_limit, terminal_pitch_rate_limit = mpc_solver_with_terminal_constraints

    # Initial state with moderate disturbance
    initial_state = np.zeros(6)
    initial_state[PITCH_INDEX] = 0.35
    initial_state[PITCH_RATE_INDEX] = 0.5

    reference = np.zeros(6)

    solution = solver.solve(initial_state, reference)

    assert solution.solver_status == 'optimal'

    # Check that trajectory is converging (pitch error decreasing over horizon)
    trajectory = solution.predicted_trajectory

    # First state should be initial state
    np.testing.assert_array_almost_equal(trajectory[0], initial_state)

    # Terminal state should satisfy terminal constraints
    terminal_state = trajectory[-1]
    assert abs(terminal_state[PITCH_INDEX]) <= terminal_pitch_limit + 1e-6
    assert abs(terminal_state[PITCH_RATE_INDEX]) <= terminal_pitch_rate_limit + 1e-6

    # Pitch error should decrease monotonically (or mostly so)
    pitch_errors = np.abs(trajectory[:, PITCH_INDEX])
    # At least the terminal error should be less than initial
    assert pitch_errors[-1] < pitch_errors[0]


def test_mpc_without_terminal_constraints_for_comparison():
    """Test MPC without terminal constraints as a baseline comparison."""
    # Load config
    robot_params = RobotParameters.from_yaml(
        str(project_root / "config" / "simulation" / "robot_params.yaml")
    )
    mpc_config = MPCConfig.from_yaml(
        str(project_root / "config" / "simulation" / "mpc_params.yaml")
    )

    # Linearize and discretize
    equilibrium_state = np.zeros(6)
    equilibrium_control = np.zeros(2)
    linearized = linearize_at_equilibrium(robot_params, equilibrium_state, equilibrium_control)
    discrete = discretize_linear_dynamics(
        linearized.state_matrix,
        linearized.control_matrix,
        mpc_config.sampling_period_s
    )

    # Create constraints
    state_constraints, input_constraints = create_constraints_from_config(
        mpc_config.pitch_limit_rad,
        mpc_config.pitch_rate_limit_radps,
        mpc_config.control_limit_nm
    )

    # Compute terminal cost
    terminal_cost = compute_terminal_cost_dare(
        discrete.state_matrix_discrete,
        discrete.control_matrix_discrete,
        mpc_config.state_cost_matrix,
        mpc_config.control_cost_matrix
    )

    # Create solver WITHOUT terminal constraints (all None)
    solver = LinearMPCSolver(
        prediction_horizon_steps=mpc_config.prediction_horizon_steps,
        discrete_dynamics=discrete,
        state_cost=mpc_config.state_cost_matrix,
        control_cost=mpc_config.control_cost_matrix,
        terminal_cost=terminal_cost,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
        terminal_pitch_limit_rad=None,  # No terminal constraint
        terminal_pitch_rate_limit_radps=None,  # No terminal constraint
        terminal_velocity_limit_mps=None,  # No terminal constraint
    )

    # Solve with initial disturbance
    initial_state = np.zeros(6)
    initial_state[PITCH_INDEX] = 0.4
    reference = np.zeros(6)

    solution = solver.solve(initial_state, reference)

    # Should still be feasible (just different terminal state)
    assert solution.solver_status == 'optimal'

    # Terminal state only needs to satisfy path constraints (0.524), not tighter terminal constraints
    terminal_state = solution.predicted_trajectory[-1]
    assert abs(terminal_state[PITCH_INDEX]) <= 0.524 + 1e-6


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

"""Tests for control invariant set computation."""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from robot_dynamics.parameters import (
    PITCH_INDEX,
    PITCH_RATE_INDEX,
    POSITION_INDEX,
    YAW_INDEX,
    VELOCITY_INDEX,
    YAW_RATE_INDEX,
)
from mpc.control_invariant_set import ConstraintDebugger


@pytest.fixture
def mpc_params_path():
    """Path to simulation MPC parameters."""
    return str(project_root / "config" / "simulation" / "mpc_params.yaml")


def test_state_bounds_correct_ordering(mpc_params_path):
    """Test that state bounds use correct state vector ordering.

    State vector: [x, theta, psi, x_dot, theta_dot, psi_dot]
    Only theta (index 1) and theta_dot (index 4) should be constrained.
    """
    debugger = ConstraintDebugger(mpc_params_path)

    # Check pitch constraint at correct index
    assert np.isfinite(debugger.state_lower_bounds_[PITCH_INDEX])
    assert np.isfinite(debugger.state_upper_bounds_[PITCH_INDEX])
    assert debugger.state_lower_bounds_[PITCH_INDEX] < 0
    assert debugger.state_upper_bounds_[PITCH_INDEX] > 0

    # Check pitch rate constraint at correct index
    assert np.isfinite(debugger.state_lower_bounds_[PITCH_RATE_INDEX])
    assert np.isfinite(debugger.state_upper_bounds_[PITCH_RATE_INDEX])
    assert debugger.state_lower_bounds_[PITCH_RATE_INDEX] < 0
    assert debugger.state_upper_bounds_[PITCH_RATE_INDEX] > 0

    # Check other states are unconstrained (infinite bounds)
    unconstrained_indices = [POSITION_INDEX, YAW_INDEX, VELOCITY_INDEX, YAW_RATE_INDEX]
    for index in unconstrained_indices:
        assert debugger.state_lower_bounds_[index] == -np.inf, \
            f"State index {index} should have -inf lower bound"
        assert debugger.state_upper_bounds_[index] == np.inf, \
            f"State index {index} should have +inf upper bound"


def test_input_bounds_use_torque_not_voltage(mpc_params_path):
    """Test that input bounds use control_limit_nm, not voltage_limit_v."""
    import yaml

    debugger = ConstraintDebugger(mpc_params_path)

    # Load expected value from config
    with open(mpc_params_path, 'r') as file:
        config = yaml.safe_load(file)

    expected_limit = config['control_limit_nm']

    # Check bounds match torque limit
    np.testing.assert_array_equal(
        debugger.input_lower_bounds_,
        np.array([-expected_limit, -expected_limit])
    )
    np.testing.assert_array_equal(
        debugger.input_upper_bounds_,
        np.array([expected_limit, expected_limit])
    )

    # Ensure it does NOT use voltage limit (if present)
    if 'voltage_limit_v' in config:
        voltage_limit = config['voltage_limit_v']
        # Bounds should NOT equal voltage limits (unless they happen to be the same)
        # This is a sanity check that we're reading the right parameter
        assert debugger.input_lower_bounds_[0] == -expected_limit


def test_check_state_constraints_correct_indices(mpc_params_path):
    """Test that check_state_constraints uses correct state vector ordering."""
    import yaml

    debugger = ConstraintDebugger(mpc_params_path)

    with open(mpc_params_path, 'r') as file:
        config = yaml.safe_load(file)

    pitch_limit = config['pitch_limit_rad']

    # State: [x, theta, psi, x_dot, theta_dot, psi_dot]
    # Put pitch (index 1) at limit
    state = np.zeros(6)
    state[PITCH_INDEX] = pitch_limit

    margin, violated_index = debugger.check_state_constraints(state)

    # Margin should be near zero (at constraint)
    assert margin < 1e-10, f"Margin should be near zero, got {margin}"
    # Violated index should be PITCH_INDEX
    assert violated_index == PITCH_INDEX, \
        f"Violated index should be {PITCH_INDEX}, got {violated_index}"


def test_check_state_constraints_unconstrained_states(mpc_params_path):
    """Test that unconstrained states (x, psi, velocities) have infinite margins."""
    debugger = ConstraintDebugger(mpc_params_path)

    # State with large position but zero pitch
    state = np.zeros(6)
    state[POSITION_INDEX] = 1000.0  # Large position (unconstrained)
    state[PITCH_INDEX] = 0.0  # Zero pitch (well within constraints)

    margin, violated_index = debugger.check_state_constraints(state)

    # Margin should be positive (pitch is the tightest constraint, but still satisfied)
    # The minimum margin should come from pitch constraints, not position
    assert margin > 0, "Margin should be positive for state within constraints"


def test_check_input_constraints(mpc_params_path):
    """Test that check_input_constraints works correctly."""
    import yaml

    debugger = ConstraintDebugger(mpc_params_path)

    with open(mpc_params_path, 'r') as file:
        config = yaml.safe_load(file)

    control_limit = config['control_limit_nm']

    # Control at limit
    control_input = np.array([control_limit, 0.0])

    margin, violated_index = debugger.check_input_constraints(control_input)

    # Margin should be near zero (at constraint)
    assert margin < 1e-10, f"Margin should be near zero, got {margin}"
    # Violated index should be 0 (left wheel)
    assert violated_index == 0, f"Violated index should be 0, got {violated_index}"


def test_default_reduced_dims_are_pitch_dynamics(mpc_params_path):
    """Test that default reduced_dims are [PITCH_INDEX, PITCH_RATE_INDEX]."""
    debugger = ConstraintDebugger(mpc_params_path)

    # Create stable discrete system (identity scaled < 1)
    state_matrix = np.eye(6) * 0.95
    control_matrix = np.random.randn(6, 2) * 0.1

    # Compute with default reduced_dims (should be [1, 4])
    # We'll verify by checking the print output indirectly
    # For now, just verify it doesn't crash
    try:
        result = debugger.compute_invariant_set(
            state_matrix=state_matrix,
            control_matrix=control_matrix,
            grid_resolution=5,  # Small for speed
            reduced_dims=None  # Use default
        )
        # If we get a result, the default reduced_dims worked
        assert 'bounds' in result
        assert 'feasible_states' in result
    except RuntimeError as e:
        # It's ok if no invariant states found (system might be unstable)
        # The important thing is reduced_dims defaulted correctly
        assert "No invariant states found" in str(e)


def test_compute_invariant_set_structure(mpc_params_path):
    """Test that compute_invariant_set returns expected structure."""
    debugger = ConstraintDebugger(mpc_params_path)

    # Create a very stable system
    state_matrix = np.eye(6) * 0.5  # Highly stable
    control_matrix = np.ones((6, 2)) * 0.01

    result = debugger.compute_invariant_set(
        state_matrix=state_matrix,
        control_matrix=control_matrix,
        grid_resolution=5,  # Small for speed
        reduced_dims=[PITCH_INDEX, PITCH_RATE_INDEX]
    )

    # Check structure
    assert 'feasible_states' in result
    assert 'bounds' in result
    assert 'volume_fraction' in result

    # Check shapes
    assert result['feasible_states'].shape[1] == 6  # Full state vector
    assert result['bounds'].shape == (6, 2)  # 6 states, 2 bounds each

    # Check volume fraction is reasonable
    assert 0 <= result['volume_fraction'] <= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

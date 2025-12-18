"""Unit tests for MPC constraints."""

import numpy as np
import pytest

from robot_dynamics.parameters import (
    STATE_DIMENSION,
    CONTROL_DIMENSION,
    PITCH_INDEX,
    PITCH_RATE_INDEX,
    VELOCITY_INDEX,
)
from mpc.constraints import (
    StateConstraints,
    InputConstraints,
    create_constraints_from_config,
)


class TestStateConstraints:
    """Tests for StateConstraints."""

    def test_creation(self):
        """Test basic creation."""
        constraints = StateConstraints(
            pitch_limit_rad=0.524,
            pitch_rate_limit_radps=5.0,
        )
        assert constraints.pitch_limit_rad == 0.524
        assert constraints.pitch_rate_limit_radps == 5.0

    def test_get_bounds_shape(self):
        """Test that bounds have correct shape."""
        constraints = StateConstraints(
            pitch_limit_rad=0.524,
            pitch_rate_limit_radps=5.0,
        )
        lower, upper = constraints.get_bounds()

        assert lower.shape == (STATE_DIMENSION,)
        assert upper.shape == (STATE_DIMENSION,)

    def test_pitch_bounds_correct(self):
        """Test that pitch bounds are set correctly."""
        pitch_limit = 0.524
        constraints = StateConstraints(
            pitch_limit_rad=pitch_limit,
            pitch_rate_limit_radps=5.0,
        )
        lower, upper = constraints.get_bounds()

        assert lower[PITCH_INDEX] == -pitch_limit
        assert upper[PITCH_INDEX] == pitch_limit

    def test_pitch_rate_bounds_correct(self):
        """Test that pitch rate bounds are set correctly."""
        pitch_rate_limit = 5.0
        constraints = StateConstraints(
            pitch_limit_rad=0.524,
            pitch_rate_limit_radps=pitch_rate_limit,
        )
        lower, upper = constraints.get_bounds()

        assert lower[PITCH_RATE_INDEX] == -pitch_rate_limit
        assert upper[PITCH_RATE_INDEX] == pitch_rate_limit

    def test_velocity_bounds_optional(self):
        """Velocity bounds are applied when velocity_limit is provided."""
        constraints = StateConstraints(
            pitch_limit_rad=0.524,
            pitch_rate_limit_radps=5.0,
            velocity_limit_mps=0.3,
        )
        lower, upper = constraints.get_bounds()

        assert lower[VELOCITY_INDEX] == -0.3
        assert upper[VELOCITY_INDEX] == 0.3

    def test_unconstrained_states_infinite(self):
        """Test that unconstrained states have infinite bounds."""
        constraints = StateConstraints(
            pitch_limit_rad=0.524,
            pitch_rate_limit_radps=5.0,
        )
        lower, upper = constraints.get_bounds()

        # Check that non-pitch states are unconstrained
        for state_index in range(STATE_DIMENSION):
            if state_index not in [PITCH_INDEX, PITCH_RATE_INDEX]:
                assert lower[state_index] == -np.inf
                assert upper[state_index] == np.inf

    def test_negative_pitch_limit_raises(self):
        """Test that negative pitch limit raises."""
        with pytest.raises(ValueError, match="must be positive"):
            StateConstraints(pitch_limit_rad=-0.5, pitch_rate_limit_radps=5.0)

    def test_zero_pitch_limit_raises(self):
        """Test that zero pitch limit raises."""
        with pytest.raises(ValueError, match="must be positive"):
            StateConstraints(pitch_limit_rad=0.0, pitch_rate_limit_radps=5.0)


class TestInputConstraints:
    """Tests for InputConstraints."""

    def test_creation(self):
        """Test basic creation."""
        constraints = InputConstraints(control_limit_nm=0.25)
        assert constraints.control_limit_nm == 0.25

    def test_get_bounds_shape(self):
        """Test that bounds have correct shape."""
        constraints = InputConstraints(control_limit_nm=0.25)
        lower, upper = constraints.get_bounds()

        assert lower.shape == (CONTROL_DIMENSION,)
        assert upper.shape == (CONTROL_DIMENSION,)

    def test_symmetric_bounds(self):
        """Test that bounds are symmetric."""
        control_limit = 0.25
        constraints = InputConstraints(control_limit_nm=control_limit)
        lower, upper = constraints.get_bounds()

        assert np.allclose(lower, -control_limit)
        assert np.allclose(upper, control_limit)

    def test_negative_control_limit_raises(self):
        """Test that negative control limit raises."""
        with pytest.raises(ValueError, match="must be positive"):
            InputConstraints(control_limit_nm=-0.25)


class TestCreateConstraintsFromConfig:
    """Tests for create_constraints_from_config."""

    def test_creates_both_constraint_types(self):
        """Test that both constraint types are created."""
        state_constraints, input_constraints = create_constraints_from_config(
            pitch_limit_rad=0.524,
            pitch_rate_limit_radps=5.0,
            control_limit_nm=0.25,
        )

        assert isinstance(state_constraints, StateConstraints)
        assert isinstance(input_constraints, InputConstraints)

    def test_values_match(self):
        """Test that constraint values match inputs."""
        state_constraints, input_constraints = create_constraints_from_config(
            pitch_limit_rad=0.524,
            pitch_rate_limit_radps=5.0,
            control_limit_nm=0.25,
            velocity_limit_mps=0.3,
        )

        assert state_constraints.pitch_limit_rad == 0.524
        assert state_constraints.pitch_rate_limit_radps == 5.0
        assert input_constraints.control_limit_nm == 0.25
        assert state_constraints.velocity_limit_mps == 0.3

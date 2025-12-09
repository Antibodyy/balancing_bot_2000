"""Unit tests for reference generator."""

import numpy as np
import pytest

from robot_dynamics.parameters import (
    STATE_DIMENSION,
    POSITION_INDEX,
    PITCH_INDEX,
    YAW_INDEX,
    VELOCITY_INDEX,
)
from mpc.reference_generator import (
    ReferenceGenerator,
    ReferenceCommand,
    ReferenceMode,
)


@pytest.fixture
def ref_gen():
    """Create reference generator."""
    return ReferenceGenerator(
        sampling_period_s=0.02,
        prediction_horizon_steps=20,
    )


class TestReferenceGeneratorCreation:
    """Tests for ReferenceGenerator creation."""

    def test_creation(self, ref_gen):
        """Test basic creation."""
        assert ref_gen.sampling_period_s == 0.02
        assert ref_gen.prediction_horizon_steps == 20

    def test_negative_sampling_period_raises(self):
        """Test that negative sampling period raises."""
        with pytest.raises(ValueError):
            ReferenceGenerator(sampling_period_s=-0.02, prediction_horizon_steps=20)

    def test_zero_horizon_raises(self):
        """Test that zero horizon raises."""
        with pytest.raises(ValueError):
            ReferenceGenerator(sampling_period_s=0.02, prediction_horizon_steps=0)


class TestBalanceReference:
    """Tests for balance reference generation."""

    def test_shape(self, ref_gen):
        """Test that output has correct shape."""
        reference = ref_gen.generate_balance_reference()

        assert reference.shape == (21, STATE_DIMENSION)

    def test_all_zeros(self, ref_gen):
        """Test that balance reference is all zeros."""
        reference = ref_gen.generate_balance_reference()

        assert np.allclose(reference, 0)

    def test_via_command(self, ref_gen):
        """Test generation via ReferenceCommand."""
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)
        reference = ref_gen.generate(command)

        assert np.allclose(reference, 0)


class TestVelocityReference:
    """Tests for velocity reference generation."""

    def test_shape(self, ref_gen):
        """Test that output has correct shape."""
        reference = ref_gen.generate_velocity_reference(
            desired_velocity_mps=0.2,
            desired_yaw_rate_radps=0.0,
        )

        assert reference.shape == (21, STATE_DIMENSION)

    def test_velocity_is_set(self, ref_gen):
        """Test that velocity is set correctly."""
        velocity = 0.2
        reference = ref_gen.generate_velocity_reference(
            desired_velocity_mps=velocity,
            desired_yaw_rate_radps=0.0,
        )

        assert np.allclose(reference[:, VELOCITY_INDEX], velocity)

    def test_position_increases(self, ref_gen):
        """Test that position increases over horizon."""
        velocity = 0.2
        reference = ref_gen.generate_velocity_reference(
            desired_velocity_mps=velocity,
            desired_yaw_rate_radps=0.0,
        )

        # Position should increase monotonically
        positions = reference[:, POSITION_INDEX]
        assert np.all(np.diff(positions) >= 0)

        # Final position should match integrated velocity
        expected_final = velocity * 20 * 0.02
        assert np.isclose(positions[-1], expected_final)

    def test_pitch_is_zero(self, ref_gen):
        """Test that pitch reference is zero (upright while moving)."""
        reference = ref_gen.generate_velocity_reference(
            desired_velocity_mps=0.2,
            desired_yaw_rate_radps=0.0,
        )

        assert np.allclose(reference[:, PITCH_INDEX], 0)

    def test_yaw_rate_causes_heading_change(self, ref_gen):
        """Test that yaw rate causes heading to change."""
        yaw_rate = 0.5
        reference = ref_gen.generate_velocity_reference(
            desired_velocity_mps=0.0,
            desired_yaw_rate_radps=yaw_rate,
        )

        # Heading should increase
        headings = reference[:, YAW_INDEX]
        assert np.all(np.diff(headings) >= 0)

    def test_via_command(self, ref_gen):
        """Test generation via ReferenceCommand."""
        command = ReferenceCommand(
            mode=ReferenceMode.VELOCITY,
            velocity_mps=0.2,
            yaw_rate_radps=0.1,
        )
        reference = ref_gen.generate(command)

        assert np.allclose(reference[:, VELOCITY_INDEX], 0.2)


class TestPositionReference:
    """Tests for position reference generation."""

    def test_shape(self, ref_gen):
        """Test that output has correct shape."""
        current_state = np.zeros(STATE_DIMENSION)
        reference = ref_gen.generate_position_reference(
            current_state=current_state,
            target_position_m=0.5,
            target_heading_rad=0.0,
        )

        assert reference.shape == (21, STATE_DIMENSION)

    def test_starts_at_current_position(self, ref_gen):
        """Test that reference starts at current position."""
        current_state = np.zeros(STATE_DIMENSION)
        current_state[POSITION_INDEX] = 0.1

        reference = ref_gen.generate_position_reference(
            current_state=current_state,
            target_position_m=0.5,
            target_heading_rad=0.0,
        )

        assert np.isclose(reference[0, POSITION_INDEX], 0.1)

    def test_approaches_target(self, ref_gen):
        """Test that reference approaches target position."""
        current_state = np.zeros(STATE_DIMENSION)
        target = 0.5

        reference = ref_gen.generate_position_reference(
            current_state=current_state,
            target_position_m=target,
            target_heading_rad=0.0,
        )

        # Position should be moving toward target
        positions = reference[:, POSITION_INDEX]
        assert positions[-1] > positions[0]

    def test_heading_interpolation(self, ref_gen):
        """Test that heading is interpolated."""
        current_state = np.zeros(STATE_DIMENSION)
        target_heading = np.deg2rad(30)

        reference = ref_gen.generate_position_reference(
            current_state=current_state,
            target_position_m=0.5,
            target_heading_rad=target_heading,
        )

        # Heading should end near target
        assert reference[-1, YAW_INDEX] > 0

    def test_via_command_requires_current_state(self, ref_gen):
        """Test that POSITION mode requires current_state."""
        command = ReferenceCommand(
            mode=ReferenceMode.POSITION,
            target_position_m=0.5,
        )

        with pytest.raises(ValueError, match="current_state is required"):
            ref_gen.generate(command, current_state=None)


class TestReferenceCommand:
    """Tests for ReferenceCommand dataclass."""

    def test_default_is_balance(self):
        """Test that default mode is BALANCE."""
        command = ReferenceCommand()
        assert command.mode == ReferenceMode.BALANCE

    def test_velocity_mode(self):
        """Test velocity mode creation."""
        command = ReferenceCommand(
            mode=ReferenceMode.VELOCITY,
            velocity_mps=0.2,
            yaw_rate_radps=0.1,
        )
        assert command.mode == ReferenceMode.VELOCITY
        assert command.velocity_mps == 0.2
        assert command.yaw_rate_radps == 0.1

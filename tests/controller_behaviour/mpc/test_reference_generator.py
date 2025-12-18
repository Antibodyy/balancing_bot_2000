"""Unit tests for reference generator."""

import numpy as np
import pytest

from robot_dynamics.parameters import (
    STATE_DIMENSION,
    POSITION_INDEX,
    PITCH_INDEX,
    YAW_INDEX,
    VELOCITY_INDEX,
    PITCH_RATE_INDEX,
    YAW_RATE_INDEX,
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
        sampling_period_s=0.065,
        prediction_horizon_steps=20,
    )


class TestReferenceGeneratorCreation:
    """Tests for ReferenceGenerator creation."""

    def test_creation(self, ref_gen):
        """Test basic creation."""
        assert ref_gen.sampling_period_s == 0.065
        assert ref_gen.prediction_horizon_steps == 20

    def test_negative_sampling_period_raises(self):
        """Test that negative sampling period raises."""
        with pytest.raises(ValueError):
            ReferenceGenerator(sampling_period_s=-0.065, prediction_horizon_steps=20)

    def test_zero_horizon_raises(self):
        """Test that zero horizon raises."""
        with pytest.raises(ValueError):
            ReferenceGenerator(sampling_period_s=0.065, prediction_horizon_steps=0)


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

    def test_via_command_defaults_to_origin(self, ref_gen):
        """Test generation via ReferenceCommand with default position."""
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)
        reference = ref_gen.generate(command, np.zeros(STATE_DIMENSION))

        assert np.allclose(reference[:, POSITION_INDEX], 0.0)
        assert np.allclose(reference[:, VELOCITY_INDEX], 0.0)

    def test_balance_latches_current_position(self, ref_gen):
        """Balance mode should hold the current position when it begins."""
        balance_command = ReferenceCommand(mode=ReferenceMode.BALANCE)
        current_state = np.zeros(STATE_DIMENSION)
        current_state[POSITION_INDEX] = 0.42

        reference = ref_gen.generate(balance_command, current_state)

        assert np.allclose(reference[:, POSITION_INDEX], 0.42)
        assert np.allclose(reference[:, VELOCITY_INDEX], 0.0)

    def test_balance_hold_persists_until_mode_change(self, ref_gen):
        """Latched balance position should persist while remaining in balance mode."""
        balance_command = ReferenceCommand(mode=ReferenceMode.BALANCE)
        first_state = np.zeros(STATE_DIMENSION)
        first_state[POSITION_INDEX] = 0.15
        ref_gen.generate(balance_command, first_state)

        # Simulate estimator drift while still commanding balance.
        second_state = np.zeros(STATE_DIMENSION)
        second_state[POSITION_INDEX] = 1.25
        reference = ref_gen.generate(balance_command, second_state)

        assert np.allclose(reference[:, POSITION_INDEX], 0.15)
        assert np.allclose(reference[:, VELOCITY_INDEX], 0.0)


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
        expected_final = velocity * 20 * 0.065
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


class TestCircularReference:
    """Tests for circular reference generation."""

    def test_shape(self, ref_gen):
        """Test output shape is correct."""
        reference = ref_gen.generate_circular_reference(
            radius_m=1.0,
            target_velocity_mps=0.2,
        )
        assert reference.shape == (21, STATE_DIMENSION)

    def test_radius_too_small_raises(self, ref_gen):
        """Test that small radius raises ValueError."""
        with pytest.raises(ValueError, match="radius must be >= 0.3"):
            ref_gen.generate_circular_reference(
                radius_m=0.1,
                target_velocity_mps=0.2,
            )

    def test_negative_velocity_raises(self, ref_gen):
        """Test that negative velocity raises."""
        with pytest.raises(ValueError, match="non-negative"):
            ref_gen.generate_circular_reference(
                radius_m=1.0,
                target_velocity_mps=-0.1,
            )

    def test_velocity_is_constant(self, ref_gen):
        """Test that velocity reference is constant."""
        velocity = 0.3
        reference = ref_gen.generate_circular_reference(
            radius_m=1.0,
            target_velocity_mps=velocity,
        )
        assert np.allclose(reference[:, VELOCITY_INDEX], velocity)

    def test_yaw_rate_matches_circle_kinematics(self, ref_gen):
        """Test that yaw rate = v/r."""
        velocity = 0.2
        radius = 0.5
        expected_yaw_rate = velocity / radius  # 0.4 rad/s

        reference = ref_gen.generate_circular_reference(
            radius_m=radius,
            target_velocity_mps=velocity,
        )

        assert np.allclose(reference[:, YAW_RATE_INDEX], expected_yaw_rate)

    def test_clockwise_has_negative_yaw_rate(self, ref_gen):
        """Test clockwise motion has negative yaw rate."""
        reference = ref_gen.generate_circular_reference(
            radius_m=1.0,
            target_velocity_mps=0.2,
            clockwise=True,
        )

        yaw_rate = reference[0, YAW_RATE_INDEX]
        assert yaw_rate < 0  # Clockwise = negative

    def test_heading_increases_over_horizon(self, ref_gen):
        """Test that heading increases for counter-clockwise."""
        reference = ref_gen.generate_circular_reference(
            radius_m=1.0,
            target_velocity_mps=0.2,
            clockwise=False,
        )

        headings = reference[:, YAW_INDEX]
        # For CCW, heading should increase monotonically
        assert np.all(np.diff(headings) >= 0)

    def test_position_follows_sine_curve(self, ref_gen):
        """Test that X position follows x = x_c + r*sin(ψ)."""
        radius = 1.0
        center_x = 2.0

        reference = ref_gen.generate_circular_reference(
            radius_m=radius,
            target_velocity_mps=0.2,
            center_x_m=center_x,
        )

        # Verify: x[k] = x_c + r * sin(ψ[k])
        for step in range(len(reference)):
            expected_x = center_x + radius * np.sin(reference[step, YAW_INDEX])
            actual_x = reference[step, POSITION_INDEX]
            assert np.isclose(expected_x, actual_x, atol=1e-6)

    def test_pitch_is_zero(self, ref_gen):
        """Test balance is maintained (zero pitch)."""
        reference = ref_gen.generate_circular_reference(
            radius_m=1.0,
            target_velocity_mps=0.2,
        )
        assert np.allclose(reference[:, PITCH_INDEX], 0)
        assert np.allclose(reference[:, PITCH_RATE_INDEX], 0)

    def test_via_command(self, ref_gen):
        """Test generation via ReferenceCommand."""
        command = ReferenceCommand(
            mode=ReferenceMode.CIRCULAR,
            radius_m=0.8,
            target_velocity_mps=0.25,
            center_x_m=1.0,
            center_y_m=0.5,
            clockwise=False,
        )

        current_state = np.zeros(STATE_DIMENSION)
        reference = ref_gen.generate(command, current_state=current_state)

        # Verify it's a valid reference
        assert reference.shape == (21, STATE_DIMENSION)
        assert np.allclose(reference[:, VELOCITY_INDEX], 0.25)
        expected_yaw_rate = 0.25 / 0.8
        assert np.allclose(reference[:, YAW_RATE_INDEX], expected_yaw_rate)

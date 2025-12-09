"""Unit tests for state estimation module."""

import numpy as np
import pytest

from state_estimation import EstimatorConfig, ComplementaryFilter, IMUReading
from state_estimation._internal.imu_fusion import (
    pitch_from_accelerometer,
    integrate_gyroscope,
    compute_complementary_filter_alpha,
)


class TestPitchFromAccelerometer:
    """Tests for pitch_from_accelerometer."""

    def test_upright_returns_zero(self):
        """Test that upright robot returns zero pitch."""
        gravity = 9.81
        pitch = pitch_from_accelerometer(
            acceleration_x_mps2=0.0,
            acceleration_z_mps2=gravity,
        )
        assert np.isclose(pitch, 0.0)

    def test_tilted_forward(self):
        """Test pitch when tilted forward."""
        gravity = 9.81
        expected_pitch = np.deg2rad(10)

        pitch = pitch_from_accelerometer(
            acceleration_x_mps2=gravity * np.sin(expected_pitch),
            acceleration_z_mps2=gravity * np.cos(expected_pitch),
        )
        assert np.isclose(pitch, expected_pitch, atol=1e-6)

    def test_tilted_backward(self):
        """Test pitch when tilted backward."""
        gravity = 9.81
        expected_pitch = np.deg2rad(-10)

        pitch = pitch_from_accelerometer(
            acceleration_x_mps2=gravity * np.sin(expected_pitch),
            acceleration_z_mps2=gravity * np.cos(expected_pitch),
        )
        assert np.isclose(pitch, expected_pitch, atol=1e-6)


class TestIntegrateGyroscope:
    """Tests for integrate_gyroscope."""

    def test_zero_rate_unchanged(self):
        """Test that zero rate leaves pitch unchanged."""
        initial_pitch = 0.1
        new_pitch = integrate_gyroscope(
            previous_pitch_rad=initial_pitch,
            pitch_rate_radps=0.0,
            timestep_s=0.02,
        )
        assert new_pitch == initial_pitch

    def test_positive_rate_increases_pitch(self):
        """Test that positive rate increases pitch."""
        initial_pitch = 0.1
        rate = 1.0  # rad/s
        timestep = 0.02

        new_pitch = integrate_gyroscope(
            previous_pitch_rad=initial_pitch,
            pitch_rate_radps=rate,
            timestep_s=timestep,
        )

        expected = initial_pitch + rate * timestep
        assert np.isclose(new_pitch, expected)


class TestComplementaryFilterAlpha:
    """Tests for compute_complementary_filter_alpha."""

    def test_typical_values(self):
        """Test alpha for typical values."""
        alpha = compute_complementary_filter_alpha(
            time_constant_s=0.1,
            sampling_period_s=0.02,
        )
        # alpha = 0.1 / (0.1 + 0.02) = 0.833...
        assert np.isclose(alpha, 0.1 / 0.12)

    def test_large_time_constant_trusts_gyro(self):
        """Test that large time constant gives high alpha (trust gyro)."""
        alpha = compute_complementary_filter_alpha(
            time_constant_s=1.0,
            sampling_period_s=0.02,
        )
        assert alpha > 0.95

    def test_small_time_constant_trusts_accel(self):
        """Test that small time constant gives low alpha (trust accel)."""
        alpha = compute_complementary_filter_alpha(
            time_constant_s=0.01,
            sampling_period_s=0.02,
        )
        assert alpha < 0.5

    def test_negative_time_constant_raises(self):
        """Test that negative time constant raises."""
        with pytest.raises(ValueError):
            compute_complementary_filter_alpha(
                time_constant_s=-0.1,
                sampling_period_s=0.02,
            )


class TestIMUReading:
    """Tests for IMUReading dataclass."""

    def test_creation(self):
        """Test basic creation."""
        reading = IMUReading(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
        )
        assert reading.acceleration_mps2.shape == (3,)
        assert reading.angular_velocity_radps.shape == (3,)

    def test_wrong_accel_shape_raises(self):
        """Test that wrong acceleration shape raises."""
        with pytest.raises(ValueError, match="acceleration_mps2 must have shape"):
            IMUReading(
                acceleration_mps2=np.array([0.0, 9.81]),  # Wrong shape
                angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            )

    def test_wrong_gyro_shape_raises(self):
        """Test that wrong gyroscope shape raises."""
        with pytest.raises(ValueError, match="angular_velocity_radps must have shape"):
            IMUReading(
                acceleration_mps2=np.array([0.0, 0.0, 9.81]),
                angular_velocity_radps=np.array([0.0, 0.0]),  # Wrong shape
            )


class TestComplementaryFilter:
    """Tests for ComplementaryFilter."""

    @pytest.fixture
    def filter_obj(self):
        """Create filter with typical parameters."""
        return ComplementaryFilter(
            time_constant_s=0.1,
            sampling_period_s=0.02,
        )

    def test_creation(self, filter_obj):
        """Test basic creation."""
        assert filter_obj.time_constant_s == 0.1
        assert np.isclose(filter_obj.alpha, 0.1 / 0.12)
        assert not filter_obj.is_initialized

    def test_first_update_initializes_from_accel(self, filter_obj):
        """Test that first update initializes from accelerometer."""
        gravity = 9.81
        expected_pitch = np.deg2rad(10)

        reading = IMUReading(
            acceleration_mps2=np.array([
                gravity * np.sin(expected_pitch),
                0.0,
                gravity * np.cos(expected_pitch),
            ]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
        )

        pitch = filter_obj.update(reading)

        assert filter_obj.is_initialized
        assert np.isclose(pitch, expected_pitch, atol=1e-6)

    def test_subsequent_updates_fuse(self, filter_obj):
        """Test that subsequent updates perform fusion."""
        gravity = 9.81
        pitch_rad = np.deg2rad(10)

        # First update (initialization)
        reading1 = IMUReading(
            acceleration_mps2=np.array([
                gravity * np.sin(pitch_rad),
                0.0,
                gravity * np.cos(pitch_rad),
            ]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
        )
        filter_obj.update(reading1)

        # Second update with gyro indicating rotation
        reading2 = IMUReading(
            acceleration_mps2=np.array([
                gravity * np.sin(pitch_rad),
                0.0,
                gravity * np.cos(pitch_rad),
            ]),
            angular_velocity_radps=np.array([0.0, 0.5, 0.0]),  # Rotating
        )
        pitch2 = filter_obj.update(reading2)

        # Pitch should have increased due to gyro integration
        assert pitch2 > pitch_rad

    def test_reset(self, filter_obj):
        """Test filter reset."""
        # First update
        reading = IMUReading(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
        )
        filter_obj.update(reading)
        assert filter_obj.is_initialized

        # Reset
        filter_obj.reset(initial_pitch_rad=0.5)
        assert not filter_obj.is_initialized
        assert filter_obj.pitch_estimate_rad == 0.5

    def test_filters_noisy_accelerometer(self, filter_obj):
        """Test that filter smooths noisy accelerometer data."""
        gravity = 9.81
        true_pitch = np.deg2rad(10)
        np.random.seed(42)

        # Initialize
        reading = IMUReading(
            acceleration_mps2=np.array([
                gravity * np.sin(true_pitch),
                0.0,
                gravity * np.cos(true_pitch),
            ]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
        )
        filter_obj.update(reading)

        # Apply noisy readings
        pitch_estimates = []
        for _ in range(20):
            noise = np.random.normal(0, 0.5)
            noisy_reading = IMUReading(
                acceleration_mps2=np.array([
                    gravity * np.sin(true_pitch) + noise,
                    0.0,
                    gravity * np.cos(true_pitch),
                ]),
                angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            )
            pitch = filter_obj.update(noisy_reading)
            pitch_estimates.append(pitch)

        # Filter should keep estimate reasonably close to true value
        # despite noisy accelerometer
        assert np.abs(np.mean(pitch_estimates) - true_pitch) < np.deg2rad(5)


class TestEstimatorConfig:
    """Tests for EstimatorConfig."""

    def test_from_yaml(self):
        """Test loading from YAML."""
        config = EstimatorConfig.from_yaml('config/estimator_params.yaml')

        assert config.complementary_filter_time_constant_s > 0
        assert config.sampling_period_s > 0

    def test_negative_time_constant_raises(self):
        """Test that negative time constant raises."""
        with pytest.raises(ValueError):
            EstimatorConfig(
                complementary_filter_time_constant_s=-0.1,
                sampling_period_s=0.02,
            )

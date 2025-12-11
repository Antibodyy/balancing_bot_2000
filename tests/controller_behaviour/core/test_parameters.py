"""Tests for robot parameters module.

Tests YAML loading, validation, computed properties, and immutability.
"""

import pytest
import numpy as np
from robot_dynamics import RobotParameters


def test_yaml_loading():
    """Test loading parameters from YAML config file."""
    params = RobotParameters.from_yaml('config/robot_params.yaml')

    # Verify expected placeholder values
    assert params.body_mass_kg == 1.5
    assert params.wheel_mass_kg == 0.5
    assert params.com_distance_m == 0.2
    assert params.wheel_radius_m == 0.05
    assert params.track_width_m == 0.3
    assert params.gravity_mps2 == 9.81
    assert params.ground_slope_rad == 0.0


def test_positive_mass_validation():
    """Test that negative masses are rejected."""
    with pytest.raises(ValueError, match="body_mass_kg must be positive"):
        RobotParameters(
            body_mass_kg=-1.0,  # Invalid
            wheel_mass_kg=0.5,
            com_distance_m=0.2,
            wheel_radius_m=0.05,
            track_width_m=0.3,
            body_pitch_inertia_kg_m2=0.05,
            body_yaw_inertia_kg_m2=0.01,
            wheel_inertia_kg_m2=0.001
        )


def test_positive_dimension_validation():
    """Test that zero or negative dimensions are rejected."""
    with pytest.raises(ValueError, match="wheel_radius_m must be positive"):
        RobotParameters(
            body_mass_kg=1.5,
            wheel_mass_kg=0.5,
            com_distance_m=0.2,
            wheel_radius_m=0.0,  # Invalid
            track_width_m=0.3,
            body_pitch_inertia_kg_m2=0.05,
            body_yaw_inertia_kg_m2=0.01,
            wheel_inertia_kg_m2=0.001
        )


def test_non_negative_inertia_validation():
    """Test that negative inertias are rejected."""
    with pytest.raises(ValueError, match="wheel_inertia_kg_m2 must be non-negative"):
        RobotParameters(
            body_mass_kg=1.5,
            wheel_mass_kg=0.5,
            com_distance_m=0.2,
            wheel_radius_m=0.05,
            track_width_m=0.3,
            body_pitch_inertia_kg_m2=0.05,
            body_yaw_inertia_kg_m2=0.01,
            wheel_inertia_kg_m2=-0.001  # Invalid
        )


def test_effective_mass_computation():
    """Test computed effective mass property."""
    params = RobotParameters.from_yaml('config/robot_params.yaml')

    # M_eff = M + 2*m (MuJoCo model: wheels are separate bodies with own DOFs)
    # NOTE: Does NOT include 2*J_w/r^2 term (that's for quasi-static wheels)
    expected_effective_mass = (
        params.body_mass_kg +
        2 * params.wheel_mass_kg
    )

    assert abs(params.effective_mass_kg - expected_effective_mass) < 1e-10


def test_effective_pitch_inertia_computation():
    """Test computed effective pitch inertia property."""
    params = RobotParameters.from_yaml('config/robot_params.yaml')

    # I_eff = I_y + m*l^2
    expected_pitch_inertia = (
        params.body_pitch_inertia_kg_m2 +
        params.body_mass_kg * (params.com_distance_m ** 2)
    )

    assert abs(
        params.effective_pitch_inertia_kg_m2 - expected_pitch_inertia
    ) < 1e-10


def test_immutability():
    """Test that parameters are immutable (frozen dataclass)."""
    params = RobotParameters.from_yaml('config/robot_params.yaml')

    with pytest.raises(Exception):  # FrozenInstanceError
        params.body_mass_kg = 2.0

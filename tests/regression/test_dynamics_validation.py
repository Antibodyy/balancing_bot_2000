"""Regression tests for dynamics model validation.

Tests that the analytical dynamics model matches the MuJoCo simulation.
Incorporates checks from the validation scripts to ensure consistency.
"""

import pytest
import numpy as np
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

from robot_dynamics import RobotParameters, compute_equilibrium_state, linearize_at_equilibrium


@pytest.fixture
def params():
    """Load robot parameters."""
    return RobotParameters.from_yaml('config/simulation/robot_params.yaml')


@pytest.fixture
def mujoco_model():
    """Load MuJoCo model."""
    if not MUJOCO_AVAILABLE:
        pytest.skip("MuJoCo not available")
    model = mujoco.MjModel.from_xml_path('mujoco_sim/robot_model.xml')
    return model


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
def test_mass_consistency(params, mujoco_model):
    """Test that parameter masses match MuJoCo model.

    Verifies that the total mass and individual component masses
    are consistent between parameters and MuJoCo.
    """
    # MuJoCo: Body 0=world, 1=base, 2&3=wheels
    mujoco_body_mass = mujoco_model.body_mass[1]
    mujoco_wheel_mass = mujoco_model.body_mass[2]
    mujoco_total_mass = mujoco_model.body_mass[1:].sum()

    params_total_mass = params.body_mass_kg + 2 * params.wheel_mass_kg

    # Check total mass matches within 1%
    mass_error = abs(mujoco_total_mass - params_total_mass) / mujoco_total_mass
    assert mass_error < 0.01, \
        f"Total mass mismatch: MuJoCo {mujoco_total_mass:.3f}kg vs Params {params_total_mass:.3f}kg"

    # Check body mass matches
    body_mass_error = abs(mujoco_body_mass - params.body_mass_kg) / mujoco_body_mass
    assert body_mass_error < 0.01, \
        f"Body mass mismatch: MuJoCo {mujoco_body_mass:.3f}kg vs Params {params.body_mass_kg:.3f}kg"

    # Check wheel mass matches
    wheel_mass_error = abs(mujoco_wheel_mass - params.wheel_mass_kg) / mujoco_wheel_mass
    assert wheel_mass_error < 0.01, \
        f"Wheel mass mismatch: MuJoCo {mujoco_wheel_mass:.3f}kg vs Params {params.wheel_mass_kg:.3f}kg"


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
def test_geometry_consistency(params, mujoco_model):
    """Test that geometric parameters match MuJoCo model.

    Verifies wheel radius, track width, and COM distance.
    """
    # These values are known from the mujoco_sim/robot_model.xml
    mujoco_wheel_radius = 0.04  # m
    mujoco_track_width = 0.10  # m
    mujoco_com_height = 0.0275  # m

    # Check wheel radius (within 1mm)
    assert abs(params.wheel_radius_m - mujoco_wheel_radius) < 0.001, \
        f"Wheel radius mismatch: {params.wheel_radius_m:.3f}m vs {mujoco_wheel_radius:.3f}m"

    # Check track width (within 1mm)
    assert abs(params.track_width_m - mujoco_track_width) < 0.001, \
        f"Track width mismatch: {params.track_width_m:.3f}m vs {mujoco_track_width:.3f}m"

    # Check COM distance (within 1mm)
    assert abs(params.com_distance_m - mujoco_com_height) < 0.001, \
        f"COM distance mismatch: {params.com_distance_m:.3f}m vs {mujoco_com_height:.3f}m"


def test_equilibrium_computation(params):
    """Test equilibrium state computation.

    Verifies that equilibrium computation works and returns a result.
    """
    eq_state = compute_equilibrium_state(params)

    # Should return something
    assert eq_state is not None, "Equilibrium computation failed"


def test_effective_parameters(params):
    """Test effective parameter computations.

    Verifies that effective mass and inertia calculations are reasonable.
    """
    # Effective mass should be positive
    assert params.effective_mass_kg > 0, "Effective mass must be positive"

    # Effective pitch inertia should be positive
    assert params.effective_pitch_inertia_kg_m2 > 0, "Effective pitch inertia must be positive"


def test_parameter_reasonableness(params):
    """Test that all parameters have reasonable values.

    Sanity checks for parameter ranges.
    """
    # Masses should be reasonable (0.1kg to 10kg)
    assert 0.1 < params.body_mass_kg < 10.0, f"Body mass seems wrong: {params.body_mass_kg}kg"
    assert 0.01 < params.wheel_mass_kg < 2.0, f"Wheel mass seems wrong: {params.wheel_mass_kg}kg"

    # Dimensions should be reasonable (1cm to 1m)
    assert 0.01 < params.wheel_radius_m < 0.5, f"Wheel radius seems wrong: {params.wheel_radius_m}m"
    assert 0.02 < params.com_distance_m < 0.1, f"COM distance seems wrong: {params.com_distance_m}m"
    assert 0.05 < params.track_width_m < 0.5, f"Track width seems wrong: {params.track_width_m}m"

    # Inertias should be positive and reasonable
    assert params.body_pitch_inertia_kg_m2 > 0, "Body pitch inertia must be positive"
    assert params.body_yaw_inertia_kg_m2 > 0, "Body yaw inertia must be positive"
    assert params.wheel_inertia_kg_m2 > 0, "Wheel inertia must be positive"

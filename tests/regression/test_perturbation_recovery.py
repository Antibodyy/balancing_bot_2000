"""Regression tests for perturbation recovery behavior.

Tests that the MPC controller can recover from various perturbation magnitudes
and maintain stability. This consolidates test_from_equilibrium.py and
test_small_perturbation.py.
"""

import pytest
import numpy as np
from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode


@pytest.fixture
def sim_config():
    """Create a standard simulation configuration."""
    return SimulationConfig(
        model_path='mujoco_sim/robot_model.xml',
        robot_params_path='config/simulation/robot_params.yaml',
        mpc_params_path='config/simulation/mpc_params.yaml',
        estimator_params_path='config/simulation/estimator_params.yaml',
        duration_s=2.0,
    )


@pytest.mark.parametrize("perturbation_deg", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_perturbation_recovery(sim_config, perturbation_deg):
    """Test recovery from various perturbation magnitudes.

    Verifies that the controller can:
    - Survive at least 2 seconds
    - Maintain velocity estimation accuracy
    - Keep final pitch reasonable
    - Avoid excessive control saturation
    """
    sim = MPCSimulation(sim_config)
    perturbation_rad = np.deg2rad(perturbation_deg)

    result = sim.run(
        duration_s=2.0,
        initial_pitch_rad=perturbation_rad,
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    # Should complete successfully
    assert result.success, f"Failed to survive 2s with {perturbation_deg}° perturbation"
    assert len(result.state_history) > 0, "No state data recorded"

    # Check velocity estimation accuracy
    if len(result.state_estimate_history) > 0:
        velocity_errors = result.state_estimate_history[:, 3] - result.state_history[:, 3]
        mean_vel_error = np.mean(np.abs(velocity_errors))
        assert mean_vel_error < 0.01, f"Velocity estimation error too high: {mean_vel_error:.4f} m/s"

    # Check final pitch is reasonable (not diverging)
    final_pitch_deg = np.rad2deg(result.state_history[-1, 1])
    assert abs(final_pitch_deg) < 10.0, f"Final pitch too large: {final_pitch_deg:.2f}°"

    # Check control isn't constantly saturating
    if len(result.control_history) > 0:
        total_torque = result.control_history[:, 0] + result.control_history[:, 1]
        max_torque = 0.25 * 2  # Max per motor is 0.25 N⋅m
        saturation_ratio = np.mean(np.abs(total_torque) > 0.9 * max_torque)
        assert saturation_ratio < 0.5, f"Control saturated {saturation_ratio*100:.1f}% of time"


def test_equilibrium_stability(sim_config):
    """Test stability when starting exactly at equilibrium.

    Verifies that starting with zero perturbation doesn't cause drift
    or instability over time.
    """
    sim = MPCSimulation(sim_config)

    result = sim.run(
        duration_s=2.0,
        initial_pitch_rad=0.0,  # Perfect equilibrium
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    assert result.success, "Failed to maintain equilibrium"
    assert len(result.state_history) > 0, "No state data recorded"

    # Pitch should remain very small
    pitch_history = result.state_history[:, 1]
    max_pitch = np.rad2deg(np.max(np.abs(pitch_history)))
    assert max_pitch < 1.0, f"Pitch drifted too much from equilibrium: {max_pitch:.4f}°"

    # Check for pitch growth over time (instability indicator)
    if len(pitch_history) > 100:
        pitch_first_half = np.abs(pitch_history[:len(pitch_history)//2])
        pitch_second_half = np.abs(pitch_history[len(pitch_history)//2:])
        mean_first = np.mean(pitch_first_half)
        mean_second = np.mean(pitch_second_half)

        # Second half shouldn't have significantly larger pitch amplitude
        assert mean_second < 2.0 * mean_first, \
            f"Pitch amplitude growing: {mean_first:.4f}° → {mean_second:.4f}°"

    # Velocity should remain near zero
    velocity_history = result.state_history[:, 3]
    max_velocity = np.max(np.abs(velocity_history))
    assert max_velocity < 0.1, f"Velocity too high at equilibrium: {max_velocity:.4f} m/s"


def test_recovery_statistics(sim_config):
    """Test statistics across multiple perturbations.

    Ensures consistent performance across different perturbation sizes.
    """
    perturbations_deg = [0.5, 1.0, 2.0, 3.0]
    results_data = []

    for perturb_deg in perturbations_deg:
        sim = MPCSimulation(sim_config)
        perturb_rad = np.deg2rad(perturb_deg)

        result = sim.run(
            duration_s=2.0,
            initial_pitch_rad=perturb_rad,
            reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
        )

        if result.success and len(result.state_history) > 0:
            pitch_rad = result.state_history[:, 1]
            drift_deg = np.rad2deg(pitch_rad[-1] - pitch_rad[0])
            max_pitch_deg = np.rad2deg(np.max(np.abs(pitch_rad)))

            results_data.append({
                'perturbation': perturb_deg,
                'drift': abs(drift_deg),
                'max_pitch': max_pitch_deg,
                'success': result.success
            })

    # All should succeed
    assert len(results_data) == len(perturbations_deg), "Some tests failed to complete"

    # Drift should be reasonable for all
    for data in results_data:
        assert data['drift'] < 5.0, \
            f"{data['perturbation']}° perturbation drifted {data['drift']:.2f}°"

"""Regression tests for perturbation recovery behavior with plots.

Tests that the MPC controller can recover from various perturbation magnitudes
and maintain stability. Plots are saved to test_and_debug_output/perturbation_recovery.
"""

from pathlib import Path

import pytest
import numpy as np

# Headless-safe plotting
import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt

from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode

OUTPUT_DIR = Path("test_and_debug_output/perturbation_recovery")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def sim_config():
    """Create a standard simulation configuration."""
    return SimulationConfig(
        model_path='Mujoco sim/robot_model.xml',
        robot_params_path='config/simulation/robot_params.yaml',
        mpc_params_path='config/simulation/mpc_params.yaml',
        estimator_params_path='config/simulation/estimator_params.yaml',
        duration_s=2.0,
    )


def _plot_result(result, perturbation_deg: float) -> None:
    """Save basic plots for a single perturbation case."""
    fname = OUTPUT_DIR / f"perturb_{perturbation_deg:.1f}deg.png"
    time_s = result.time_s
    pitch_deg = np.rad2deg(result.state_history[:, 1])
    velocity = result.state_history[:, 3]
    control = result.control_history if len(result.control_history) > 0 else np.zeros((len(time_s), 2))

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axs[0].plot(time_s, pitch_deg, label="Pitch (deg)")
    axs[0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axs[0].set_ylabel("Pitch (deg)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    axs[1].plot(time_s, velocity, label="Velocity (m/s)")
    axs[1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    axs[2].plot(time_s[:control.shape[0]], control[:, 0], label="Tau L")
    axs[2].plot(time_s[:control.shape[0]], control[:, 1], label="Tau R", linestyle="--")
    axs[2].axhline(0, color="k", linestyle="--", alpha=0.5)
    axs[2].set_ylabel("Torque (Nm)")
    axs[2].set_xlabel("Time (s)")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    fig.suptitle(f"Perturbation {perturbation_deg:.1f} deg recovery")
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)


@pytest.mark.parametrize("perturbation_deg", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_perturbation_recovery(sim_config, perturbation_deg):
    """Test recovery from various perturbation magnitudes."""
    sim = MPCSimulation(sim_config)
    perturbation_rad = np.deg2rad(perturbation_deg)

    result = sim.run(
        duration_s=2.0,
        initial_pitch_rad=perturbation_rad,
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )
    _plot_result(result, perturbation_deg)

    # Should complete successfully
    assert result.success, f"Failed to survive 2s with {perturbation_deg} deg perturbation"
    assert len(result.state_history) > 0, "No state data recorded"

    # Check velocity estimation accuracy
    if len(result.state_estimate_history) > 0:
        velocity_errors = result.state_estimate_history[:, 3] - result.state_history[:, 3]
        mean_vel_error = np.mean(np.abs(velocity_errors))
        assert mean_vel_error < 0.01, f"Velocity estimation error too high: {mean_vel_error:.4f} m/s"

    # Check final pitch is reasonable (not diverging)
    final_pitch_deg = np.rad2deg(result.state_history[-1, 1])
    assert abs(final_pitch_deg) < 10.0, f"Final pitch too large: {final_pitch_deg:.2f} deg"

    # Check control isn't constantly saturating
    if len(result.control_history) > 0:
        total_torque = result.control_history[:, 0] + result.control_history[:, 1]
        max_torque = 0.25 * 2  # Max per motor is 0.25 N·m
        saturation_ratio = np.mean(np.abs(total_torque) > 0.9 * max_torque)
        assert saturation_ratio < 0.5, f"Control saturated {saturation_ratio*100:.1f}% of time"


def test_equilibrium_stability(sim_config):
    """Test stability when starting exactly at equilibrium."""
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
    assert max_pitch < 1.0, f"Pitch drifted too much from equilibrium: {max_pitch:.4f} deg"

    # Check for pitch growth over time (instability indicator)
    if len(pitch_history) > 100:
        pitch_first_half = np.abs(pitch_history[:len(pitch_history)//2])
        pitch_second_half = np.abs(pitch_history[len(pitch_history)//2:])
        mean_first = np.mean(pitch_first_half)
        mean_second = np.mean(pitch_second_half)

        # Second half shouldn't have significantly larger pitch amplitude
        assert mean_second < 2.0 * mean_first, \
            f"Pitch amplitude growing: {mean_first:.4f} deg vs {mean_second:.4f} deg"

    # Velocity should remain near zero
    velocity_history = result.state_history[:, 3]
    max_velocity = np.max(np.abs(velocity_history))
    assert max_velocity < 0.1, f"Velocity too high at equilibrium: {max_velocity:.4f} m/s"


def test_recovery_statistics(sim_config):
    """Test statistics across multiple perturbations and plot aggregate results."""
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

    # Plot aggregate drift and max pitch vs perturbation
    if results_data:
        pts = [d['perturbation'] for d in results_data]
        drift = [d['drift'] for d in results_data]
        max_pitch = [d['max_pitch'] for d in results_data]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(pts, drift, marker="o", label="Drift (deg)")
        ax.plot(pts, max_pitch, marker="s", label="Max pitch (deg)")
        ax.set_xlabel("Perturbation (deg)")
        ax.set_ylabel("Error (deg)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.suptitle("Perturbation recovery statistics")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "aggregate_stats.png", dpi=150)
        plt.close(fig)

    # All should succeed
    assert len(results_data) == len(perturbations_deg), "Some tests failed to complete"

    # Drift should be reasonable for all (lenient, scaled with perturbation)
    for data in results_data:
        drift_limit = 2.0 + 1.5 * data['perturbation']
        assert data['drift'] < drift_limit, \
            f"{data['perturbation']} deg perturbation drifted {data['drift']:.2f} deg (limit {drift_limit:.2f} deg)"

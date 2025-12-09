"""Regression tests for long-duration stability.

Tests that the MPC controller maintains stability over longer durations
and that MPC solve times remain acceptable. This consolidates
test_long_headless.py.
"""

import pytest
import numpy as np
from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode


@pytest.fixture
def sim_config():
    """Create a standard simulation configuration."""
    return SimulationConfig(
        model_path='robot_model.xml',
        robot_params_path='config/robot_params.yaml',
        mpc_params_path='config/mpc_params.yaml',
        estimator_params_path='config/estimator_params.yaml',
        duration_s=10.0,
    )


def test_long_duration_stability(sim_config):
    """Test 10-second simulation in headless mode.

    Verifies that the controller maintains stability over longer durations
    without drift or performance degradation.
    """
    sim = MPCSimulation(sim_config)

    result = sim.run(
        duration_s=10.0,
        initial_pitch_rad=np.deg2rad(3.0),
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    # Should complete successfully
    assert result.success, "Failed to complete 10-second simulation"
    assert len(result.state_history) > 0, "No state data recorded"
    assert result.time_s[-1] >= 9.5, f"Simulation ended early at {result.time_s[-1]:.1f}s"

    # Check MPC solve times are reasonable
    assert result.mean_solve_time_ms < 20.0, \
        f"Mean solve time too high: {result.mean_solve_time_ms:.1f}ms"
    # Max solve time can be higher due to occasional spikes
    assert result.max_solve_time_ms < 200.0, \
        f"Max solve time too high: {result.max_solve_time_ms:.1f}ms"
    # Allow a few deadline violations due to timing variability
    assert result.deadline_violations < 5, \
        f"Too many deadline violations: {result.deadline_violations}"

    # Check pitch remains bounded
    pitch_history = result.state_history[:, 1]
    max_pitch = np.rad2deg(np.max(np.abs(pitch_history)))
    final_pitch = np.rad2deg(pitch_history[-1])

    assert max_pitch < 15.0, f"Pitch exceeded bounds: max {max_pitch:.2f}°"
    assert abs(final_pitch) < 10.0, f"Final pitch too large: {final_pitch:.2f}°"

    # Check velocity estimation accuracy
    if len(result.state_estimate_history) > 0:
        velocity_errors = result.state_estimate_history[:, 3] - result.state_history[:, 3]
        mean_vel_error = np.mean(np.abs(velocity_errors))
        max_vel_error = np.max(np.abs(velocity_errors))

        assert mean_vel_error < 0.01, \
            f"Mean velocity error too high: {mean_vel_error:.4f} m/s"
        assert max_vel_error < 0.05, \
            f"Max velocity error too high: {max_vel_error:.4f} m/s"

    # Check for drift (pitch should not systematically drift)
    initial_pitch = np.rad2deg(pitch_history[0])
    drift = abs(final_pitch - initial_pitch)
    assert drift < 5.0, f"Pitch drifted {drift:.2f}° over 10 seconds"


def test_equilibrium_long_duration(sim_config):
    """Test long-duration stability from perfect equilibrium.

    Verifies that starting at equilibrium doesn't cause drift over 10 seconds.
    """
    sim = MPCSimulation(sim_config)

    result = sim.run(
        duration_s=10.0,
        initial_pitch_rad=0.0,  # Perfect equilibrium
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    assert result.success, "Failed to maintain equilibrium over 10 seconds"
    assert len(result.state_history) > 0, "No state data recorded"

    # Pitch drift should be minimal
    pitch_history = result.state_history[:, 1]
    pitch_std = np.rad2deg(np.std(pitch_history))
    max_pitch = np.rad2deg(np.max(np.abs(pitch_history)))

    assert pitch_std < 0.5, f"Pitch variation too high: std={pitch_std:.4f}°"
    assert max_pitch < 1.0, f"Pitch drifted from equilibrium: max={max_pitch:.4f}°"

    # Position drift should be small
    position_history = result.state_history[:, 0]
    position_drift = abs(position_history[-1] - position_history[0])
    assert position_drift < 0.01, f"Position drifted {position_drift:.4f}m from equilibrium"


def test_solve_time_consistency(sim_config):
    """Test that solve times remain consistent over duration.

    Verifies that solve times don't degrade as simulation progresses.
    """
    sim = MPCSimulation(sim_config)

    result = sim.run(
        duration_s=10.0,
        initial_pitch_rad=np.deg2rad(3.0),
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    assert result.success, "Simulation failed"
    assert len(result.solve_time_history) > 0, "No solve time data"

    # Split solve times into quarters
    solve_times_ms = result.solve_time_history * 1000
    n = len(solve_times_ms)
    quarters = [
        solve_times_ms[:n//4],
        solve_times_ms[n//4:n//2],
        solve_times_ms[n//2:3*n//4],
        solve_times_ms[3*n//4:]
    ]

    quarter_means = [np.mean(q) for q in quarters]

    # Each quarter should have acceptable mean solve time
    for i, mean_time in enumerate(quarter_means):
        assert mean_time < 20.0, \
            f"Quarter {i+1} mean solve time too high: {mean_time:.1f}ms"

    # Solve times shouldn't increase significantly over time
    first_half_mean = np.mean(solve_times_ms[:n//2])
    second_half_mean = np.mean(solve_times_ms[n//2:])

    assert second_half_mean < 1.5 * first_half_mean, \
        f"Solve times degraded: {first_half_mean:.1f}ms → {second_half_mean:.1f}ms"

"""Regression tests for long-duration stability.

Tests that the MPC controller maintains stability over longer durations
and that MPC solve times remain acceptable. This consolidates
test_long_headless.py.
"""

from pathlib import Path
from typing import Optional
import subprocess

import pytest
import numpy as np
import yaml
from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import (
    PITCH_INDEX,
    PITCH_RATE_INDEX,
    POSITION_INDEX,
    VELOCITY_INDEX,
)


TORQUE_SATURATION_CONSECUTIVE_STEPS = 5
ESTIMATOR_DIVERGENCE_THRESHOLD_RAD = np.deg2rad(5.0)  # 5 degrees
ESTIMATOR_DIVERGENCE_STEPS = 5
POSITION_DRIFT_LIMIT_M = 0.5  # Balance mode should remain near origin
VELOCITY_DRIFT_LIMIT_MPS = 0.5


def _load_sampling_period() -> float:
    """Load MPC sampling period from config to set timing budgets."""
    with open('config/simulation/mpc_params.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    return float(config.get('sampling_period_s', 0.065))


SAMPLING_PERIOD_S = _load_sampling_period()
MEAN_SOLVE_LIMIT_MS = SAMPLING_PERIOD_S * 1000 * 0.5  # Require solves faster than half the period
MAX_SOLVE_LIMIT_MS = SAMPLING_PERIOD_S * 1000 * 1.5   # Allow brief spikes but still below 1.5x period
FAILURE_ARTIFACT_PATH = Path("test_and_debug_output/balance_failure_snapshot.npz")


def _first_consecutive_true(flags: np.ndarray, run_length: int) -> Optional[int]:
    """Return index of first window with consecutive True values."""
    if run_length <= 0:
        return None
    count = 0
    for idx, flag in enumerate(flags):
        if flag:
            count += 1
            if count >= run_length:
                return idx - run_length + 1
        else:
            count = 0
    return None


def _handle_balance_failure(result, test_name: str) -> None:
    """Persist diagnostic artifact and print failure summary."""
    FAILURE_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = FAILURE_ARTIFACT_PATH

    reference_history = np.array(result.reference_history, dtype=object)
    artifact_payload = {
        "time_s": result.time_s,
        "state_history": result.state_history,
        "true_state_history": result.true_state_history,
        "state_estimate_history": result.state_estimate_history,
        "u_applied_history": result.u_applied_history,
        "torque_saturation_history": result.torque_saturation_history,
        "infeasible_history": result.infeasible_history,
        "qp_solve_time_history": result.qp_solve_time_history,
        "iteration_history": result.iteration_history,
        "desired_pitch_history": result.desired_pitch_history,
        "desired_velocity_history": result.desired_velocity_history,
        "reference_history": reference_history,
        "pred_state_history": result.pred_state_history,
        "ground_slope_rad": result.ground_slope_rad,
        "equilibrium_state": result.equilibrium_state,
        "equilibrium_control": result.equilibrium_control,
        "mpc_config_yaml": np.array(result.mpc_config_yaml),
        "velocity_slack_history": result.velocity_slack_history,
        "success": np.array(result.success),
    }
    np.savez(artifact_path, **artifact_payload)

    summary = _summarize_failure(result)
    print(f"[FAILURE SUMMARY] {summary}")
    print(f"[FAILURE ARTIFACT] Saved to {artifact_path.resolve()}")
    try:
        subprocess.run(
            [
                "python3",
                "scripts/analyze_balance_failure_snapshot.py",
                str(artifact_path),
            ],
            check=False,
        )
    except FileNotFoundError:
        print("[FAILURE ANALYSIS] analyze_balance_failure_snapshot.py not found.")


def _summarize_failure(result) -> str:
    """Determine which failure condition triggered first."""
    if result.time_s.size == 0:
        return "No samples recorded."

    try:
        config_data = yaml.safe_load(result.mpc_config_yaml) or {}
    except yaml.YAMLError:
        config_data = {}

    pitch_limit = float(config_data.get("pitch_limit_rad", np.inf))
    pitch_rate_limit = float(config_data.get("pitch_rate_limit_radps", np.inf))

    events: list[tuple[int, str]] = []
    time_s = result.time_s

    # Event A: sustained torque saturation
    torque_idx = _first_consecutive_true(result.torque_saturation_history.astype(bool), TORQUE_SATURATION_CONSECUTIVE_STEPS)
    if torque_idx is not None:
        events.append((
            torque_idx,
            f"A) torque saturation for >= {TORQUE_SATURATION_CONSECUTIVE_STEPS} steps starting at "
            f"t={time_s[torque_idx]:.3f}s; command={result.u_applied_history[torque_idx]}"
        ))

    # Event B: estimator divergence
    min_len = min(len(result.true_state_history), len(result.state_estimate_history))
    if min_len > 0:
        pitch_true = result.true_state_history[:min_len, PITCH_INDEX]
        pitch_est = result.state_estimate_history[:min_len, PITCH_INDEX]
        diff = np.abs(pitch_true - pitch_est)
        divergence_mask = diff > ESTIMATOR_DIVERGENCE_THRESHOLD_RAD
        divergence_idx = _first_consecutive_true(divergence_mask, ESTIMATOR_DIVERGENCE_STEPS)
        if divergence_idx is not None:
            events.append((
                divergence_idx,
                f"B) estimator diverged |theta_true - theta_est|={diff[divergence_idx]:.3f} rad at "
                f"t={time_s[divergence_idx]:.3f}s"
            ))

    # Event C: unbounded position/velocity drift
    if result.true_state_history.size > 0:
        pos = np.abs(result.true_state_history[:, POSITION_INDEX])
        vel = np.abs(result.true_state_history[:, VELOCITY_INDEX])
        pos_mask = pos > POSITION_DRIFT_LIMIT_M
        vel_mask = vel > VELOCITY_DRIFT_LIMIT_MPS
        if pos.size > 0 and np.any(pos_mask):
            pos_idx = int(np.argmax(pos_mask))
            events.append((
                pos_idx,
                f"C) |x| drifted to {pos[pos_idx]:.3f} m (> {POSITION_DRIFT_LIMIT_M} m) at t={time_s[pos_idx]:.3f}s"
            ))
        if vel.size > 0 and np.any(vel_mask):
            vel_idx = int(np.argmax(vel_mask))
            events.append((
                vel_idx,
                f"C) |dx| drifted to {vel[vel_idx]:.3f} m/s (> {VELOCITY_DRIFT_LIMIT_MPS} m/s) at "
                f"t={time_s[vel_idx]:.3f}s"
            ))

    # Event D: solver infeasible while state within bounds
    infeasible_indices = np.where(result.infeasible_history)[0]
    for idx in infeasible_indices:
        if idx >= len(result.state_history):
            continue
        pitch = abs(result.state_history[idx, PITCH_INDEX])
        pitch_rate = abs(result.state_history[idx, PITCH_RATE_INDEX])
        if pitch < pitch_limit and pitch_rate < pitch_rate_limit:
            events.append((
                idx,
                f"D) solver infeasible at t={time_s[idx]:.3f}s while pitch={pitch:.3f} rad, dtheta={pitch_rate:.3f} rad/s"
            ))
            break

    # Event E: state exceeds pitch/pitch-rate limits
    for idx, state in enumerate(result.state_history):
        pitch = abs(state[PITCH_INDEX])
        pitch_rate = abs(state[PITCH_RATE_INDEX])
        if pitch >= pitch_limit or pitch_rate >= pitch_rate_limit:
            events.append((
                idx,
                f"E) pitch limit reached at t={time_s[idx]:.3f}s "
                f"(pitch={pitch:.3f} rad, dtheta={pitch_rate:.3f} rad/s)"
            ))
            break

    if not events:
        return "No diagnostic triggers detected."

    first_event = min(events, key=lambda item: item[0])
    return first_event[1]


@pytest.fixture
def sim_config():
    """Create a standard simulation configuration."""
    return SimulationConfig(
        model_path='mujoco_sim/robot_model.xml',
        robot_params_path='config/simulation/robot_params.yaml',
        mpc_params_path='config/simulation/mpc_params.yaml',
        estimator_params_path='config/simulation/estimator_params.yaml',
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

    if not result.success:
        _handle_balance_failure(result, "test_long_duration_stability")

    # Should complete successfully
    assert result.success, "Failed to complete 10-second simulation"
    assert len(result.state_history) > 0, "No state data recorded"
    assert result.time_s[-1] >= 9.5, f"Simulation ended early at {result.time_s[-1]:.1f}s"

    # Check MPC solve times are reasonable
    assert result.mean_solve_time_ms < MEAN_SOLVE_LIMIT_MS, \
        f"Mean solve time too high: {result.mean_solve_time_ms:.1f}ms (limit {MEAN_SOLVE_LIMIT_MS:.1f}ms)"
    # Max solve time can be higher due to occasional spikes
    assert result.max_solve_time_ms < MAX_SOLVE_LIMIT_MS, \
        f"Max solve time too high: {result.max_solve_time_ms:.1f}ms (limit {MAX_SOLVE_LIMIT_MS:.1f}ms)"
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

    if not result.success:
        _handle_balance_failure(result, "test_equilibrium_long_duration")

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

    if not result.success:
        _handle_balance_failure(result, "test_solve_time_consistency")

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
        assert mean_time < MEAN_SOLVE_LIMIT_MS, \
            f"Quarter {i+1} mean solve time too high: {mean_time:.1f}ms (limit {MEAN_SOLVE_LIMIT_MS:.1f}ms)"

    # Solve times shouldn't increase significantly over time
    first_half_mean = np.mean(solve_times_ms[:n//2])
    second_half_mean = np.mean(solve_times_ms[n//2:])

    assert second_half_mean < 1.5 * first_half_mean, \
        f"Solve times degraded: {first_half_mean:.1f}ms → {second_half_mean:.1f}ms"

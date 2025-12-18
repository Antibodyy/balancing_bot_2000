"""Regression tests for MuJoCo sign conventions.

These tests bypass the controller and directly apply wheel torques to the
MuJoCo plant to verify that the measured forward velocity and pitch follow
the same sign conventions assumed by the analytic dynamics.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.mpc_simulation import (
    MPCSimulation,
    SimulationConfig,
    MUJOCO_AVAILABLE,
    mujoco,
    VELOCITY_INDEX,
    PITCH_INDEX,
)


pytestmark = pytest.mark.skipif(
    not MUJOCO_AVAILABLE, reason="MuJoCo not installed; sign test requires physics engine."
)


def _make_simulation() -> MPCSimulation:
    """Create a MuJoCo-backed simulation with default configuration."""
    sim = MPCSimulation(SimulationConfig(duration_s=0.5, use_virtual_physics=False))
    sim._load_model()
    return sim


def _reset_robot_pose(sim: MPCSimulation, pitch_offset_rad: float = 0.0) -> None:
    """Reset the free joint pose to the upright equilibrium plus offset."""
    assert sim._model is not None and sim._data is not None
    mujoco.mj_resetData(sim._model, sim._data)
    data = sim._data

    # Position base at nominal height above ground (matches run() initialization).
    data.qpos[sim.FREE_JOINT_POS_START : sim.FREE_JOINT_POS_START + 3] = [0.0, 0.0, 0.25]

    pitch = sim._equilibrium_state[PITCH_INDEX] + pitch_offset_rad
    qw = np.cos(pitch / 2.0)
    qy = np.sin(pitch / 2.0)
    data.qpos[sim.FREE_JOINT_QUAT_START : sim.FREE_JOINT_QUAT_START + 4] = [qw, 0.0, qy, 0.0]

    data.qpos[sim.MOTOR_L_JOINT] = 0.0
    data.qpos[sim.MOTOR_R_JOINT] = 0.0
    data.qvel[:] = 0.0


def _collect_states(
    sim: MPCSimulation,
    torque_left_nm: float,
    torque_right_nm: float,
    duration_s: float = 0.5,
) -> np.ndarray:
    """Apply constant torques for the requested duration and log states every MPC period."""
    assert sim._model is not None and sim._data is not None
    model = sim._model
    data = sim._data
    mpc_period = sim._mpc_config.sampling_period_s
    steps_per_period = max(1, int(round(mpc_period / model.opt.timestep)))
    periods = max(1, int(round(duration_s / mpc_period)))

    history = []
    for _ in range(periods):
        sim._apply_control(torque_left_nm, torque_right_nm)
        for _ in range(steps_per_period):
            mujoco.mj_step(model, data)
        history.append(sim._extract_state())
    return np.asarray(history)


def test_positive_torque_drives_forward():
    """Applying equal positive torque should yield positive forward velocity."""
    sim = _make_simulation()
    _reset_robot_pose(sim)
    states = _collect_states(sim, torque_left_nm=0.05, torque_right_nm=0.05, duration_s=0.6)
    assert states.size > 0

    dx = states[:, VELOCITY_INDEX]
    steady_mean = np.mean(dx[len(dx) // 2 :])
    assert steady_mean > 0.02, f"Expected positive velocity, got mean {steady_mean:.4f} m/s"


def test_negative_torque_drives_backward():
    """Applying equal negative torque should yield negative forward velocity."""
    sim = _make_simulation()
    _reset_robot_pose(sim)
    states = _collect_states(sim, torque_left_nm=-0.05, torque_right_nm=-0.05, duration_s=0.6)
    assert states.size > 0

    dx = states[:, VELOCITY_INDEX]
    steady_mean = np.mean(dx[len(dx) // 2 :])
    assert steady_mean < -0.02, f"Expected negative velocity, got mean {steady_mean:.4f} m/s"


def test_pitch_sign_matches_gravity():
    """A forward lean should continue falling forward (pitch increases) under gravity."""
    sim = _make_simulation()
    initial_offset = 0.05  # radians
    _reset_robot_pose(sim, pitch_offset_rad=initial_offset)
    initial_pitch = sim._extract_state()[PITCH_INDEX]

    states = _collect_states(sim, torque_left_nm=0.0, torque_right_nm=0.0, duration_s=0.4)
    final_pitch = states[-1, PITCH_INDEX]
    assert final_pitch - initial_pitch > 0.01, (
        f"Expected pitch to increase due to gravity, but Δpitch={final_pitch - initial_pitch:.4f} rad"
    )

"""Regression test to ensure MuJoCo model has no open-loop bias.

The robot is initialized upright at equilibrium, zero torque is applied
for several seconds, and we assert that forward velocity and pitch remain
near zero. This catches sign mismatches or persistent biases in the
MuJoCo XML that would violate the assumptions used by the analytic model.
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
    not MUJOCO_AVAILABLE,
    reason="MuJoCo not installed; open-loop bias test requires physics engine.",
)


def _make_simulation() -> MPCSimulation:
    sim = MPCSimulation(SimulationConfig(duration_s=3.0, use_virtual_physics=False))
    sim._load_model()
    return sim


def _reset_robot_pose(sim: MPCSimulation) -> None:
    """Reset robot to upright equilibrium pose with zero velocities."""
    assert sim._model is not None and sim._data is not None
    mujoco.mj_resetData(sim._model, sim._data)
    data = sim._data

    data.qpos[sim.FREE_JOINT_POS_START : sim.FREE_JOINT_POS_START + 3] = [0.0, 0.0, 0.25]
    pitch = sim._equilibrium_state[PITCH_INDEX]
    qw = np.cos(pitch / 2.0)
    qy = np.sin(pitch / 2.0)
    data.qpos[sim.FREE_JOINT_QUAT_START : sim.FREE_JOINT_QUAT_START + 4] = [qw, 0.0, qy, 0.0]
    data.qpos[sim.MOTOR_L_JOINT] = 0.0
    data.qpos[sim.MOTOR_R_JOINT] = 0.0
    data.qvel[:] = 0.0


def _collect_states(sim: MPCSimulation, duration_s: float = 3.0) -> np.ndarray:
    assert sim._model is not None and sim._data is not None
    model = sim._model
    data = sim._data
    mpc_period = sim._mpc_config.sampling_period_s
    steps_per_period = max(1, int(round(mpc_period / model.opt.timestep)))
    periods = max(1, int(round(duration_s / mpc_period)))
    history = []
    for _ in range(periods):
        sim._apply_control(0.0, 0.0)
        for _ in range(steps_per_period):
            mujoco.mj_step(model, data)
        history.append(sim._extract_state())
    return np.asarray(history)


def test_open_loop_bias_is_small():
    sim = _make_simulation()
    _reset_robot_pose(sim)
    states = _collect_states(sim, duration_s=3.0)
    dx = states[:, VELOCITY_INDEX]
    pitch = states[:, PITCH_INDEX]
    dx_max = float(np.max(np.abs(dx)))
    pitch_max = float(np.max(np.abs(pitch)))
    eps_vel = 0.05
    eps_pitch = 0.05
    assert dx_max < eps_vel, (
        f"Open-loop drift detected: max |dx|={dx_max:.4f} m/s, "
        f"mean dx={np.mean(dx):.4f} m/s"
    )
    assert pitch_max < eps_pitch, (
        f"Pitch drift detected: max |pitch|={pitch_max:.4f} rad, "
        f"final pitch={pitch[-1]:.4f} rad"
    )

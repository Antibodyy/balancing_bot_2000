"""Regression test verifying MuJoCo control effectiveness matches analytic model."""

import numpy as np
import pytest

from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None
from robot_dynamics import linearize_at_equilibrium, discretize_linear_dynamics
from robot_dynamics.parameters import (
    VELOCITY_INDEX,
    PITCH_RATE_INDEX,
)

pytestmark = pytest.mark.skipif(
    not MUJOCO_AVAILABLE,
    reason="MuJoCo not installed; control effectiveness test requires physics engine.",
)


def _reset_to_equilibrium(sim: MPCSimulation) -> None:
    assert sim._model is not None and sim._data is not None
    mujoco.mj_resetData(sim._model, sim._data)
    data = sim._data
    # Position base
    data.qpos[sim.FREE_JOINT_POS_START : sim.FREE_JOINT_POS_START + 3] = [0.0, 0.0, 0.25]
    pitch = sim._equilibrium_state[1]
    qw = np.cos(pitch / 2.0)
    qy = np.sin(pitch / 2.0)
    data.qpos[sim.FREE_JOINT_QUAT_START : sim.FREE_JOINT_QUAT_START + 4] = [qw, 0.0, qy, 0.0]
    data.qpos[sim.MOTOR_L_JOINT] = 0.0
    data.qpos[sim.MOTOR_R_JOINT] = 0.0
    data.qvel[:] = 0.0


def _apply_bias_and_step(sim: MPCSimulation, torque: float, duration_s: float) -> np.ndarray:
    assert sim._model is not None and sim._data is not None
    model = sim._model
    steps = max(1, int(round(duration_s / model.opt.timestep)))
    sim._apply_control(torque, torque)
    for _ in range(steps):
        mujoco.mj_step(model, sim._data)
    return sim._extract_state()


def test_control_effectiveness_sign():
    sim = MPCSimulation(SimulationConfig(duration_s=0.2, use_virtual_physics=False))
    sim._load_model()
    _reset_to_equilibrium(sim)
    dt = sim._mpc_config.sampling_period_s
    steps = int(round(dt / sim._model.opt.timestep))
    epsilon = 0.01

    state_pre = sim._extract_state()
    state_pos = _apply_bias_and_step(sim, epsilon, dt)
    _reset_to_equilibrium(sim)
    state_neg = _apply_bias_and_step(sim, -epsilon, dt)

    deriv = (state_pos - state_neg) / (2 * epsilon)
    dx_deriv = deriv[VELOCITY_INDEX]
    dtheta_deriv = deriv[PITCH_RATE_INDEX]

    lin = linearize_at_equilibrium(sim._robot_params, sim._equilibrium_state, sim._equilibrium_control)
    disc = discretize_linear_dynamics(
        lin.state_matrix,
        lin.control_matrix,
        sim._mpc_config.sampling_period_s,
    )
    b_bias = disc.control_matrix_discrete[:, 0] + disc.control_matrix_discrete[:, 1]

    assert dx_deriv * b_bias[VELOCITY_INDEX] > 0, (
        f"Velocity effectiveness sign mismatch: sim={dx_deriv}, model={b_bias[VELOCITY_INDEX]}"
    )
    assert dtheta_deriv * b_bias[PITCH_RATE_INDEX] > 0, (
        f"Pitch-rate effectiveness sign mismatch: sim={dtheta_deriv}, model={b_bias[PITCH_RATE_INDEX]}"
    )

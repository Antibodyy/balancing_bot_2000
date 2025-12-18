"""Regression test ensuring zero-error balance has no torque bias or drift."""

from pathlib import Path

import numpy as np
import pytest

from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import VELOCITY_INDEX


SNAPSHOT_PATH = Path("test_and_debug_output/balance_zero_bias_snapshot.npz")


def _save_snapshot(result) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time_s": result.time_s,
        "state_history": result.state_history,
        "true_state_history": result.true_state_history,
        "u_applied_history": result.u_applied_history,
        "success": np.array(result.success),
    }
    np.savez(SNAPSHOT_PATH, **payload)


def test_balance_zero_error_has_no_bias():
    sim = MPCSimulation(
        SimulationConfig(
            duration_s=3.0,
            model_path='mujoco_sim/robot_model.xml',
            robot_params_path='config/simulation/robot_params.yaml',
            mpc_params_path='config/simulation/mpc_params.yaml',
            estimator_params_path='config/simulation/estimator_params.yaml',
        )
    )

    result = sim.run(
        duration_s=3.0,
        initial_pitch_rad=0.0,
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    dx = result.true_state_history[:, VELOCITY_INDEX]
    mean_bias = float(
        np.mean((result.u_applied_history[:, 0] + result.u_applied_history[:, 1]) / 2.0)
    )

    try:
        assert result.success, "Simulation failed with zero-error setup"
        assert abs(mean_bias) < 2e-3, f"Mean torque bias {mean_bias:.6f} Nm"
        assert np.max(np.abs(dx)) < 0.05, f"Unexpected drift: max|dx|={np.max(np.abs(dx)):.4f} m/s"
    except AssertionError:
        _save_snapshot(result)
        raise

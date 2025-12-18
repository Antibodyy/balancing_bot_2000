"""Regression test for balance with ground-truth state estimates."""

import numpy as np

from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode


def test_balance_with_groundtruth_state():
    """Run 10-second balance simulation using MuJoCo ground-truth state."""
    sim = MPCSimulation(
        SimulationConfig(
            model_path='mujoco_sim/robot_model.xml',
            robot_params_path='config/simulation/robot_params.yaml',
            mpc_params_path='config/simulation/mpc_params.yaml',
            estimator_params_path='config/simulation/estimator_params.yaml',
            duration_s=10.0,
        )
    )

    result = sim.run(
        duration_s=10.0,
        initial_pitch_rad=np.deg2rad(3.0),
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    assert result.success, "Simulation failed even with ground-truth state estimate"
    assert len(result.state_history) > 0

    pitch_history = result.state_history[:, 1]
    pitch_deg = np.rad2deg(np.max(np.abs(pitch_history)))
    assert pitch_deg < 30.0, f"Pitch exceeded limit: {pitch_deg:.2f} deg"

    dx = result.state_history[:, 3]
    max_dx = float(np.max(np.abs(dx)))
    assert max_dx < 0.5, f"Forward velocity exceeded bound: {max_dx:.3f} m/s"

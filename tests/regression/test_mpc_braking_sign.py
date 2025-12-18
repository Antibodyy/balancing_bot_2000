"""Regression test verifying MPC brakes when forward velocity is positive."""

import numpy as np

from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode
from control_pipeline import SensorData


def test_mpc_brakes_positive_velocity():
    """MPC should command negative mean torque when dx > 0 for balance."""
    sim = MPCSimulation(SimulationConfig(duration_s=0.1))
    controller = sim._controller

    # Ground-truth state injected via simulation path.
    state = np.zeros(6)
    state[3] = 0.2  # positive forward velocity
    controller.update_simulation_state(state)

    zeros = np.zeros(3)
    sensor = SensorData(
        acceleration_mps2=zeros,
        angular_velocity_radps=zeros,
        encoder_left_rad=0.0,
        encoder_right_rad=0.0,
        timestamp_s=0.0,
    )
    cmd = ReferenceCommand(mode=ReferenceMode.BALANCE)

    output = controller.step(sensor, cmd)
    avg_torque = 0.5 * (output.torque_left_nm + output.torque_right_nm)
    assert avg_torque < 0.0, f"MPC failed to brake; mean torque {avg_torque:.4f} Nm"

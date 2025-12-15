"""Empirical search for the steepest slope the MPC can balance on.

This utility test runs the MPC simulator on incrementally steeper slopes
until the robot can no longer recover (the simulation marks success=False).
It reports the last successful slope and the first failure slope so we can
track regressions when tuning the controller.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import pytest
import yaml

from mpc import MPCConfig, ReferenceCommand, ReferenceMode
from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE

OUTPUT_DIR = Path("test_and_debug_output/slope_failure")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_ROBOT_PARAMS = "config/simulation/robot_params.yaml"
BASE_MPC_PARAMS = "config/simulation/mpc_params.yaml"
BASE_EST_PARAMS = "config/simulation/estimator_params.yaml"


def _write_robot_params_with_slope(slope_deg: float) -> str:
    params = yaml.safe_load(Path(BASE_ROBOT_PARAMS).read_text())
    params["ground_slope_rad"] = math.radians(slope_deg)
    tmp_path = OUTPUT_DIR / f"robot_params_slope_{slope_deg:.3f}.yaml"
    with tmp_path.open("w") as fh:
        yaml.safe_dump(params, fh)
    return str(tmp_path)


def _reference_for_slope(slope_rad: float):
    def _cb(_: float) -> ReferenceCommand:
        return ReferenceCommand(mode=ReferenceMode.BALANCE, desired_pitch_rad=-slope_rad)

    return _cb


def _run_simulation(slope_deg: float, duration_s: float = 6.0) -> bool:
    cfg = MPCConfig.from_yaml(BASE_MPC_PARAMS)
    robot_params_path = _write_robot_params_with_slope(slope_deg)
    sim_cfg = SimulationConfig(
        model_path="mujoco_sim/robot_model.xml",
        robot_params_path=robot_params_path,
        mpc_params_path=BASE_MPC_PARAMS,
        estimator_params_path=BASE_EST_PARAMS,
        duration_s=duration_s,
        render=False,
        use_virtual_physics=True,
    )
    sim = MPCSimulation(sim_cfg)
    reference_cb = _reference_for_slope(math.radians(slope_deg))
    result = sim.run(
        duration_s=duration_s,
        initial_pitch_rad=-math.radians(slope_deg),
        reference_command_callback=reference_cb,
    )
    return result.success


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
def test_find_maximum_stable_slope():
    """Binary search for the steepest slope the robot can balance on."""
    max_search_deg = 30.0
    tolerance_deg = 0.5
    evaluations: dict[float, bool] = {}

    def eval_slope(slope: float) -> bool:
        if slope in evaluations:
            return evaluations[slope]
        success = _run_simulation(slope)
        evaluations[slope] = success
        return success

    if eval_slope(max_search_deg):
        pytest.skip("Controller remained stable up to 30°, no failure to bracket.")

    low = 0.0
    high = max_search_deg
    while (high - low) > tolerance_deg:
        mid = round((low + high) / 2.0, 3)
        if eval_slope(mid):
            low = mid
        else:
            high = mid

    print(
        f"[SLOPE FAILURE] last success: {low:.2f}°, "
        f"first failure <= {high:.2f}° (evaluations={len(evaluations)})"
    )
    summary_path = OUTPUT_DIR / "slope_failure_summary.txt"
    with summary_path.open("w") as fh:
        fh.write(
            f"last_success_deg: {low:.2f}\n"
            f"first_failure_deg: {high:.2f}\n"
            f"evaluations: {len(evaluations)}\n"
            f"evaluation_table:\n"
        )
        for slope, success in sorted(evaluations.items()):
            fh.write(f"  slope={slope:.3f} success={success}\n")
    print(f"[SLOPE FAILURE] summary written to {summary_path}")
    assert high > 0.0

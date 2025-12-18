#!/usr/bin/env python3
"""Sweep MPC balance weights to evaluate forward-velocity drift.

For each combination of Q_x and Q_dx multipliers, the script writes a temporary
MPC YAML, runs a short MuJoCo simulation, and logs the maximum |dx| and |theta|.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import yaml

from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode


BASE_CONFIG = Path("config/simulation/mpc_params.yaml")
DX_MULTIPLIERS = [3, 5, 7, 10]
X_MULTIPLIERS = [1]
MAX_DURATION_S = 10.0
DX_THRESHOLD = 0.5
PITCH_THRESHOLD = 0.524  # 30 degrees


def load_base_yaml() -> dict:
    with open(BASE_CONFIG, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_sim_with_config(config_path: Path) -> tuple[float, float]:
    sim = MPCSimulation(
        SimulationConfig(
            mpc_params_path=str(config_path),
            duration_s=MAX_DURATION_S,
        )
    )
    result = sim.run(
        duration_s=MAX_DURATION_S,
        initial_pitch_rad=np.deg2rad(3.0),
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )
    max_dx = float(np.max(np.abs(result.state_history[:, 3])))
    max_pitch = float(np.max(np.abs(result.state_history[:, 1])))
    return max_dx, max_pitch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dx-multipliers",
        nargs="*",
        type=float,
        default=DX_MULTIPLIERS,
        help="Multipliers applied to Q_dx entry.",
    )
    parser.add_argument(
        "--x-multipliers",
        nargs="*",
        type=float,
        default=X_MULTIPLIERS,
        help="Multipliers applied to Q_x entry.",
    )
    args = parser.parse_args()

    base_yaml = load_base_yaml()
    base_state_cost = list(base_yaml["state_cost_diagonal"])

    best = None
    print("dx_mult,x_mult,max|dx|,max|theta|")
    for dx_mult in args.dx_multipliers:
        for x_mult in args.x_multipliers:
            cfg = dict(base_yaml)
            state_cost = base_state_cost.copy()
            state_cost[0] = base_state_cost[0] * x_mult
            state_cost[3] = base_state_cost[3] * dx_mult
            cfg["state_cost_diagonal"] = state_cost

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as tmp:
                yaml.safe_dump(cfg, tmp)
                tmp_path = Path(tmp.name)

            try:
                max_dx, max_pitch = run_sim_with_config(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

            print(f"{dx_mult},{x_mult},{max_dx:.3f},{max_pitch:.3f}")
            if max_dx < DX_THRESHOLD and max_pitch < PITCH_THRESHOLD:
                if best is None or max_dx < best[2]:
                    best = (dx_mult, x_mult, max_dx, max_pitch)

    if best:
        print(
            f"\nBest candidate: dx_mult={best[0]}, x_mult={best[1]}, "
            f"max|dx|={best[2]:.3f}, max|theta|={best[3]:.3f}"
        )
    else:
        print("\nNo candidate met the drift/pitch thresholds.")


if __name__ == "__main__":
    main()

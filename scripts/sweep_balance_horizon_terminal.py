#!/usr/bin/env python3
"""Sweep MPC horizon and terminal velocity limit for balance drift diagnosis."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import yaml

from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import (
    VELOCITY_INDEX,
    PITCH_INDEX,
)


BASE_CONFIG = Path("config/simulation/mpc_params.yaml")
HORIZON_CANDIDATES = [15, 20, 25, 30, 35]
TERMINAL_VEL_CANDIDATES = [None, 0.5, 0.25, 0.15, 0.10]
DURATION = 10.0
INITIAL_PITCH_DEG = 3.0
DX_THRESHOLD = 0.5
PITCH_LIMIT_RAD = 0.524


def load_base_config() -> dict:
    with open(BASE_CONFIG, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_sim(config_path: Path) -> MPCSimulation:
    sim = MPCSimulation(
        SimulationConfig(
            mpc_params_path=str(config_path),
            duration_s=DURATION,
        )
    )
    result = sim.run(
        duration_s=DURATION,
        initial_pitch_rad=np.deg2rad(INITIAL_PITCH_DEG),
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )
    return result


def summarize(result) -> dict:
    time = result.time_s
    dx = result.state_history[:, VELOCITY_INDEX]
    pitch = result.state_history[:, PITCH_INDEX]
    abs_dx = np.abs(dx)
    abs_pitch = np.abs(pitch)
    first_dx_idx = np.argmax(abs_dx > DX_THRESHOLD) if np.any(abs_dx > DX_THRESHOLD) else None
    first_dx_time = time[first_dx_idx] if first_dx_idx is not None else None

    infeasible_idx = np.argmax(result.infeasible_history) if np.any(result.infeasible_history) else None
    infeasible_time = time[infeasible_idx] if infeasible_idx is not None else None

    bias_window = result.time_s <= 3.0
    if not np.any(bias_window):
        bias_window[:] = True
    torques = result.u_applied_history
    mean_bias = float(np.mean((torques[bias_window, 0] + torques[bias_window, 1]) / 2.0))

    return {
        "success": bool(result.success),
        "first_dx_time": first_dx_time,
        "max_dx": float(np.max(abs_dx)),
        "max_pitch": float(np.max(abs_pitch)),
        "infeasible_time": infeasible_time,
        "mean_bias": mean_bias,
    }


def main() -> None:
    base = load_base_config()
    best = None

    print("horizon,term_vel,success,first_dx_time,max|dx|,max|pitch|,infeasible_time,mean_bias")
    for horizon in HORIZON_CANDIDATES:
        for term in TERMINAL_VEL_CANDIDATES:
            cfg = copy.deepcopy(base)
            cfg["prediction_horizon_steps"] = horizon
            cfg["terminal_velocity_limit_mps"] = term

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                yaml.safe_dump(cfg, tmp)
                tmp_path = Path(tmp.name)

            try:
                result = run_sim(tmp_path)
                summary = summarize(result)
            except Exception as exc:  # pragma: no cover
                summary = {
                    "success": False,
                    "first_dx_time": None,
                    "max_dx": float("inf"),
                    "max_pitch": float("inf"),
                    "infeasible_time": None,
                    "mean_bias": float("nan"),
                }
                print(f"# ERROR horizon {horizon}, term {term}: {exc}")
            finally:
                tmp_path.unlink(missing_ok=True)

            fd_str = f"{summary['first_dx_time']:.3f}" if summary["first_dx_time"] is not None else "none"
            inf_str = f"{summary['infeasible_time']:.3f}" if summary["infeasible_time"] is not None else "none"
            print(
                f"{horizon},{term},{summary['success']},{fd_str},"
                f"{summary['max_dx']:.3f},{summary['max_pitch']:.3f},"
                f"{inf_str},{summary['mean_bias']:.5f}"
            )

            if (
                summary["success"]
                and summary["max_dx"] < DX_THRESHOLD
                and summary["max_pitch"] < PITCH_LIMIT_RAD
            ):
                if best is None or summary["max_dx"] < best["summary"]["max_dx"]:
                    best = {"horizon": horizon, "term": term, "summary": summary}

    if best:
        s = best["summary"]
        print(
            "\nBest: horizon={}, term={}, max|dx|={:.3f}, max|pitch|={:.3f}, mean_bias={:.5f}".format(
                best["horizon"], best["term"], s["max_dx"], s["max_pitch"], s["mean_bias"]
            )
        )
    else:
        print("\nNo configuration satisfied the constraints.")


if __name__ == "__main__":
    main()

"""
Sweep MPC prediction horizon to balance performance vs real-time cost.

For N in [10, 15, 20, 25, 30, 35, 40] with Ts=0.02 s, simulate a closed-loop
recovery from a 0.1 rad pitch perturbation and collect:
 - Settling time (|pitch| < tol for 0.5 s window)
 - Overshoot (max |pitch| excursion)
 - Control effort (RMS torque)
 - Average solve time (per MPC iteration)

Outputs plots and a simple recommendation where performance plateaus while
average solve time stays below Ts (real-time feasible).

Usage:
    PYTHONPATH=. python tests/controller_behaviour/mpc/prediction_horizon.py
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from mpc import MPCConfig, ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import PITCH_INDEX
from simulation import MPCSimulation, SimulationConfig


def compute_settling_time(time_s: np.ndarray, pitch: np.ndarray, tol_rad: float, window_s: float) -> float:
    """First time after which |pitch| stays within tol for a full window."""
    dt = np.mean(np.diff(time_s)) if len(time_s) > 1 else 0.0
    if dt <= 0:
        return float("nan")
    window_n = max(int(window_s / dt), 1)
    within = np.abs(pitch) < tol_rad
    for i in range(len(within) - window_n):
        if within[i : i + window_n].all():
            return time_s[i]
    return float("nan")


def metric_from_result(result) -> Dict[str, float]:
    pitch = result.state_history[:, PITCH_INDEX]
    time_s = result.time_s
    control = result.control_history
    settle = compute_settling_time(time_s, pitch, tol_rad=0.01, window_s=0.5)
    overshoot = float(np.max(np.abs(pitch)))
    control_rms = float(np.sqrt(np.mean(control**2)))
    solve_mean = float(np.mean(result.solve_time_history))
    return {
        "settling_time_s": settle,
        "overshoot_rad": overshoot,
        "control_rms_nm": control_rms,
        "solve_mean_s": solve_mean,
    }


def write_temp_mpc_config(base_config: MPCConfig, horizon: int) -> str:
    cfg = replace(base_config, prediction_horizon_steps=horizon)
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    import yaml

    yaml.safe_dump(cfg_dict, tmp)
    tmp.flush()
    tmp.close()
    return tmp.name


def main() -> None:
    horizons = [10, 15, 20, 25, 30, 35, 40]
    Ts = 0.02
    duration_s = 5.0
    init_pitch = 0.1
    output_dir = Path("test_and_debug_output/horizon_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_mpc_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")
    metrics = []
    trajectories = []

    for N in horizons:
        mpc_path = write_temp_mpc_config(base_mpc_cfg, N)
        sim_cfg = SimulationConfig(
            model_path="Mujoco sim/robot_model.xml",
            robot_params_path="config/simulation/robot_params.yaml",
            mpc_params_path=mpc_path,
            estimator_params_path="config/simulation/estimator_params.yaml",
            duration_s=duration_s,
            render=False,
        )
        sim = MPCSimulation(sim_cfg)
        result = sim.run(
            duration_s=duration_s,
            initial_pitch_rad=init_pitch,
            reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
        )
        metrics.append(metric_from_result(result))
        trajectories.append((result.time_s, result.state_history[:, PITCH_INDEX]))

    # Plots
    plt.figure(figsize=(8, 4))
    plt.plot(horizons, [m["settling_time_s"] for m in metrics], marker="o", label="Settling time (s)")
    plt.plot(horizons, [m["overshoot_rad"] for m in metrics], marker="o", label="Overshoot (rad)")
    plt.plot(horizons, [m["control_rms_nm"] for m in metrics], marker="o", label="Control RMS (Nm)")
    plt.title("Performance vs Prediction Horizon")
    plt.xlabel("Horizon N (samples)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "performance_vs_horizon.png", dpi=150)

    plt.figure(figsize=(8, 4))
    solve_ms = [m["solve_mean_s"] * 1e3 for m in metrics]
    plt.plot(horizons, solve_ms, marker="o", label="Avg solve time (ms)")
    plt.axhline(Ts * 1e3, color="r", linestyle="--", label=f"Ts={Ts*1e3:.1f} ms")
    plt.title("Computation vs Prediction Horizon")
    plt.xlabel("Horizon N (samples)")
    plt.ylabel("Solve time (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "solve_time_vs_horizon.png", dpi=150)

    plt.figure(figsize=(8, 4))
    for (t, p), N in zip(trajectories, horizons):
        plt.plot(t, p, label=f"N={N}")
    plt.title("Pitch Trajectories vs Horizon")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (rad)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pitch_trajectories.png", dpi=150)

    # Simple recommendation: first N where settling time within 2% of best and solve time < Ts
    best_settle = min(m["settling_time_s"] for m in metrics if not np.isnan(m["settling_time_s"]))
    recommendation = None
    for N, m in zip(horizons, metrics):
        if np.isnan(m["settling_time_s"]):
            continue
        if m["settling_time_s"] <= 1.02 * best_settle and m["solve_mean_s"] < Ts:
            recommendation = N
            break

    print("\nHorizon sweep results (Ts = 0.02 s):")
    for N, m in zip(horizons, metrics):
        print(
            f"N={N:2d} | settle={m['settling_time_s']:.3f}s | overshoot={m['overshoot_rad']:.3f}rad | "
            f"u_rms={m['control_rms_nm']:.3f}Nm | solve_mean={m['solve_mean_s']*1e3:.2f}ms"
        )
    if recommendation is not None:
        print(f"\nRecommended N: {recommendation} (performance plateau, solve time < Ts)")
    else:
        print("\nNo horizon meets both performance plateau and solve-time < Ts criteria.")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

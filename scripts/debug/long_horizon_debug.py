"""
Stress-test very long MPC horizons to observe performance, solve times, and the feasibility limit.
Stops after the first failure (infeasible solve, crash, or fall).

Beware file takes long to run as it is a lot of simulations with a long horizon each.

Horizon sweep:
    Start at N = 50 and increment by 10 up to 100, then continue stepping by 10
    until the controller fails (infeasible solve, crash, or fall). Ts = 0.02 s.

Scenario:
    Closed-loop recovery from a 0.1 rad pitch perturbation, duration 6 s.

Metrics per N:
    - Settling time (|pitch| < 0.01 rad for 0.5 s window)
    - Overshoot (max |pitch|)
    - Control effort (RMS torque)
    - Average solve time
    - Success flag (simulation completed without fall/exception)

Outputs:
    Plots to test_and_debug_output/long_horizon_debug/

Usage:
    PYTHONPATH=. python scripts/debug/long_horizon_debug.py
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

from mpc import MPCConfig, ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import PITCH_INDEX
from simulation import MPCSimulation, SimulationConfig


def compute_settling_time(time_s: np.ndarray, pitch: np.ndarray, tol_rad: float, window_s: float) -> float:
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
    start_horizon = 50
    step = 10
    max_initial = 100
    Ts = 0.02
    duration_s = 6.0
    init_pitch = 0.1
    output_dir = Path("test_and_debug_output/long_horizon_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_mpc_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")
    metrics: List[Dict[str, float]] = []
    trajectories: List[Tuple[np.ndarray, np.ndarray]] = []
    horizons: List[int] = []
    failure_horizon: Optional[int] = None

    N = start_horizon
    while True:
        horizons.append(N)
        try:
            mpc_path = write_temp_mpc_config(base_mpc_cfg, N)
            sim_cfg = SimulationConfig(
                model_path="mujoco_sim/robot_model.xml",
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
            metrics.append(metric_from_result(result) | {"success": result.success})
            trajectories.append((result.time_s, result.state_history[:, PITCH_INDEX]))
            if not result.success:
                failure_horizon = N
                break
        except Exception as exc:  # Capture infeasibility or other crashes
            metrics.append({
                "settling_time_s": np.nan,
                "overshoot_rad": np.nan,
                "control_rms_nm": np.nan,
                "solve_mean_s": np.nan,
                "success": False,
                "error": str(exc),
            })
            failure_horizon = N
            break

        # Stop if we reached the initial max and still stable; keep going otherwise
        if N >= max_initial:
            # continue stepping until failure to find limit
            pass
        N += step
        # hard cap to avoid infinite loop
        if N > 200:
            break

    # Performance plot
    plt.figure(figsize=(8, 4))
    plt.plot(horizons, [m.get("settling_time_s", np.nan) for m in metrics], marker="o", label="Settling time (s)")
    plt.plot(horizons, [m.get("overshoot_rad", np.nan) for m in metrics], marker="o", label="Overshoot (rad)")
    plt.plot(horizons, [m.get("control_rms_nm", np.nan) for m in metrics], marker="o", label="Control RMS (Nm)")
    plt.title("Performance vs Long Prediction Horizons")
    plt.xlabel("Horizon N (samples)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "performance_vs_horizon.png", dpi=150)

    # Solve time plot (no frequency cap)
    plt.figure(figsize=(8, 4))
    solve_ms = [m.get("solve_mean_s", np.nan) * 1e3 for m in metrics]
    plt.plot(horizons, solve_ms, marker="o", label="Avg solve time (ms)")
    plt.axhline(Ts * 1e3, color="r", linestyle="--", label=f"Ts={Ts*1e3:.1f} ms")
    plt.title("Computation vs Long Horizons")
    plt.xlabel("Horizon N (samples)")
    plt.ylabel("Solve time (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "solve_time_vs_horizon.png", dpi=150)

    # Trajectories
    plt.figure(figsize=(8, 4))
    for (t, p), N in zip(trajectories, horizons[: len(trajectories)]):
        plt.plot(t, p, label=f"N={N}")
    plt.title("Pitch Trajectories vs Long Horizons")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (rad)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pitch_trajectories.png", dpi=150)

    print("\nLong-horizon sweep results (Ts = 0.02 s):")
    for N, m in zip(horizons, metrics):
        print(
            f"N={N:3d} | settle={m.get('settling_time_s', np.nan):.3f}s | "
            f"overshoot={m.get('overshoot_rad', np.nan):.3f}rad | "
            f"u_rms={m.get('control_rms_nm', np.nan):.3f}Nm | "
            f"solve_mean={m.get('solve_mean_s', np.nan)*1e3:.2f}ms | "
            f"success={m.get('success', False)}"
        )
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()

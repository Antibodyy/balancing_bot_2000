"""
Objective vs relative linearization comparison.

This modeling test contrasts MPC performance when the dynamics stay fixed
at the equilibrium operating point ("objective" linearization) versus
when we re-linearize around the current state every iteration ("relative"
linearization). It runs each configuration on multiple slopes while keeping
all other parameters (warm start, horizon, reference profile, MuJoCo model)
identical, then reports tracking accuracy, computation time, and solver health.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from mpc import MPCConfig, ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import PITCH_INDEX, VELOCITY_INDEX
from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE

OUTPUT_DIR = Path("test_and_debug_output/linearization_modes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLOPES_DEG = [0.0, 5.0, 10.0]
LINEARIZATION_MODES = {
    "objective": False,
    "relative": True,
}
MODE_LABELS = {
    "objective": "Objective linearization",
    "relative": "Relative linearization",
}
WARMUP_STEPS = 3
SEED = 12345
TS_MS = 20.0


def _make_reference_profile(slope_rad: float):
    command = ReferenceCommand(mode=ReferenceMode.BALANCE, desired_pitch_rad=-slope_rad)

    def _profile(_: float) -> ReferenceCommand:
        return command

    return _profile


def _write_robot_params_with_slope(base_yaml: str, slope_deg: float) -> str:
    params = yaml.safe_load(Path(base_yaml).read_text())
    params["ground_slope_rad"] = math.radians(slope_deg)
    tmp = OUTPUT_DIR / f"robot_params_slope_{slope_deg:.1f}.yaml"
    with tmp.open("w") as fh:
        yaml.safe_dump(params, fh)
    return str(tmp)


def _write_temp_config(base: MPCConfig, mode_key: str, online: bool) -> str:
    cfg = replace(
        base,
        prediction_horizon_steps=60,
        warm_start_enabled=True,
        online_linearization_enabled=online,
        preserve_warm_start_on_linearization=online,
    )
    cfg_dict = asdict(cfg)
    tmp = OUTPUT_DIR / f"mpc_params_{mode_key}.yaml"
    with tmp.open("w") as fh:
        yaml.safe_dump(cfg_dict, fh)
    return str(tmp)


def _trim(array: np.ndarray) -> np.ndarray:
    if array.size <= WARMUP_STEPS:
        return np.array([], dtype=array.dtype)
    return array[WARMUP_STEPS:]


def _desired_velocity_series(time_s: np.ndarray) -> np.ndarray:
    return np.zeros_like(time_s)


def _run_case(
    base_cfg: MPCConfig,
    robot_params_path: str,
    slope_deg: float,
    mode_key: str,
    online_linearization: bool,
) -> Tuple[Dict[str, float], MPCSimulation]:
    cfg_path = _write_temp_config(base_cfg, f"{mode_key}_slope_{slope_deg:.1f}", online_linearization)
    sim_cfg = SimulationConfig(
        model_path="mujoco_sim/robot_model.xml",
        robot_params_path=robot_params_path,
        mpc_params_path=cfg_path,
        estimator_params_path="config/simulation/estimator_params.yaml",
        duration_s=6.0,
        render=False,
        use_virtual_physics=False,
    )
    sim = MPCSimulation(sim_cfg)
    reference_callback = _make_reference_profile(math.radians(slope_deg))
    result = sim.run(
        duration_s=6.0,
        initial_pitch_rad=0.12,
        reference_command_callback=reference_callback,
    )

    trimmed_qp = _trim(result.qp_solve_time_history) * 1e3
    trimmed_model = _trim(result.model_update_time_history) * 1e3
    trimmed_pitch_err = _trim(result.pitch_error_history)
    trimmed_vel_err = _trim(result.velocity_error_history)
    trimmed_sat = _trim(result.torque_saturation_history.astype(float))
    trimmed_infeasible = _trim(result.infeasible_history.astype(float))

    def _safe_stats(arr: np.ndarray, percentile: float = 95.0) -> Tuple[float, float, float]:
        if arr.size == 0:
            return float("nan"), float("nan"), float("nan")
        return (
            float(np.mean(arr)),
            float(np.percentile(arr, percentile)),
            float(np.max(arr)),
        )

    qp_mean, qp_p95, qp_max = _safe_stats(trimmed_qp)
    mu_mean, mu_p95, mu_max = _safe_stats(trimmed_model)
    pitch_err_rms = float(np.sqrt(np.mean(trimmed_pitch_err**2))) if trimmed_pitch_err.size else float("nan")
    velocity_err_rms = float(np.sqrt(np.mean(trimmed_vel_err**2))) if trimmed_vel_err.size else float("nan")
    sat_pct = float(np.mean(trimmed_sat) * 100.0) if trimmed_sat.size else float("nan")
    infeasible_pct = float(np.mean(trimmed_infeasible) * 100.0) if trimmed_infeasible.size else float("nan")

    metrics = {
        "slope_deg": slope_deg,
        "mode": mode_key,
        "seed": SEED,
        "mean_qp_solve_ms": qp_mean,
        "p95_qp_solve_ms": qp_p95,
        "max_qp_solve_ms": qp_max,
        "mean_model_update_ms": mu_mean,
        "p95_model_update_ms": mu_p95,
        "max_model_update_ms": mu_max,
        "pitch_err_rms_rad": pitch_err_rms,
        "velocity_err_rms_mps": velocity_err_rms,
        "torque_saturation_pct": sat_pct,
        "infeasible_pct": infeasible_pct,
    }
    return metrics, result


def _plot_timeseries(
    slope_deg: float,
    results: Dict[str, MPCSimulation],
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].set_title(f"Slope {slope_deg:.1f}° – pitch error")
    for mode_key, result in results.items():
        label = MODE_LABELS[mode_key]
        axes[0].plot(result.time_s, result.pitch_error_history, label=label)
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Pitch err (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    desired_velocity = _desired_velocity_series(next(iter(results.values())).time_s)
    axes[1].set_title("Forward velocity")
    axes[1].plot(next(iter(results.values())).time_s, desired_velocity, color="k", linestyle="--", label="Reference")
    for mode_key, result in results.items():
        axes[1].plot(result.time_s, result.state_history[:, VELOCITY_INDEX], label=MODE_LABELS[mode_key])
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("Model update time")
    for mode_key, result in results.items():
        axes[2].plot(result.time_s, result.model_update_time_history * 1e3, label=MODE_LABELS[mode_key])
    axes[2].set_ylabel("ms")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].set_title("QP solve time")
    for mode_key, result in results.items():
        axes[3].plot(result.time_s, result.qp_solve_time_history * 1e3, label=MODE_LABELS[mode_key])
    axes[3].axhline(TS_MS, color="r", linestyle="--", label="Ts = 20 ms")
    axes[3].set_ylabel("ms")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"slope_{slope_deg:.1f}_timeseries.png", dpi=150)
    plt.close(fig)


def _plot_bar_chart(
    slope_deg: float,
    metrics: Dict[str, Dict[str, float]],
) -> None:
    categories = [
        ("Mean QP (ms)", "mean_qp_solve_ms"),
        ("p95 QP (ms)", "p95_qp_solve_ms"),
        ("Mean model (ms)", "mean_model_update_ms"),
        ("p95 model (ms)", "p95_model_update_ms"),
        ("Pitch err RMS (rad)", "pitch_err_rms_rad"),
    ]
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    objective_vals = [metrics["objective"][key] for _, key in categories]
    relative_vals = [metrics["relative"][key] for _, key in categories]
    ax.bar(x - width / 2, objective_vals, width, label="Objective")
    ax.bar(x + width / 2, relative_vals, width, label="Relative")
    ax.set_xticks(x, [name for name, _ in categories], rotation=15)
    ax.set_title(f"Slope {slope_deg:.1f}° aggregated metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"slope_{slope_deg:.1f}_bar.png", dpi=150)
    plt.close(fig)


def _plot_summary_figures(
    plot_results: Dict[Tuple[float, str], MPCSimulation],
    metrics_by_slope: Dict[float, Dict[str, Dict[str, float]]],
) -> None:
    base_slope = SLOPES_DEG[0]
    base_results = {
        mode: plot_results[(base_slope, mode)] for mode in LINEARIZATION_MODES
    }

    plt.figure(figsize=(9, 4))
    for mode_key, result in base_results.items():
        label = MODE_LABELS[mode_key]
        plt.plot(result.time_s, result.state_history[:, PITCH_INDEX], label=label)
    plt.axhline(-math.radians(base_slope), color="k", linestyle="--", linewidth=1, label="Slope reference")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (rad)")
    plt.title(f"Pitch comparison (slope {base_slope:.1f}°)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pitch_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    for mode_key, result in base_results.items():
        label = MODE_LABELS[mode_key]
        plt.plot(result.time_s, result.qp_solve_time_history * 1e3, label=label)
    plt.axhline(TS_MS, color="r", linestyle="--", label="Ts = 20 ms")
    plt.xlabel("Time (s)")
    plt.ylabel("QP solve (ms)")
    plt.title(f"Solve time comparison (slope {base_slope:.1f}°)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "solve_time_comparison.png", dpi=150)
    plt.close()

    categories = [
        ("Mean QP (ms)", "mean_qp_solve_ms"),
        ("p95 QP (ms)", "p95_qp_solve_ms"),
        ("Pitch err RMS (rad)", "pitch_err_rms_rad"),
    ]
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    objective_vals = [metrics_by_slope[base_slope]["objective"][key] for _, key in categories]
    relative_vals = [metrics_by_slope[base_slope]["relative"][key] for _, key in categories]
    ax.bar(x - width / 2, objective_vals, width, label="Objective")
    ax.bar(x + width / 2, relative_vals, width, label="Relative")
    ax.set_xticks(x, [name for name, _ in categories], rotation=10)
    ax.set_title(f"Metrics comparison (slope {base_slope:.1f}°)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=150)
    plt.close(fig)


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
@pytest.mark.slow
def test_objective_vs_relative_linearization():
    np.random.seed(SEED)
    base_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")
    rows: List[Dict[str, float]] = []
    plot_results: Dict[Tuple[float, str], MPCSimulation] = {}
    metrics_by_slope: Dict[float, Dict[str, Dict[str, float]]] = {}

    for slope_deg in SLOPES_DEG:
        robot_params_path = _write_robot_params_with_slope("config/simulation/robot_params.yaml", slope_deg)
        metrics_by_slope[slope_deg] = {}
        for mode_key, online_flag in LINEARIZATION_MODES.items():
            metrics, result = _run_case(base_cfg, robot_params_path, slope_deg, mode_key, online_flag)
            rows.append(metrics)
            metrics_by_slope[slope_deg][mode_key] = metrics
            plot_results[(slope_deg, mode_key)] = result

        slope_results = {mode: plot_results[(slope_deg, mode)] for mode in LINEARIZATION_MODES}
        _plot_timeseries(slope_deg, slope_results)
        _plot_bar_chart(slope_deg, metrics_by_slope[slope_deg])
    _plot_summary_figures(plot_results, metrics_by_slope)

    csv_path = OUTPUT_DIR / "linearization_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nObjective vs relative metrics written to {csv_path}")

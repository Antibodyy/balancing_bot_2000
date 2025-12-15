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
import random

from matplotlib.lines import Line2D
from mpc import MPCConfig, ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import PITCH_INDEX, VELOCITY_INDEX
from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE

OUTPUT_DIR = Path("test_and_debug_output/linearization_modes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLOPES_DEG = [0.0, 5.0, 10.0, 15.466, 15.467]
LINEARIZATION_MODES = {
    "objective": False,
    "relative": True,
}
MODE_LABELS = {
    "objective": "Objective linearization (fixed A,B)",
    "relative": "Relative linearization (online A,B updates)",
}
SEED = 12345
TS_MS = 65.0
USE_VIRTUAL_PHYSICS = True
TRIM_CUTOFF_S = 0.2

random.seed(SEED)


def _make_reference_profile(slope_rad: float):
    def _profile(time_s: float) -> ReferenceCommand:
        if time_s < 1.0:
            return ReferenceCommand(mode=ReferenceMode.BALANCE, desired_pitch_rad=-slope_rad)
        if time_s < 3.0:
            return ReferenceCommand(
                mode=ReferenceMode.VELOCITY,
                velocity_mps=0.5,
                desired_pitch_rad=-slope_rad,
            )
        if time_s < 4.0:
            return ReferenceCommand(
                mode=ReferenceMode.VELOCITY,
                velocity_mps=-0.3,
                desired_pitch_rad=-slope_rad,
            )
        return ReferenceCommand(mode=ReferenceMode.BALANCE, desired_pitch_rad=-slope_rad)

    return _profile


def _write_robot_params_with_slope(base_yaml: str, slope_deg: float) -> str:
    params = yaml.safe_load(Path(base_yaml).read_text())
    params["ground_slope_rad"] = math.radians(slope_deg)
    tmp = OUTPUT_DIR / f"robot_params_slope_{slope_deg:.3f}.yaml"
    with tmp.open("w") as fh:
        yaml.safe_dump(params, fh)
    return str(tmp)


def _write_temp_config(base: MPCConfig, mode_key: str, online: bool) -> str:
    cfg = replace(
        base,
        prediction_horizon_steps=60,
        warm_start_enabled=True,
        online_linearization_enabled=online,
        preserve_warm_start_on_linearization=True,
    )
    cfg_dict = asdict(cfg)
    tmp = OUTPUT_DIR / f"mpc_params_{mode_key}.yaml"
    with tmp.open("w") as fh:
        yaml.safe_dump(cfg_dict, fh)
    return str(tmp)


def _desired_velocity_series(time_s: np.ndarray) -> np.ndarray:
    velocities = np.zeros_like(time_s)
    velocities[(time_s >= 1.0) & (time_s < 3.0)] = 0.5
    velocities[(time_s >= 3.0) & (time_s < 4.0)] = -0.3
    return velocities


def _mask_after_warmup(result: MPCSimulation, drop_infeasible: bool = False) -> np.ndarray:
    """Return mask excluding the first timestep, early transients, and (optionally) infeasible solves."""
    idx = np.arange(len(result.time_s))
    mask = (result.time_s > TRIM_CUTOFF_S) & (idx > 0)
    if drop_infeasible:
        mask &= (~result.infeasible_history)
    return mask


def _masked_array(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to a 1-D array."""
    if data.size == 0:
        return data
    return data[mask]


def _run_case(
    base_cfg: MPCConfig,
    robot_params_path: str,
    slope_deg: float,
    mode_key: str,
    online_linearization: bool,
) -> Tuple[Dict[str, float], MPCSimulation]:
    slope_rad = math.radians(slope_deg)
    cfg_path = _write_temp_config(base_cfg, f"{mode_key}_slope_{slope_deg:.3f}", online_linearization)
    sim_cfg = SimulationConfig(
        model_path="mujoco_sim/robot_model.xml",
        robot_params_path=robot_params_path,
        mpc_params_path=cfg_path,
        estimator_params_path="config/simulation/estimator_params.yaml",
        duration_s=6.0,
        render=False,
        use_virtual_physics=USE_VIRTUAL_PHYSICS,
    )
    sim = MPCSimulation(sim_cfg)
    reference_callback = _make_reference_profile(slope_rad)
    result = sim.run(
        duration_s=6.0,
        initial_pitch_rad=-slope_rad,
        reference_command_callback=reference_callback,
    )

    desired_pitch = np.full_like(result.time_s, -slope_rad)
    desired_velocity = _desired_velocity_series(result.time_s)
    pitch_error = result.state_history[:, PITCH_INDEX] - desired_pitch
    velocity_error = result.state_history[:, VELOCITY_INDEX] - desired_velocity

    qp_mask = _mask_after_warmup(result, drop_infeasible=True)
    model_mask = _mask_after_warmup(result, drop_infeasible=False)

    trimmed_qp = result.qp_solve_time_history[qp_mask] * 1e3
    trimmed_model = result.model_update_time_history[model_mask] * 1e3
    trimmed_pitch_err = pitch_error[model_mask]
    trimmed_vel_err = velocity_error[model_mask]
    trimmed_sat = result.torque_saturation_history[model_mask].astype(float)
    trimmed_infeasible = result.infeasible_history[model_mask].astype(float)

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

    # Attach derived series for downstream plotting/analysis.
    result.slope_deg = slope_deg
    result.desired_pitch = desired_pitch
    result.desired_velocity = desired_velocity
    result.pitch_error_history = pitch_error
    result.velocity_error_history = velocity_error
    result.qp_mask = qp_mask
    result.model_mask = model_mask
    result.filtered_qp_ms = np.where(qp_mask, result.qp_solve_time_history * 1e3, np.nan)
    result.filtered_model_ms = np.where(model_mask, result.model_update_time_history * 1e3, np.nan)

    return metrics, result


def _plot_timeseries(
    slope_deg: float,
    results: Dict[str, MPCSimulation],
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].set_title(f"Slope {slope_deg:.3f}° – pitch error")
    for mode_key, result in results.items():
        label = MODE_LABELS[mode_key]
        axes[0].plot(result.time_s, result.pitch_error_history, label=label)
    axes[0].axhline(0.0, color="k", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Pitch err (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    sample_result = next(iter(results.values()))
    axes[1].set_title("Forward velocity")
    axes[1].plot(
        sample_result.time_s,
        sample_result.desired_velocity,
        color="k",
        linestyle="--",
        linewidth=1.5,
        label="Desired velocity",
    )
    for mode_key, result in results.items():
        axes[1].plot(result.time_s, result.state_history[:, VELOCITY_INDEX], label=MODE_LABELS[mode_key])
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("Model update time")
    for mode_key, result in results.items():
        axes[2].plot(result.time_s, result.filtered_model_ms, label=MODE_LABELS[mode_key])
    axes[2].set_ylabel("ms")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].set_title("QP solve time")
    for mode_key, result in results.items():
        axes[3].plot(result.time_s, result.filtered_qp_ms, label=MODE_LABELS[mode_key])
    axes[3].axhline(TS_MS, color="r", linestyle=":", label="Ts (65 ms)")
    axes[3].set_ylabel("ms")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"slope_{slope_deg:.3f}_timeseries.png", dpi=150)
    plt.close(fig)


def _plot_tracking_overlays(
    slope_deg: float,
    results: Dict[str, MPCSimulation],
) -> None:
    sample_result = next(iter(results.values()))

    plt.figure(figsize=(9, 4))
    for mode_key, result in results.items():
        plt.plot(result.time_s, result.state_history[:, PITCH_INDEX], label=MODE_LABELS[mode_key])
    plt.plot(
        sample_result.time_s,
        sample_result.desired_pitch,
        "k--",
        linewidth=1.5,
        label="Desired pitch",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (rad)")
    plt.title(f"Pitch tracking — slope {slope_deg:.3f}°")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"slope_{slope_deg:.3f}_pitch.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(
        sample_result.time_s,
        sample_result.desired_velocity,
        "k--",
        linewidth=1.5,
        label="Desired velocity",
    )
    for mode_key, result in results.items():
        plt.plot(result.time_s, result.state_history[:, VELOCITY_INDEX], label=MODE_LABELS[mode_key])
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title(f"Velocity tracking — slope {slope_deg:.3f}°")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"slope_{slope_deg:.3f}_velocity.png", dpi=150)
    plt.close()


def _one_step_rms(result: MPCSimulation, idx: int) -> float:
    """Compute RMS error between 1-step prediction and actual state."""
    pred = np.asarray(getattr(result, "pred_state_history", None))
    actual = np.asarray(result.state_history)
    if pred is None or pred.ndim != 3 or pred.shape[0] < 2:
        return float("nan")
    pred_1 = pred[:-1, 1, idx]
    actual_1 = actual[1:, idx]
    if pred_1.size == 0 or actual_1.size == 0:
        return float("nan")
    err = pred_1 - actual_1
    return float(np.sqrt(np.mean(err**2)))


def _print_e1_stats(mode_label: str, result: MPCSimulation) -> None:
    e1 = np.asarray(getattr(result, "e1_history", None))
    if e1 is None or e1.ndim != 2 or e1.size == 0:
        print(f"[DEBUG] {mode_label}: 1-step error unavailable")
        return
    norms = np.linalg.norm(e1, axis=1)
    if norms.size == 0:
        print(f"[DEBUG] {mode_label}: 1-step error unavailable")
        return
    mean_val = float(np.mean(norms))
    p95_val = float(np.percentile(norms, 95))
    max_val = float(np.max(norms))
    print(
        f"[DEBUG] {mode_label}: 1-step error mean={mean_val:.6f}, "
        f"p95={p95_val:.6f}, max={max_val:.6f}"
    )


def plot_predicted_vs_actual(
    slope_deg: float,
    results: Dict[str, MPCSimulation],
    pitch_index: int,
    velocity_index: int,
    outdir: Path,
    step_stride: int = 20,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"Slope {slope_deg:.3f}° — objective vs relative predictions", y=0.98)

    pitch_handles: List[Line2D] = []
    vel_handles: List[Line2D] = []

    for mode_idx, (mode_key, result) in enumerate(results.items()):
        mode_label = MODE_LABELS[mode_key]
        t = np.asarray(result.time_s)
        X = np.asarray(result.state_history)
        Xpred = np.asarray(getattr(result, "pred_state_history", None))
        if Xpred is None or Xpred.ndim != 3 or t.size == 0 or Xpred.shape[1] < 2:
            continue

        if not np.allclose(Xpred[:, 0, :], X, atol=1e-8, rtol=1e-6):
            raise AssertionError("pred_state_history[:,0,:] must equal state_history")

        Ts = float(getattr(result, "sample_time_s", 0.0) or 0.0)
        if Ts <= 0.0:
            dt = np.diff(t)
            Ts = float(np.median(dt if dt.size > 0 else [1.0]))
        j_grid = np.arange(Xpred.shape[1]) * Ts
        color = f"C{mode_idx}"

        ax = axes[0]
        actual_line, = ax.plot(t, X[:, pitch_index], linewidth=2.0, color=color, label=f"{mode_label} actual")
        pred_line = None
        for k in range(0, t.shape[0], step_stride):
            t_pred = t[k] + j_grid
            x_pred = Xpred[k, :, pitch_index]
            mask = t_pred <= t[-1]
            ax.plot(t_pred[mask], x_pred[mask], color=color, alpha=0.25, linewidth=1.0)
            if pred_line is None and mask.any():
                pred_line = Line2D(
                    [0],
                    [0],
                    color=color,
                    alpha=0.4,
                    linewidth=1.5,
                    label=f"{mode_label} predicted (every {step_stride} steps)",
                )
        if pred_line is not None:
            pitch_handles.extend([actual_line, pred_line])
        else:
            pitch_handles.append(actual_line)

        ax = axes[1]
        actual_line, = ax.plot(t, X[:, velocity_index], linewidth=2.0, color=color, label=f"{mode_label} actual")
        pred_line = None
        for k in range(0, t.shape[0], step_stride):
            t_pred = t[k] + j_grid
            v_pred = Xpred[k, :, velocity_index]
            mask = t_pred <= t[-1]
            ax.plot(t_pred[mask], v_pred[mask], color=color, alpha=0.25, linewidth=1.0)
            if pred_line is None and mask.any():
                pred_line = Line2D(
                    [0],
                    [0],
                    color=color,
                    alpha=0.4,
                    linewidth=1.5,
                    label=f"{mode_label} predicted (every {step_stride} steps)",
                )
        if pred_line is not None:
            vel_handles.extend([actual_line, pred_line])
        else:
            vel_handles.append(actual_line)

        pitch_rms = _one_step_rms(result, pitch_index)
        vel_rms = _one_step_rms(result, velocity_index)
        print(
            f"Slope {slope_deg:.3f}°, {mode_label}: 1-step RMS pitch={pitch_rms:.4f} rad,"
            f" velocity={vel_rms:.4f} m/s"
        )
        _print_e1_stats(mode_label, result)

    axes[0].set_ylabel("Pitch (rad)")
    axes[0].grid(True, alpha=0.3)
    if pitch_handles:
        axes[0].legend(handles=pitch_handles, loc="upper left", fontsize=8)
    axes[0].set_title("Pitch — actual vs MPC predictions (every Nth step)")

    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    if vel_handles:
        axes[1].legend(handles=vel_handles, loc="upper left", fontsize=8)
    axes[1].set_title("Velocity — actual vs MPC predictions (every Nth step)")

    fig.tight_layout()
    fig.savefig(outdir / f"pred_vs_actual_slope_{slope_deg:.3f}.png", dpi=150)
    plt.close(fig)


def _plot_predicted_vs_actual_single(
    slope_deg: float,
    mode_key: str,
    result: MPCSimulation,
    pitch_index: int,
    velocity_index: int,
    outdir: Path,
    step_stride: int = 20,
) -> None:
    mode_label = MODE_LABELS[mode_key]
    t = np.asarray(result.time_s)
    X = np.asarray(result.state_history)
    Xpred = np.asarray(getattr(result, "pred_state_history", None))
    if Xpred is None or Xpred.ndim != 3 or t.size == 0 or Xpred.shape[1] < 2:
        return
    if not np.allclose(Xpred[:, 0, :], X, atol=1e-8, rtol=1e-6):
        raise AssertionError("pred_state_history[:,0,:] must equal state_history")

    Ts = float(getattr(result, "sample_time_s", 0.0) or 0.0)
    if Ts <= 0.0:
        dt = np.diff(t)
        Ts = float(np.median(dt if dt.size > 0 else [1.0]))
    j_grid = np.arange(Xpred.shape[1]) * Ts

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Slope {slope_deg:.3f}° — {mode_label}", y=0.97)

    ax = axes[0]
    actual_pitch_line, = ax.plot(t, X[:, pitch_index], linewidth=2.0, label="Actual")
    pred_handle = None
    for k in range(0, t.shape[0], step_stride):
        t_pred = t[k] + j_grid
        x_pred = Xpred[k, :, pitch_index]
        mask = t_pred <= t[-1]
        line, = ax.plot(t_pred[mask], x_pred[mask], alpha=0.3, linewidth=1.0)
        if pred_handle is None and mask.any():
            pred_handle = Line2D(
                [0],
                [0],
                color=line.get_color(),
                alpha=0.4,
                linewidth=1.5,
                label="Predicted",
            )
    ax.set_ylabel("Pitch (rad)")
    ax.grid(True, alpha=0.3)
    if pred_handle is not None:
        ax.legend(handles=[actual_pitch_line, pred_handle], loc="upper left", fontsize=8)
    else:
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Pitch — actual vs MPC predictions")

    ax = axes[1]
    actual_vel_line, = ax.plot(t, X[:, velocity_index], linewidth=2.0, label="Actual")
    pred_handle = None
    for k in range(0, t.shape[0], step_stride):
        t_pred = t[k] + j_grid
        v_pred = Xpred[k, :, velocity_index]
        mask = t_pred <= t[-1]
        line, = ax.plot(t_pred[mask], v_pred[mask], alpha=0.3, linewidth=1.0)
        if pred_handle is None and mask.any():
            pred_handle = Line2D(
                [0],
                [0],
                color=line.get_color(),
                alpha=0.4,
                linewidth=1.5,
                label="Predicted",
            )
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    if pred_handle is not None:
        ax.legend(handles=[actual_vel_line, pred_handle], loc="upper left", fontsize=8)
    else:
        ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Velocity — actual vs MPC predictions")

    fig.tight_layout()
    fig.savefig(outdir / f"pred_vs_actual_slope_{slope_deg:.3f}_{mode_key}.png", dpi=150)
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
    ax.set_title(f"Slope {slope_deg:.3f}° aggregated metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"slope_{slope_deg:.3f}_bar.png", dpi=150)
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
    sample_result = next(iter(base_results.values()))
    plt.plot(
        sample_result.time_s,
        sample_result.desired_pitch,
        "k--",
        linewidth=1.5,
        label="Desired pitch",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (rad)")
    plt.title(f"Pitch comparison (slope {base_slope:.3f}°)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pitch_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    for mode_key, result in base_results.items():
        label = MODE_LABELS[mode_key]
        plt.plot(result.time_s, result.filtered_qp_ms, label=label)
    plt.axhline(TS_MS, color="r", linestyle=":", label="Ts (65 ms)")
    plt.xlabel("Time (s)")
    plt.ylabel("QP solve (ms)")
    plt.title(f"Solve time comparison (slope {base_slope:.3f}°)")
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
    ax.set_title(f"Metrics comparison (slope {base_slope:.3f}°)")
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
        _plot_tracking_overlays(slope_deg, slope_results)
        _plot_bar_chart(slope_deg, metrics_by_slope[slope_deg])
        for mode_key in LINEARIZATION_MODES:
            _plot_predicted_vs_actual_single(
                slope_deg=slope_deg,
                mode_key=mode_key,
                result=slope_results[mode_key],
                pitch_index=PITCH_INDEX,
                velocity_index=VELOCITY_INDEX,
                outdir=OUTPUT_DIR,
                step_stride=20,
            )
        plot_predicted_vs_actual(
            slope_deg=slope_deg,
            results=slope_results,
            pitch_index=PITCH_INDEX,
            velocity_index=VELOCITY_INDEX,
            outdir=OUTPUT_DIR,
            step_stride=20,
        )
    _plot_summary_figures(plot_results, metrics_by_slope)

    csv_path = OUTPUT_DIR / "linearization_metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nObjective vs relative metrics written to {csv_path}")

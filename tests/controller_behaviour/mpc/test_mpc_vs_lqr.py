"""
Direct comparison of MPC vs LQR control performance across slopes.

This test reuses the MPCSimulation infrastructure for MPC runs and adds a
lightweight LQR controller/plant loop that shares the same sampling period,
reference commands, and slope-specific robot parameters. Metrics and plots are
emitted to test_and_debug_output/mpc_vs_lqr for further analysis.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from scipy.linalg import solve_discrete_are

from mpc import MPCConfig, ReferenceCommand, ReferenceMode
from robot_dynamics import RobotParameters
from robot_dynamics.parameters import PITCH_INDEX, VELOCITY_INDEX
from robot_dynamics.continuous_dynamics import (
    compute_state_derivative,
    compute_equilibrium_state,
)
from robot_dynamics.linearization import linearize_at_equilibrium
from robot_dynamics.discretization import discretize_linear_dynamics
from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE

OUTPUT_DIR = Path("test_and_debug_output/mpc_vs_lqr")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLOPES_DEG = [0.0, 5.0, 10.0]
DURATION_S = 6.0
USE_VIRTUAL_PHYSICS = True
PREDICTION_STEP_STRIDE = 20


def _make_reference_profile(slope_rad: float):
    def _profile(time_s: float) -> ReferenceCommand:
        if time_s < 1.0:
            return ReferenceCommand(
                mode=ReferenceMode.BALANCE,
                desired_pitch_rad=-slope_rad,
            )
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
        return ReferenceCommand(
            mode=ReferenceMode.BALANCE,
            desired_pitch_rad=-slope_rad,
        )

    return _profile


def _write_robot_params_with_slope(base_yaml: str, slope_deg: float) -> str:
    params = yaml.safe_load(Path(base_yaml).read_text())
    params["ground_slope_rad"] = math.radians(slope_deg)
    tmp = OUTPUT_DIR / f"robot_params_slope_{slope_deg:.3f}.yaml"
    with tmp.open("w") as fh:
        yaml.safe_dump(params, fh)
    return str(tmp)


def _desired_pitch_velocity(command: ReferenceCommand, fallback_pitch: float) -> Tuple[float, float]:
    desired_pitch = command.desired_pitch_rad if hasattr(command, "desired_pitch_rad") else fallback_pitch
    desired_velocity = command.velocity_mps if command.mode == ReferenceMode.VELOCITY else 0.0
    return desired_pitch, desired_velocity


@dataclass
class LQRResult:
    time_s: np.ndarray
    state_history: np.ndarray
    control_history: np.ndarray
    desired_pitch: np.ndarray
    desired_velocity: np.ndarray
    pitch_error_history: np.ndarray
    velocity_error_history: np.ndarray
    torque_saturation_history: np.ndarray
    solve_time_history: np.ndarray
    success: bool


class LQRController:
    """Discrete-time LQR controller aligned with the MPC linearization."""

    def __init__(self, params: RobotParameters, mpc_cfg: MPCConfig) -> None:
        self._params = params
        self._mpc_cfg = mpc_cfg
        self.control_limit = mpc_cfg.control_limit_nm
        self.eq_state, self.eq_control = compute_equilibrium_state(params)
        print(f"Effective Ts = {mpc_cfg.sampling_period_s}")

        lin = linearize_at_equilibrium(params, self.eq_state, self.eq_control)
        discrete = discretize_linear_dynamics(
            lin.state_matrix,
            lin.control_matrix,
            mpc_cfg.sampling_period_s,
        )

        Q = mpc_cfg.state_cost_matrix
        R = mpc_cfg.control_cost_matrix
        A_d = discrete.state_matrix_discrete
        B_d = discrete.control_matrix_discrete
        P = solve_discrete_are(A_d, B_d, Q, R)
        self._K = np.linalg.solve(B_d.T @ P @ B_d + R, B_d.T @ P @ A_d)
        self._A_d = A_d
        self._B_d = B_d
        self.Ts = discrete.sampling_period_s

    def compute_control(self, state: np.ndarray, ref_state: np.ndarray) -> np.ndarray:
        err = state - ref_state
        delta_u = -self._K @ err
        total = self.eq_control + delta_u
        return np.clip(total, -self.control_limit, self.control_limit)


def _run_mpc_case(
    slope_deg: float,
    base_cfg: MPCConfig,
    robot_params_path: str,
    reference_cb,
) -> Tuple[Dict[str, float], MPCSimulation]:
    slope_rad = math.radians(slope_deg)
    sim_cfg = SimulationConfig(
        model_path="mujoco_sim/robot_model.xml",
        robot_params_path=robot_params_path,
        mpc_params_path="config/simulation/mpc_params.yaml",
        estimator_params_path="config/simulation/estimator_params.yaml",
        duration_s=DURATION_S,
        render=False,
        use_virtual_physics=USE_VIRTUAL_PHYSICS,
    )
    sim = MPCSimulation(sim_cfg)
    result = sim.run(
        duration_s=DURATION_S,
        initial_pitch_rad=-slope_rad,
        reference_command_callback=reference_cb,
    )

    def _safe_rms(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")

    pitch_rms = _safe_rms(result.pitch_error_history)
    velocity_rms = _safe_rms(result.velocity_error_history)
    torque_pct = float(np.mean(result.torque_saturation_history) * 100.0) if result.torque_saturation_history.size else 0.0
    solve_mean = float(np.mean(result.qp_solve_time_history) * 1e3) if result.qp_solve_time_history.size else 0.0
    infeasible_pct = float(np.mean(result.infeasible_history) * 100.0) if result.infeasible_history.size else 0.0

    metrics = {
        "slope_deg": slope_deg,
        "controller": "MPC",
        "pitch_err_rms_rad": pitch_rms,
        "velocity_err_rms_mps": velocity_rms,
        "torque_saturation_pct": torque_pct,
        "mean_solve_ms": solve_mean,
        "constraint_violation_pct": infeasible_pct,
    }
    return metrics, result


def _run_lqr_case(
    slope_deg: float,
    base_cfg: MPCConfig,
    robot_params_path: str,
    reference_cb,
) -> Tuple[Dict[str, float], LQRResult]:
    params = RobotParameters.from_yaml(robot_params_path)
    controller = LQRController(params, base_cfg)
    slope_rad = math.radians(slope_deg)
    Ts = base_cfg.sampling_period_s
    steps = int(DURATION_S / Ts)
    substeps = 5
    dt = Ts / substeps

    state = controller.eq_state.copy()
    state[PITCH_INDEX] += -slope_rad

    time_hist: List[float] = []
    state_hist: List[np.ndarray] = []
    control_hist: List[np.ndarray] = []
    desired_pitch_hist: List[float] = []
    desired_vel_hist: List[float] = []
    pitch_err_hist: List[float] = []
    vel_err_hist: List[float] = []
    torque_sat_hist: List[bool] = []
    solve_hist: List[float] = []
    success = True

    for step in range(steps):
        t = step * Ts
        time_hist.append(t)
        state_hist.append(state.copy())

        command = reference_cb(t)
        desired_pitch, desired_vel = _desired_pitch_velocity(command, -slope_rad)
        desired_pitch_hist.append(desired_pitch)
        desired_vel_hist.append(desired_vel)
        pitch_err_hist.append(state[PITCH_INDEX] - desired_pitch)
        vel_err_hist.append(state[VELOCITY_INDEX] - desired_vel)

        ref_state = controller.eq_state.copy()
        ref_state[PITCH_INDEX] = desired_pitch
        ref_state[VELOCITY_INDEX] = desired_vel

        start = time.perf_counter()
        control = controller.compute_control(state, ref_state)
        solve_hist.append(time.perf_counter() - start)
        control_hist.append(control.copy())
        torque_sat_hist.append(bool(np.any(np.abs(control) >= controller.control_limit - 1e-6)))

        for _ in range(substeps):
            deriv = compute_state_derivative(state, control, params)
            state = state + deriv * dt

        if abs(state[PITCH_INDEX]) > base_cfg.pitch_limit_rad:
            success = False
            break

    result = LQRResult(
        time_s=np.asarray(time_hist),
        state_history=np.asarray(state_hist),
        control_history=np.asarray(control_hist),
        desired_pitch=np.asarray(desired_pitch_hist),
        desired_velocity=np.asarray(desired_vel_hist),
        pitch_error_history=np.asarray(pitch_err_hist),
        velocity_error_history=np.asarray(vel_err_hist),
        torque_saturation_history=np.asarray(torque_sat_hist, dtype=bool),
        solve_time_history=np.asarray(solve_hist),
        success=success,
    )

    def _safe_rms(arr: np.ndarray) -> float:
        return float(np.sqrt(np.mean(arr**2))) if arr.size else float("nan")

    metrics = {
        "slope_deg": slope_deg,
        "controller": "LQR",
        "pitch_err_rms_rad": _safe_rms(result.pitch_error_history),
        "velocity_err_rms_mps": _safe_rms(result.velocity_error_history),
        "torque_saturation_pct": float(np.mean(result.torque_saturation_history) * 100.0) if result.torque_saturation_history.size else 0.0,
        "mean_solve_ms": float(np.mean(result.solve_time_history) * 1e3) if result.solve_time_history.size else 0.0,
        "constraint_violation_pct": 0.0,
    }
    return metrics, result


def _plot_timeseries(
    slope_deg: float,
    mpc_res: MPCSimulation,
    lqr_res: LQRResult,
    mpc_cfg: MPCConfig,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)
    fig.suptitle(f"Slope {slope_deg:.3f}° — MPC vs LQR timeseries")

    axes[0].plot(mpc_res.time_s, mpc_res.pitch_error_history, label="MPC", linewidth=2)
    axes[0].plot(lqr_res.time_s, lqr_res.pitch_error_history, label="LQR", linewidth=2, linestyle="--")
    axes[0].set_ylabel("Pitch err (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(mpc_res.time_s, mpc_res.state_history[:, VELOCITY_INDEX], label="MPC velocity")
    axes[1].plot(lqr_res.time_s, lqr_res.state_history[:, VELOCITY_INDEX], label="LQR velocity", linestyle="--")
    axes[1].plot(lqr_res.time_s, lqr_res.desired_velocity, label="Reference", linestyle=":", color="k")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    mpc_u = mpc_res.u_applied_history[:, 0] if hasattr(mpc_res, "u_applied_history") and mpc_res.u_applied_history.size else np.zeros_like(mpc_res.time_s)
    axes[2].plot(mpc_res.time_s[: len(mpc_u)], mpc_u, label="MPC torque", linewidth=2)
    axes[2].plot(lqr_res.time_s[: len(lqr_res.control_history)], lqr_res.control_history[:, 0], label="LQR torque", linestyle="--")
    axes[2].set_ylabel("Left torque (Nm)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(mpc_res.time_s, mpc_res.qp_solve_time_history * 1e3, label="MPC solve", linewidth=2)
    axes[3].plot(lqr_res.time_s, lqr_res.solve_time_history * 1e3, label="LQR solve", linestyle="--")
    axes[3].axhline(mpc_cfg.sampling_period_s * 1e3, color="r", linestyle=":", label="Ts (65 ms)")
    axes[3].set_ylabel("Solve time (ms)")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"slope_{slope_deg:.3f}_timeseries.png", dpi=150)
    plt.close(fig)


def _plot_prediction_accuracy(
    slope_deg: float,
    mpc_res: MPCSimulation,
    lqr_res: LQRResult,
) -> None:
    if mpc_res.pred_state_history.size == 0 or not mpc_res.time_s.size:
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Slope {slope_deg:.3f}° — MPC prediction accuracy")

    Ts = float(getattr(mpc_res, "sample_time_s", 0.0) or np.median(np.diff(mpc_res.time_s)))
    horizon = mpc_res.pred_state_history.shape[1]
    j_grid = np.arange(horizon) * Ts

    axes[0].plot(mpc_res.time_s, mpc_res.state_history[:, PITCH_INDEX], label="MPC actual", linewidth=2)
    axes[0].plot(lqr_res.time_s, lqr_res.state_history[:, PITCH_INDEX], label="LQR actual", linestyle="--")
    for k in range(0, mpc_res.pred_state_history.shape[0], PREDICTION_STEP_STRIDE):
        t_pred = mpc_res.time_s[k] + j_grid
        x_pred = mpc_res.pred_state_history[k, :, PITCH_INDEX]
        mask = t_pred <= mpc_res.time_s[-1]
        axes[0].plot(t_pred[mask], x_pred[mask], alpha=0.2, color="tab:blue")
    axes[0].set_ylabel("Pitch (rad)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(mpc_res.time_s, mpc_res.state_history[:, VELOCITY_INDEX], label="MPC actual", linewidth=2)
    axes[1].plot(lqr_res.time_s, lqr_res.state_history[:, VELOCITY_INDEX], label="LQR actual", linestyle="--")
    axes[1].plot(lqr_res.time_s, lqr_res.desired_velocity, label="Reference", linestyle=":", color="k")
    for k in range(0, mpc_res.pred_state_history.shape[0], PREDICTION_STEP_STRIDE):
        t_pred = mpc_res.time_s[k] + j_grid
        v_pred = mpc_res.pred_state_history[k, :, VELOCITY_INDEX]
        mask = t_pred <= mpc_res.time_s[-1]
        axes[1].plot(t_pred[mask], v_pred[mask], alpha=0.2, color="tab:orange")
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"slope_{slope_deg:.3f}_prediction.png", dpi=150)
    plt.close(fig)


def _plot_metric_bar(slope_deg: float, mpc_metrics: Dict[str, float], lqr_metrics: Dict[str, float]) -> None:
    categories = [
        ("Pitch RMS (rad)", "pitch_err_rms_rad"),
        ("Velocity RMS (m/s)", "velocity_err_rms_mps"),
        ("Torque sat (%)", "torque_saturation_pct"),
        ("Mean solve (ms)", "mean_solve_ms"),
    ]
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    mpc_vals = [mpc_metrics[key] for _, key in categories]
    lqr_vals = [lqr_metrics[key] for _, key in categories]
    ax.bar(x - width / 2, mpc_vals, width, label="MPC")
    ax.bar(x + width / 2, lqr_vals, width, label="LQR")
    ax.set_xticks(x, [name for name, _ in categories], rotation=15)
    ax.set_title(f"Slope {slope_deg:.3f}° metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"slope_{slope_deg:.3f}_metrics.png", dpi=150)
    plt.close(fig)


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
@pytest.mark.slow
def test_mpc_vs_lqr():
    base_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")
    metrics_rows: List[Dict[str, float]] = []

    for slope_deg in SLOPES_DEG:
        robot_params_path = _write_robot_params_with_slope("config/simulation/robot_params.yaml", slope_deg)
        reference_cb = _make_reference_profile(math.radians(slope_deg))

        mpc_metrics, mpc_res = _run_mpc_case(slope_deg, base_cfg, robot_params_path, reference_cb)
        lqr_metrics, lqr_res = _run_lqr_case(slope_deg, base_cfg, robot_params_path, reference_cb)

        metrics_rows.append(mpc_metrics)
        metrics_rows.append(lqr_metrics)

        _plot_timeseries(slope_deg, mpc_res, lqr_res, base_cfg)
        _plot_prediction_accuracy(slope_deg, mpc_res, lqr_res)
        _plot_metric_bar(slope_deg, mpc_metrics, lqr_metrics)

    csv_path = OUTPUT_DIR / "mpc_vs_lqr_metrics.csv"
    fieldnames = [
        "slope_deg",
        "controller",
        "pitch_err_rms_rad",
        "velocity_err_rms_mps",
        "torque_saturation_pct",
        "mean_solve_ms",
        "constraint_violation_pct",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_rows)

    print(f"MPC vs LQR comparison written to {csv_path}")

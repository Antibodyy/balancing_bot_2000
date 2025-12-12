"""
Comparative tests: LQR vs MPC under identical scenarios.

These tests run MPC with the MuJoCo simulation as the "real" system and run a
linear discrete LQR controller on a linearized + discretized model as the
"expected" trajectory. Plots overlay both for visual inspection.

Note: MuJoCo must be available; otherwise tests are skipped.
"""

import math
from pathlib import Path
import sys
from dataclasses import replace

# Ensure project root on path for direct script execution
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pytest

# Headless-safe plotting (important for CI / no-display environments)
import matplotlib
matplotlib.use("Agg")  # noqa: E402

from control import dlqr  # python-control package

from robot_dynamics.parameters import RobotParameters
from robot_dynamics.linearization import linearize_at_state, linearize_at_equilibrium
from robot_dynamics.discretization import discretize_linear_dynamics
from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE
from mpc import MPCConfig
from LQR.LQR_Dynamics_update import linearize_shared_dynamics, solve_discrete_lqr


OUTPUT_DIR = Path("test_and_debug_output/LQR_vs_MPC")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_lqr_gain(
    params: RobotParameters,
    mpc_cfg: MPCConfig,
    Ts: float,
    ref_state: np.ndarray | None = None,
):
    """
    Build discrete-time LQR gain K using the same Q/R as MPC.

    - If ref_state is None: linearize at equilibrium (typical balance point).
    - Else: linearize at provided operating point (relative linearization).
    """
    if ref_state is None:
        # Shared linearization matching MPC dynamics
        A, B, _, _ = linearize_shared_dynamics(params)
    else:
        lin = linearize_at_state(params, ref_state, np.zeros(2))
        A, B = lin.state_matrix, lin.control_matrix

    disc = discretize_linear_dynamics(A, B, Ts)

    Q = mpc_cfg.state_cost_matrix
    R = mpc_cfg.control_cost_matrix

    # Use shared discrete LQR solver (DARE) for consistency with MPC Ts
    K, _ = solve_discrete_lqr(disc.state_matrix_discrete, disc.control_matrix_discrete, Q, R)
    return disc, K


def simulate_lqr(disc, K, x0, steps, ref=None, u_limit=None):
    x = np.array(x0, dtype=float)
    traj = [x.copy()]
    for k in range(steps):
        r = np.zeros_like(x)
        if ref is not None:
            r = ref(k * disc.sampling_period_s)

        u = -K @ (x - r)

        if u_limit is not None:
            u = np.clip(u, -u_limit, u_limit)

        x = disc.state_matrix_discrete @ x + disc.control_matrix_discrete @ u
        traj.append(x.copy())
    return np.array(traj)


def run_mpc(duration_s: float, initial_pitch: float, ref_cb):
    cfg = SimulationConfig(
        model_path="Mujoco sim/robot_model.xml",
        robot_params_path="config/simulation/robot_params.yaml",
        mpc_params_path="config/simulation/mpc_params.yaml",
        estimator_params_path="config/simulation/estimator_params.yaml",
        duration_s=duration_s,
        render=False,
    )
    sim = MPCSimulation(cfg)
    result = sim.run(
        duration_s=duration_s,
        initial_pitch_rad=initial_pitch,
        reference_command_callback=ref_cb,
    )
    return result


def plot_overlay(time, mpc_state, lqr_state, fname, title, pred_time=None, pred_state=None):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = [("Position x (m)", 0), ("Pitch (deg)", 1), ("Yaw (deg)", 2)]
    for ax, (label, idx) in zip(axs, labels):
        ax.plot(time, mpc_state[:, idx], label="MPC (true)", linewidth=2)
        ax.plot(time, lqr_state[: len(time), idx], "--", label="LQR (linear model)", linewidth=2)
        if pred_time is not None and pred_state is not None:
            ax.plot(pred_time[: len(pred_state)], pred_state[: len(pred_time), idx], ":", label="MPC predicted (1-step)", linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    axs[0].legend()
    fname.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close(fig)


def build_reference_callback(mode, velocity=0.0, yaw_rate=0.0):
    from mpc import ReferenceCommand, ReferenceMode

    def cb(t):
        if mode == "balance":
            return ReferenceCommand(mode=ReferenceMode.BALANCE)
        if mode == "drive_stop":
            if t < 2.0:
                return ReferenceCommand(mode=ReferenceMode.VELOCITY, velocity_mps=velocity)
            return ReferenceCommand(mode=ReferenceMode.BALANCE)
        if mode == "circle":
            return ReferenceCommand(mode=ReferenceMode.VELOCITY, velocity_mps=velocity, yaw_rate_radps=yaw_rate)
        if mode == "slope":
            return ReferenceCommand(mode=ReferenceMode.BALANCE)
        return ReferenceCommand(mode=ReferenceMode.BALANCE)

    return cb


def make_lqr_ref(mode, velocity=0.0, yaw_rate=0.0):
    def ref(t):
        r = np.zeros(6)
        if mode == "drive_stop" and t < 2.0:
            r[3] = velocity
        if mode == "circle":
            r[3] = velocity
            r[5] = yaw_rate
        return r

    return ref


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
@pytest.mark.slow
def test_lqr_vs_mpc_balance():
    duration = 3.0
    initial_pitch = math.radians(3.0)
    ref_cb = build_reference_callback("balance")
    mpc_res = run_mpc(duration, initial_pitch, ref_cb)

    params = RobotParameters.from_yaml("config/simulation/robot_params.yaml")
    mpc_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")
    disc, K = build_lqr_gain(params, mpc_cfg, mpc_cfg.sampling_period_s)

    x0 = np.zeros(6)
    x0[1] = initial_pitch
    lqr_traj = simulate_lqr(disc, K, x0, len(mpc_res.time_s))

    # Predicted one-step-ahead from MPC
    pred = np.array([traj[1] for traj in mpc_res.reference_history]) if mpc_res.reference_history else None
    pred_time = mpc_res.time_s + mpc_cfg.sampling_period_s if mpc_res.reference_history else None

    print(f"Balance: mean solve {mpc_res.mean_solve_time_ms:.2f}ms, max solve {mpc_res.max_solve_time_ms:.2f}ms")

    plot_overlay(
        mpc_res.time_s,
        mpc_res.state_history,
        lqr_traj,
        OUTPUT_DIR / "balance.png",
        "Balance: LQR vs MPC",
        pred_time=pred_time,
        pred_state=pred,
    )


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
@pytest.mark.slow
def test_lqr_vs_mpc_drive_stop():
    duration = 4.0
    velocity = 0.3
    ref_cb = build_reference_callback("drive_stop", velocity=velocity)
    mpc_res = run_mpc(duration, 0.0, ref_cb)

    params = RobotParameters.from_yaml("config/simulation/robot_params.yaml")
    mpc_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")
    disc, K = build_lqr_gain(params, mpc_cfg, mpc_cfg.sampling_period_s)

    lqr_traj = simulate_lqr(disc, K, np.zeros(6), len(mpc_res.time_s), ref=make_lqr_ref("drive_stop", velocity))

    pred = np.array([traj[1] for traj in mpc_res.reference_history]) if mpc_res.reference_history else None
    pred_time = mpc_res.time_s + mpc_cfg.sampling_period_s if mpc_res.reference_history else None

    print(f"Drive/Stop: mean solve {mpc_res.mean_solve_time_ms:.2f}ms, max solve {mpc_res.max_solve_time_ms:.2f}ms")

    plot_overlay(
        mpc_res.time_s,
        mpc_res.state_history,
        lqr_traj,
        OUTPUT_DIR / "drive_stop.png",
        "Drive/Stop: LQR vs MPC",
        pred_time=pred_time,
        pred_state=pred,
    )


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
@pytest.mark.slow
def test_lqr_vs_mpc_circle():
    duration = 5.0
    velocity = 0.2
    yaw_rate = 0.4
    ref_cb = build_reference_callback("circle", velocity=velocity, yaw_rate=yaw_rate)
    mpc_res = run_mpc(duration, 0.0, ref_cb)

    params = RobotParameters.from_yaml("config/simulation/robot_params.yaml")
    mpc_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")

    ref_state = np.zeros(6)
    ref_state[3] = velocity
    ref_state[5] = yaw_rate

    disc, K = build_lqr_gain(params, mpc_cfg, mpc_cfg.sampling_period_s, ref_state=ref_state)
    lqr_traj = simulate_lqr(disc, K, np.zeros(6), len(mpc_res.time_s), ref=make_lqr_ref("circle", velocity, yaw_rate))

    pred = np.array([traj[1] for traj in mpc_res.reference_history]) if mpc_res.reference_history else None
    pred_time = mpc_res.time_s + mpc_cfg.sampling_period_s if mpc_res.reference_history else None

    print(f"Circle: mean solve {mpc_res.mean_solve_time_ms:.2f}ms, max solve {mpc_res.max_solve_time_ms:.2f}ms")

    plot_overlay(
        mpc_res.time_s,
        mpc_res.state_history,
        lqr_traj,
        OUTPUT_DIR / "circle.png",
        "Circle: LQR vs MPC",
        pred_time=pred_time,
        pred_state=pred,
    )


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
@pytest.mark.slow
def test_lqr_vs_mpc_slope_balance():
    duration = 3.0
    slope_deg = 5.0
    initial_pitch = 0.0
    ref_cb = build_reference_callback("slope")

    mpc_res = run_mpc(duration, initial_pitch, ref_cb)

    params = RobotParameters.from_yaml("config/simulation/robot_params.yaml")
    params = replace(params, ground_slope_rad=math.radians(slope_deg))

    mpc_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")
    disc, K = build_lqr_gain(params, mpc_cfg, mpc_cfg.sampling_period_s)

    lqr_traj = simulate_lqr(disc, K, np.zeros(6), len(mpc_res.time_s))
    pred = np.array([traj[1] for traj in mpc_res.reference_history]) if mpc_res.reference_history else None
    pred_time = mpc_res.time_s + mpc_cfg.sampling_period_s if mpc_res.reference_history else None

    print(f"Slope {slope_deg}deg: mean solve {mpc_res.mean_solve_time_ms:.2f}ms, max solve {mpc_res.max_solve_time_ms:.2f}ms")

    plot_overlay(
        mpc_res.time_s,
        mpc_res.state_history,
        lqr_traj,
        OUTPUT_DIR / "slope.png",
        f"Slope {slope_deg} deg: LQR vs MPC",
        pred_time=pred_time,
        pred_state=pred,
    )


def main() -> None:
    """Run all scenarios to generate plots when invoked directly."""
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available; skipping LQR vs MPC plots.")
        return

    scenarios = [
        ("balance", test_lqr_vs_mpc_balance),
        ("drive_stop", test_lqr_vs_mpc_drive_stop),
        ("circle", test_lqr_vs_mpc_circle),
        ("slope", test_lqr_vs_mpc_slope_balance),
    ]
    print("Running LQR vs MPC scenarios...")
    for name, fn in scenarios:
        try:
            fn()
            print(f"  OK {name} plot generated.")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {name} failed: {exc}")
    print(f"Plots saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

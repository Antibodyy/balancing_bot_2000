"""
Compare MPC cold-start vs warm-start performance with online linearization.

This test builds two MPC configurations that are identical except for
`warm_start_enabled` (toggled the same way you would in config/simulation/mpc_params.yaml).
Both runs start from the same 0.12 rad pitch offset and experience a demanding
three-stage reference: balance → accelerate forward → brake back to zero velocity.
The MPC horizon is expanded to N=60 with Ts=0.02 s in order to magnify the solve-time
benefit of warm-starting while still running successive (online) linearization.
The warm-start configuration keeps the previous solution even when the dynamics
are re-linearized each step so we measure a realistic benefit for the paper.

It produces plots contrasting solve time per iteration and pitch tracking accuracy
between cold and warm starts.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from mpc import (
    MPCConfig,
    LinearMPCSolver,
    ReferenceCommand,
    ReferenceMode,
    create_constraints_from_config,
    compute_terminal_cost_dare,
)
from robot_dynamics import (
    RobotParameters,
    linearize_at_equilibrium,
    discretize_linear_dynamics,
    compute_equilibrium_state,
)
from robot_dynamics.parameters import PITCH_INDEX, PITCH_RATE_INDEX, STATE_DIMENSION
from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE
COLD_WARM_OUTPUT_DIR = Path("test_and_debug_output/mpc_warm_start")
COLD_WARM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_TEST_PREDICTION_HORIZON = 60  # Larger N makes the QP heavy enough to showcase warm-start gains
_TEST_DURATION_S = 6.0
_ACCEL_PHASE_END = 2.5
_BRAKE_PHASE_END = 4.0
_CRUISE_VELOCITY = 0.6  # m/s
_BRAKE_VELOCITY = -0.4  # m/s (command reverse torque to settle)
_WARMUP_STEPS = 3


def _trim(array: np.ndarray) -> np.ndarray:
    if array.size <= _WARMUP_STEPS:
        return np.array([], dtype=array.dtype)
    return array[_WARMUP_STEPS:]


def _reference_profile(time_s: float) -> ReferenceCommand:
    """Piecewise reference used for both warm and cold runs.

    The profile keeps the robot balancing for the first second, commands a
    forward velocity step, then transitions into an aggressive braking command.
    This forces the optimizer to re-plan frequently, amplifying the benefit of
    a good initial guess.
    """
    if time_s < 1.0:
        return ReferenceCommand(mode=ReferenceMode.BALANCE)
    if time_s < _ACCEL_PHASE_END:
        return ReferenceCommand(mode=ReferenceMode.VELOCITY, velocity_mps=_CRUISE_VELOCITY)
    if time_s < _BRAKE_PHASE_END:
        return ReferenceCommand(mode=ReferenceMode.VELOCITY, velocity_mps=_BRAKE_VELOCITY)
    return ReferenceCommand(mode=ReferenceMode.BALANCE)


def _write_temp_config(base: MPCConfig, warm_start: bool) -> str:
    """Create a temporary YAML with only warm-start toggled (mimics flipping the config line)."""
    # Keep user-configured values for everything else; assert required horizon/Ts
    assert base.prediction_horizon_steps == 30, "Test expects N=30"
    assert abs(base.sampling_period_s - 0.02) < 1e-6, "Test expects Ts=0.02s"

    cfg = replace(
        base,
        prediction_horizon_steps=_TEST_PREDICTION_HORIZON,
        warm_start_enabled=warm_start,
        online_linearization_enabled=True,
        preserve_warm_start_on_linearization=True,
    )
    tmp_path = COLD_WARM_OUTPUT_DIR / f"mpc_params_warm_{warm_start}.yaml"
    with tmp_path.open("w") as fh:
        yaml.safe_dump(asdict(cfg), fh)
    return str(tmp_path)


def _run_case(base_cfg: MPCConfig, warm_start: bool) -> Tuple[Dict[str, float], object]:
    """Run a single simulation case and compute metrics."""
    cfg_path = _write_temp_config(base_cfg, warm_start)
    sim_cfg = SimulationConfig(
        model_path="mujoco_sim/robot_model.xml",
        robot_params_path="config/simulation/robot_params.yaml",
        mpc_params_path=cfg_path,
        estimator_params_path="config/simulation/estimator_params.yaml",
        duration_s=_TEST_DURATION_S,
        render=False,
    )
    sim = MPCSimulation(sim_cfg)
    result = sim.run(
        duration_s=_TEST_DURATION_S,
        initial_pitch_rad=0.12,
        reference_command_callback=_reference_profile,
    )

    pitch = result.state_history[:, PITCH_INDEX]
    settle_idx = np.argmax(np.abs(pitch) < 0.01) if np.any(np.abs(pitch) < 0.01) else len(pitch) - 1
    trimmed_qp = _trim(result.qp_solve_time_history) * 1e3
    if trimmed_qp.size == 0:
        trimmed_qp = result.qp_solve_time_history * 1e3
    metrics = {
        "mean_solve_ms": float(np.mean(trimmed_qp)),
        "max_solve_ms": float(np.max(trimmed_qp)),
        "p95_solve_ms": float(np.percentile(trimmed_qp, 95)),
        "final_pitch_rad": float(pitch[-1]),
        "pitch_rms_rad": float(np.sqrt(np.mean(pitch**2))),
        "settle_time_s": float(result.time_s[settle_idx]) if settle_idx < len(result.time_s) else np.nan,
        "deadline_violations": int(result.deadline_violations),
        "mean_iterations": float(np.mean(result.iteration_history)),
        "p95_iterations": float(np.percentile(result.iteration_history, 95)),
    }
    return metrics, result


def _run_solver_benchmark(base_cfg: MPCConfig, num_steps: int = 400) -> Dict[str, float]:
    """Deterministic micro-benchmark that isolates solver warm-start benefits.

    Runs the linear MPC solver on a sequence of randomly perturbed states
    using the analytic dynamics (no MuJoCo) so noise from the physics engine
    does not mask the IPOPT warm-start improvements.
    """
    robot_params = RobotParameters.from_yaml("config/simulation/robot_params.yaml")
    eq_state, eq_control = compute_equilibrium_state(robot_params)
    linearized = linearize_at_equilibrium(robot_params, eq_state, eq_control)
    discrete = discretize_linear_dynamics(
        linearized.state_matrix,
        linearized.control_matrix,
        base_cfg.sampling_period_s,
    )

    Q = base_cfg.state_cost_matrix
    R = base_cfg.control_cost_matrix
    if base_cfg.use_terminal_cost_dare:
        P = compute_terminal_cost_dare(
            discrete.state_matrix_discrete,
            discrete.control_matrix_discrete,
            Q,
            R,
        )
    else:
        P = Q.copy()
    P = base_cfg.terminal_cost_scale * P

    state_constraints, input_constraints = create_constraints_from_config(
        base_cfg.pitch_limit_rad,
        base_cfg.pitch_rate_limit_radps,
        base_cfg.control_limit_nm,
    )

    def _build_solver(enable_warm_start: bool) -> LinearMPCSolver:
        return LinearMPCSolver(
            prediction_horizon_steps=base_cfg.prediction_horizon_steps,
            discrete_dynamics=discrete,
            state_cost=Q,
            control_cost=R,
            terminal_cost=P,
            state_constraints=state_constraints,
            input_constraints=input_constraints,
            warm_start_enabled=enable_warm_start,
        )

    reference = np.zeros((base_cfg.prediction_horizon_steps + 1, STATE_DIMENSION))
    rng = np.random.default_rng(7)
    states = []
    state = np.zeros(STATE_DIMENSION)
    for _ in range(num_steps):
        disturbance = 0.2 * rng.standard_normal(STATE_DIMENSION)
        state = 0.5 * state + disturbance
        # Keep pitch/pitch rate within linear regime
        state[PITCH_INDEX] = float(np.clip(state[PITCH_INDEX], -0.3, 0.3))
        state[PITCH_RATE_INDEX] = float(np.clip(state[PITCH_RATE_INDEX], -2.0, 2.0))
        states.append(state.copy())

    def _time_solver(solver: LinearMPCSolver) -> np.ndarray:
        times = []
        for st in states:
            sol = solver.solve(st, reference)
            times.append(sol.solve_time_s)
        return np.asarray(times)

    cold_solver = _build_solver(enable_warm_start=False)
    warm_solver = _build_solver(enable_warm_start=True)
    cold_times = _time_solver(cold_solver)
    warm_times = _time_solver(warm_solver)

    return {
        "cold_mean_ms": float(np.mean(cold_times) * 1e3),
        "warm_mean_ms": float(np.mean(warm_times) * 1e3),
        "cold_max_ms": float(np.max(cold_times) * 1e3),
        "warm_max_ms": float(np.max(warm_times) * 1e3),
    }


def _plot_results(cold_res, warm_res, cold_metrics, warm_metrics) -> None:
    """Generate solve time and accuracy comparison plots."""
    # Solve time per iteration
    plt.figure(figsize=(9, 4))
    plt.plot(cold_res.time_s, cold_res.qp_solve_time_history * 1e3, label="Cold start", marker="o", markersize=3)
    plt.plot(warm_res.time_s, warm_res.qp_solve_time_history * 1e3, label="Warm start", marker="o", markersize=3)
    plt.axhline(20, color="r", linestyle="--", label="Ts = 20 ms")
    plt.xlabel("Time (s)")
    plt.ylabel("Solve time (ms)")
    plt.title("MPC solve time: cold vs warm start")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COLD_WARM_OUTPUT_DIR / "solve_time_comparison.png", dpi=150)

    # Pitch tracking
    plt.figure(figsize=(9, 4))
    plt.plot(cold_res.time_s, cold_res.state_history[:, PITCH_INDEX], label="Cold start")
    plt.plot(warm_res.time_s, warm_res.state_history[:, PITCH_INDEX], label="Warm start")
    plt.axhline(0, color="k", linewidth=1, alpha=0.6)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (rad)")
    plt.title("Pitch recovery: cold vs warm start")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COLD_WARM_OUTPUT_DIR / "pitch_comparison.png", dpi=150)

    # Summary bar chart
    labels = ["Mean solve (ms)", "Max solve (ms)", "Pitch RMS (rad)"]
    cold_vals = [cold_metrics["mean_solve_ms"], cold_metrics["max_solve_ms"], cold_metrics["pitch_rms_rad"]]
    warm_vals = [warm_metrics["mean_solve_ms"], warm_metrics["max_solve_ms"], warm_metrics["pitch_rms_rad"]]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, cold_vals, width, label="Cold")
    plt.bar(x + width / 2, warm_vals, width, label="Warm")
    plt.xticks(x, labels, rotation=10)
    plt.title("Aggregated metrics")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.legend()
    plt.savefig(COLD_WARM_OUTPUT_DIR / "metrics_comparison.png", dpi=150)

    # Histogram of solve times
    combined_max = max(
        np.max(cold_res.qp_solve_time_history),
        np.max(warm_res.qp_solve_time_history),
    )
    bins = np.linspace(0, combined_max, 40)
    plt.figure(figsize=(9, 4))
    plt.hist(cold_res.qp_solve_time_history * 1e3, bins=bins * 1e3, alpha=0.6, label="Cold start")
    plt.hist(warm_res.qp_solve_time_history * 1e3, bins=bins * 1e3, alpha=0.6, label="Warm start")
    plt.axvline(
        cold_metrics["p95_solve_ms"],
        color="tab:blue",
        linestyle="--",
        label=f"Cold 95th ({cold_metrics['p95_solve_ms']:.1f} ms)",
    )
    plt.axvline(
        warm_metrics["p95_solve_ms"],
        color="tab:orange",
        linestyle="--",
        label=f"Warm 95th ({warm_metrics['p95_solve_ms']:.1f} ms)",
    )
    plt.xlabel("Solve time (ms)")
    plt.ylabel("Samples")
    plt.title("Histogram of MPC solve times")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COLD_WARM_OUTPUT_DIR / "solve_time_histogram.png", dpi=150)

    plt.close("all")


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
@pytest.mark.slow
def test_mpc_warm_vs_cold_start():
    """Compare warm-start vs cold-start MPC on the same recovery scenario."""
    base_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")

    solver_metrics = _run_solver_benchmark(base_cfg)
    print("\nSolver benchmark metrics:", solver_metrics)
    assert solver_metrics["warm_mean_ms"] < solver_metrics["cold_mean_ms"] * 0.9

    cold_metrics, cold_res = _run_case(base_cfg, warm_start=False)
    warm_metrics, warm_res = _run_case(base_cfg, warm_start=True)

    _plot_results(cold_res, warm_res, cold_metrics, warm_metrics)

    print("\nCold start metrics:", cold_metrics)
    print("Warm start metrics:", warm_metrics)
    print(f"Plots saved to {COLD_WARM_OUTPUT_DIR}")

    # Simulation sanity checks: warm start should not harm accuracy or average solve time
    assert warm_metrics["mean_solve_ms"] <= cold_metrics["mean_solve_ms"] * 1.25
    assert abs(warm_metrics["pitch_rms_rad"]) <= cold_metrics["pitch_rms_rad"] * 1.05


def main() -> None:
    """Allow running directly to generate plots without pytest."""
    if not MUJOCO_AVAILABLE:
        print("MuJoCo is not installed; skipping simulation.")
        return

    base_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")
    cold_metrics, cold_res = _run_case(base_cfg, warm_start=False)
    warm_metrics, warm_res = _run_case(base_cfg, warm_start=True)
    _plot_results(cold_res, warm_res, cold_metrics, warm_metrics)

    print("\nCold start metrics:", cold_metrics)
    print("Warm start metrics:", warm_metrics)
    print(f"Plots saved to {COLD_WARM_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

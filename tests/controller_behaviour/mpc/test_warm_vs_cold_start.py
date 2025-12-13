"""
Compare MPC cold-start vs warm-start performance with online linearization.

This test builds two MPC configurations that are identical except for
`warm_start_enabled` (toggled the same way you would in config/simulation/mpc_params.yaml).
Both runs start from the same 0.12 rad pitch offset and balance toward pitch = 0
using relative (online) linearization of the continuous dynamics. Horizon is N=30,
Ts=0.02 s.

It produces plots contrasting solve time per iteration and pitch tracking accuracy
between cold and warm starts.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml

from mpc import MPCConfig, ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import PITCH_INDEX
from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE


COLD_WARM_OUTPUT_DIR = Path("test_and_debug_output/mpc_warm_start")
COLD_WARM_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_temp_config(base: MPCConfig, warm_start: bool) -> str:
    """Create a temporary YAML with only warm-start toggled (mimics flipping the config line)."""
    # Keep user-configured values for everything else; assert required horizon/Ts

    cfg = replace(base, warm_start_enabled=warm_start, online_linearization_enabled=True)
    tmp_path = COLD_WARM_OUTPUT_DIR / f"mpc_params_warm_{warm_start}.yaml"
    with tmp_path.open("w") as fh:
        yaml.safe_dump(cfg.__dict__, fh)
    return str(tmp_path)


def _run_case(base_cfg: MPCConfig, warm_start: bool) -> Tuple[Dict[str, float], object]:
    """Run a single simulation case and compute metrics."""
    cfg_path = _write_temp_config(base_cfg, warm_start)
    sim_cfg = SimulationConfig(
        model_path="Mujoco sim/robot_model.xml",
        robot_params_path="config/simulation/robot_params.yaml",
        mpc_params_path=cfg_path,
        estimator_params_path="config/simulation/estimator_params.yaml",
        duration_s=60.0,
        render=False,
    )
    sim = MPCSimulation(sim_cfg)
    result = sim.run(
        duration_s=40.0,
        initial_pitch_rad=0.12,
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    pitch = result.state_history[:, PITCH_INDEX]
    settle_idx = np.argmax(np.abs(pitch) < 0.01) if np.any(np.abs(pitch) < 0.01) else len(pitch) - 1
    metrics = {
        "mean_solve_ms": float(np.mean(result.solve_time_history) * 1e3),
        "max_solve_ms": float(np.max(result.solve_time_history) * 1e3),
        "final_pitch_rad": float(pitch[-1]),
        "pitch_rms_rad": float(np.sqrt(np.mean(pitch**2))),
        "settle_time_s": float(result.time_s[settle_idx]) if settle_idx < len(result.time_s) else np.nan,
    }
    return metrics, result


def _plot_results(cold_res, warm_res, cold_metrics, warm_metrics) -> None:
    """Generate solve time and accuracy comparison plots."""
    # Solve time per iteration
    plt.figure(figsize=(9, 4))
    plt.plot(cold_res.time_s, cold_res.solve_time_history * 1e3, label="Cold start", marker="o", markersize=3)
    plt.plot(warm_res.time_s, warm_res.solve_time_history * 1e3, label="Warm start", marker="o", markersize=3)
    plt.xlabel("Time (s)")
    plt.ylabel("Solve time (ms)")
    plt.title("MPC solve time: cold vs warm start")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COLD_WARM_OUTPUT_DIR / "solve_time_comparison.png", dpi=150)

    # Histogram of solve times (critical for real-time systems)
    plt.figure(figsize=(10, 5))
    cold_times_ms = cold_res.solve_time_history * 1e3
    warm_times_ms = warm_res.solve_time_history * 1e3

    # Focus on 0-50ms range where most results lie
    hist_range = (0, 50)
    bins = np.linspace(hist_range[0], hist_range[1], 50)

    # Count outliers beyond the range
    cold_outliers = np.sum(cold_times_ms > hist_range[1])
    warm_outliers = np.sum(warm_times_ms > hist_range[1])

    plt.hist(cold_times_ms, bins=bins, range=hist_range, alpha=0.6,
             label=f"Cold start (max={cold_times_ms.max():.1f}ms, {cold_outliers} outliers>{hist_range[1]}ms)",
             color="C0", edgecolor="black")
    plt.hist(warm_times_ms, bins=bins, range=hist_range, alpha=0.6,
             label=f"Warm start (max={warm_times_ms.max():.1f}ms, {warm_outliers} outliers>{hist_range[1]}ms)",
             color="C1", edgecolor="black")

    # Mark percentiles
    cold_99 = np.percentile(cold_times_ms, 99)
    warm_99 = np.percentile(warm_times_ms, 99)
    plt.axvline(cold_99, color="C0", linestyle=":", alpha=0.7, label=f"Cold 99th %ile ({cold_99:.1f}ms)")
    plt.axvline(warm_99, color="C1", linestyle=":", alpha=0.7, label=f"Warm 99th %ile ({warm_99:.1f}ms)")
    plt.xlabel("Solve time (ms)")
    plt.ylabel("Count")
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xlim(hist_range)
    plt.tight_layout()
    plt.savefig(COLD_WARM_OUTPUT_DIR / "solve_time_histogram.png", dpi=150)

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

    plt.close("all")


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
@pytest.mark.slow
def test_mpc_warm_vs_cold_start():
    """Compare warm-start vs cold-start MPC on the same recovery scenario."""
    base_cfg = MPCConfig.from_yaml("config/simulation/mpc_params.yaml")

    cold_metrics, cold_res = _run_case(base_cfg, warm_start=False)
    warm_metrics, warm_res = _run_case(base_cfg, warm_start=True)

    _plot_results(cold_res, warm_res, cold_metrics, warm_metrics)

    print("\nCold start metrics:", cold_metrics)
    print("Warm start metrics:", warm_metrics)

    # Print percentile statistics (critical for real-time systems)
    cold_times = cold_res.solve_time_history * 1e3
    warm_times = warm_res.solve_time_history * 1e3
    print("\n=== Real-Time Performance Analysis ===")
    print(f"Samples: {len(cold_times)} timesteps")
    print(f"\nCold start percentiles (ms):")
    print(f"  50th (median): {np.percentile(cold_times, 50):.1f}")
    print(f"  95th:          {np.percentile(cold_times, 95):.1f}")
    print(f"  99th:          {np.percentile(cold_times, 99):.1f}")
    print(f"  Max:           {cold_times.max():.1f}")
    print(f"\nWarm start percentiles (ms):")
    print(f"  50th (median): {np.percentile(warm_times, 50):.1f}")
    print(f"  95th:          {np.percentile(warm_times, 95):.1f}")
    print(f"  99th:          {np.percentile(warm_times, 99):.1f}")
    print(f"  Max:           {warm_times.max():.1f}")
    print(f"\nWorst-case improvement: {(1 - warm_times.max()/cold_times.max())*100:.1f}%")

    print(f"\nPlots saved to {COLD_WARM_OUTPUT_DIR}")

    # Expect warm start to improve solve time without degrading accuracy
    assert warm_metrics["mean_solve_ms"] < cold_metrics["mean_solve_ms"] * 0.9
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

    # Print percentile statistics (critical for real-time systems)
    cold_times = cold_res.solve_time_history * 1e3
    warm_times = warm_res.solve_time_history * 1e3
    print("\n=== Real-Time Performance Analysis ===")
    print(f"Samples: {len(cold_times)} timesteps")
    print(f"\nCold start percentiles (ms):")
    print(f"  50th (median): {np.percentile(cold_times, 50):.1f}")
    print(f"  95th:          {np.percentile(cold_times, 95):.1f}")
    print(f"  99th:          {np.percentile(cold_times, 99):.1f}")
    print(f"  Max:           {cold_times.max():.1f}")
    print(f"\nWarm start percentiles (ms):")
    print(f"  50th (median): {np.percentile(warm_times, 50):.1f}")
    print(f"  95th:          {np.percentile(warm_times, 95):.1f}")
    print(f"  99th:          {np.percentile(warm_times, 99):.1f}")
    print(f"  Max:           {warm_times.max():.1f}")
    print(f"\nWorst-case improvement: {(1 - warm_times.max()/cold_times.max())*100:.1f}%")

    print(f"\nPlots saved to {COLD_WARM_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

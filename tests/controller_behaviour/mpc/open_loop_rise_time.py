"""
Measure open-loop rise time and propose MPC sampling periods.

This script:
1) Simulates the continuous dynamics under a small torque step.
2) Computes rise time from 10% to 90% of the final value (pitch angle).
3) Suggests sampling periods that yield 10–20 samples across that rise.
4) Saves plots of the response and the sampling-period options.

Usage:
    PYTHONPATH=. python scripts/debug/open_loop_rise_time.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np

from robot_dynamics.parameters import RobotParameters
from robot_dynamics.continuous_dynamics import build_dynamics_model


@dataclass
class RiseTimeResult:
    time_s: np.ndarray
    pitch_rad: np.ndarray
    t10_s: float
    t90_s: float
    rise_time_s: float
    sampling_periods_s: List[float]


def simulate_open_loop(
    params: RobotParameters,
    torque_step_nm: float = 0.05,
    t_final_s: float = 2.0,
    dt_s: float = 0.0005,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward-Euler simulate open-loop dynamics under a torque step."""
    dyn = build_dynamics_model(params)
    u = np.array([torque_step_nm, torque_step_nm], dtype=float)

    times = np.arange(0.0, t_final_s + dt_s, dt_s)
    state = np.zeros(6, dtype=float)  # start at equilibrium
    pitch_history = np.zeros_like(times)

    for i, t in enumerate(times):
        pitch_history[i] = state[1]
        # Forward Euler integration
        deriv = np.array(dyn(state, u)).astype(float).ravel()
        state = state + dt_s * deriv

    return times, pitch_history


def compute_rise_time(
    time_s: np.ndarray,
    signal: np.ndarray,
    settling_window_s: float = 0.05,
    min_span_rad: float = 1e-4,
) -> Tuple[float, float, float]:
    """Return (t10, t90, rise_time) for a monotonic-ish signal."""
    final_span = max(int(settling_window_s / (time_s[1] - time_s[0])), 1)
    final_value = float(np.mean(signal[-final_span:]))
    initial_value = float(signal[0])
    delta = final_value - initial_value

    if abs(delta) < min_span_rad:
        raise RuntimeError(
            f"Signal change too small for rise-time computation (Δ={delta:.2e})"
        )

    target10 = initial_value + 0.1 * delta
    target90 = initial_value + 0.9 * delta

    def first_cross(target: float) -> float:
        mask = (signal - target) >= 0 if delta > 0 else (signal - target) <= 0
        idx = np.argmax(mask)
        if not mask[idx]:
            raise RuntimeError(f"Signal never crosses target {target:.3e}")
        return float(time_s[idx])

    t10 = first_cross(target10)
    t90 = first_cross(target90)

    if t90 <= t10:
        raise RuntimeError(f"Non-increasing rise: t10={t10:.4f}, t90={t90:.4f}")

    return t10, t90, t90 - t10


def propose_sampling_periods(
    rise_time_s: float,
    min_freq_hz: float = 50.0,
) -> List[float]:
    """Return sampling periods that give 10–20 samples across rise time.

    Enforces a hard minimum control frequency (default 50 Hz ⇒ Ts ≤ 20 ms).
    """
    max_period_s = 1.0 / min_freq_hz
    periods = [rise_time_s / n for n in range(10, 21)]
    return [Ts for Ts in periods if Ts <= max_period_s]


def make_plots(result: RiseTimeResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Response plot
    plt.figure(figsize=(8, 4))
    plt.plot(result.time_s, np.rad2deg(result.pitch_rad), label="Pitch (deg)")
    plt.axvline(result.t10_s, color="g", linestyle="--", label="t10")
    plt.axvline(result.t90_s, color="r", linestyle="--", label="t90")
    plt.title("Open-loop pitch response")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (deg)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "open_loop_response.png", dpi=150)

    # Sampling-period options
    if result.sampling_periods_s:
        periods_ms = np.array(result.sampling_periods_s) * 1e3
        samples = np.array(range(10, 10 + len(result.sampling_periods_s)))

        plt.figure(figsize=(8, 4))
        plt.plot(samples, periods_ms, marker="o")
        plt.title("Sampling periods for 10–20 samples over rise time")
        plt.xlabel("Samples across rise time (filtered by 50 Hz min)")
        plt.ylabel("Sampling period (ms)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "sampling_periods.png", dpi=150)


def main() -> None:
    params = RobotParameters.from_yaml("config/simulation/robot_params.yaml")
    time_s, pitch_rad = simulate_open_loop(params)
    t10, t90, rise = compute_rise_time(time_s, pitch_rad)
    sampling_periods = propose_sampling_periods(rise, min_freq_hz=50.0)

    result = RiseTimeResult(
        time_s=time_s,
        pitch_rad=pitch_rad,
        t10_s=t10,
        t90_s=t90,
        rise_time_s=rise,
        sampling_periods_s=sampling_periods,
    )

    out_dir = Path("test_and_debug_output/open_loop_rise_time")
    make_plots(result, out_dir)

    print(f"Final pitch (deg): {np.rad2deg(pitch_rad[-1]):.4f}")
    print(f"t10 = {t10:.4f}s, t90 = {t90:.4f}s, rise = {rise:.4f}s")
    if sampling_periods:
        print("Sampling periods (ms) for 10–20 samples across rise (>=50 Hz):")
        start_n = 10
        for idx, Ts in enumerate(sampling_periods):
            print(f"  {start_n + idx:2d} samples -> {Ts*1e3:.3f} ms")
    else:
        print("No sampling periods satisfy both 10–20 samples across rise and >=50 Hz.")
    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

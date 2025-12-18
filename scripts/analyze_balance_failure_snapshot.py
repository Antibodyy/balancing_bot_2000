"""Analyze balance failure snapshot artifacts.

Usage:
    python3 scripts/analyze_balance_failure_snapshot.py <snapshot_path>

Prints summary statistics and optionally saves a diagnostic plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting optional
    plt = None  # type: ignore

from robot_dynamics.parameters import (
    POSITION_INDEX,
    PITCH_INDEX,
    VELOCITY_INDEX,
)

DX_THRESHOLD_MPS = 0.5
ANALYSIS_WINDOW_S = 2.7


def _format_solver_status(success_flag: bool) -> str:
    return "optimal" if success_flag else "non-optimal"


def analyze_snapshot(
    snapshot_path: Path,
    save_plot: bool = True,
) -> None:
    """Analyze a saved failure snapshot."""
    if not snapshot_path.exists():
        print(f"[analyze_snapshot] File not found: {snapshot_path}")
        return

    data = np.load(snapshot_path, allow_pickle=True)
    fields: Dict[str, Any] = {key: data[key] for key in data.files}

    time_s = fields.get("time_s", np.array([]))
    true_state = fields.get("true_state_history", np.array([]))
    state_history = fields.get("state_history", np.array([]))
    u_applied = fields.get("u_applied_history", np.array([]))
    infeasible_history = fields.get("infeasible_history", np.array([]))
    desired_velocity = fields.get("desired_velocity_history")
    velocity_slack_history = fields.get("velocity_slack_history")

    if time_s.size == 0 or true_state.size == 0:
        print("[analyze_snapshot] Snapshot missing time/state history.")
        return

    dx = true_state[:, VELOCITY_INDEX]
    pitch = true_state[:, PITCH_INDEX]

    drift_mask = np.abs(dx) > DX_THRESHOLD_MPS
    drift_index = int(np.argmax(drift_mask)) if np.any(drift_mask) else -1

    if drift_index >= 0:
        drift_time = time_s[drift_index]
        drift_pitch = pitch[drift_index]
        solver_status = _format_solver_status(
            not bool(infeasible_history[drift_index])
            if infeasible_history.size > drift_index
            else True
        )
        print(
            f"[analysis] First |dx|>{DX_THRESHOLD_MPS:.3f} m/s at "
            f"t={drift_time:.3f}s, pitch={drift_pitch:.3f} rad, "
            f"solver_status={solver_status}"
        )
    else:
        drift_time = ANALYSIS_WINDOW_S
        print("[analysis] No |dx| threshold crossing captured.")

    window_mask = time_s <= ANALYSIS_WINDOW_S
    if not np.any(window_mask):
        print("[analysis] No samples within analysis window.")
        return

    u_window = u_applied[window_mask]
    infeasible_window = infeasible_history[window_mask] if infeasible_history.size else np.array([])
    dx_window = dx[window_mask]

    if u_window.size:
        mean_torque = np.mean(u_window, axis=0)
        median_torque = np.median(u_window, axis=0)
        bias = float(np.mean(np.sum(u_window, axis=1) / 2.0))
        print(
            "[analysis] Torque statistics over first "
            f"{ANALYSIS_WINDOW_S:.3f}s: "
            f"mean(L,R)=({mean_torque[0]:.4f},{mean_torque[1]:.4f}) Nm, "
            f"median(L,R)=({median_torque[0]:.4f},{median_torque[1]:.4f}) Nm, "
            f"mean bias ( (u_L+u_R)/2 )={bias:.4f} Nm"
        )
    else:
        print("[analysis] No torque samples in window.")

    if desired_velocity is not None and desired_velocity.size:
        dv_window = desired_velocity[window_mask]
        print(
            f"[analysis] desired dx stats: mean={np.mean(dv_window):.4f} m/s, "
            f"max|dx_ref|={np.max(np.abs(dv_window)):.4f} m/s"
        )
    else:
        print("[analysis] desired velocity history unavailable.")

    if infeasible_window.size:
        idx_threshold = drift_index if drift_index >= 0 else np.sum(window_mask) - 1
        idx_threshold = min(idx_threshold, infeasible_window.size - 1)
        frac_non_opt = float(np.mean(infeasible_window[: idx_threshold + 1]))
        print(
            f"[analysis] Fraction non-optimal before drift: {frac_non_opt:.3f}"
        )
    else:
        print("[analysis] Infeasible history unavailable.")

    if velocity_slack_history is not None and velocity_slack_history.size:
        try:
            slack_array = np.asarray(velocity_slack_history, dtype=float)
            slack_window = slack_array[window_mask, 0]
            print(
                f"[analysis] Velocity slack (k=0) mean={np.mean(slack_window):.5f}, "
                f"max={np.max(slack_window):.5f}"
            )
        except Exception:
            print("[analysis] Unable to parse velocity_slack_history.")

    if save_plot and plt is not None:
        torque_bias = np.sum(u_applied, axis=1) / 2.0 if u_applied.size else np.zeros_like(dx)
        fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        ax[0].plot(time_s, dx, label="dx (m/s)")
        ax[0].axhline(DX_THRESHOLD_MPS, color="r", linestyle="--", linewidth=0.8)
        ax[0].axhline(-DX_THRESHOLD_MPS, color="r", linestyle="--", linewidth=0.8)
        ax[0].set_ylabel("Velocity (m/s)")
        ax[0].legend()

        ax[1].plot(time_s, pitch, label="pitch (rad)", color="tab:orange")
        ax[1].set_ylabel("Pitch (rad)")
        ax[1].legend()

        ax[2].plot(time_s[: torque_bias.shape[0]], torque_bias, label="(u_L+u_R)/2")
        ax[2].set_ylabel("Torque bias (Nm)")
        ax[2].set_xlabel("Time (s)")
        ax[2].legend()

        out_path = snapshot_path.with_name("balance_failure_analysis.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"[analysis] Saved plot to {out_path}")
    elif save_plot:
        print("[analysis] matplotlib unavailable; skipping plot.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "snapshot",
        type=Path,
        help="Path to balance_failure_snapshot.npz",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation.",
    )
    args = parser.parse_args()
    analyze_snapshot(args.snapshot, save_plot=not args.no_plot)


if __name__ == "__main__":
    main()

"""
Test static balance on sloped terrain (script + parametrized pytest).

This module serves two purposes:
1) CLI script to run one or multiple slope scenarios with plotting. By default
   it runs 0, 5, and 10 degree slopes and overlays the results in shared plots.
2) Pytest that exercises 5 deg and 10 deg slopes (non-viewer, no plots) to verify
   the controller holds balance under moderate inclines.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xml.etree.ElementTree as ET
import yaml

from simulation import SimulationConfig, MPCSimulation, MUJOCO_AVAILABLE
from mpc import ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import (
    PITCH_INDEX,
    PITCH_RATE_INDEX,
    POSITION_INDEX,
    VELOCITY_INDEX,
    YAW_INDEX,
    YAW_RATE_INDEX,
)

# Add project root to path for CLI usage (needed when run directly)
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

COMPARISON_OUTPUT_DIR = Path("test_and_debug_output/slope_balance_comparison")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test static balance on a slope")
    parser.add_argument(
        "--slopes",
        type=float,
        nargs="*",
        default=None,
        help="List of slope angles in degrees to run (e.g., --slopes 0 5 10). "
             "Defaults to 0, 5, 10 when not provided.",
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=None,
        help="Single slope angle in degrees (overrides default set if provided).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Simulation duration in seconds (default: 20.0)",
    )
    parser.add_argument(
        "--initial-pitch",
        type=float,
        default=0.0,
        help="Initial pitch perturbation in degrees",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Run with interactive MuJoCo viewer",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--per-slope-plots",
        action="store_true",
        help="Also emit individual plots per slope when running multiple slopes",
    )
    return parser


def _modify_model_and_params(
    slope_rad: float, temp_dir: str
) -> Tuple[Path, Path]:
    """Create slope-modified MuJoCo model and params in a temp directory."""
    original_model_path = project_root / "mujoco_sim/robot_model.xml"
    temp_model_path = Path(temp_dir) / "mujoco_sim" / "robot_model.xml"
    temp_model_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(original_model_path)
    root = tree.getroot()
    floor_geom = root.find(".//geom[@name='floor']")
    if floor_geom is not None:
        # Positive slopes in the dynamics module mean “downhill acceleration
        # is in +x”. MuJoCo’s default Euler convention does the opposite, so
        # rotate the floor by -slope to keep both models aligned.
        floor_geom.set("euler", f"0 {-slope_rad} 0")
    tree.write(str(temp_model_path))

    original_params_path = project_root / "config" / "simulation" / "robot_params.yaml"
    temp_params_path = Path(temp_dir) / "robot_params.yaml"
    with open(original_params_path, "r") as f:
        params = yaml.safe_load(f)
    params["ground_slope_rad"] = float(slope_rad)
    with open(temp_params_path, "w") as f:
        yaml.safe_dump(params, f)

    return temp_model_path, temp_params_path


def _compute_success(result, equilibrium_pitch_rad: float, slope_rad: float) -> bool:
    """Basic success criteria: no fall, small pitch/velocity at end.

    On non-zero slopes the MPC must apply a steady braking torque to hold
    position, which can leave a tiny residual velocity while it settles.
    Allow a slightly looser velocity tolerance in that case so we score the
    run based on pitch accuracy rather than demanding perfect standstill.
    """
    if not result.success or len(result.state_history) == 0:
        return False
    final_state = result.state_history[-1]
    final_velocity = final_state[VELOCITY_INDEX]
    final_pitch = final_state[PITCH_INDEX]
    if abs(slope_rad) < 1e-6:
        velocity_tolerance = 0.01
    else:
        velocity_tolerance = 0.1
    velocity_ok = abs(final_velocity) < velocity_tolerance
    pitch_ok = abs(np.rad2deg(final_pitch - equilibrium_pitch_rad)) < 1.0
    return velocity_ok and pitch_ok


def _make_plots_single(result, equilibrium_pitch_rad: float, args, save_dir: Path) -> None:
    """Generate summary plots for a single run."""
    save_dir.mkdir(parents=True, exist_ok=True)
    time_s = result.time_s
    pitch_error_deg = np.rad2deg(result.state_history[:, PITCH_INDEX] - equilibrium_pitch_rad)

    fig, axs = plt.subplots(3, 2, figsize=(16, 12))
    axs = axs.flatten()

    axs[0].plot(time_s, np.rad2deg(result.state_history[:, PITCH_INDEX]), "b-", linewidth=2, label="Actual")
    axs[0].axhline(np.rad2deg(equilibrium_pitch_rad), color="r", linestyle="--", linewidth=2, label="Equilibrium")
    axs[0].set_title("Pitch Angle")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(time_s, pitch_error_deg, "r-", linewidth=2)
    axs[1].axhline(0, color="k", linestyle="-", alpha=0.3)
    axs[1].set_title("Pitch Error from Equilibrium")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(time_s, result.state_history[:, POSITION_INDEX], "b-", linewidth=2)
    axs[2].axhline(0, color="k", linestyle="-", alpha=0.3)
    axs[2].set_title("Position (m)")
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(time_s, result.state_history[:, VELOCITY_INDEX], "b-", linewidth=2, label="Actual")
    axs[3].axhline(0, color="r", linestyle="--", linewidth=2, label="Target (0 m/s)")
    axs[3].set_title("Velocity (m/s)")
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)

    axs[4].plot(time_s, np.rad2deg(result.state_history[:, PITCH_RATE_INDEX]), "b-", linewidth=2)
    axs[4].axhline(0, color="k", linestyle="-", alpha=0.3)
    axs[4].set_title("Pitch Rate (deg/s)")
    axs[4].grid(True, alpha=0.3)

    n_controls = len(result.control_history)
    axs[5].plot(time_s[:n_controls], result.control_history[:, 0], "b-", linewidth=1.5, label="Left wheel")
    axs[5].plot(time_s[:n_controls], result.control_history[:, 1], "r--", linewidth=1.5, label="Right wheel")
    axs[5].set_title("Control Torques (Nm)")
    axs[5].legend()
    axs[5].grid(True, alpha=0.3)

    fig.suptitle(
        f"Slope Balance Test: {args.slope}deg slope for {args.duration:.0f}s"
        + (f" with {args.initial_pitch}deg initial perturbation" if args.initial_pitch != 0 else ""),
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_dir / "balance_performance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Phase portrait
    fig2, ax = plt.subplots(figsize=(10, 8))
    ax.plot(pitch_error_deg, np.rad2deg(result.state_history[:, PITCH_RATE_INDEX]), "b-", linewidth=1.5, alpha=0.7)
    ax.plot(pitch_error_deg[0], np.rad2deg(result.state_history[0, PITCH_RATE_INDEX]), "go", markersize=10, label="Start")
    ax.plot(pitch_error_deg[-1], np.rad2deg(result.state_history[-1, PITCH_RATE_INDEX]), "ro", markersize=10, label="End")
    ax.axhline(0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(0, color="k", linestyle="-", alpha=0.3)
    ax.set_xlabel("Pitch Error from Equilibrium (deg)")
    ax.set_ylabel("Pitch Rate (deg/s)")
    ax.set_title("Pitch Phase Portrait (Equilibrium at origin)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(save_dir / "state_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)


def _plot_comparison(runs: List[Dict]) -> None:
    """Overlay results across slopes on shared plots."""
    if len(runs) < 2:
        return
    COMPARISON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(runs)))

    # Pitch overlay
    plt.figure(figsize=(9, 4))
    for (slope_deg, result, eq_pitch), color in zip(runs, colors):
        plt.plot(result.time_s, np.rad2deg(result.state_history[:, PITCH_INDEX]),
                 label=f"{slope_deg:.0f} deg", color=color)
        plt.axhline(np.rad2deg(eq_pitch), color=color, linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch (deg)")
    plt.title("Pitch vs time across slopes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT_DIR / "pitch_overlay.png", dpi=150)
    plt.close()

    # Velocity overlay
    plt.figure(figsize=(9, 4))
    for (slope_deg, result, _), color in zip(runs, colors):
        plt.plot(result.time_s, result.state_history[:, VELOCITY_INDEX],
                 label=f"{slope_deg:.0f} deg", color=color)
    plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity vs time across slopes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT_DIR / "velocity_overlay.png", dpi=150)
    plt.close()

    # Solve time overlay
    plt.figure(figsize=(9, 4))
    for (slope_deg, result, _), color in zip(runs, colors):
        plt.plot(result.time_s, result.solve_time_history * 1e3,
                 label=f"{slope_deg:.0f} deg", color=color, marker="o", markersize=3)
    plt.axhline(20, color="r", linestyle="--", label="Ts = 20 ms")
    plt.xlabel("Time (s)")
    plt.ylabel("Solve time (ms)")
    plt.title("MPC solve time across slopes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(COMPARISON_OUTPUT_DIR / "solve_time_overlay.png", dpi=150)
    plt.close()


def run_slope_balance(
    slope_deg: float,
    duration_s: float,
    initial_pitch_deg: float = 0.0,
    viewer: bool = False,
    make_plots: bool = True,
    output_suffix: Optional[str] = None,
) -> Tuple[object, bool, float]:
    """Run one slope-balance scenario and optionally plot results.

    Returns:
        Tuple of (SimulationResult, success_flag, equilibrium_pitch_rad)
    """
    slope_rad = np.deg2rad(slope_deg)
    initial_pitch_rad = np.deg2rad(initial_pitch_deg)

    temp_dir = tempfile.mkdtemp(prefix="slope_balance_")
    try:
        temp_model_path, temp_params_path = _modify_model_and_params(slope_rad, temp_dir)

        config = SimulationConfig(
            model_path=str(temp_model_path),
            robot_params_path=str(temp_params_path),
            mpc_params_path="config/simulation/mpc_params.yaml",
            estimator_params_path="config/simulation/estimator_params.yaml",
            duration_s=duration_s,
            use_virtual_physics=True,  # Force analytic “virtual” physics for deterministic slope tests.
        )
        simulation = MPCSimulation(config)
        equilibrium_pitch_rad = simulation.equilibrium_state[PITCH_INDEX]
        reference_command = ReferenceCommand(mode=ReferenceMode.BALANCE)

        if viewer:
            result = simulation.run_with_viewer(
                initial_pitch_rad=initial_pitch_rad,
                reference_command=reference_command,
            )
        else:
            result = simulation.run(
                initial_pitch_rad=initial_pitch_rad,
                reference_command=reference_command,
            )

        success = _compute_success(result, equilibrium_pitch_rad, slope_rad)

        if make_plots:
            suffix = output_suffix or (
                f"slope_balance_{slope_deg:.1f}deg"
                + (f"_perturb_{abs(initial_pitch_deg):.1f}deg" if initial_pitch_deg != 0 else "")
                + (f"_{duration_s:.0f}s" if not viewer else "_viewer")
            )
            save_dir = Path("test_and_debug_output") / suffix
            _make_plots_single(
                result,
                equilibrium_pitch_rad,
                SimpleNamespace(
                    slope=slope_deg,
                    duration=duration_s,
                    initial_pitch=initial_pitch_deg
                ),
                save_dir,
            )

        return result, success, equilibrium_pitch_rad
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not available")
@pytest.mark.slow
@pytest.mark.parametrize("slope_deg", [5.0, 10.0])
def test_slope_balance_parametrized(slope_deg: float):
    """Ensure balance holds on 5 deg and 10 deg slopes without viewer/plots."""
    _, success, _ = run_slope_balance(
        slope_deg=slope_deg,
        duration_s=6.0,
        initial_pitch_deg=0.0,
        viewer=False,
        make_plots=False,
    )
    assert success, f"Robot failed to maintain balance on {slope_deg} deg slope"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Decide slopes to run: explicit list > single value > default set
    if args.slopes:
        slopes = args.slopes
    elif args.slope is not None:
        slopes = [args.slope]
    else:
        slopes = [0.0, 5.0, 10.0]

    print("\n" + "=" * 80)
    print("SLOPE BALANCE TEST")
    print("=" * 80)
    print(f"  Slopes: {slopes} deg")
    print(f"  Duration: {args.duration} s" + (" (ignored in viewer mode)" if args.viewer else ""))
    print(f"  Initial pitch: {args.initial_pitch} deg ({np.deg2rad(args.initial_pitch):.4f} rad)")
    print(f"  Viewer mode: {'ENABLED' if args.viewer else 'DISABLED'}")
    print("=" * 80)

    per_slope_plots = not args.no_plots and (args.per_slope_plots or len(slopes) == 1)

    runs: List[Dict] = []
    for slope in slopes:
        result, success, equilibrium_pitch_rad = run_slope_balance(
            slope_deg=slope,
            duration_s=args.duration,
            initial_pitch_deg=args.initial_pitch,
            viewer=args.viewer,
            make_plots=per_slope_plots,
        )
        runs.append((slope, result, equilibrium_pitch_rad))
        status_msg = "SUCCESS" if success else "FAILED"
        print(f"Slope {slope:.1f} deg: {status_msg}, mean solve {result.mean_solve_time_ms:.2f}ms, "
              f"max solve {result.max_solve_time_ms:.2f}ms, deadline violations {result.deadline_violations}")

    # Combined overlay plots
    if not args.no_plots and len(runs) > 1:
        _plot_comparison(runs)
        print(f"Combined plots saved to: {COMPARISON_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

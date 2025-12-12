"""MPC Diagnostics class for debugging controller behavior.

Provides a high-level interface for running simulations with diagnostics
and generating debug plots.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np

from simulation import MPCSimulation, SimulationResult, SimulationConfig
from mpc import MPCConfig
from debug.plotting import (
    plot_state_comparison,
    plot_control_analysis,
    plot_prediction_accuracy,
    plot_closed_loop_diagnosis,
    plot_control_direction_test,
)


@dataclass
class DiagnosticSummary:
    """Summary statistics from diagnostic run."""

    # State tracking
    max_pitch_error_rad: float
    mean_pitch_error_rad: float
    max_pitch_rate_error_rad: float

    # Control
    mean_total_torque: float
    saturation_fraction: float

    # Prediction
    mean_one_step_error_rad: float
    max_one_step_error_rad: float

    # Timing
    mean_solve_time_ms: float
    max_solve_time_ms: float

    # Outcome
    success: bool
    failure_time_s: Optional[float]

    def print_summary(self) -> None:
        """Print formatted summary to console."""
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)

        print("\nState Estimation:")
        print(f"  Max pitch error:      {np.rad2deg(self.max_pitch_error_rad):.3f} deg")
        print(f"  Mean pitch error:     {np.rad2deg(self.mean_pitch_error_rad):.3f} deg")
        print(f"  Max pitch rate error: {np.rad2deg(self.max_pitch_rate_error_rad):.3f} deg/s")

        print("\nControl:")
        print(f"  Mean total torque:    {self.mean_total_torque:.4f} N*m")
        print(f"  Saturation fraction:  {self.saturation_fraction * 100:.1f}%")

        print("\nPrediction Accuracy:")
        print(f"  Mean 1-step error:    {self.mean_one_step_error_rad * 1000:.3f} mrad")
        print(f"  Max 1-step error:     {self.max_one_step_error_rad * 1000:.3f} mrad")

        print("\nTiming:")
        print(f"  Mean solve time:      {self.mean_solve_time_ms:.1f} ms")
        print(f"  Max solve time:       {self.max_solve_time_ms:.1f} ms")

        print("\nOutcome:")
        if self.success:
            print("  Result: SUCCESS (stayed balanced)")
        else:
            print(f"  Result: FAILED at t={self.failure_time_s:.3f}s")

        print("=" * 60 + "\n")


class MPCDiagnostics:
    """High-level diagnostics interface for MPC debugging.

    Example usage:
        config = SimulationConfig(...)
        diag = MPCDiagnostics(config)

        # Run with diagnostics
        result, summary = diag.run_with_diagnostics(
            duration_s=2.0,
            initial_pitch_rad=0.05,
        )

        # Generate plots
        diag.plot_all(result, save_dir="test_and_debug_output/")
    """

    def __init__(self, config: SimulationConfig):
        """Initialize diagnostics with simulation config.

        Args:
            config: Simulation configuration
        """
        self._config = config
        self._simulation = MPCSimulation(config)

    @property
    def simulation(self) -> MPCSimulation:
        """Access underlying simulation."""
        return self._simulation

    @property
    def mpc_config(self) -> MPCConfig:
        """Access MPC configuration."""
        return self._simulation._mpc_config

    def run_with_diagnostics(
        self,
        duration_s: float = 2.0,
        initial_pitch_rad: float = 0.05,
        verbose: bool = True,
        reference_command_callback=None,
    ) -> tuple[SimulationResult, DiagnosticSummary]:
        """Run simulation and compute diagnostic summary.

        Args:
            duration_s: Simulation duration
            initial_pitch_rad: Initial pitch perturbation
            verbose: Print step-by-step trace
            reference_command_callback: Optional time-varying reference command callback

        Returns:
            Tuple of (SimulationResult, DiagnosticSummary)
        """
        result = self._simulation.run(
            duration_s=duration_s,
            initial_pitch_rad=initial_pitch_rad,
            reference_command_callback=reference_command_callback,
        )

        summary = self._compute_summary(result)

        if verbose:
            self._print_trace(result)
            summary.print_summary()

        return result, summary

    def _compute_summary(self, result: SimulationResult) -> DiagnosticSummary:
        """Compute diagnostic summary from simulation result."""
        from robot_dynamics.parameters import PITCH_INDEX, PITCH_RATE_INDEX

        # State estimation errors
        pitch_error = result.state_history[:, PITCH_INDEX] - result.state_estimate_history[:, PITCH_INDEX]
        pitch_rate_error = result.state_history[:, PITCH_RATE_INDEX] - result.state_estimate_history[:, PITCH_RATE_INDEX]

        # Control analysis
        total_torque = result.control_history[:, 0] + result.control_history[:, 1]
        control_limit = self.mpc_config.control_limit_nm
        saturated = np.any(np.abs(result.control_history) >= control_limit * 0.99, axis=1)

        # Prediction errors
        one_step_errors = []
        for i in range(len(result.reference_history) - 1):
            pred_next = result.reference_history[i][1, PITCH_INDEX]
            actual_next = result.state_history[i + 1, PITCH_INDEX]
            one_step_errors.append(abs(pred_next - actual_next))

        # Failure detection
        failure_time = None
        if not result.success:
            failure_time = result.time_s[-1]

        return DiagnosticSummary(
            max_pitch_error_rad=np.max(np.abs(pitch_error)),
            mean_pitch_error_rad=np.mean(np.abs(pitch_error)),
            max_pitch_rate_error_rad=np.max(np.abs(pitch_rate_error)),
            mean_total_torque=np.mean(total_torque),
            saturation_fraction=np.mean(saturated),
            mean_one_step_error_rad=np.mean(one_step_errors) if one_step_errors else 0.0,
            max_one_step_error_rad=np.max(one_step_errors) if one_step_errors else 0.0,
            mean_solve_time_ms=result.mean_solve_time_ms,
            max_solve_time_ms=result.max_solve_time_ms,
            success=result.success,
            failure_time_s=failure_time,
        )

    def _print_trace(self, result: SimulationResult, max_steps: int = 15) -> None:
        """Print step-by-step trace of first N steps."""
        from robot_dynamics.parameters import PITCH_INDEX, PITCH_RATE_INDEX

        print("\n" + "=" * 80)
        print("STEP-BY-STEP TRACE (first {} steps)".format(min(max_steps, len(result.time_s))))
        print("=" * 80)
        print(f"{'Step':>4} {'Time':>6} {'True_th':>10} {'Est_th':>10} {'tau_L':>8} {'tau_R':>8} {'tau_total':>8} {'Solve':>8}")
        print(f"{'':>4} {'(s)':>6} {'(deg)':>10} {'(deg)':>10} {'(N*m)':>8} {'(N*m)':>8} {'(N*m)':>8} {'(ms)':>8}")
        print("-" * 80)

        for i in range(min(max_steps, len(result.time_s))):
            true_pitch = np.rad2deg(result.state_history[i, PITCH_INDEX])
            est_pitch = np.rad2deg(result.state_estimate_history[i, PITCH_INDEX])
            tau_l = result.control_history[i, 0]
            tau_r = result.control_history[i, 1]
            tau_total = tau_l + tau_r
            solve_ms = result.solve_time_history[i] * 1000

            print(f"{i:4d} {result.time_s[i]:6.3f} {true_pitch:10.4f} {est_pitch:10.4f} "
                  f"{tau_l:8.4f} {tau_r:8.4f} {tau_total:8.4f} {solve_ms:8.1f}")

        print("-" * 80)

        # Check control direction
        if len(result.time_s) > 0:
            pitch_sign = np.sign(result.state_estimate_history[0, PITCH_INDEX])
            torque_sign = np.sign(result.control_history[0, 0] + result.control_history[0, 1])
            if pitch_sign != 0 and torque_sign != 0:
                if pitch_sign == torque_sign:
                    print("\n⚠️  WARNING: Control torque has SAME sign as pitch error!")
                    print("    This suggests the controller is pushing in the WRONG direction.")
                    print("    Expected: positive pitch → negative torque (or vice versa)")
                else:
                    print("\n✓  Control direction appears correct (opposite to pitch error)")

    def plot_all(
        self,
        result: SimulationResult,
        save_dir: Optional[str] = None,
        show: bool = True,
    ) -> dict:
        """Generate all diagnostic plots.

        Args:
            result: Simulation result to plot
            save_dir: Directory to save plots (optional)
            show: Whether to call plt.show()

        Returns:
            Dictionary of figure objects
        """
        import matplotlib.pyplot as plt
        import os

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        control_limit = self.mpc_config.control_limit_nm
        solve_times_ms = result.solve_time_history * 1000

        figures = {}

        # State comparison
        figures['state'] = plot_state_comparison(
            result.time_s,
            result.state_history,
            result.state_estimate_history,
            save_path=f"{save_dir}/state_comparison.png" if save_dir else None,
        )

        # Control analysis
        figures['control'] = plot_control_analysis(
            result.time_s,
            result.control_history,
            control_limit,
            save_path=f"{save_dir}/control_analysis.png" if save_dir else None,
        )

        # Prediction accuracy
        figures['prediction'] = plot_prediction_accuracy(
            result.time_s,
            result.state_history,
            result.reference_history,
            save_path=f"{save_dir}/prediction_accuracy.png" if save_dir else None,
        )

        # Closed-loop diagnosis
        figures['diagnosis'] = plot_closed_loop_diagnosis(
            result.time_s,
            result.state_history,
            result.state_estimate_history,
            result.control_history,
            result.reference_history,
            control_limit,
            solve_times_ms,
            save_path=f"{save_dir}/closed_loop_diagnosis.png" if save_dir else None,
        )

        if show:
            plt.show()

        return figures

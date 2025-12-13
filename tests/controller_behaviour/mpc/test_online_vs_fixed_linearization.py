"""Test comparing online linearization vs. fixed linearization around 0 degrees.

This test demonstrates the difference in MPC behavior when:
1. Online linearization: Re-linearizing the dynamics at the current state at each solve
2. Fixed linearization: Using a fixed linearization around 0 degrees (equilibrium)

The test applies an initial pitch perturbation and compares:
- Actual pitch evolution over time
- Predicted pitch horizons at each MPC solve instance
- Dynamics matrices (A, B) evolution
- Numerical differences in predictions
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from simulation import MPCSimulation, SimulationConfig, MUJOCO_AVAILABLE
from mpc import ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import PITCH_INDEX, STATE_DIMENSION


# Skip all tests if MuJoCo is not available
pytestmark = pytest.mark.skipif(
    not MUJOCO_AVAILABLE,
    reason="MuJoCo not installed"
)


class MatrixTracker:
    """Track dynamics matrices (A, B) over time for analysis."""

    def __init__(self):
        """Initialize matrix tracker."""
        self.state_matrices: List[np.ndarray] = []
        self.control_matrices: List[np.ndarray] = []
        self.timestamps: List[float] = []
        self.update_count: int = 0

    def record(self, state_matrix: np.ndarray, control_matrix: np.ndarray, time: float):
        """Record a matrix snapshot."""
        self.state_matrices.append(state_matrix.copy())
        self.control_matrices.append(control_matrix.copy())
        self.timestamps.append(time)
        self.update_count += 1

    def compute_matrix_differences(self, baseline_tracker: 'MatrixTracker') -> dict:
        """Compute differences from baseline matrices.

        Args:
            baseline_tracker: Tracker with baseline (fixed) matrices

        Returns:
            Dictionary with difference statistics
        """
        n_steps = len(self.state_matrices)
        baseline_steps = len(baseline_tracker.state_matrices)

        state_matrix_diffs = []
        control_matrix_diffs = []

        for i in range(n_steps):
            if baseline_steps == 0:
                break

            # If the baseline is shorter (fixed linearization records only the initial
            # matrices), keep comparing against its last available snapshot so we can
            # still observe how the online matrices drift over time.
            baseline_idx = min(i, baseline_steps - 1)

            # Frobenius norm of difference
            state_diff = np.linalg.norm(
                self.state_matrices[i] - baseline_tracker.state_matrices[baseline_idx], 'fro'
            )
            control_diff = np.linalg.norm(
                self.control_matrices[i] - baseline_tracker.control_matrices[baseline_idx], 'fro'
            )

            state_matrix_diffs.append(state_diff)
            control_matrix_diffs.append(control_diff)

        return {
            'state_matrix_diffs': np.array(state_matrix_diffs),
            'control_matrix_diffs': np.array(control_matrix_diffs),
            'max_state_diff': np.max(state_matrix_diffs) if state_matrix_diffs else 0.0,
            'mean_state_diff': np.mean(state_matrix_diffs) if state_matrix_diffs else 0.0,
            'max_control_diff': np.max(control_matrix_diffs) if control_matrix_diffs else 0.0,
            'mean_control_diff': np.mean(control_matrix_diffs) if control_matrix_diffs else 0.0,
            'baseline_steps': baseline_steps,
            'comparison_steps': len(state_matrix_diffs),
        }


def analyze_prediction_differences(result_fixed, result_online) -> dict:
    """Analyze numerical differences between predictions.

    Args:
        result_fixed: SimulationResult from fixed linearization
        result_online: SimulationResult from online linearization

    Returns:
        Dictionary with difference statistics
    """
    n_predictions = min(len(result_fixed.reference_history), len(result_online.reference_history))

    max_diff_all = 0.0
    mean_diffs = []
    identical_count = 0

    for i in range(n_predictions):
        pred_fixed = result_fixed.reference_history[i]
        pred_online = result_online.reference_history[i]

        # Compute differences
        diff = np.abs(pred_fixed - pred_online)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        max_diff_all = max(max_diff_all, max_diff)
        mean_diffs.append(mean_diff)

        # Check if identical (within floating point tolerance)
        if np.allclose(pred_fixed, pred_online, atol=1e-10):
            identical_count += 1

    return {
        'n_predictions': n_predictions,
        'max_diff': max_diff_all,
        'mean_diff': np.mean(mean_diffs) if mean_diffs else 0.0,
        'identical_count': identical_count,
        'identical_percentage': 100 * identical_count / n_predictions if n_predictions > 0 else 0.0,
        'mean_diffs_array': np.array(mean_diffs),
    }


class TestOnlineVsFixedLinearization:
    """Compare online linearization vs fixed linearization around 0 degrees."""

    def test_compare_linearization_strategies(self):
        """Compare MPC behavior with online vs. fixed linearization.

        This test runs two simulations with different linearization strategies
        and plots the pitch evolution and predicted horizons to visualize
        the differences in MPC behavior.
        """
        # Test parameters
        duration_s = 3.0
        initial_pitch_deg = 20.0  # 10 degree initial perturbation
        initial_pitch_rad = np.deg2rad(initial_pitch_deg)

        # Create two simulations with different linearization strategies
        print(f"\n{'='*80}")
        print(f"Running MPC comparison with {initial_pitch_deg}° initial pitch perturbation")
        print(f"{'='*80}\n")

        # Create matrix trackers
        tracker_fixed = MatrixTracker()
        tracker_online = MatrixTracker()

        # Simulation 1: Fixed linearization around 0 degrees
        print("1. Running FIXED linearization (around 0°)...")
        config_fixed = SimulationConfig(
            model_path='Mujoco sim/robot_model.xml',
            robot_params_path='config/simulation/robot_params.yaml',
            mpc_params_path='config/simulation/mpc_params.yaml',
            estimator_params_path='config/simulation/estimator_params.yaml',
            duration_s=duration_s,
        )
        sim_fixed = MPCSimulation(config_fixed)
        # Ensure online linearization is disabled
        sim_fixed.controller._online_linearization_enabled = False

        # Instrument to track matrices
        sim_fixed.controller._matrix_tracker = tracker_fixed

        result_fixed = self._run_with_matrix_tracking(
            sim_fixed,
            tracker_fixed,
            duration_s,
            initial_pitch_rad,
        )

        if result_fixed.success:
            final_pitch_fixed = np.rad2deg(result_fixed.state_history[-1, PITCH_INDEX])
            max_pitch_fixed = np.rad2deg(np.max(np.abs(result_fixed.state_history[:, PITCH_INDEX])))
            print(f"   ✓ Success! Final pitch: {final_pitch_fixed:.2f}°, Max pitch: {max_pitch_fixed:.2f}°")
            print(f"   ✓ Mean solve time: {result_fixed.mean_solve_time_ms:.2f}ms")
        else:
            print(f"   ✗ Robot fell!")
            pytest.skip(f"Fixed linearization: Robot fell - cannot compare")

        # Simulation 2: Online linearization
        print("\n2. Running ONLINE linearization (at current state)...")
        config_online = SimulationConfig(
            model_path='Mujoco sim/robot_model.xml',
            robot_params_path='config/simulation/robot_params.yaml',
            mpc_params_path='config/simulation/mpc_params.yaml',
            estimator_params_path='config/simulation/estimator_params.yaml',
            duration_s=duration_s,
        )
        sim_online = MPCSimulation(config_online)
        # Enable online linearization
        sim_online.controller._online_linearization_enabled = True
        # Cache Jacobian functions for efficient linearization
        from robot_dynamics.linearization import build_jacobian_functions
        sim_online.controller._jacobian_functions = build_jacobian_functions(
            sim_online.controller._robot_params
        )

        # Instrument to track matrices
        sim_online.controller._matrix_tracker = tracker_online

        result_online = self._run_with_matrix_tracking(
            sim_online,
            tracker_online,
            duration_s,
            initial_pitch_rad,
        )

        if result_online.success:
            final_pitch_online = np.rad2deg(result_online.state_history[-1, PITCH_INDEX])
            max_pitch_online = np.rad2deg(np.max(np.abs(result_online.state_history[:, PITCH_INDEX])))
            print(f"   ✓ Success! Final pitch: {final_pitch_online:.2f}°, Max pitch: {max_pitch_online:.2f}°")
            print(f"   ✓ Mean solve time: {result_online.mean_solve_time_ms:.2f}ms")
        else:
            print(f"   ✗ Robot fell!")
            pytest.skip(f"Online linearization: Robot fell - cannot compare")

        # Analyze differences
        print(f"\n{'='*80}")
        print("NUMERICAL ANALYSIS")
        print(f"{'='*80}\n")

        # Prediction differences
        pred_analysis = analyze_prediction_differences(result_fixed, result_online)
        print(f"Prediction differences:")
        print(f"  - Number of predictions: {pred_analysis['n_predictions']}")
        print(f"  - Max difference: {pred_analysis['max_diff']:.6e}")
        print(f"  - Mean difference: {pred_analysis['mean_diff']:.6e}")
        print(f"  - Identical predictions: {pred_analysis['identical_count']}/{pred_analysis['n_predictions']}")
        print(f"  - Identical percentage: {pred_analysis['identical_percentage']:.1f}%")

        if pred_analysis['identical_percentage'] > 90:
            print(f"\n  ⚠️  WARNING: {pred_analysis['identical_percentage']:.1f}% of predictions are identical!")
            print(f"  This suggests online linearization may not be working correctly.")
        elif pred_analysis['max_diff'] < 1e-6:
            print(f"\n  ⚠️  WARNING: Max difference is very small ({pred_analysis['max_diff']:.3e})")
            print(f"  Differences may be too subtle to observe in plots.")
        else:
            print(f"\n  ✓ Predictions show meaningful differences.")

        # Matrix differences
        matrix_analysis = tracker_online.compute_matrix_differences(tracker_fixed)
        print(f"\nMatrix differences (Online vs Fixed):")
        print(f"  - Matrix updates tracked: {tracker_fixed.update_count} (fixed), {tracker_online.update_count} (online)")
        print(f"  - State matrix (A) max diff: {matrix_analysis['max_state_diff']:.6e}")
        print(f"  - State matrix (A) mean diff: {matrix_analysis['mean_state_diff']:.6e}")
        print(f"  - Control matrix (B) max diff: {matrix_analysis['max_control_diff']:.6e}")
        print(f"  - Control matrix (B) mean diff: {matrix_analysis['mean_control_diff']:.6e}")

        if matrix_analysis['max_state_diff'] < 1e-10:
            print(f"\n  ⚠️  BUG DETECTED: State matrices are identical!")
            print(f"  Online linearization is NOT updating the matrices.")
        elif matrix_analysis['max_state_diff'] < 1e-6:
            print(f"\n  ⚠️  Matrix differences are very small ({matrix_analysis['max_state_diff']:.3e})")
            print(f"  This may be expected at small perturbations (system is quite linear).")
        else:
            print(f"\n  ✓ Matrices show meaningful differences.")

        # Generate comparison plots
        print(f"\n{'='*80}")
        print("Generating comparison plots...")
        print(f"{'='*80}\n")

        self._plot_comparison(
            result_fixed,
            result_online,
            initial_pitch_deg,
            tracker_fixed,
            tracker_online,
            pred_analysis,
            matrix_analysis,
        )

        # Print comparison summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"Initial pitch perturbation: {initial_pitch_deg}°")
        print(f"\nFixed linearization (0°):")
        print(f"  - Final pitch:     {final_pitch_fixed:6.2f}°")
        print(f"  - Max pitch:       {max_pitch_fixed:6.2f}°")
        print(f"  - Mean solve time: {result_fixed.mean_solve_time_ms:6.2f}ms")
        print(f"\nOnline linearization (current state):")
        print(f"  - Final pitch:     {final_pitch_online:6.2f}°")
        print(f"  - Max pitch:       {max_pitch_online:6.2f}°")
        print(f"  - Mean solve time: {result_online.mean_solve_time_ms:6.2f}ms")
        print(f"\nDifference:")
        print(f"  - Final pitch diff: {abs(final_pitch_fixed - final_pitch_online):6.2f}°")
        print(f"  - Max pitch diff:   {abs(max_pitch_fixed - max_pitch_online):6.2f}°")
        print(f"  - Prediction max diff: {pred_analysis['max_diff']:.6e}")
        print(f"  - Matrix max diff: {matrix_analysis['max_state_diff']:.6e}")
        print("="*80 + "\n")

    def _run_with_matrix_tracking(self, simulation, tracker, duration_s, initial_pitch_rad):
        """Run simulation while tracking matrices at each step.

        Args:
            simulation: MPCSimulation instance
            tracker: MatrixTracker to record matrices
            duration_s: Simulation duration
            initial_pitch_rad: Initial pitch perturbation

        Returns:
            SimulationResult
        """
        # Monkey-patch the controller to record matrices after each update
        original_update = simulation.controller._mpc_solver.update_dynamics

        def tracked_update(discrete_dynamics):
            # Record matrices
            tracker.record(
                discrete_dynamics.state_matrix_discrete,
                discrete_dynamics.control_matrix_discrete,
                0.0  # Time will be updated by step tracking
            )
            # Call original
            return original_update(discrete_dynamics)

        simulation.controller._mpc_solver.update_dynamics = tracked_update

        # Also track initial matrices
        tracker.record(
            simulation.controller._mpc_solver._state_matrix,
            simulation.controller._mpc_solver._control_matrix,
            0.0
        )

        # Run simulation
        result = simulation.run(
            duration_s=duration_s,
            initial_pitch_rad=initial_pitch_rad,
            reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
        )

        return result

    def _plot_comparison(
        self,
        result_fixed,
        result_online,
        initial_pitch_deg: float,
        tracker_fixed: MatrixTracker,
        tracker_online: MatrixTracker,
        pred_analysis: dict,
        matrix_analysis: dict,
    ):
        """Generate comparison plots for online vs. fixed linearization.

        Creates a figure with 9 subplots showing:
        - Pitch evolution, predicted horizons, matrix differences, and diagnostics
        """
        fig = plt.figure(figsize=(20, 15))

        # Subplot 1: Actual pitch evolution comparison
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(
            result_fixed.time_s,
            np.rad2deg(result_fixed.state_history[:, PITCH_INDEX]),
            'b-',
            linewidth=2,
            label='Fixed linearization (0°)',
        )
        ax1.plot(
            result_online.time_s,
            np.rad2deg(result_online.state_history[:, PITCH_INDEX]),
            'r-',
            linewidth=2,
            label='Online linearization',
        )
        ax1.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Time [s]', fontsize=11)
        ax1.set_ylabel('Pitch angle [deg]', fontsize=11)
        ax1.set_title(f'Actual Pitch Evolution\n(Initial: {initial_pitch_deg}°)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)

        # Subplot 2: Fixed linearization - predicted horizons
        ax2 = plt.subplot(3, 3, 2)
        self._plot_predicted_horizons(
            ax2,
            result_fixed,
            'Fixed linearization (0°)',
            'Blues',
        )

        # Subplot 3: Online linearization - predicted horizons
        ax3 = plt.subplot(3, 3, 3)
        self._plot_predicted_horizons(
            ax3,
            result_online,
            'Online linearization',
            'Reds',
        )

        # Subplot 4: Pitch rate comparison
        ax4 = plt.subplot(3, 3, 4)
        from robot_dynamics.parameters import PITCH_RATE_INDEX
        ax4.plot(
            result_fixed.time_s,
            np.rad2deg(result_fixed.state_history[:, PITCH_RATE_INDEX]),
            'b-',
            linewidth=2,
            label='Fixed linearization',
        )
        ax4.plot(
            result_online.time_s,
            np.rad2deg(result_online.state_history[:, PITCH_RATE_INDEX]),
            'r-',
            linewidth=2,
            label='Online linearization',
        )
        ax4.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Time [s]', fontsize=11)
        ax4.set_ylabel('Pitch rate [deg/s]', fontsize=11)
        ax4.set_title('Pitch Rate Evolution', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)

        # Subplot 5: First prediction comparison
        ax5 = plt.subplot(3, 3, 5)
        self._plot_first_prediction_comparison(
            ax5,
            result_fixed,
            result_online,
        )

        # Subplot 6: Control effort comparison
        ax6 = plt.subplot(3, 3, 6)
        from robot_dynamics.parameters import CONTROL_DIMENSION
        # Plot sum of absolute control effort
        control_effort_fixed = np.sum(np.abs(result_fixed.control_history), axis=1)
        control_effort_online = np.sum(np.abs(result_online.control_history), axis=1)
        ax6.plot(
            result_fixed.time_s,
            control_effort_fixed,
            'b-',
            linewidth=2,
            label='Fixed linearization',
        )
        ax6.plot(
            result_online.time_s,
            control_effort_online,
            'r-',
            linewidth=2,
            label='Online linearization',
        )
        ax6.grid(True, alpha=0.3)
        ax6.set_xlabel('Time [s]', fontsize=11)
        ax6.set_ylabel('Control effort [N·m]', fontsize=11)
        ax6.set_title('Total Control Effort (|τ_L| + |τ_R|)', fontsize=12, fontweight='bold')
        ax6.legend(loc='best', fontsize=9)

        # Subplot 7: Matrix difference evolution
        ax7 = plt.subplot(3, 3, 7)
        if len(matrix_analysis['state_matrix_diffs']) > 0:
            time_steps = np.arange(len(matrix_analysis['state_matrix_diffs']))
            ax7.plot(time_steps, matrix_analysis['state_matrix_diffs'], 'g-', linewidth=2, label='||A_online - A_fixed||_F')
            ax7.plot(time_steps, matrix_analysis['control_matrix_diffs'], 'm-', linewidth=2, label='||B_online - B_fixed||_F')
            ax7.grid(True, alpha=0.3)
            ax7.set_xlabel('MPC Step', fontsize=11)
            ax7.set_ylabel('Frobenius Norm', fontsize=11)
            ax7.set_title('Matrix Differences Over Time', fontsize=12, fontweight='bold')
            ax7.legend(loc='best', fontsize=9)
            ax7.set_yscale('log')
        else:
            ax7.text(0.5, 0.5, 'No matrix data', ha='center', va='center', transform=ax7.transAxes)

        # Subplot 8: Prediction difference evolution
        ax8 = plt.subplot(3, 3, 8)
        if len(pred_analysis['mean_diffs_array']) > 0:
            time_steps = np.arange(len(pred_analysis['mean_diffs_array']))
            ax8.plot(time_steps, pred_analysis['mean_diffs_array'], 'purple', linewidth=2)
            ax8.grid(True, alpha=0.3)
            ax8.set_xlabel('MPC Step', fontsize=11)
            ax8.set_ylabel('Mean Prediction Difference', fontsize=11)
            ax8.set_title('Prediction Differences Over Time', fontsize=12, fontweight='bold')
            ax8.set_yscale('log')
        else:
            ax8.text(0.5, 0.5, 'No prediction data', ha='center', va='center', transform=ax8.transAxes)

        # Subplot 9: Diagnostic summary (text)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        # Create diagnostic text
        diag_text = f"DIAGNOSTIC SUMMARY\n\n"
        diag_text += f"Predictions:\n"
        diag_text += f"  Max diff: {pred_analysis['max_diff']:.3e}\n"
        diag_text += f"  Mean diff: {pred_analysis['mean_diff']:.3e}\n"
        diag_text += f"  Identical: {pred_analysis['identical_percentage']:.1f}%\n\n"
        diag_text += f"Matrices:\n"
        diag_text += f"  A max diff: {matrix_analysis['max_state_diff']:.3e}\n"
        diag_text += f"  A mean diff: {matrix_analysis['mean_state_diff']:.3e}\n"
        diag_text += f"  B max diff: {matrix_analysis['max_control_diff']:.3e}\n"
        diag_text += f"  B mean diff: {matrix_analysis['mean_control_diff']:.3e}\n\n"
        diag_text += f"Updates:\n"
        diag_text += f"  Fixed: {tracker_fixed.update_count}\n"
        diag_text += f"  Online: {tracker_online.update_count}\n\n"

        # Determine status
        if matrix_analysis['max_state_diff'] < 1e-10:
            status = "⚠️ BUG: Matrices identical!"
            status_color = 'red'
        elif pred_analysis['identical_percentage'] > 90:
            status = "⚠️ WARNING: Predictions identical"
            status_color = 'orange'
        elif matrix_analysis['max_state_diff'] < 1e-6:
            status = "⚠️ Small differences"
            status_color = 'orange'
        else:
            status = "✓ Working correctly"
            status_color = 'green'

        diag_text += f"Status: {status}"

        ax9.text(0.1, 0.9, diag_text, transform=ax9.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # Save plot
        output_dir = Path('test_results')
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f'online_vs_fixed_linearization_{initial_pitch_deg}deg.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")

        # Show plot (comment out if running in CI)
        plt.show()

    def _plot_predicted_horizons(
        self,
        ax,
        result,
        title: str,
        colormap: str,
    ):
        """Plot predicted pitch horizons over time.

        Args:
            ax: Matplotlib axis
            result: SimulationResult
            title: Plot title
            colormap: Matplotlib colormap name
        """
        # Extract sampling period from time array
        if len(result.time_s) > 1:
            dt = result.time_s[1] - result.time_s[0]
        else:
            dt = 0.065  # Default

        # Plot every Nth horizon to avoid overcrowding
        n_horizons = len(result.reference_history)
        plot_every = max(1, n_horizons // 15)  # Plot ~15 horizons max

        # Get colormap
        cmap = plt.get_cmap(colormap)

        for i in range(0, n_horizons, plot_every):
            predicted_traj = result.reference_history[i]
            horizon_length = predicted_traj.shape[0]

            # Time axis for this prediction
            t_start = result.time_s[i]
            t_horizon = t_start + np.arange(horizon_length) * dt

            # Predicted pitch
            predicted_pitch_deg = np.rad2deg(predicted_traj[:, PITCH_INDEX])

            # Color fades from dark to light as time progresses
            color = cmap(0.3 + 0.6 * (i / n_horizons))

            ax.plot(
                t_horizon,
                predicted_pitch_deg,
                color=color,
                linewidth=0.8,
                alpha=0.5,
            )

        # Overlay actual trajectory
        ax.plot(
            result.time_s,
            np.rad2deg(result.state_history[:, PITCH_INDEX]),
            'k-',
            linewidth=2.5,
            label='Actual trajectory',
            zorder=100,
        )

        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('Pitch angle [deg]', fontsize=11)
        ax.set_title(f'Predicted Horizons\n{title}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)

    def _plot_first_prediction_comparison(
        self,
        ax,
        result_fixed,
        result_online,
    ):
        """Plot comparison of first prediction from each method.

        Shows how the predicted horizon differs between online and fixed
        linearization at the first MPC solve (highest pitch angle).
        """
        # Extract first predictions
        pred_fixed = result_fixed.reference_history[0]
        pred_online = result_online.reference_history[0]

        # Extract sampling period
        if len(result_fixed.time_s) > 1:
            dt = result_fixed.time_s[1] - result_fixed.time_s[0]
        else:
            dt = 0.065

        # Time horizon
        horizon_length = min(pred_fixed.shape[0], pred_online.shape[0])
        t_horizon = np.arange(horizon_length) * dt

        # Predicted pitch
        pitch_fixed_deg = np.rad2deg(pred_fixed[:horizon_length, PITCH_INDEX])
        pitch_online_deg = np.rad2deg(pred_online[:horizon_length, PITCH_INDEX])

        ax.plot(
            t_horizon,
            pitch_fixed_deg,
            'b-',
            linewidth=2,
            marker='o',
            markersize=4,
            label='Fixed linearization',
        )
        ax.plot(
            t_horizon,
            pitch_online_deg,
            'r-',
            linewidth=2,
            marker='s',
            markersize=4,
            label='Online linearization',
        )

        # Plot actual trajectory for first few steps
        n_actual = min(horizon_length, len(result_fixed.time_s))
        ax.plot(
            result_fixed.time_s[:n_actual],
            np.rad2deg(result_fixed.state_history[:n_actual, PITCH_INDEX]),
            'k--',
            linewidth=2,
            marker='x',
            markersize=6,
            label='Actual trajectory',
        )

        ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time from first solve [s]', fontsize=11)
        ax.set_ylabel('Pitch angle [deg]', fontsize=11)
        ax.set_title('First Predicted Horizon Comparison', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)


if __name__ == '__main__':
    """Run the test standalone and generate plots."""
    if not MUJOCO_AVAILABLE:
        print("ERROR: MuJoCo is not installed. Cannot run test.")
        exit(1)

    test = TestOnlineVsFixedLinearization()
    test.test_compare_linearization_strategies()

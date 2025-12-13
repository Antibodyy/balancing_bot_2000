"""Debug script for analyzing perturbation recovery."""

import argparse
import numpy as np
from simulation import SimulationConfig, MUJOCO_AVAILABLE
from debug import MPCDiagnostics
from mpc import ReferenceCommand, ReferenceMode

# Parse command line arguments
parser = argparse.ArgumentParser(description='Debug perturbation recovery')
parser.add_argument('--viewer', action='store_true',
                    help='Show MuJoCo viewer (disables batch plotting)')
parser.add_argument('--perturbation', type=float, default=None,
                    help='Single perturbation to test (in degrees)')
args = parser.parse_args()

if args.viewer and not MUJOCO_AVAILABLE:
    print("MuJoCo is not installed. Install with: pip install mujoco")
    exit(1)

# Configuration
config = SimulationConfig(
    model_path='mujoco_sim/robot_model.xml',
    robot_params_path='config/simulation/robot_params.yaml',
    mpc_params_path='config/simulation/mpc_params.yaml',
    estimator_params_path='config/simulation/estimator_params.yaml',
    duration_s=10.0,  # 10 second simulation for viewer mode
)

# Create diagnostics
print("Creating MPC Diagnostics...")
diag = MPCDiagnostics(config)

# Test different perturbation magnitudes
if args.perturbation is not None:
    perturbations = [np.deg2rad(args.perturbation)]
else:
    perturbations = [0.01, 0.05, 0.1]  # radians (~0.6°, 3°, 6°)

print("\n" + "="*80)
print("PERTURBATION RECOVERY ANALYSIS")
print("="*80)

if args.viewer:
    # Interactive viewer mode - single perturbation
    initial_pitch = perturbations[0]
    print(f"\nTesting with initial pitch: {np.rad2deg(initial_pitch):.2f} degrees")
    print("Mode: MuJoCo Viewer (interactive)")
    print("\nStarting MuJoCo viewer... (close window to stop)")
    print("Controls: Space=pause, Backspace=reset, Mouse=rotate, Scroll=zoom\n")

    # Run with viewer
    result = diag.simulation.run_with_viewer(
        initial_pitch_rad=initial_pitch,
        reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
    )

    # Print results
    print("\n" + "=" * 50)
    print("Simulation Results")
    print("=" * 50)
    print(f"  Success (no fall): {result.success}")
    print(f"  Duration: {result.time_s[-1]:.1f}s" if len(result.time_s) > 0 else "  Duration: 0s")
    print(f"  Mean solve time: {result.mean_solve_time_ms:.1f}ms")
    print(f"  Max solve time: {result.max_solve_time_ms:.1f}ms")
    print(f"  Deadline violations: {result.deadline_violations}")

    if len(result.state_history) > 0:
        max_pitch = np.rad2deg(np.max(np.abs(result.state_history[:, 1])))
        final_pitch = np.rad2deg(result.state_history[-1, 1])
        print(f"  Max pitch: {max_pitch:.1f} degrees")
        print(f"  Final pitch: {final_pitch:.1f} degrees")

else:
    # Batch analysis mode - multiple perturbations with plots
    for initial_pitch in perturbations:
        print(f"\n\n{'='*80}")
        print(f"Testing with initial pitch: {np.rad2deg(initial_pitch):.2f} degrees")
        print('='*80)

        # Run simulation with diagnostics
        result, summary = diag.run_with_diagnostics(
            duration_s=2.0,
            initial_pitch_rad=initial_pitch,
            verbose=True,  # Print step-by-step trace
        )

        # Print summary statistics
        summary.print_summary()

        # Generate plots for this perturbation
        save_dir = f"test_and_debug_output/perturbation_{np.rad2deg(initial_pitch):.1f}deg"
        print(f"\nGenerating diagnostic plots...")
        diag.plot_all(result, save_dir=save_dir, show=False)
        print(f"Plots saved to: {save_dir}/")
        print(f"  - state_comparison.png")
        print(f"  - control_analysis.png")
        print(f"  - prediction_accuracy.png")
        print(f"  - closed_loop_diagnosis.png")

        # Analysis
        if not result.success:
            print(f"\n⚠️  FAILED: Robot fell after {result.time_s[-1]:.3f}s")
            final_pitch = np.rad2deg(result.state_history[-1, 1])
            print(f"   Final pitch: {final_pitch:.2f}°")
        else:
            print(f"\n✓  SUCCESS: Robot remained balanced")
            final_pitch = np.rad2deg(result.state_history[-1, 1])
            max_pitch = np.rad2deg(np.max(np.abs(result.state_history[:, 1])))
            print(f"   Final pitch: {final_pitch:.2f}°")
            print(f"   Max pitch excursion: {max_pitch:.2f}°")

    print("\n" + "="*80)
    print("Analysis complete! Check test_and_debug_output/ for detailed plots.")
    print("="*80)

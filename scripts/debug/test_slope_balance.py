"""Test static balance on a slope.

This script tests the robot's ability to maintain balance (zero velocity)
on an inclined surface. It dynamically modifies the MuJoCo model and robot
parameters to create a slope at the specified angle.

Usage:
    python3 scripts/debug/test_slope_balance.py
    python3 scripts/debug/test_slope_balance.py --slope 5.0 --duration 20.0
    python3 scripts/debug/test_slope_balance.py --slope 10.0 --initial-pitch 3.0
    python3 scripts/debug/test_slope_balance.py --slope 5.0 --viewer
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import tempfile
import shutil
import yaml
import xml.etree.ElementTree as ET
from simulation import SimulationConfig, MPCSimulation
from mpc import ReferenceCommand, ReferenceMode
from robot_dynamics.parameters import (
    PITCH_INDEX, PITCH_RATE_INDEX,
    POSITION_INDEX, VELOCITY_INDEX,
    YAW_INDEX, YAW_RATE_INDEX
)

# Parse arguments
parser = argparse.ArgumentParser(description='Test static balance on a slope')
parser.add_argument('--slope', type=float, default=0.0,
                    help='Slope angle in degrees (default: 0.0 for flat ground)')
parser.add_argument('--duration', type=float, default=20.0,
                    help='Simulation duration in seconds (default: 20.0, ignored in viewer mode)')
parser.add_argument('--initial-pitch', type=float, default=0.0,
                    help='Initial pitch perturbation in degrees (default: 0.0)')
parser.add_argument('--viewer', action='store_true',
                    help='Run with interactive MuJoCo viewer (runs until closed or robot falls)')
args = parser.parse_args()

# Convert angles to radians
slope_rad = np.deg2rad(args.slope)
initial_pitch_rad = np.deg2rad(args.initial_pitch)
equilibrium_pitch_rad = -slope_rad  # Robot leans opposite to slope

print("\n" + "="*80)
print("SLOPE BALANCE TEST")
print("="*80)
print(f"Configuration:")
print(f"  Slope angle: {args.slope} deg ({slope_rad:.4f} rad)")
print(f"  Equilibrium pitch: {np.rad2deg(equilibrium_pitch_rad):.2f} deg (robot leans {'back' if args.slope > 0 else 'forward' if args.slope < 0 else 'upright'} to balance)")
print(f"  Duration: {args.duration} s" + (" (ignored in viewer mode)" if args.viewer else ""))
print(f"  Initial pitch: {args.initial_pitch} deg ({initial_pitch_rad:.4f} rad)")
if args.initial_pitch != 0:
    perturbation_from_eq = args.initial_pitch - np.rad2deg(equilibrium_pitch_rad)
    print(f"  Initial perturbation from equilibrium: {perturbation_from_eq:.2f} deg")
print(f"  Viewer mode: {'ENABLED' if args.viewer else 'DISABLED'}")
print("="*80)

# Create temporary directory for modified files
temp_dir = tempfile.mkdtemp(prefix="slope_balance_")
print(f"\nCreating temporary files in: {temp_dir}")

try:
    # 1. Modify robot_model.xml to add physical slope
    original_model_path = project_root / 'robot_model.xml'
    temp_model_path = Path(temp_dir) / 'robot_model.xml'

    print(f"Modifying MuJoCo model to add {args.slope}° physical slope...")

    # Parse XML
    tree = ET.parse(original_model_path)
    root = tree.getroot()

    # Find floor geom and update euler attribute
    floor_geom = root.find(".//geom[@name='floor']")
    if floor_geom is not None:
        # Set euler attribute: euler="rx ry rz" where ry is the slope angle
        floor_geom.set('euler', f'0 {slope_rad} 0')
        print(f"  Updated floor geom euler to: 0 {slope_rad:.6f} 0")
    else:
        print("  WARNING: Could not find floor geom in XML")

    # Write modified XML
    tree.write(temp_model_path)
    print(f"  Saved modified model to: {temp_model_path}")

    # 2. Modify robot_params.yaml to set ground_slope_rad
    original_params_path = project_root / 'config' / 'simulation' / 'robot_params.yaml'
    temp_params_path = Path(temp_dir) / 'robot_params.yaml'

    print(f"Modifying robot parameters to set ground_slope_rad={slope_rad:.6f}...")

    # Load YAML
    with open(original_params_path, 'r') as f:
        params = yaml.safe_load(f)

    # Set ground_slope_rad to 0 (slope is handled physically in MuJoCo)
    # The physical slope in MuJoCo provides the correct gravity and contact forces
    params['ground_slope_rad'] = 0.0
    print(f"  Set ground_slope_rad to: 0.0 rad (physical slope in MuJoCo)")

    # Write modified YAML
    with open(temp_params_path, 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    print(f"  Saved modified params to: {temp_params_path}")

    # 3. Create simulation configuration with modified files
    config = SimulationConfig(
        model_path=str(temp_model_path),  # Modified model with sloped floor
        robot_params_path=str(temp_params_path),  # Modified params with slope
        mpc_params_path='config/simulation/mpc_params.yaml',
        estimator_params_path='config/simulation/estimator_params.yaml',
        duration_s=args.duration,
    )

    # Create simulation
    simulation = MPCSimulation(config)

    # Balance reference command (zero velocity)
    reference_command = ReferenceCommand(mode=ReferenceMode.BALANCE)

    # Run simulation
    if args.viewer:
        print("\nLaunching viewer...")
        print("Close the viewer window to end simulation and see results.")
        result = simulation.run_with_viewer(
            initial_pitch_rad=initial_pitch_rad,
            reference_command=reference_command,
        )
    else:
        print("\nRunning simulation...")
        result = simulation.run(
            initial_pitch_rad=initial_pitch_rad,
            reference_command=reference_command,
        )

    # Print results
    print("\n" + "="*80)
    print("SIMULATION RESULTS")
    print("="*80)

    # Determine success
    success = result.success
    if success and len(result.state_history) > 0:
        final_velocity = result.state_history[-1, VELOCITY_INDEX]
        final_pitch = result.state_history[-1, PITCH_INDEX]
        final_pitch_error = final_pitch - equilibrium_pitch_rad

        # Check success criteria
        velocity_ok = abs(final_velocity) < 0.01  # m/s
        pitch_ok = abs(np.rad2deg(final_pitch_error)) < 1.0  # degrees
        success = velocity_ok and pitch_ok

    status_msg = 'SUCCESS (Robot stayed balanced)' if success else 'FAILED'
    print(f"Status: {status_msg}")
    print(f"Duration: {result.total_duration_s:.2f}s")
    print(f"Mean solve time: {result.mean_solve_time_ms:.2f}ms")
    print(f"Max solve time: {result.max_solve_time_ms:.2f}ms")
    print(f"Deadline violations: {result.deadline_violations}")

    # Final state values
    if len(result.state_history) > 0:
        final_state = result.state_history[-1]
        print(f"\nFinal State:")
        print(f"  Position: {final_state[POSITION_INDEX]:.2f} m (drift)")
        print(f"  Velocity: {final_state[VELOCITY_INDEX]:.4f} m/s (target: 0.0000 m/s) {'✓' if abs(final_state[VELOCITY_INDEX]) < 0.01 else '✗'}")
        print(f"  Pitch: {np.rad2deg(final_state[PITCH_INDEX]):.4f} deg (equilibrium: {np.rad2deg(equilibrium_pitch_rad):.2f} deg) {'✓' if abs(np.rad2deg(final_state[PITCH_INDEX] - equilibrium_pitch_rad)) < 1.0 else '✗'}")
        print(f"  Pitch rate: {np.rad2deg(final_state[PITCH_RATE_INDEX]):.4f} deg/s {'✓' if abs(np.rad2deg(final_state[PITCH_RATE_INDEX])) < 5.0 else '✗'}")
        print(f"  Yaw: {np.rad2deg(final_state[YAW_INDEX]):.4f} deg")

        # Compute tracking error statistics
        pitch_error = result.state_history[:, PITCH_INDEX] - equilibrium_pitch_rad
        pitch_error_deg = np.rad2deg(pitch_error)
        velocity_error = result.state_history[:, VELOCITY_INDEX] - 0.0

        print(f"\nBalance Performance:")
        print(f"  Pitch equilibrium: {np.rad2deg(equilibrium_pitch_rad):.2f} deg")
        print(f"  Mean pitch error: {np.mean(np.abs(pitch_error_deg)):.4f} deg (from equilibrium)")
        print(f"  Pitch RMSE: {np.sqrt(np.mean(pitch_error_deg**2)):.4f} deg")
        print(f"  Max pitch error: {np.max(np.abs(pitch_error_deg)):.4f} deg")
        print(f"  Position drift: {abs(final_state[POSITION_INDEX]):.4f} m")
        print(f"  Mean velocity: {np.mean(np.abs(velocity_error)):.4f} m/s {'✓' if np.mean(np.abs(velocity_error)) < 0.01 else '✗'}")
        print(f"  Max velocity: {np.max(np.abs(velocity_error)):.4f} m/s")

        # Perturbation recovery analysis (if initial perturbation was applied)
        if args.initial_pitch != 0.0:
            initial_perturbation = args.initial_pitch - np.rad2deg(equilibrium_pitch_rad)

            # Find settling time (time to stay within 2% of equilibrium)
            settling_threshold_deg = 0.02 * abs(initial_perturbation) if abs(initial_perturbation) > 0 else 1.0
            settled = np.abs(pitch_error_deg) < settling_threshold_deg

            settling_time = None
            if len(result.time_s) > 0:
                # Find first time where it stays settled
                for i in range(len(settled) - 10):
                    if np.all(settled[i:]):
                        settling_time = result.time_s[i]
                        break

            print(f"\nPerturbation Recovery:")
            print(f"  Initial perturbation: {initial_perturbation:.2f} deg from equilibrium")
            print(f"  Peak deviation: {np.max(np.abs(pitch_error_deg)):.2f} deg")
            if settling_time is not None:
                print(f"  Settling time (2% band): {settling_time:.2f} s")
            else:
                print(f"  Settling time: Did not settle within simulation duration")
            print(f"  Final error: {np.abs(pitch_error_deg[-1]):.2f} deg {'✓' if np.abs(pitch_error_deg[-1]) < 1.0 else '✗'}")

        # Control effort
        control_rms = np.sqrt(np.mean(result.control_history**2))
        max_control = np.max(np.abs(result.control_history))

        print(f"\nControl Effort:")
        print(f"  RMS torque: {control_rms:.4f} Nm")
        print(f"  Max torque: {max_control:.4f} Nm")

        # Generate plots
        slope_str = f"{args.slope:.1f}deg" if args.slope != 0 else "flat"
        save_dir = f"debug_output/slope_balance_{slope_str}"
        if args.initial_pitch != 0:
            save_dir += f"_perturb_{abs(args.initial_pitch):.1f}deg"
        if args.viewer:
            save_dir += "_viewer"
        else:
            save_dir += f"_{args.duration:.0f}s"
        os.makedirs(save_dir, exist_ok=True)

        print(f"\nGenerating plots...")

        # Create figure with 6 subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

        time_s = result.time_s

        # 1. Pitch angle tracking
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_s, np.rad2deg(result.state_history[:, PITCH_INDEX]),
                 'b-', linewidth=2, label='Actual')
        ax1.axhline(np.rad2deg(equilibrium_pitch_rad), color='r', linestyle='--',
                    linewidth=2, label=f'Equilibrium ({np.rad2deg(equilibrium_pitch_rad):.1f}°)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Pitch (deg)')
        ax1.set_title('Pitch Angle', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Pitch error from equilibrium
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_s, pitch_error_deg, 'r-', linewidth=2)
        ax2.axhline(0, color='k', linestyle='-', alpha=0.3)
        pitch_rmse = np.sqrt(np.mean(pitch_error_deg**2))
        ax2.axhline(pitch_rmse, color='orange', linestyle='--',
                    alpha=0.5, label=f'RMSE: {pitch_rmse:.2f}°')
        ax2.axhline(-pitch_rmse, color='orange', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Pitch Error (deg)')
        ax2.set_title('Pitch Error from Equilibrium', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Position (should show minimal drift)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_s, result.state_history[:, POSITION_INDEX], 'b-', linewidth=2)
        ax3.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position (m)')
        ax3.set_title('Position (Should Have Minimal Drift)', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Velocity (should converge to zero)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(time_s, result.state_history[:, VELOCITY_INDEX],
                 'b-', linewidth=2, label='Actual')
        ax4.axhline(0, color='r', linestyle='--', linewidth=2, label='Target (0 m/s)')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.set_title('Velocity (KEY METRIC - Should Converge to Zero)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Pitch rate
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time_s, np.rad2deg(result.state_history[:, PITCH_RATE_INDEX]),
                 'b-', linewidth=2)
        ax5.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Pitch Rate (deg/s)')
        ax5.set_title('Pitch Rate', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # 6. Control torques
        ax6 = fig.add_subplot(gs[2, 1])
        n_controls = len(result.control_history)
        time_controls = time_s[:n_controls]
        ax6.plot(time_controls, result.control_history[:, 0],
                 'b-', linewidth=1.5, label='Left wheel')
        ax6.plot(time_controls, result.control_history[:, 1],
                 'r--', linewidth=1.5, label='Right wheel')
        ax6.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Torque (Nm)')
        ax6.set_title('Control Torques', fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # Overall title
        if args.viewer:
            mode_str = f'{args.slope}° slope (viewer mode)'
        else:
            mode_str = f'{args.slope}° slope for {args.duration}s'
        if args.initial_pitch != 0:
            mode_str += f' with {args.initial_pitch}° initial perturbation'
        fig.suptitle(f'Slope Balance Test: {mode_str}',
                     fontsize=16, fontweight='bold', y=0.995)

        plt.savefig(f"{save_dir}/balance_performance.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_dir}/balance_performance.png")

        # Create state space plot (pitch phase portrait)
        fig2, ax = plt.subplots(figsize=(10, 8))

        # Plot trajectory in pitch-pitch_rate space (relative to equilibrium)
        ax.plot(pitch_error_deg,
                np.rad2deg(result.state_history[:, PITCH_RATE_INDEX]),
                'b-', linewidth=1.5, alpha=0.7)
        ax.plot(pitch_error_deg[0],
                np.rad2deg(result.state_history[0, PITCH_RATE_INDEX]),
                'go', markersize=10, label='Start')
        ax.plot(pitch_error_deg[-1],
                np.rad2deg(result.state_history[-1, PITCH_RATE_INDEX]),
                'ro', markersize=10, label='End')
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Pitch Error from Equilibrium (deg)')
        ax.set_ylabel('Pitch Rate (deg/s)')
        ax.set_title(f'Pitch Phase Portrait (Equilibrium at origin)\n{mode_str}',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.savefig(f"{save_dir}/state_space.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_dir}/state_space.png")

        plt.close('all')

        print(f"\nPlots saved to: {save_dir}/")
    else:
        print("\nNo data collected (simulation ended immediately)")

    print("\n" + "="*80)
    if success:
        print("Test complete - Robot successfully maintained balance on slope!")
    else:
        print("Test complete - Robot failed to maintain balance")
    print("="*80)

finally:
    # Clean up temporary directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory: {temp_dir}")

#!/usr/bin/env python3
"""Compute terminal control-invariant set for MPC.

This script computes a numerically approximated maximal control-invariant
set for the pitch dynamics (theta, theta_dot) of the balancing robot. The
resulting bounds guarantee that if the terminal state is within the set,
the robot can be kept stable indefinitely.

Usage:
    # For simulation parameters:
    python3 scripts/compute_terminal_set.py --config simulation

    # For hardware parameters:
    python3 scripts/compute_terminal_set.py --config hardware

    # Custom grid resolution (default: 40):
    python3 scripts/compute_terminal_set.py --config simulation --grid-resolution 50

The computed bounds are saved to:
    config/simulation/terminal_set.yaml  (for simulation)
    config/hardware/terminal_set.yaml    (for hardware)

WARNING: Recompute this set whenever robot parameters change (mass, inertia,
geometry, control limits, etc.)!
"""

import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from robot_dynamics.parameters import RobotParameters, PITCH_INDEX, PITCH_RATE_INDEX
from robot_dynamics.linearization import linearize_at_equilibrium
from robot_dynamics.discretization import discretize_linear_dynamics
from mpc.config import MPCConfig
from mpc.control_invariant_set import ConstraintDebugger


def compute_upright_equilibrium(robot_params: RobotParameters) -> tuple:
    """Compute upright equilibrium state and control.

    For a balancing robot, upright equilibrium is:
    - Position: arbitrary (we set to 0)
    - Pitch: 0 rad (upright)
    - Yaw: 0 rad (arbitrary heading)
    - All velocities: 0
    - Control: 0 torque (no actuation needed at equilibrium)

    Args:
        robot_params: Robot physical parameters

    Returns:
        Tuple of (equilibrium_state, equilibrium_control)
    """
    # State vector: [x, theta, psi, x_dot, theta_dot, psi_dot]
    equilibrium_state = np.zeros(6)
    equilibrium_state[PITCH_INDEX] = 0.0  # Upright

    # Control vector: [tau_L, tau_R]
    equilibrium_control = np.zeros(2)

    return equilibrium_state, equilibrium_control


def main():
    """Main computation script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute terminal control-invariant set for MPC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config',
        type=str,
        choices=['simulation', 'hardware'],
        required=True,
        help='Configuration to use (simulation or hardware)'
    )
    parser.add_argument(
        '--grid-resolution',
        type=int,
        default=40,
        help='Grid resolution (points per dimension). Higher = more accurate but slower. Default: 40'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=100,
        help='Maximum fixed-point iterations for invariant set computation. Default: 100'
    )

    args = parser.parse_args()
    config_type = args.config
    grid_resolution = args.grid_resolution
    max_iterations = args.max_iterations

    print(f"\n{'='*70}")
    print(f"Terminal Control-Invariant Set Computation")
    print(f"{'='*70}")
    print(f"Configuration: {config_type}")
    print(f"Grid resolution: {grid_resolution}")
    print(f"Max iterations: {max_iterations}")
    print(f"{'='*70}\n")

    # Define paths
    config_dir = project_root / 'config' / config_type
    robot_params_path = str(config_dir / 'robot_params.yaml')
    mpc_params_path = str(config_dir / 'mpc_params.yaml')
    output_path = str(config_dir / 'terminal_set.yaml')

    # Load parameters
    print(f"Loading robot parameters from: {robot_params_path}")
    robot_params = RobotParameters.from_yaml(robot_params_path)

    print(f"Loading MPC parameters from: {mpc_params_path}")
    mpc_config = MPCConfig.from_yaml(mpc_params_path)

    # Compute upright equilibrium
    print("\nComputing upright equilibrium...")
    equilibrium_state, equilibrium_control = compute_upright_equilibrium(robot_params)
    print(f"  Equilibrium state: {equilibrium_state}")
    print(f"  Equilibrium control: {equilibrium_control}")

    # Linearize dynamics around upright equilibrium
    print("\nLinearizing dynamics around upright equilibrium...")
    linearized_dynamics = linearize_at_equilibrium(
        robot_params,
        equilibrium_state,
        equilibrium_control
    )
    print(f"  State matrix A shape: {linearized_dynamics.state_matrix.shape}")
    print(f"  Control matrix B shape: {linearized_dynamics.control_matrix.shape}")

    # Discretize linear dynamics
    print(f"\nDiscretizing with sampling period: {mpc_config.sampling_period_s} s")
    discrete_dynamics = discretize_linear_dynamics(
        linearized_dynamics.state_matrix,
        linearized_dynamics.control_matrix,
        mpc_config.sampling_period_s
    )

    # Create constraint debugger
    debugger = ConstraintDebugger(mpc_params_path=mpc_params_path)

    # Compute invariant set for pitch dynamics only
    invariant_set = debugger.compute_invariant_set(
        state_matrix=discrete_dynamics.state_matrix_discrete,
        control_matrix=discrete_dynamics.control_matrix_discrete,
        grid_resolution=grid_resolution,
        reduced_dims=[PITCH_INDEX, PITCH_RATE_INDEX],
        max_iterations=max_iterations
    )

    # Print results
    bounds = invariant_set['bounds']
    iterations = invariant_set.get('iterations', 'N/A')
    print(f"\n{'='*70}")
    print("RESULTS: Terminal Set Bounds")
    print(f"{'='*70}")
    print(f"Convergence: {iterations} iterations")
    print(f"\nPitch (theta):")
    print(f"  Lower bound: {bounds[PITCH_INDEX, 0]:.6f} rad ({np.rad2deg(bounds[PITCH_INDEX, 0]):.2f} deg)")
    print(f"  Upper bound: {bounds[PITCH_INDEX, 1]:.6f} rad ({np.rad2deg(bounds[PITCH_INDEX, 1]):.2f} deg)")
    print(f"\nPitch rate (theta_dot):")
    print(f"  Lower bound: {bounds[PITCH_RATE_INDEX, 0]:.6f} rad/s ({np.rad2deg(bounds[PITCH_RATE_INDEX, 0]):.2f} deg/s)")
    print(f"  Upper bound: {bounds[PITCH_RATE_INDEX, 1]:.6f} rad/s ({np.rad2deg(bounds[PITCH_RATE_INDEX, 1]):.2f} deg/s)")
    print(f"\nVolume fraction: {invariant_set['volume_fraction']*100:.1f}% of search space")
    print(f"Feasible states: {invariant_set['feasible_states'].shape[0]}")
    print(f"{'='*70}\n")

    # Save to YAML
    computation_metadata = {
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'robot_params_path': robot_params_path,
        'mpc_params_path': mpc_params_path,
    }

    debugger.save_terminal_set_config(
        bounds=bounds,
        grid_resolution=grid_resolution,
        computation_metadata=computation_metadata,
        iterations=iterations,
        volume_fraction=invariant_set['volume_fraction'],
        output_path=output_path
    )

    print(f"\nNext steps:")
    print(f"1. Review the computed bounds above")
    print(f"2. Update {mpc_params_path}:")
    print(f"     terminal_pitch_limit_rad: {bounds[PITCH_INDEX, 1]:.6f}")
    print(f"     terminal_pitch_rate_limit_radps: {bounds[PITCH_RATE_INDEX, 1]:.6f}")
    print(f"3. Test MPC with terminal constraints enabled")
    print(f"4. WARNING: Recompute if robot parameters change!\n")


if __name__ == "__main__":
    main()

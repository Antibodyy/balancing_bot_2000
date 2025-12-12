"""
Debug utilities for MPC constraint analysis and control-invariant set computation to ensure feasibility

This module provides tools to:
1. Log constraint violations and margins during MPC execution
2. Compute numerical approximations of control-invariant sets
3. Validate terminal constraints

"""

import numpy as np
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from datetime import datetime

from robot_dynamics.parameters import (
    PITCH_INDEX,
    PITCH_RATE_INDEX,
    VELOCITY_INDEX,
    STATE_DIMENSION,
    CONTROL_DIMENSION,
)


class ConstraintDebugger:
    """Analyze MPC constraints and compute control-invariant sets.

    This class provides two main functionalities:
    1. Logging constraint violations during MPC execution (for debugging)
    2. Computing numerical approximations of control-invariant sets (offline)

    The control-invariant set computation uses a grid-based search to find
    states from which the system can be kept within constraints indefinitely.
    This is critical for terminal constraints in MPC to guarantee feasibility.

    State vector ordering: [x, theta, psi, x_dot, theta_dot, psi_dot]
    Control vector ordering: [tau_L, tau_R] in N·m

    Example offline computation:
        See scripts/compute_terminal_set.py for complete workflow.
    """
    
    def __init__(self, mpc_params_path: str, output_dir: str = "debug/logs"):
        """
        Initialize constraint debugger.
        
        Args:
            mpc_params_path: Path to mpc_params.yaml
            output_dir: Directory for debug output files
        """
        self.output_dir_ = Path(output_dir)
        self.output_dir_.mkdir(parents=True, exist_ok=True)
        
        with open(mpc_params_path, 'r') as file:
            self.params_ = yaml.safe_load(file)
        
        self._extract_constraints()
        self.log_file_path_ = None
        self.csv_writer_ = None
        self.csv_file_ = None
        
    def _extract_constraints(self) -> None:
        """Extract constraint bounds from MPC configuration.

        State vector ordering: [x, theta, psi, x_dot, theta_dot, psi_dot]
        Control vector ordering: [tau_L, tau_R] in N·m

        Only pitch and pitch rate are constrained. Other states have infinite bounds.
        """
        # Initialize all states as unconstrained
        self.state_lower_bounds_ = np.full(STATE_DIMENSION, -np.inf)
        self.state_upper_bounds_ = np.full(STATE_DIMENSION, np.inf)

        # Constrain pitch (theta) at index 1
        self.state_lower_bounds_[PITCH_INDEX] = -self.params_['pitch_limit_rad']
        self.state_upper_bounds_[PITCH_INDEX] = self.params_['pitch_limit_rad']

        # Constrain pitch rate (theta_dot) at index 4
        self.state_lower_bounds_[PITCH_RATE_INDEX] = -self.params_['pitch_rate_limit_radps']
        self.state_upper_bounds_[PITCH_RATE_INDEX] = self.params_['pitch_rate_limit_radps']

        # Constrain velocity (x_dot) at index 3
        # Use large initial bounds - invariant set algorithm will tighten based on coupling
        self.state_lower_bounds_[VELOCITY_INDEX] = -5.0  # m/s (initial, algorithm will shrink)
        self.state_upper_bounds_[VELOCITY_INDEX] = 5.0   # m/s

        # Input constraints: [tau_L, tau_R] in N·m
        control_limit_nm = self.params_['control_limit_nm']
        self.input_lower_bounds_ = np.array([-control_limit_nm, -control_limit_nm])
        self.input_upper_bounds_ = np.array([control_limit_nm, control_limit_nm])
        
    def initialize_log(self, experiment_name: str) -> None:
        """
        Initialize CSV log file for constraint debugging.
        
        Args:
            experiment_name: Name identifier for this experiment
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"constraint_log_{experiment_name}_{timestamp}.csv"
        self.log_file_path_ = self.output_dir_ / filename
        
        self.csv_file_ = open(self.log_file_path_, 'w', newline='')
        self.csv_writer_ = csv.writer(self.csv_file_)
        
        # Write header
        # State vector: [x, theta, psi, x_dot, theta_dot, psi_dot]
        header = [
            'timestep', 'horizon_index',
            'state_norm', 'input_norm',
            'min_state_margin', 'violated_state_index',
            'min_input_margin', 'violated_input_index',
            'x_m', 'theta_rad', 'psi_rad',
            'x_dot_mps', 'theta_dot_radps', 'psi_dot_radps',
            'tau_left_nm', 'tau_right_nm',
            'is_terminal'
        ]
        self.csv_writer_.writerow(header)
        
    def check_state_constraints(self, state: np.ndarray) -> Tuple[float, int]:
        """
        Check state constraint margins.

        Args:
            state: State vector [x, theta, psi, x_dot, theta_dot, psi_dot]

        Returns:
            (min_margin, violated_index): Minimum margin and index of tightest constraint
                Negative margin indicates violation
        """
        if state.shape[0] != STATE_DIMENSION:
            raise ValueError(f"Expected state dimension {STATE_DIMENSION}, got {state.shape[0]}")
            
        lower_margins = state - self.state_lower_bounds_
        upper_margins = self.state_upper_bounds_ - state
        
        all_margins = np.concatenate([lower_margins, upper_margins])
        min_margin_index = np.argmin(all_margins)
        min_margin = all_margins[min_margin_index]
        
        # Map back to state index
        violated_index = min_margin_index % 6
        
        return min_margin, violated_index
        
    def check_input_constraints(self, control_input: np.ndarray) -> Tuple[float, int]:
        """
        Check input constraint margins.

        Args:
            control_input: Control vector [tau_L, tau_R] in N·m

        Returns:
            (min_margin, violated_index): Minimum margin and index of tightest constraint
                Negative margin indicates violation
        """
        if control_input.shape[0] != CONTROL_DIMENSION:
            raise ValueError(f"Expected input dimension {CONTROL_DIMENSION}, got {control_input.shape[0]}")
            
        lower_margins = control_input - self.input_lower_bounds_
        upper_margins = self.input_upper_bounds_ - control_input
        
        all_margins = np.concatenate([lower_margins, upper_margins])
        min_margin_index = np.argmin(all_margins)
        min_margin = all_margins[min_margin_index]
        
        violated_index = min_margin_index % 2
        
        return min_margin, violated_index
        
    def log_step(self, 
                 timestep: int,
                 state_trajectory: np.ndarray,
                 input_trajectory: np.ndarray) -> None:
        """
        Log constraint information for one MPC solve.
        
        Args:
            timestep: Current simulation timestep
            state_trajectory: Predicted states shape (N+1, 6)
            input_trajectory: Predicted inputs shape (N, 2)
        """
        if self.csv_writer_ is None:
            raise RuntimeError("Call initialize_log() before logging")
            
        horizon_length = state_trajectory.shape[0]
        
        for horizon_index in range(horizon_length):
            state = state_trajectory[horizon_index]
            state_norm = np.linalg.norm(state)
            
            state_margin, state_viol_idx = self.check_state_constraints(state)
            
            # Handle terminal state (no corresponding input)
            is_terminal = (horizon_index == horizon_length - 1)
            if is_terminal:
                input_norm = 0.0
                input_margin = 0.0
                input_viol_idx = -1
                u_left = 0.0
                u_right = 0.0
            else:
                control_input = input_trajectory[horizon_index]
                input_norm = np.linalg.norm(control_input)
                input_margin, input_viol_idx = self.check_input_constraints(control_input)
                u_left = control_input[0]
                u_right = control_input[1]
            
            row = [
                timestep, horizon_index,
                state_norm, input_norm,
                state_margin, state_viol_idx,
                input_margin, input_viol_idx,
                state[0], state[1], state[2], state[3], state[4], state[5],
                u_left, u_right,
                is_terminal
            ]
            self.csv_writer_.writerow(row)
            
        self.csv_file_.flush()
        
    def close_log(self) -> None:
        """Close log file."""
        if self.csv_file_ is not None:
            self.csv_file_.close()
            print(f"Constraint log saved to: {self.log_file_path_}")
            
    def compute_invariant_set(self,
                             state_matrix: np.ndarray,
                             control_matrix: np.ndarray,
                             grid_resolution: int = 40,
                             reduced_dims: Optional[List[int]] = None,
                             max_iterations: int = 100) -> Dict:
        """
        Numerically approximate maximal control-invariant set using fixed-point iteration.

        Computes the set C where: C ⊆ X and ∀x ∈ C, ∃u ∈ U : Ax + Bu ∈ C

        Uses backward reachability algorithm (Saint-Pierre's viability kernel):
        C₀ = X, C_{k+1} = C_k ∩ Pre(C_k) until convergence.

        For balancing robots, typically focus on pitch dynamics [theta, theta_dot]
        since other states (position, yaw) are unconstrained.

        Args:
            state_matrix: Discrete state matrix A_d (6x6)
            control_matrix: Discrete control matrix B_d (6x2)
            grid_resolution: Points per dimension for grid search (higher = more
                accurate but slower). Recommended: 30-50 for 2D problems.
            reduced_dims: State indices to analyze. Default: [PITCH_INDEX,
                PITCH_RATE_INDEX] for pitch dynamics only.
            max_iterations: Maximum fixed-point iterations (default: 100)

        Returns:
            Dictionary containing:
                - 'feasible_states': Array of states in invariant set (N, 6)
                - 'bounds': Tight box bounds [lower, upper] for each dimension (6, 2)
                - 'volume_fraction': Fraction of search space that is invariant
                - 'iterations': Number of iterations to convergence

        Raises:
            RuntimeError: If no invariant states found (constraints too tight)
        """
        if reduced_dims is None:
            # Focus on pitch dynamics AND velocity for balancing robot
            # State vector: [x, theta, psi, x_dot, theta_dot, psi_dot]
            # Including velocity captures pitch-velocity coupling
            reduced_dims = [PITCH_INDEX, PITCH_RATE_INDEX, VELOCITY_INDEX]  # [1, 4, 3] - 3D

        num_dims = len(reduced_dims)
        print(f"\nComputing maximal control-invariant set")
        print(f"  Dimensions: {reduced_dims} (pitch dynamics)")
        print(f"  Grid resolution: {grid_resolution} points per dimension")
        print(f"  Algorithm: Fixed-point iteration (backward reachability)")

        # Create search grid within state bounds
        grid_axes = []
        for dim_index in reduced_dims:
            axis = np.linspace(
                self.state_lower_bounds_[dim_index],
                self.state_upper_bounds_[dim_index],
                grid_resolution
            )
            grid_axes.append(axis)

        # Generate grid points (reduced dimensional)
        grid_meshes = np.meshgrid(*grid_axes, indexing='ij')
        grid_points = np.stack([mesh.flatten() for mesh in grid_meshes], axis=1)
        total_points = grid_points.shape[0]

        # Initialize C₀ = X (all grid points in constraint set)
        C_k_states = []
        for point_index in range(total_points):
            state = np.zeros(6)
            for local_dim_index, global_dim_index in enumerate(reduced_dims):
                state[global_dim_index] = grid_points[point_index, local_dim_index]
            C_k_states.append(state.copy())
        C_k = np.array(C_k_states)

        print(f"\n  Initial set C₀: {C_k.shape[0]} states (100.0% of grid)")

        # Fixed-point iteration: C_{k+1} = C_k ∩ Pre(C_k)
        for iteration in range(max_iterations):
            # Compute Pre(C_k): states that can reach C_k in one step
            C_k_plus_1 = self._compute_predecessor_set(
                C_k,
                grid_points,
                state_matrix,
                control_matrix,
                reduced_dims
            )

            # Check convergence
            if C_k_plus_1.shape[0] == C_k.shape[0]:
                print(f"  Iteration {iteration + 1}: {C_k_plus_1.shape[0]} states - CONVERGED")
                C_k = C_k_plus_1
                break
            elif C_k_plus_1.shape[0] == 0:
                raise RuntimeError(
                    f"No invariant states found after {iteration + 1} iterations. "
                    "System may be unstable or constraints too tight."
                )
            else:
                fraction = C_k_plus_1.shape[0] / total_points * 100
                print(f"  Iteration {iteration + 1}: {C_k_plus_1.shape[0]} states ({fraction:.1f}% of grid)")
                C_k = C_k_plus_1
        else:
            # Max iterations reached without exact convergence
            print(f"  WARNING: Max iterations ({max_iterations}) reached without convergence")

        final_states = C_k

        # Compute tight bounds
        bounds = np.zeros((6, 2))
        for dim_index in range(6):
            if dim_index in reduced_dims:
                bounds[dim_index, 0] = np.min(final_states[:, dim_index])
                bounds[dim_index, 1] = np.max(final_states[:, dim_index])
            else:
                bounds[dim_index, 0] = self.state_lower_bounds_[dim_index]
                bounds[dim_index, 1] = self.state_upper_bounds_[dim_index]

        volume_fraction = final_states.shape[0] / total_points

        print(f"\nFinal maximal control-invariant set:")
        print(f"  States: {final_states.shape[0]}/{total_points} ({volume_fraction*100:.1f}% of grid)")
        print(f"  Tight bounds on analyzed dimensions:")
        for dim_index in reduced_dims:
            print(f"    Dimension {dim_index}: [{bounds[dim_index, 0]:.4f}, {bounds[dim_index, 1]:.4f}]")

        return {
            'feasible_states': final_states,
            'bounds': bounds,
            'volume_fraction': volume_fraction,
            'iterations': iteration + 1
        }
        
    def _is_state_in_set(self,
                        state: np.ndarray,
                        state_set: np.ndarray,
                        reduced_dims: List[int],
                        grid_spacing: Optional[float] = None) -> bool:
        """Check if state is in the given set (accounting for grid discretization).

        Args:
            state: Full state vector (6,)
            state_set: Array of states in the set (N, 6)
            reduced_dims: Dimensions being analyzed
            grid_spacing: Grid spacing for tolerance (if None, uses 10% of smallest spacing)

        Returns:
            True if state projects onto any point in state_set within tolerance
        """
        if state_set.shape[0] == 0:
            return False

        # Project state onto reduced dimensions
        state_reduced = np.array([state[dim] for dim in reduced_dims])

        # Project state_set onto reduced dimensions
        set_reduced = state_set[:, reduced_dims]

        # Auto-compute tolerance based on grid spacing if not provided
        if grid_spacing is None:
            # Estimate grid spacing from smallest difference in set
            if set_reduced.shape[0] > 1:
                # Find minimum non-zero distance between points (approx grid spacing)
                unique_sorted = np.unique(set_reduced[:, 0])
                if len(unique_sorted) > 1:
                    grid_spacing = np.min(np.diff(unique_sorted))
                else:
                    grid_spacing = 0.1  # Fallback
            else:
                grid_spacing = 0.1  # Fallback

        # Use tolerance of half grid spacing (to account for numerical errors)
        tolerance = 0.5 * grid_spacing

        # Check if state is close to any point in the set
        distances = np.linalg.norm(set_reduced - state_reduced, axis=1)
        return np.min(distances) <= tolerance

    def _compute_predecessor_set(self,
                                current_set_states: np.ndarray,
                                grid_points: np.ndarray,
                                state_matrix: np.ndarray,
                                control_matrix: np.ndarray,
                                reduced_dims: List[int]) -> np.ndarray:
        """Compute one-step backward reachable set Pre(current_set).

        Returns states x ∈ X such that ∃u ∈ U : Ax + Bu ∈ current_set.

        Args:
            current_set_states: States in current set C_k (N_k, 6)
            grid_points: Grid of reduced-dimensional points to test (N_grid, len(reduced_dims))
            state_matrix: Discrete state matrix A_d (6x6)
            control_matrix: Discrete control matrix B_d (6x2)
            reduced_dims: Dimensions being analyzed

        Returns:
            Array of states in Pre(current_set) (N_pre, 6)
        """
        # Control input grid resolution
        u_grid_resolution = 10
        u_left_values = np.linspace(self.input_lower_bounds_[0],
                                     self.input_upper_bounds_[0],
                                     u_grid_resolution)
        u_right_values = np.linspace(self.input_lower_bounds_[1],
                                      self.input_upper_bounds_[1],
                                      u_grid_resolution)

        predecessor_states = []

        for point_index in range(grid_points.shape[0]):
            # Construct full state vector (zeros for non-analyzed dims)
            state = np.zeros(6)
            for local_dim_index, global_dim_index in enumerate(reduced_dims):
                state[global_dim_index] = grid_points[point_index, local_dim_index]

            # Check if ∃u ∈ U : Ax + Bu ∈ current_set
            is_predecessor = False
            for u_left in u_left_values:
                if is_predecessor:
                    break
                for u_right in u_right_values:
                    control_input = np.array([u_left, u_right])

                    # Compute next state
                    next_state = state_matrix @ state + control_matrix @ control_input

                    # Check if next state is in current_set
                    if self._is_state_in_set(next_state, current_set_states, reduced_dims):
                        is_predecessor = True
                        break

            if is_predecessor:
                predecessor_states.append(state.copy())

        if len(predecessor_states) == 0:
            return np.array([]).reshape(0, 6)

        return np.array(predecessor_states)

    def _is_one_step_invariant(self,
                               state: np.ndarray,
                               state_matrix: np.ndarray,
                               control_matrix: np.ndarray,
                               reduced_dims: List[int]) -> bool:
        """
        Check if state has feasible one-step successor in analyzed subspace.

        DEPRECATED: This method only checks Pre(X), not the invariant set property.
        Use _compute_predecessor_set() instead for proper fixed-point iteration.

        Args:
            state: Full state vector
            state_matrix: A matrix
            control_matrix: B matrix
            reduced_dims: Dimensions being analyzed

        Returns:
            True if exists u such that next state satisfies constraints
        """
        # Try grid of control inputs (simple but effective for 2D input)
        u_grid_resolution = 10
        u_left_values = np.linspace(self.input_lower_bounds_[0], 
                                     self.input_upper_bounds_[0], 
                                     u_grid_resolution)
        u_right_values = np.linspace(self.input_lower_bounds_[1],
                                      self.input_upper_bounds_[1],
                                      u_grid_resolution)
        
        for u_left in u_left_values:
            for u_right in u_right_values:
                control_input = np.array([u_left, u_right])
                
                # Compute next state
                next_state = state_matrix @ state + control_matrix @ control_input
                
                # Check if next state satisfies constraints in analyzed dimensions
                satisfies_constraints = True
                for dim_index in reduced_dims:
                    if (next_state[dim_index] < self.state_lower_bounds_[dim_index] or
                        next_state[dim_index] > self.state_upper_bounds_[dim_index]):
                        satisfies_constraints = False
                        break
                        
                if satisfies_constraints:
                    return True
                    
        return False
        
    def save_terminal_set_config(self,
                                 bounds: np.ndarray,
                                 grid_resolution: int,
                                 computation_metadata: Dict,
                                 iterations: int,
                                 volume_fraction: float,
                                 output_path: str = "config/simulation/terminal_set.yaml") -> None:
        """
        Save computed terminal set bounds to YAML configuration.

        Saves only pitch dynamics bounds (theta, theta_dot) with metadata
        for traceability and recomputation guidance.

        Args:
            bounds: Computed invariant set bounds (6x2 array)
            grid_resolution: Grid resolution used for computation
            computation_metadata: Dict with robot_params_path, mpc_params_path, date
            iterations: Number of iterations to convergence
            volume_fraction: Fraction of grid points in invariant set
            output_path: Path for output YAML file
        """
        terminal_set_config = {
            'terminal_set': {
                'description': 'Maximal control-invariant terminal set for pitch dynamics',
                'algorithm': 'Fixed-point iteration (backward reachability)',
                'warning': (
                    'These bounds were computed offline. If robot parameters '
                    '(mass, inertia, geometry) change, RECOMPUTE using '
                    'scripts/compute_terminal_set.py'
                ),
                'computation_date': computation_metadata['date'],
                'grid_resolution': grid_resolution,
                'convergence_iterations': iterations,
                'volume_fraction': float(volume_fraction),
                'robot_params_path': computation_metadata['robot_params_path'],
                'mpc_params_path': computation_metadata['mpc_params_path'],

                # Terminal constraint values for pitch dynamics AND velocity
                # Copy these to mpc_params.yaml
                'terminal_pitch_limit_rad': float(bounds[PITCH_INDEX, 1]),  # Upper bound (symmetric)
                'terminal_pitch_rate_limit_radps': float(bounds[PITCH_RATE_INDEX, 1]),
                'terminal_velocity_limit_mps': float(bounds[VELOCITY_INDEX, 1]),  # Velocity coupling
            }
        }

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as file:
            yaml.dump(terminal_set_config, file, default_flow_style=False, sort_keys=False)

        print(f"Terminal set configuration saved to: {output_path}")


def main():
    """Example usage of constraint debugger."""
    # Initialize debugger
    debugger = ConstraintDebugger(
        mpc_params_path="config/simulation/mpc_params.yaml"
    )
    
    # Example: Compute invariant set
    # TODO: Replace with actual linearized A, B from robot_dynamics/linearization.py
    # Example placeholder - replace with your actual matrices:
    # from robot_dynamics.linearization import get_linearized_dynamics
    # equilibrium_state = np.zeros(6)
    # state_matrix, control_matrix = get_linearized_dynamics(equilibrium_state)
    
    state_matrix = np.eye(6)  # REPLACE: Load from linearization.py
    control_matrix = np.zeros((6, 2))  # REPLACE: Load from linearization.py
    
    print("Computing control-invariant set...")
    invariant_set = debugger.compute_invariant_set(
        state_matrix=state_matrix,
        control_matrix=control_matrix,
        grid_resolution=20,
        reduced_dims=[0, 1]  # Analyze theta and theta_dot only
    )
    
    # Save terminal set configuration
    debugger.save_terminal_set_config(
        bounds=invariant_set['bounds'],
        output_path="config/simulation/terminal_set.yaml"
    )
    
    # Example: Log constraints during simulation
    debugger.initialize_log(experiment_name="test_terminal_constraints")
    
    # Simulate one step (REPLACE with actual MPC trajectories)
    horizon_length = 10
    state_traj = np.random.randn(horizon_length + 1, 6) * 0.1
    input_traj = np.random.randn(horizon_length, 2) * 2.0
    
    debugger.log_step(
        timestep=0,
        state_trajectory=state_traj,
        input_trajectory=input_traj
    )
    
    debugger.close_log()
    
    print("\nNext steps:")
    print("1. Replace placeholder A, B matrices with actual linearization")
    print("2. Integrate log_step() into your MPC simulation loop")
    print("3. Load terminal_set.yaml bounds in your MPC formulation")


if __name__ == "__main__":
    main()

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


class ConstraintDebugger:
    """Analyze MPC constraints and compute control-invariant sets."""
    
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
        """Extract constraint bounds from MPC configuration."""
        # State constraints: [theta, theta_dot, x, y, psi, psi_dot]
        self.state_lower_bounds_ = np.array([
            -self.params_['pitch_limit_rad'],
            -self.params_['pitch_rate_limit_radps'],
            -self.params_['position_limit_m'],
            -self.params_['position_limit_m'],
            -self.params_['yaw_limit_rad'],
            -self.params_['yaw_rate_limit_radps']
        ])
        
        self.state_upper_bounds_ = np.array([
            self.params_['pitch_limit_rad'],
            self.params_['pitch_rate_limit_radps'],
            self.params_['position_limit_m'],
            self.params_['position_limit_m'],
            self.params_['yaw_limit_rad'],
            self.params_['yaw_rate_limit_radps']
        ])
        
        # Input constraints: [u_left, u_right] in volts
        voltage_limit = self.params_['voltage_limit_v']
        self.input_lower_bounds_ = np.array([-voltage_limit, -voltage_limit])
        self.input_upper_bounds_ = np.array([voltage_limit, voltage_limit])
        
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
        header = [
            'timestep', 'horizon_index', 
            'state_norm', 'input_norm',
            'min_state_margin', 'violated_state_index',
            'min_input_margin', 'violated_input_index',
            'theta_rad', 'theta_dot_rad_per_s',
            'x_m', 'y_m', 'psi_rad', 'psi_dot_rad_per_s',
            'u_left_v', 'u_right_v',
            'is_terminal'
        ]
        self.csv_writer_.writerow(header)
        
    def check_state_constraints(self, state: np.ndarray) -> Tuple[float, int]:
        """
        Check state constraint margins.
        
        Args:
            state: State vector [theta, theta_dot, x, y, psi, psi_dot]
            
        Returns:
            (min_margin, violated_index): Minimum margin and index of tightest constraint
                Negative margin indicates violation
        """
        if state.shape[0] != 6:
            raise ValueError(f"Expected state dimension 6, got {state.shape[0]}")
            
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
            control_input: Control vector [u_left, u_right] in volts
            
        Returns:
            (min_margin, violated_index): Minimum margin and index of tightest constraint
                Negative margin indicates violation
        """
        if control_input.shape[0] != 2:
            raise ValueError(f"Expected input dimension 2, got {control_input.shape[0]}")
            
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
                             grid_resolution: int = 20,
                             reduced_dims: Optional[List[int]] = None) -> Dict:
        """
        Numerically approximate maximal control-invariant set.
        
        Computes set C where: x in C => exists u such that Ax + Bu in C
        
        Args:
            state_matrix: Linearized A matrix (6x6)
            control_matrix: Linearized B matrix (6x2)
            grid_resolution: Points per dimension for grid search
            reduced_dims: State indices to analyze (default: [0,1] for theta, theta_dot)
            
        Returns:
            Dictionary containing:
                - 'feasible_states': Array of states in invariant set
                - 'bounds': Tight box bounds [lower, upper] for each dimension
                - 'volume_fraction': Fraction of initial search space that is invariant
        """
        if reduced_dims is None:
            # Focus on pitch dynamics for balancing robot
            reduced_dims = [0, 1]  # theta, theta_dot
            
        num_dims = len(reduced_dims)
        print(f"Computing invariant set for dimensions: {reduced_dims}")
        print(f"Grid resolution: {grid_resolution} points per dimension")
        
        # Create search grid within state bounds
        grid_axes = []
        for dim_index in reduced_dims:
            axis = np.linspace(
                self.state_lower_bounds_[dim_index],
                self.state_upper_bounds_[dim_index],
                grid_resolution
            )
            grid_axes.append(axis)
            
        # Generate grid points
        grid_meshes = np.meshgrid(*grid_axes, indexing='ij')
        grid_points = np.stack([mesh.flatten() for mesh in grid_meshes], axis=1)
        total_points = grid_points.shape[0]
        
        print(f"Testing {total_points} grid points...")
        
        feasible_states = []
        
        for point_index in range(total_points):
            # Construct full state vector (zeros for non-analyzed dims)
            state = np.zeros(6)
            for local_dim_index, global_dim_index in enumerate(reduced_dims):
                state[global_dim_index] = grid_points[point_index, local_dim_index]
                
            # Check if state is one-step invariant
            if self._is_one_step_invariant(state, state_matrix, control_matrix, reduced_dims):
                feasible_states.append(state.copy())
                
        feasible_states = np.array(feasible_states)
        
        if feasible_states.shape[0] == 0:
            raise RuntimeError("No invariant states found. Check constraints and dynamics.")
            
        # Compute tight bounds
        bounds = np.zeros((6, 2))
        for dim_index in range(6):
            if dim_index in reduced_dims:
                bounds[dim_index, 0] = np.min(feasible_states[:, dim_index])
                bounds[dim_index, 1] = np.max(feasible_states[:, dim_index])
            else:
                bounds[dim_index, 0] = self.state_lower_bounds_[dim_index]
                bounds[dim_index, 1] = self.state_upper_bounds_[dim_index]
                
        volume_fraction = feasible_states.shape[0] / total_points
        
        print(f"Found {feasible_states.shape[0]} invariant states ({volume_fraction*100:.1f}% of grid)")
        print("\nTight bounds on analyzed dimensions:")
        for dim_index in reduced_dims:
            print(f"  Dimension {dim_index}: [{bounds[dim_index, 0]:.4f}, {bounds[dim_index, 1]:.4f}]")
            
        return {
            'feasible_states': feasible_states,
            'bounds': bounds,
            'volume_fraction': volume_fraction
        }
        
    def _is_one_step_invariant(self,
                               state: np.ndarray,
                               state_matrix: np.ndarray,
                               control_matrix: np.ndarray,
                               reduced_dims: List[int]) -> bool:
        """
        Check if state has feasible one-step successor in analyzed subspace.
        
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
                                 output_path: str = "config/control/terminal_set.yaml") -> None:
        """
        Save computed terminal set bounds to YAML configuration.
        
        Args:
            bounds: Computed invariant set bounds (6x2 array)
            output_path: Path for output YAML file
        """
        terminal_set_config = {
            'terminal_set': {
                'description': 'Control-invariant set bounds for terminal constraints',
                'state_bounds': {
                    'theta_min_rad': float(bounds[0, 0]),
                    'theta_max_rad': float(bounds[0, 1]),
                    'theta_dot_min_rad_per_s': float(bounds[1, 0]),
                    'theta_dot_max_rad_per_s': float(bounds[1, 1]),
                    'x_min_m': float(bounds[2, 0]),
                    'x_max_m': float(bounds[2, 1]),
                    'y_min_m': float(bounds[3, 0]),
                    'y_max_m': float(bounds[3, 1]),
                    'psi_min_rad': float(bounds[4, 0]),
                    'psi_max_rad': float(bounds[4, 1]),
                    'psi_dot_min_rad_per_s': float(bounds[5, 0]),
                    'psi_dot_max_rad_per_s': float(bounds[5, 1])
                }
            }
        }
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as file:
            yaml.dump(terminal_set_config, file, default_flow_style=False)
            
        print(f"Terminal set configuration saved to: {output_path}")


def main():
    """Example usage of constraint debugger."""
    # Initialize debugger
    debugger = ConstraintDebugger(
        mpc_params_path="config/mpc_params.yaml"
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
        output_path="config/control/terminal_set.yaml"
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

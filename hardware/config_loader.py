"""Configuration loader for hardware MPC controller.

This module provides functions to load and configure the MPC controller
specifically for the physical Pololu Balboa robot hardware, using parameters
from config/hardware/ (separate from simulation configs).
"""

from pathlib import Path
from typing import Optional

from robot_dynamics import RobotParameters
from robot_dynamics import (
    linearize_at_equilibrium,
    discretize_linear_dynamics,
    compute_equilibrium_state,
)
from mpc import (
    MPCConfig,
    LinearMPCSolver,
    compute_terminal_cost_dare,
    create_constraints_from_config,
    ReferenceGenerator,
)
from state_estimation import ComplementaryFilter, EstimatorConfig
from control_pipeline import BalanceController


def load_hardware_mpc(
    robot_params_path: str = 'config/hardware/robot_params.yaml',
    mpc_params_path: str = 'config/hardware/mpc_params.yaml',
    estimator_params_path: str = 'config/hardware/estimator_params.yaml'
) -> BalanceController:
    """Load MPC controller with HARDWARE parameters (not simulation).

    This function creates a complete BalanceController instance configured
    for the physical robot, reusing all the simulation code but with
    hardware-specific parameters.

    Args:
        robot_params_path: Path to hardware robot parameters YAML
        mpc_params_path: Path to hardware MPC configuration YAML
        estimator_params_path: Path to hardware estimator configuration YAML

    Returns:
        BalanceController: Fully configured controller ready for hardware deployment

    Example:
        >>> controller = load_hardware_mpc()
        >>> # Use with serial interface to control robot
    """
    # Load hardware parameters (separate from MuJoCo simulation!)
    robot_params = RobotParameters.from_yaml(robot_params_path)
    mpc_config = MPCConfig.from_yaml(mpc_params_path)
    estimator_config = EstimatorConfig.from_yaml(estimator_params_path)

    # Compute equilibrium (upright, stationary) - reuse from simulation!
    eq_state, eq_control = compute_equilibrium_state(robot_params)

    # Linearize around equilibrium point - reuse from simulation!
    linearized = linearize_at_equilibrium(robot_params, eq_state, eq_control)

    # Discretize dynamics using zero-order hold - reuse from simulation!
    discrete = discretize_linear_dynamics(
        linearized.state_matrix,
        linearized.control_matrix,
        mpc_config.sampling_period_s
    )

    # Build cost matrices
    Q = mpc_config.state_cost_matrix
    R = mpc_config.control_cost_matrix
    P = compute_terminal_cost_dare(
        discrete.state_matrix_discrete,
        discrete.control_matrix_discrete,
        Q, R
    )

    # Create state and control constraints - reuse from simulation!
    state_constraints = create_constraints_from_config(
        mpc_config,
        robot_params,
        constraint_type='state'
    )
    control_constraints = create_constraints_from_config(
        mpc_config,
        robot_params,
        constraint_type='control'
    )

    # Create MPC solver - reuse from simulation!
    mpc_solver = LinearMPCSolver(
        A=discrete.state_matrix_discrete,
        B=discrete.control_matrix_discrete,
        Q=Q,
        R=R,
        P=P,
        prediction_horizon=mpc_config.prediction_horizon_steps,
        state_constraints=state_constraints,
        control_constraints=control_constraints,
        sampling_period_s=mpc_config.sampling_period_s,
        solver_name=mpc_config.solver_name,
        warm_start=mpc_config.warm_start_enabled
    )

    # Create state estimator (reuse from simulation!)
    complementary_filter = ComplementaryFilter(
        time_constant_s=estimator_config.time_constant_s,
        sampling_period_s=estimator_config.sampling_period_s
    )

    # Create reference generator (reuse from simulation!)
    reference_generator = ReferenceGenerator(
        prediction_horizon=mpc_config.prediction_horizon_steps,
        sampling_period_s=mpc_config.sampling_period_s
    )

    # Assemble BalanceController (reuse from simulation!)
    return BalanceController(
        mpc_solver=mpc_solver,
        state_estimator=complementary_filter,
        reference_generator=reference_generator,
        sampling_period_s=mpc_config.sampling_period_s,
        wheel_radius_m=robot_params.wheel_radius_m,
        track_width_m=robot_params.track_width_m,
        use_simulation_velocity=False  # Use encoder-based velocity on hardware!
    )


def load_hardware_mpc_with_custom_params(
    robot_params: RobotParameters,
    mpc_config: MPCConfig,
    estimator_config: EstimatorConfig
) -> BalanceController:
    """Load hardware MPC with pre-loaded parameter objects.

    Useful for testing or when parameters are created programmatically.

    Args:
        robot_params: Robot parameters object
        mpc_config: MPC configuration object
        estimator_config: Estimator configuration object

    Returns:
        BalanceController: Configured controller
    """
    # Same logic as load_hardware_mpc but with pre-loaded params
    eq_state, eq_control = compute_equilibrium_state(robot_params)
    linearized = linearize_at_equilibrium(robot_params, eq_state, eq_control)

    discrete = discretize_linear_dynamics(
        linearized.state_matrix,
        linearized.control_matrix,
        mpc_config.sampling_period_s
    )

    Q = mpc_config.state_cost_matrix
    R = mpc_config.control_cost_matrix
    P = compute_terminal_cost_dare(
        discrete.state_matrix_discrete,
        discrete.control_matrix_discrete,
        Q, R
    )

    state_constraints = create_constraints_from_config(
        mpc_config,
        robot_params,
        constraint_type='state'
    )
    control_constraints = create_constraints_from_config(
        mpc_config,
        robot_params,
        constraint_type='control'
    )

    mpc_solver = LinearMPCSolver(
        A=discrete.state_matrix_discrete,
        B=discrete.control_matrix_discrete,
        Q=Q,
        R=R,
        P=P,
        prediction_horizon=mpc_config.prediction_horizon_steps,
        state_constraints=state_constraints,
        control_constraints=control_constraints,
        sampling_period_s=mpc_config.sampling_period_s,
        solver_name=mpc_config.solver_name,
        warm_start=mpc_config.warm_start_enabled
    )

    complementary_filter = ComplementaryFilter(
        time_constant_s=estimator_config.time_constant_s,
        sampling_period_s=estimator_config.sampling_period_s
    )

    reference_generator = ReferenceGenerator(
        prediction_horizon=mpc_config.prediction_horizon_steps,
        sampling_period_s=mpc_config.sampling_period_s
    )

    return BalanceController(
        mpc_solver=mpc_solver,
        state_estimator=complementary_filter,
        reference_generator=reference_generator,
        sampling_period_s=mpc_config.sampling_period_s,
        wheel_radius_m=robot_params.wheel_radius_m,
        track_width_m=robot_params.track_width_m,
        use_simulation_velocity=False
    )

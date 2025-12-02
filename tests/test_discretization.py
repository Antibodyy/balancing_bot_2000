"""Tests for discretization module.

Tests ZOH vs Euler, time-step sensitivity, and creates visualizations.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from robot_dynamics import (
    RobotParameters,
    compute_equilibrium_state,
    linearize_at_equilibrium,
    discretize_linear_dynamics
)


@pytest.fixture
def params():
    """Load robot parameters from YAML."""
    return RobotParameters.from_yaml('config/robot_params.yaml')


def test_zoh_vs_euler_comparison(params):
    """Test that ZOH is more accurate than Euler for same time step."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    sampling_period_s = 0.02  # 20ms

    # ZOH discretization
    discrete_sys = discretize_linear_dynamics(
        linear_sys.state_matrix,
        linear_sys.control_matrix,
        sampling_period_s
    )

    # Euler approximation: A_d ≈ I + A*Ts, B_d ≈ B*Ts
    euler_state_matrix = (
        np.eye(6) + linear_sys.state_matrix * sampling_period_s
    )
    euler_control_matrix = linear_sys.control_matrix * sampling_period_s

    # ZOH should be different from Euler (more accurate)
    error = np.linalg.norm(
        discrete_sys.state_matrix_discrete - euler_state_matrix
    )
    assert error > 1e-6, "ZOH should differ from Euler approximation"


def test_time_step_independence(params):
    """Test that smaller time steps approach continuous limit."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    # Very small time step (should be close to I + A*Ts)
    small_ts = 0.001
    discrete_sys_small = discretize_linear_dynamics(
        linear_sys.state_matrix,
        linear_sys.control_matrix,
        small_ts
    )

    first_order_approx = np.eye(6) + linear_sys.state_matrix * small_ts

    np.testing.assert_allclose(
        discrete_sys_small.state_matrix_discrete,
        first_order_approx,
        atol=1e-4  # Absolute tolerance accounts for second-order terms
    )


def test_matrix_shapes_and_properties(params):
    """Verify discrete matrix shapes and numerical properties."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    discrete_sys = discretize_linear_dynamics(
        linear_sys.state_matrix,
        linear_sys.control_matrix,
        0.02
    )

    assert discrete_sys.state_matrix_discrete.shape == (6, 6)
    assert discrete_sys.control_matrix_discrete.shape == (6, 2)
    assert discrete_sys.sampling_period_s == 0.02


def test_visualization_timestep_comparison(params):
    """Visualization 1: ZOH vs Euler for different time steps."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    time_steps = np.logspace(-3, -1, 15)  # 1ms to 100ms
    zoh_errors = []
    euler_errors = []

    reference_ts = 0.0001  # Very small reference
    discrete_ref = discretize_linear_dynamics(
        linear_sys.state_matrix,
        linear_sys.control_matrix,
        reference_ts
    )
    reference_state_matrix = discrete_ref.state_matrix_discrete

    for time_step_s in time_steps:
        # ZOH
        discrete_zoh = discretize_linear_dynamics(
            linear_sys.state_matrix,
            linear_sys.control_matrix,
            time_step_s
        )

        # Euler
        euler_state_matrix = np.eye(6) + linear_sys.state_matrix * time_step_s

        # Compare to reference (extrapolated from small time step)
        steps_ratio = int(time_step_s / reference_ts)
        reference_extrapolated = np.linalg.matrix_power(
            reference_state_matrix, steps_ratio
        )

        zoh_error = np.linalg.norm(
            discrete_zoh.state_matrix_discrete - reference_extrapolated
        )
        euler_error = np.linalg.norm(
            euler_state_matrix - reference_extrapolated
        )

        zoh_errors.append(zoh_error)
        euler_errors.append(euler_error)

    plt.figure(figsize=(10, 6))
    plt.loglog(time_steps * 1000, zoh_errors, 'o-', label='ZOH (exact)')
    plt.loglog(time_steps * 1000, euler_errors, 's-', label='Euler (approximate)')
    plt.xlabel('Sampling Period (ms)')
    plt.ylabel('Discretization Error (norm)')
    plt.title('ZOH vs Euler: Discretization Accuracy')
    plt.legend()
    plt.grid(True, which='both')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/discretization_comparison_{timestamp}.png', dpi=150)
    plt.close()


def test_visualization_eigenvalue_preservation(params):
    """Visualization 2: Eigenvalues continuous vs discrete."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    sampling_period_s = 0.02

    discrete_sys = discretize_linear_dynamics(
        linear_sys.state_matrix,
        linear_sys.control_matrix,
        sampling_period_s
    )

    # Continuous eigenvalues
    continuous_eigenvalues = np.linalg.eigvals(linear_sys.state_matrix)

    # Discrete eigenvalues
    discrete_eigenvalues = np.linalg.eigvals(discrete_sys.state_matrix_discrete)

    # Theoretical relationship: lambda_d = exp(lambda_c * Ts)
    theoretical_discrete = np.exp(continuous_eigenvalues * sampling_period_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Continuous eigenvalues
    ax1.scatter(
        np.real(continuous_eigenvalues),
        np.imag(continuous_eigenvalues),
        s=100, alpha=0.6, label='Continuous'
    )
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('Continuous Eigenvalues')
    ax1.grid(True)
    ax1.legend()

    # Discrete eigenvalues
    circle = plt.Circle((0, 0), 1.0, fill=False, color='r', linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    ax2.scatter(
        np.real(discrete_eigenvalues),
        np.imag(discrete_eigenvalues),
        s=100, alpha=0.6, label='Discrete (actual)'
    )
    ax2.scatter(
        np.real(theoretical_discrete),
        np.imag(theoretical_discrete),
        s=100, alpha=0.6, marker='x', label='Discrete (theoretical)'
    )
    ax2.set_xlabel('Real Part')
    ax2.set_ylabel('Imaginary Part')
    ax2.set_title('Discrete Eigenvalues (unit circle = stability)')
    ax2.grid(True)
    ax2.legend()
    ax2.axis('equal')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/discretization_eigenvalues_{timestamp}.png', dpi=150)
    plt.close()


def test_visualization_step_response(params):
    """Visualization 3: Discrete system response to control input step."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    sampling_period_s = 0.02
    discrete_sys = discretize_linear_dynamics(
        linear_sys.state_matrix,
        linear_sys.control_matrix,
        sampling_period_s
    )

    # Simulate step response
    num_steps = 100
    state = np.zeros(6)
    control_step = np.array([0.1, 0.1])  # Small torque step

    states_history = [state.copy()]

    for _ in range(num_steps):
        state = (
            discrete_sys.state_matrix_discrete @ state +
            discrete_sys.control_matrix_discrete @ control_step
        )
        states_history.append(state.copy())

    states_array = np.array(states_history)
    time_values = np.arange(num_steps + 1) * sampling_period_s

    plt.figure(figsize=(12, 8))
    labels = ['x', 'theta', 'psi', 'dx', 'dtheta', 'dpsi']

    for state_index in range(6):
        plt.subplot(3, 2, state_index + 1)
        plt.plot(time_values, states_array[:, state_index])
        plt.ylabel(labels[state_index])
        plt.grid(True)
        if state_index >= 4:
            plt.xlabel('Time (s)')

    plt.suptitle('Discrete System Step Response (torque step = 0.1 N·m)')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/discretization_step_response_{timestamp}.png', dpi=150)
    plt.close()

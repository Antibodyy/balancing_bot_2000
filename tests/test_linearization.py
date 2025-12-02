"""Tests for linearization module.

Tests small perturbation approximation, eigenvalues, and creates visualizations.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from robot_dynamics import (
    RobotParameters,
    compute_equilibrium_state,
    compute_state_derivative,
    linearize_at_equilibrium
)


@pytest.fixture
def params():
    """Load robot parameters from YAML."""
    return RobotParameters.from_yaml('config/robot_params.yaml')


def test_small_perturbation_linear_approximation(params):
    """Test that linearization approximates nonlinear dynamics near equilibrium."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    # Small perturbation
    delta_state = np.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0])  # Small pitch perturbation
    delta_control = np.zeros(2)

    perturbed_state = equilibrium_state + delta_state
    perturbed_control = equilibrium_control + delta_control

    # Nonlinear dynamics
    nonlinear_derivative = compute_state_derivative(
        perturbed_state, perturbed_control, params
    )
    equilibrium_derivative = compute_state_derivative(
        equilibrium_state, equilibrium_control, params
    )
    nonlinear_delta = nonlinear_derivative - equilibrium_derivative

    # Linear approximation
    linear_delta = (linear_sys.state_matrix @ delta_state +
                    linear_sys.control_matrix @ delta_control)

    # Should be close for small perturbations
    np.testing.assert_allclose(
        nonlinear_delta, linear_delta, rtol=0.1  # 10% relative tolerance
    )


def test_eigenvalues_unstable_pitch_mode(params):
    """Test that linearized system has unstable pitch mode."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    eigenvalues = np.linalg.eigvals(linear_sys.state_matrix)

    # Should have at least one eigenvalue with positive real part (unstable)
    has_unstable_mode = any(np.real(eigenvalues) > 0)
    assert has_unstable_mode, "Expected unstable pitch mode not found"


def test_matrix_shapes(params):
    """Verify A, B matrix shapes are correct."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    assert linear_sys.state_matrix.shape == (6, 6)
    assert linear_sys.control_matrix.shape == (6, 2)


def test_visualization_eigenvalues(params):
    """Visualization 1: Eigenvalue plot for stability analysis."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    eigenvalues = np.linalg.eigvals(linear_sys.state_matrix)

    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(eigenvalues), np.imag(eigenvalues), s=100, alpha=0.6, c='blue')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Stability boundary')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    plt.title('Eigenvalue Plot: Linearized Dynamics Stability')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/linearization_eigenvalues_{timestamp}.png', dpi=150)
    plt.close()


def test_visualization_linearization_error(params):
    """Visualization 2: Linearization error vs perturbation size."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    perturbation_sizes = np.logspace(-3, -0.5, 20)  # From 0.001 to ~0.3 rad
    errors = []

    for perturbation_size in perturbation_sizes:
        delta_state = np.array([0.0, perturbation_size, 0.0, 0.0, 0.0, 0.0])
        delta_control = np.zeros(2)

        perturbed_state = equilibrium_state + delta_state
        perturbed_control = equilibrium_control + delta_control

        nonlinear_derivative = compute_state_derivative(
            perturbed_state, perturbed_control, params
        )
        equilibrium_derivative = compute_state_derivative(
            equilibrium_state, equilibrium_control, params
        )
        nonlinear_delta = nonlinear_derivative - equilibrium_derivative

        linear_delta = (linear_sys.state_matrix @ delta_state +
                        linear_sys.control_matrix @ delta_control)

        error = np.linalg.norm(nonlinear_delta - linear_delta)
        errors.append(error)

    plt.figure(figsize=(10, 6))
    plt.loglog(perturbation_sizes, errors, 'o-')
    plt.xlabel('Perturbation Size (rad)')
    plt.ylabel('Linearization Error (norm)')
    plt.title('Linearization Error vs Perturbation Size')
    plt.grid(True, which='both')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/linearization_error_{timestamp}.png', dpi=150)
    plt.close()


def test_visualization_jacobian_heatmaps(params):
    """Visualization 3: Heatmap of A and B matrices structure."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    linear_sys = linearize_at_equilibrium(params, equilibrium_state, equilibrium_control)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # A matrix heatmap
    im1 = ax1.imshow(linear_sys.state_matrix, cmap='RdBu', aspect='auto')
    ax1.set_title('State Matrix A (6x6)')
    ax1.set_xlabel('State Index')
    ax1.set_ylabel('State Index')
    plt.colorbar(im1, ax=ax1)

    # B matrix heatmap
    im2 = ax2.imshow(linear_sys.control_matrix, cmap='RdBu', aspect='auto')
    ax2.set_title('Control Matrix B (6x2)')
    ax2.set_xlabel('Control Index')
    ax2.set_ylabel('State Index')
    plt.colorbar(im2, ax=ax2)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/linearization_jacobians_{timestamp}.png', dpi=150)
    plt.close()

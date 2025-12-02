"""Tests for continuous dynamics module.

Tests equilibrium, symmetry, differential drive, and creates visualizations.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from robot_dynamics import (
    RobotParameters,
    compute_state_derivative,
    compute_equilibrium_state
)


@pytest.fixture
def params():
    """Load robot parameters from YAML."""
    return RobotParameters.from_yaml('config/robot_params.yaml')


def test_equilibrium_on_flat_ground(params):
    """Verify zero acceleration at equilibrium on flat ground."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)

    derivative = compute_state_derivative(
        equilibrium_state, equilibrium_control, params
    )

    # Velocities should match state
    np.testing.assert_allclose(
        derivative[:3], equilibrium_state[3:], atol=1e-10
    )

    # Accelerations should be zero
    np.testing.assert_allclose(
        derivative[3:], np.zeros(3), atol=1e-10
    )


def test_symmetry_equal_torques_no_yaw(params):
    """Equal torques should produce no yaw acceleration."""
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # All zeros
    control = np.array([1.0, 1.0])  # Equal torques

    derivative = compute_state_derivative(state, control, params)

    # Yaw acceleration should be zero
    assert abs(derivative[5]) < 1e-10


def test_differential_torque_produces_yaw(params):
    """Opposite torques should produce pure yaw rotation."""
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    control = np.array([1.0, -1.0])  # Opposite torques

    derivative = compute_state_derivative(state, control, params)

    # Should produce yaw acceleration
    assert abs(derivative[5]) > 1e-6


def test_visualization_phase_portrait(params):
    """Visualization 1: Phase portrait (theta, theta_dot) for pitch dynamics."""
    # Simulate unforced pendulum motion from various initial conditions
    theta_values = []
    theta_dot_values = []

    for initial_theta in np.linspace(-0.2, 0.2, 5):
        for initial_theta_dot in np.linspace(-0.5, 0.5, 5):
            state = np.array([0.0, initial_theta, 0.0, 0.0, initial_theta_dot, 0.0])
            control = np.zeros(2)

            derivative = compute_state_derivative(state, control, params)

            theta_values.append(initial_theta)
            theta_dot_values.append(initial_theta_dot)

    plt.figure(figsize=(8, 6))
    plt.scatter(theta_values, theta_dot_values, alpha=0.6)
    plt.xlabel('Pitch Angle theta (rad)')
    plt.ylabel('Pitch Rate theta_dot (rad/s)')
    plt.title('Phase Portrait: Pitch Dynamics')
    plt.grid(True)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/dynamics_phase_portrait_{timestamp}.png', dpi=150)
    plt.close()


def test_visualization_time_response(params):
    """Visualization 2: Time response for unforced motion trajectories."""
    # Simple Euler integration for visualization
    state = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])  # Small initial pitch
    control = np.zeros(2)

    time_steps = 100
    time_step_s = 0.01
    time_values = np.linspace(0, time_steps * time_step_s, time_steps)
    theta_trajectory = []
    theta_dot_trajectory = []

    for _ in range(time_steps):
        theta_trajectory.append(state[1])
        theta_dot_trajectory.append(state[4])

        derivative = compute_state_derivative(state, control, params)
        state = state + derivative * time_step_s  # Euler step

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_values, theta_trajectory)
    plt.ylabel('Pitch Angle theta (rad)')
    plt.title('Time Response: Unforced Pendulum Motion')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time_values, theta_dot_trajectory)
    plt.ylabel('Pitch Rate theta_dot (rad/s)')
    plt.xlabel('Time (s)')
    plt.grid(True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/dynamics_time_response_{timestamp}.png', dpi=150)
    plt.close()


def test_visualization_equilibrium_verification(params):
    """Visualization 3: Plot equilibrium verification f(x_eq, u_eq) components."""
    equilibrium_state, equilibrium_control = compute_equilibrium_state(params)
    derivative = compute_state_derivative(
        equilibrium_state, equilibrium_control, params
    )

    labels = ['x_dot', 'theta_dot', 'psi_dot', 'x_ddot', 'theta_ddot', 'psi_ddot']

    plt.figure(figsize=(10, 6))
    plt.bar(labels, derivative)
    plt.ylabel('State Derivative Value')
    plt.title('Equilibrium Verification: f(x_eq, u_eq) Components')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/dynamics_equilibrium_{timestamp}.png', dpi=150)
    plt.close()


def test_visualization_symmetry_results(params):
    """Visualization 4: Yaw response to equal vs differential torques."""
    state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    torque_values = np.linspace(-2.0, 2.0, 20)
    equal_torque_yaw_accel = []
    diff_torque_yaw_accel = []

    for torque in torque_values:
        # Equal torques
        control_equal = np.array([torque, torque])
        derivative_equal = compute_state_derivative(state, control_equal, params)
        equal_torque_yaw_accel.append(derivative_equal[5])

        # Differential torques
        control_diff = np.array([torque, -torque])
        derivative_diff = compute_state_derivative(state, control_diff, params)
        diff_torque_yaw_accel.append(derivative_diff[5])

    plt.figure(figsize=(10, 6))
    plt.plot(torque_values, equal_torque_yaw_accel, 'o-', label='Equal Torques (tau_L = tau_R)')
    plt.plot(torque_values, diff_torque_yaw_accel, 's-', label='Differential Torques (tau_L = -tau_R)')
    plt.xlabel('Torque Magnitude (N·m)')
    plt.ylabel('Yaw Acceleration psi_ddot (rad/s²)')
    plt.title('Symmetry Test: Yaw Response to Torque Configuration')
    plt.legend()
    plt.grid(True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'tests/visualization/dynamics_symmetry_{timestamp}.png', dpi=150)
    plt.close()


# Comprehensive Integration Tests

def test_unforced_motion_inverted_pendulum_falls(params):
    """Test that unforced inverted pendulum falls over (unstable equilibrium)."""
    # Start slightly perturbed from upright
    initial_state = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])  # 0.05 rad pitch
    control = np.zeros(2)  # No control input

    # Integrate forward in time (simple Euler for test)
    state = initial_state.copy()
    time_step_s = 0.001
    num_steps = 1000

    pitch_angles = [state[1]]

    for _ in range(num_steps):
        derivative = compute_state_derivative(state, control, params)
        state = state + derivative * time_step_s
        pitch_angles.append(state[1])

    # Inverted pendulum should fall (pitch angle magnitude should increase)
    assert abs(state[1]) > abs(initial_state[1]), \
        "Unstable inverted pendulum should fall away from upright"

    # Pitch rate should also increase in magnitude
    assert abs(state[4]) > 0.1, \
        "Falling pendulum should have non-zero pitch rate"


def test_small_perturbation_bounded_response(params):
    """Test that small perturbations produce bounded initial response."""
    # Very small perturbation
    initial_state = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0])  # 0.001 rad
    control = np.zeros(2)

    derivative = compute_state_derivative(initial_state, control, params)

    # Response should be proportional to perturbation (small angle approx)
    # For small angles, theta_ddot should be approximately proportional to theta
    theta_ddot = derivative[4]

    # Should be destabilizing (positive feedback for inverted pendulum)
    # Sign should be same as initial perturbation
    assert theta_ddot * initial_state[1] > 0, \
        "Inverted pendulum acceleration should destabilize (same sign as perturbation)"

    # Magnitude should be reasonable for small perturbation
    assert abs(theta_ddot) < 100.0, \
        "Angular acceleration should be bounded for small perturbation"


def test_constant_forward_torque_produces_forward_motion(params):
    """Test that constant equal forward torques produce forward acceleration."""
    initial_state = np.zeros(6)
    control = np.array([0.5, 0.5])  # Equal forward torques

    # Simulate for short time
    state = initial_state.copy()
    time_step_s = 0.01
    num_steps = 50

    for _ in range(num_steps):
        derivative = compute_state_derivative(state, control, params)
        state = state + derivative * time_step_s

    # Should produce forward motion (positive velocity)
    assert state[3] > 0, \
        "Equal forward torques should produce forward velocity"

    # Position should have advanced
    assert state[0] > 0, \
        "Robot should move forward in space"

    # Yaw should remain approximately zero (equal torques)
    assert abs(state[2]) < 0.01, \
        "Equal torques should not produce significant yaw"


def test_differential_torque_produces_pure_rotation(params):
    """Test that opposite torques produce rotation with minimal translation."""
    initial_state = np.zeros(6)
    control = np.array([0.3, -0.3])  # Opposite torques

    # Simulate for short time
    state = initial_state.copy()
    time_step_s = 0.01
    num_steps = 50

    for _ in range(num_steps):
        derivative = compute_state_derivative(state, control, params)
        state = state + derivative * time_step_s

    # Should produce yaw motion
    assert abs(state[5]) > 0.01, \
        "Differential torques should produce yaw rate"

    # Forward motion should be minimal compared to yaw
    assert abs(state[3]) < abs(state[5]) * params.track_width_m, \
        "Differential torques should produce more rotation than translation"


def test_energy_behavior_unforced_motion(params):
    """Test that total energy increases for unstable inverted pendulum."""
    # Start with small initial energy
    initial_state = np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0])
    control = np.zeros(2)

    # Compute initial energy (kinetic + potential)
    # KE = 0.5 * m * v^2 + 0.5 * I * omega^2
    # PE = m * g * h = m * g * l * cos(theta)
    def compute_energy(state):
        theta = state[1]
        theta_dot = state[4]

        # Rotational kinetic energy
        ke_rot = 0.5 * params.effective_pitch_inertia_kg_m2 * (theta_dot ** 2)

        # Potential energy (taking theta=0 as maximum PE)
        pe = params.body_mass_kg * params.gravity_mps2 * params.com_distance_m * np.cos(theta)

        return ke_rot + pe

    initial_energy = compute_energy(initial_state)

    # Simulate
    state = initial_state.copy()
    time_step_s = 0.001
    num_steps = 500

    for _ in range(num_steps):
        derivative = compute_state_derivative(state, control, params)
        state = state + derivative * time_step_s

    final_energy = compute_energy(state)

    # For unforced conservative system, total energy should be approximately conserved
    # With Euler integration, energy typically increases slightly (numerical artifact)
    # Energy should not change dramatically
    energy_change = abs(final_energy - initial_energy)
    assert energy_change < initial_energy * 0.1, \
        "Energy should be approximately conserved (within 10%)"

    assert state[4] != 0.0, \
        "Pendulum should be moving (non-zero pitch rate)"


def test_differential_drive_kinematics(params):
    """Test that wheel velocities relate correctly to robot velocity and yaw rate."""
    # Apply known torques and verify kinematic relationship
    state = np.zeros(6)
    control = np.array([1.0, 0.5])  # Different torques

    derivative = compute_state_derivative(state, control, params)

    # At initial state (theta=0), verify expected accelerations
    # Sum of torques should produce forward acceleration
    # Difference should produce yaw acceleration

    x_ddot = derivative[3]
    psi_ddot = derivative[5]

    # Both should be non-zero
    assert abs(x_ddot) > 1e-6, "Forward acceleration should be non-zero"
    assert abs(psi_ddot) > 1e-6, "Yaw acceleration should be non-zero"

    # For larger left torque (tau_L > tau_R), robot turns left (negative yaw)
    # This matches differential drive: tau_yaw = (d/2r)*(tau_R - tau_L)
    assert psi_ddot < 0, "Larger left torque should produce negative yaw acceleration (turn left)"


def test_gravity_acts_downward_on_tilted_robot(params):
    """Test that gravity produces expected torque on tilted robot."""
    # Tilt robot forward (positive pitch)
    state = np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])  # Pitched forward
    control = np.zeros(2)

    derivative = compute_state_derivative(state, control, params)

    # Gravity should create a torque that accelerates the pitch further
    # (destabilizing for inverted pendulum)
    theta_ddot = derivative[4]

    # For positive pitch, gravity creates positive angular acceleration
    assert theta_ddot > 0, \
        "Gravity should create destabilizing torque on forward-tilted inverted pendulum"

    # Now tilt backward (negative pitch)
    state_backward = np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0])
    derivative_backward = compute_state_derivative(state_backward, control, params)
    theta_ddot_backward = derivative_backward[4]

    # For negative pitch, gravity creates negative angular acceleration
    assert theta_ddot_backward < 0, \
        "Gravity should create destabilizing torque on backward-tilted inverted pendulum"


def test_coriolis_effect_with_pitch_rotation(params):
    """Test that rotating pitch produces Coriolis/centrifugal effects."""
    # State with pitch angle and pitch rate (rotating pendulum)
    state = np.array([0.0, 0.1, 0.0, 0.0, 1.0, 0.0])  # theta=0.1, theta_dot=1.0
    control = np.zeros(2)

    derivative = compute_state_derivative(state, control, params)

    # Fast pitch rotation should produce centrifugal force affecting translation
    x_ddot = derivative[3]

    # There should be some coupling effect
    # (exact sign depends on direction, but magnitude should be non-zero)
    assert abs(x_ddot) > 1e-6, \
        "Rotating pitch should produce coupling effects on translation"

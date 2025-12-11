"""Continuous-time nonlinear dynamics for self-balancing robot.

Implements the coupled dynamics from dynamics.md using CasADi symbolic framework:
1. Longitudinal force balance (x direction)
2. Pitch moment balance (theta dynamics)
3. Yaw moment balance (psi dynamics)

All functions use CasADi for symbolic computation, outputting NumPy arrays.
"""

import casadi as ca
import numpy as np
from robot_dynamics.parameters import RobotParameters
from robot_dynamics._internal.validation import (
    validate_state_vector,
    validate_control_vector
)


def build_dynamics_model(params: RobotParameters) -> ca.Function:
    """Build symbolic dynamics model using CasADi.

    Creates a CasADi Function representing the nonlinear dynamics:
    x_dot = f(x, u)

    Args:
        params: Robot physical parameters

    Returns:
        CasADi Function mapping (state, control) to state_derivative
    """
    # Create symbolic variables
    state = ca.SX.sym('state', 6)      # [x, theta, psi, dx, dtheta, dpsi]
    control = ca.SX.sym('control', 2)  # [tau_L, tau_R]

    # Extract state components
    position_m = state[0]
    pitch_rad = state[1]
    yaw_rad = state[2]
    velocity_mps = state[3]
    pitch_rate_radps = state[4]
    yaw_rate_radps = state[5]

    # Extract control components
    torque_left_nm = control[0]
    torque_right_nm = control[1]

    # Build mass matrix M(theta)
    mass_matrix = _build_mass_matrix_symbolic(pitch_rad, params)

    # Build Coriolis and centrifugal terms
    coriolis_vector = _build_coriolis_symbolic(
        pitch_rad, pitch_rate_radps, yaw_rate_radps, params
    )

    # Build gravitational terms
    gravity_vector = _build_gravity_symbolic(pitch_rad, params)

    # Build generalized forces from wheel torques
    force_vector = _build_generalized_forces_symbolic(
        torque_left_nm, torque_right_nm, params
    )

    # Solve for accelerations: M·q̈ = F - C - G
    right_hand_side = force_vector - coriolis_vector - gravity_vector
    accelerations = ca.solve(mass_matrix, right_hand_side)

    # Construct state derivative [velocities, accelerations]
    state_derivative = ca.vertcat(
        velocity_mps,
        pitch_rate_radps,
        yaw_rate_radps,
        accelerations[0],
        accelerations[1],
        accelerations[2]
    )

    # Create CasADi Function
    return ca.Function(
        'robot_dynamics',
        [state, control],
        [state_derivative],
        ['state', 'control'],
        ['state_derivative']
    )


def compute_state_derivative(
    state: np.ndarray,
    control: np.ndarray,
    params: RobotParameters
) -> np.ndarray:
    """Evaluate dynamics at numeric state and control values.

    Implements first-order form of coupled dynamics: x_dot = f(x, u)

    Args:
        state: State vector (6,) - [x, theta, psi, dx, dtheta, dpsi]
        control: Control vector (2,) - [tau_L, tau_R] in N·m
        params: Robot physical parameters

    Returns:
        State derivative (6,) - [dx, dtheta, dpsi, ddx, ddtheta, ddpsi]

    Raises:
        ValueError: If inputs have incorrect shape or non-finite values

    References:
        See dynamics.md section 5 for mathematical derivation
    """
    # Validate inputs (fail fast per style_guide.md)
    validate_state_vector(state)
    validate_control_vector(control)

    # Build dynamics function (in practice, cache this)
    dynamics_function = build_dynamics_model(params)

    # Evaluate and convert to NumPy
    result = dynamics_function(state, control)
    return np.array(result).flatten()


def compute_equilibrium_state(
    params: RobotParameters,
    desired_velocity_mps: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute equilibrium state and control for constant velocity.

    For a robot on a slope, the equilibrium pitch angle compensates
    for gravity to maintain constant velocity (possibly zero).

    Args:
        params: Robot parameters
        desired_velocity_mps: Desired steady-state velocity

    Returns:
        Tuple of (equilibrium_state, equilibrium_control)
        - equilibrium_state: (6,) state vector at equilibrium
        - equilibrium_control: (2,) control vector at equilibrium

    Note:
        For zero velocity on flat ground: theta_eq = 0, tau_eq = 0
        For non-zero slope, equilibrium is computed numerically
    """
    # For flat ground or zero velocity, use simple analytical solution
    if abs(params.ground_slope_rad) < 1e-6:
        # Flat ground: equilibrium is upright with zero torque
        equilibrium_state = np.array([
            0.0,  # position (arbitrary reference)
            0.0,  # pitch
            0.0,  # yaw (arbitrary reference)
            desired_velocity_mps,
            0.0,  # pitch rate = 0
            0.0   # yaw rate = 0
        ])
        equilibrium_control = np.zeros(2)
        return equilibrium_state, equilibrium_control

    # For slopes, use numerical solver to find equilibrium
    from scipy.optimize import fsolve

    # Build dynamics function for optimization
    dynamics_function = build_dynamics_model(params)

    def equilibrium_error(variables):
        """Error function: should be zero at equilibrium.

        Variables: [pitch, tau] where tau = tau_left = tau_right
        """
        pitch = variables[0]
        tau = variables[1]

        # Build state and control
        state = np.array([0.0, pitch, 0.0, desired_velocity_mps, 0.0, 0.0])
        control = np.array([tau, tau])

        # Compute accelerations
        state_derivative = compute_state_derivative(state, control, params)

        # At equilibrium, accelerations (last 3 components) should be zero
        accelerations = state_derivative[3:]

        # Return pitch and longitudinal accelerations (yaw is always 0 for straight motion)
        return [accelerations[0], accelerations[1]]  # [d²x/dt², d²theta/dt²]

    # Initial guess: pitch ≈ -slope, tau from gravity approximation
    pitch_guess = -params.ground_slope_rad
    total_mass = params.body_mass_kg + 2 * params.wheel_mass_kg
    tau_guess = 0.5 * total_mass * params.gravity_mps2 * np.sin(params.ground_slope_rad) * params.wheel_radius_m

    initial_guess = [pitch_guess, tau_guess]

    # Solve for equilibrium
    try:
        solution = fsolve(equilibrium_error, initial_guess, full_output=True)
        variables_opt = solution[0]
        info = solution[1]

        # Check if solution converged
        residual = np.max(np.abs(info['fvec']))
        if residual > 1e-3:
            # If numerical solution didn't converge well, fall back to analytical approximation
            print(f"Warning: Equilibrium solver residual {residual:.6f} > 1e-3, using approximation")
            pitch_eq = pitch_guess
            tau_eq = tau_guess
        else:
            pitch_eq = variables_opt[0]
            tau_eq = variables_opt[1]
    except Exception as e:
        # Fall back to analytical approximation if solver fails
        print(f"Warning: Equilibrium solver failed ({e}), using approximation")
        pitch_eq = pitch_guess
        tau_eq = tau_guess

    equilibrium_state = np.array([
        0.0,  # position (arbitrary reference)
        pitch_eq,
        0.0,  # yaw (arbitrary reference)
        desired_velocity_mps,
        0.0,  # pitch rate = 0
        0.0   # yaw rate = 0
    ])

    equilibrium_control = np.array([tau_eq, tau_eq])

    return equilibrium_state, equilibrium_control


def _build_mass_matrix_symbolic(
    theta_sym: ca.SX,
    params: RobotParameters
) -> ca.SX:
    """Build configuration-dependent mass matrix M(theta).

    From dynamics.md equations 115-117:
    M = [[M_11,  M_12,    0  ],
         [M_12,  M_22,    0  ],
         [  0,     0,   M_33 ]]

    Args:
        theta_sym: Symbolic pitch angle
        params: Robot parameters

    Returns:
        Symbolic mass matrix (3, 3)
    """
    cos_pitch = ca.cos(theta_sym)

    # Effective mass (see parameters.py)
    mass_effective = params.effective_mass_kg

    # Coupling term between translation and pitch
    mass_coupling = (params.body_mass_kg +params.wheel_mass_kg)* params.com_distance_m * cos_pitch

    # Pitch inertia
    pitch_inertia = params.effective_pitch_inertia_kg_m2

    # Yaw inertia (constant for simplified model)
    yaw_inertia = params.body_yaw_inertia_kg_m2

    mass_matrix = ca.vertcat(
        ca.horzcat(mass_effective, mass_coupling, 0.0),
        ca.horzcat(mass_coupling, pitch_inertia, 0.0),
        ca.horzcat(0.0, 0.0, yaw_inertia)
    )

    return mass_matrix


def _build_coriolis_symbolic(
    theta_sym: ca.SX,
    theta_dot_sym: ca.SX,
    psi_dot_sym: ca.SX,
    params: RobotParameters
) -> ca.SX:
    """Build Coriolis and centrifugal force terms.

    From dynamics.md equation 90: C = [C_1, C_2, C_3]^T

    Args:
        theta_sym: Symbolic pitch angle
        theta_dot_sym: Symbolic pitch angular velocity
        psi_dot_sym: Symbolic yaw angular velocity
        params: Robot parameters

    Returns:
        Symbolic Coriolis vector (3,)
    """
    sin_pitch = ca.sin(theta_sym)

    # Centrifugal term from pitch rotation
    coriolis_longitudinal = (
        -params.body_mass_kg *
        params.com_distance_m *
        (theta_dot_sym ** 2) *
        sin_pitch
    )

    # Simplified model: C_2 and C_3 are zero for planar motion
    coriolis_pitch = 0.0
    coriolis_yaw = 0.0

    return ca.vertcat(
        coriolis_longitudinal,
        coriolis_pitch,
        coriolis_yaw
    )


def _build_gravity_symbolic(
    theta_sym: ca.SX,
    params: RobotParameters
) -> ca.SX:
    """Build gravitational force terms.

    From dynamics.md equations 115-117:
    - Longitudinal: gravity component on slope
    - Pitch: destabilizing gravitational torque
    - Yaw: no gravitational coupling

    Args:
        theta_sym: Symbolic pitch angle
        params: Robot parameters

    Returns:
        Symbolic gravity vector (3,)
    """
    sin_slope = ca.sin(params.ground_slope_rad)
    sin_pitch_slope = ca.sin(theta_sym + params.ground_slope_rad)

    # Longitudinal gravity on slope
    gravity_longitudinal = (
        (params.body_mass_kg + 2 * params.wheel_mass_kg) *
        params.gravity_mps2 *
        sin_slope
    )

    # Destabilizing torque from body weight
    gravity_pitch = (
        -params.body_mass_kg *
        params.gravity_mps2 *
        params.com_distance_m *
        sin_pitch_slope
    )

    gravity_yaw = 0.0

    return ca.vertcat(
        gravity_longitudinal,
        gravity_pitch,
        gravity_yaw
    )


def _build_generalized_forces_symbolic(
    tau_left_sym: ca.SX,
    tau_right_sym: ca.SX,
    params: RobotParameters
) -> ca.SX:
    """Build generalized forces from wheel torques.

    From dynamics.md equations 39-44:
    F_tr = (tau_L + tau_R) / r
    tau_psi = (d / 2r) * (tau_R - tau_L)

    Args:
        tau_left_sym: Symbolic left wheel torque
        tau_right_sym: Symbolic right wheel torque
        params: Robot parameters

    Returns:
        Symbolic generalized force vector (3,)
    """
    torque_sum = tau_left_sym + tau_right_sym
    torque_diff = tau_right_sym - tau_left_sym

    # Longitudinal force from both wheels
    force_longitudinal = torque_sum / params.wheel_radius_m

    # Reaction torque on body (opposite to wheel acceleration)
    force_pitch = -torque_sum

    # Yaw torque from differential drive
    force_yaw = (
        params.track_width_m / (2 * params.wheel_radius_m) * torque_diff
    )

    return ca.vertcat(
        force_longitudinal,
        force_pitch,
        force_yaw
    )

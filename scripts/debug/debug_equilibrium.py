"""Debug equilibrium computation for slopes."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from robot_dynamics.parameters import RobotParameters
from robot_dynamics.continuous_dynamics import (
    compute_equilibrium_state,
    compute_state_derivative
)

# Load robot params with 5 degree slope
params_dict = yaml.safe_load(open('config/simulation/robot_params.yaml'))
params_dict['ground_slope_rad'] = np.deg2rad(5.0)
params = RobotParameters(**params_dict)

print(f"Ground slope: {np.rad2deg(params.ground_slope_rad):.2f} deg")
print(f"Robot params:")
print(f"  body_mass: {params.body_mass_kg} kg")
print(f"  wheel_mass: {params.wheel_mass_kg} kg")
print(f"  wheel_radius: {params.wheel_radius_m} m")
print(f"  gravity: {params.gravity_mps2} m/s²")

# Compute equilibrium
eq_state, eq_control = compute_equilibrium_state(params, desired_velocity_mps=0.0)

print(f"\nEquilibrium state:")
print(f"  position: {eq_state[0]:.6f}")
print(f"  pitch: {np.rad2deg(eq_state[1]):.6f} deg")
print(f"  yaw: {np.rad2deg(eq_state[2]):.6f} deg")
print(f"  velocity: {eq_state[3]:.6f} m/s")
print(f"  pitch_rate: {np.rad2deg(eq_state[4]):.6f} deg/s")
print(f"  yaw_rate: {np.rad2deg(eq_state[5]):.6f} deg/s")

print(f"\nEquilibrium control:")
print(f"  tau_left: {eq_control[0]:.6f} Nm")
print(f"  tau_right: {eq_control[1]:.6f} Nm")
print(f"  tau_sum: {eq_control[0] + eq_control[1]:.6f} Nm")

# Compute derivative at equilibrium
derivative = compute_state_derivative(eq_state, eq_control, params)

print(f"\nState derivative at equilibrium:")
print(f"  dx/dt: {derivative[0]:.6f} (should be velocity)")
print(f"  dtheta/dt: {derivative[1]:.6f} (should be pitch_rate)")
print(f"  dpsi/dt: {derivative[2]:.6f} (should be yaw_rate)")
print(f"  d²x/dt²: {derivative[3]:.6f} m/s² (should be ~0)")
print(f"  d²theta/dt²: {derivative[4]:.6f} rad/s² (should be ~0)")
print(f"  d²psi/dt²: {derivative[5]:.6f} rad/s² (should be ~0)")

print(f"\nMax acceleration: {np.max(np.abs(derivative[3:])):.6f}")

# Compute expected gravity force
total_mass = params.body_mass_kg + 2 * params.wheel_mass_kg
gravity_force = total_mass * params.gravity_mps2 * np.sin(params.ground_slope_rad)
print(f"\nGravity force down slope: {gravity_force:.6f} N")
print(f"Expected wheel force: {gravity_force:.6f} N")
print(f"Expected total torque: {gravity_force * params.wheel_radius_m:.6f} Nm")
print(f"Expected torque per wheel: {0.5 * gravity_force * params.wheel_radius_m:.6f} Nm")

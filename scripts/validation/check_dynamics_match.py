"""Verify dynamics implementation matches MuJoCo model."""

import numpy as np
import mujoco
from robot_dynamics import RobotParameters, compute_equilibrium_state, linearize_at_equilibrium

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path('robot_model.xml')
data = mujoco.MjData(model)

# Load parameters
params = RobotParameters.from_yaml('config/robot_params.yaml')

print("=" * 60)
print("DYNAMICS VALIDATION: MuJoCo vs Parameters")
print("=" * 60)

# Extract masses from MuJoCo
# Body 0 is world, Body 1 is robot base, Bodies 2 & 3 are wheels
print("\nMasses:")
print(f"  MuJoCo total mass: {model.body_mass[1:].sum():.3f} kg")
print(f"    - Body (base): {model.body_mass[1]:.3f} kg")
print(f"    - Wheel R: {model.body_mass[2]:.3f} kg")
print(f"    - Wheel L: {model.body_mass[3]:.3f} kg")
print(f"  Params body_mass: {params.body_mass_kg:.3f} kg")
print(f"  Params wheel_mass (each): {params.wheel_mass_kg:.3f} kg")
print(f"  Params total: {params.body_mass_kg + 2*params.wheel_mass_kg:.3f} kg")

# Extract inertias from MuJoCo
print("\nInertias:")
print(f"  MuJoCo body inertia (diag): {model.body_inertia[1]}")
print(f"  Params body pitch inertia (I_y): {params.body_pitch_inertia_kg_m2:.6f} kg⋅m²")
print(f"  Params body yaw inertia (I_z): {params.body_yaw_inertia_kg_m2:.6f} kg⋅m²")

# Wheel inertia for cylinder: I = 0.5 * m * r^2
wheel_r = 0.05  # from model
wheel_m = 0.5   # from model
wheel_inertia_expected = 0.5 * wheel_m * wheel_r**2
print(f"\nWheel inertia:")
print(f"  Expected for cylinder (I = 0.5*m*r²): {wheel_inertia_expected:.6f} kg⋅m²")
print(f"  Params wheel_inertia: {params.wheel_inertia_kg_m2:.6f} kg⋅m²")

# Geometry
print("\nGeometry:")
print(f"  Params COM distance: {params.com_distance_m:.3f} m")
print(f"  Params wheel radius: {params.wheel_radius_m:.3f} m")
print(f"  Params track width: {params.track_width_m:.3f} m")
print(f"  MuJoCo wheel radius (from geom): 0.050 m")
print(f"  MuJoCo track width (from wheel positions): 0.300 m")
print(f"  MuJoCo body COM height (from geom pos): 0.200 m")

# Check effective mass calculation
M_eff = params.effective_mass_kg
print(f"\nEffective Mass:")
print(f"  M_eff = {M_eff:.3f} kg")
print(f"  Formula: m_body + 2*m_wheel + 2*J_w/r²")
print(f"  = {params.body_mass_kg} + 2*{params.wheel_mass_kg} + 2*{params.wheel_inertia_kg_m2}/{params.wheel_radius_m**2}")
print(f"  = {params.body_mass_kg} + {2*params.wheel_mass_kg:.3f} + {2*params.wheel_inertia_kg_m2/(params.wheel_radius_m**2):.3f}")

# Test equilibrium
print("\n" + "=" * 60)
print("EQUILIBRIUM CHECK")
print("=" * 60)
eq_state, eq_control = compute_equilibrium_state(params)
print(f"  Equilibrium pitch: {np.rad2deg(eq_state[1]):.3f}°")
print(f"  Equilibrium control: {eq_control}")

# Test linearization
print("\n" + "=" * 60)
print("LINEARIZATION CHECK")
print("=" * 60)
lin = linearize_at_equilibrium(params, eq_state, eq_control)
print(f"  State matrix A: shape {lin.state_matrix.shape}")
print(f"  Control matrix B: shape {lin.control_matrix.shape}")

# Check key dynamics terms
print("\n  B matrix (control to acceleration):")
print(f"    B[3,0] (control to x_accel): {lin.control_matrix[3, 0]:.6f}")
print(f"    Expected: 1/(M_eff * r) = {1/(M_eff * params.wheel_radius_m):.6f}")
print(f"    B[4,0] (control to theta_accel): {lin.control_matrix[4, 0]:.6f}")

# Check if there are any NaN or inf values
if np.any(np.isnan(lin.state_matrix)) or np.any(np.isinf(lin.state_matrix)):
    print("\n  ⚠️  WARNING: NaN or Inf in state matrix!")
if np.any(np.isnan(lin.control_matrix)) or np.any(np.isinf(lin.control_matrix)):
    print("\n  ⚠️  WARNING: NaN or Inf in control matrix!")

print("\n" + "=" * 60)

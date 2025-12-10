"""Extract actual inertia from MuJoCo model."""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('robot_model.xml')

# Get body index
robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot')

print("MuJoCo Model Parameters:")
print(f"  Robot body ID: {robot_body_id}")

# Get inertia from body
# MuJoCo stores inertia in model.body_inertia[body_id] as a 3-vector (Ixx, Iyy, Izz)
if robot_body_id >= 0 and robot_body_id < len(model.body_inertia):
    inertia = model.body_inertia[robot_body_id]
    print(f"  Body inertia (Ixx, Iyy, Izz): {inertia}")
    print(f"  Yaw inertia (Izz): {inertia[2]:.6f} kg·m²")
else:
    print(f"  WARNING: Could not get inertia for body {robot_body_id}")

# Get mass
body_mass = model.body_mass[robot_body_id]
print(f"  Body mass: {body_mass:.3f} kg")

# Get wheel info
wheel_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wheel_r')
wheel_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wheel_l')

if wheel_r_id >= 0:
    wheel_mass = model.body_mass[wheel_r_id]
    print(f"  Wheel mass (each): {wheel_mass:.3f} kg")

# Calculate track width from wheel positions
# Wheels are at y = -0.15 and y = +0.15
track_width = 0.30  # From model XML
wheel_radius = 0.05  # From model XML

print(f"  Track width: {track_width} m")
print(f"  Wheel radius: {wheel_radius} m")

# Calculate expected yaw control coefficient
if robot_body_id >= 0 and robot_body_id < len(model.body_inertia):
    I_z = inertia[2]
    yaw_coeff = track_width / (2 * wheel_radius * I_z)
    print(f"\nCalculated yaw control coefficient:")
    print(f"  d/(2*r*I_z) = {yaw_coeff:.1f} rad/(s²·Nm)")
else:
    print("\nCould not calculate yaw coefficient - missing inertia data")

# Note about composite inertia
print("\n⚠️  NOTE: MuJoCo uses composite body inertias that include child bodies.")
print("    The effective yaw inertia includes the robot body + wheels.")
print("    This might differ from the simplified model used for linearization.")

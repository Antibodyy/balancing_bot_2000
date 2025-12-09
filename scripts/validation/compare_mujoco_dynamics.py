"""Compare MuJoCo physics vs our dynamics model."""

import numpy as np
import mujoco
from robot_dynamics import RobotParameters, compute_state_derivative

# Load
model = mujoco.MjModel.from_xml_path('robot_model.xml')
data = mujoco.MjData(model)
params = RobotParameters.from_yaml('config/robot_params.yaml')

print("=" * 70)
print("MUJOCO vs DYNAMICS MODEL COMPARISON")
print("=" * 70)

# Test scenario: pitch = 5°, zero velocity, apply 0.1 Nm torque
test_pitch = np.deg2rad(5.0)
test_torque = 0.1  # Nm

# Set MuJoCo state
data.qpos[0] = 0.0  # x position
data.qpos[1] = test_pitch  # pitch (in MuJoCo coordinates)
data.qpos[2] = 0.0  # wheel_r
data.qpos[3] = 0.0  # wheel_l
data.qvel[:] = 0.0  # all velocities zero

# Apply control in MuJoCo
gear = 0.25
data.ctrl[0] = test_torque / gear  # left
data.ctrl[1] = test_torque / gear  # right

# Step MuJoCo forward to compute accelerations
mujoco.mj_forward(model, data)

# Extract MuJoCo accelerations
mujoco_x_accel = data.qacc[0]
mujoco_theta_accel = data.qacc[1]

print(f"\nTest Case:")
print(f"  Pitch: {np.rad2deg(test_pitch):.2f}°")
print(f"  Torque (each wheel): {test_torque:.3f} N⋅m")
print(f"  All velocities: 0")

print(f"\nMuJoCo Accelerations:")
print(f"  x_accel:     {mujoco_x_accel:+.6f} m/s²")
print(f"  theta_accel: {mujoco_theta_accel:+.6f} rad/s²")

# Our model
state = np.array([0, test_pitch, 0, 0, 0, 0])
control = np.array([test_torque, test_torque])
derivative = compute_state_derivative(state, control, params)

print(f"\nOur Model Accelerations:")
print(f"  x_accel:     {derivative[3]:+.6f} m/s²")
print(f"  theta_accel: {derivative[4]:+.6f} rad/s²")

print(f"\nDifference (MuJoCo - Ours):")
x_diff = mujoco_x_accel - derivative[3]
theta_diff = mujoco_theta_accel - derivative[4]
print(f"  Δx_accel:     {x_diff:+.6f} m/s² ({abs(x_diff/mujoco_x_accel*100):.1f}% error)")
print(f"  Δtheta_accel: {theta_diff:+.6f} rad/s² ({abs(theta_diff/mujoco_theta_accel*100):.1f}% error)")

if abs(x_diff) > 0.01 or abs(theta_diff) > 0.01:
    print(f"\n⚠️  SIGNIFICANT MISMATCH DETECTED!")
    print(f"The dynamics model does not match MuJoCo physics.")
else:
    print(f"\n✓  Models match within tolerance")

# Test with zero control (pure gravity)
print("\n" + "=" * 70)
print("ZERO CONTROL TEST (Pure Gravity)")
print("=" * 70)

data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

mujoco_x_accel_g = data.qacc[0]
mujoco_theta_accel_g = data.qacc[1]

control_zero = np.array([0.0, 0.0])
derivative_g = compute_state_derivative(state, control_zero, params)

print(f"\nMuJoCo (gravity only):")
print(f"  x_accel:     {mujoco_x_accel_g:+.6f} m/s²")
print(f"  theta_accel: {mujoco_theta_accel_g:+.6f} rad/s²")

print(f"\nOur Model (gravity only):")
print(f"  x_accel:     {derivative_g[3]:+.6f} m/s²")
print(f"  theta_accel: {derivative_g[4]:+.6f} rad/s²")

print(f"\nDifference:")
print(f"  Δx_accel:     {(mujoco_x_accel_g - derivative_g[3]):+.6f} m/s²")
print(f"  Δtheta_accel: {(mujoco_theta_accel_g - derivative_g[4]):+.6f} rad/s²")

print("\n" + "=" * 70)

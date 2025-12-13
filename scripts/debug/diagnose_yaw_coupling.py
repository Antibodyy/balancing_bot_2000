"""Diagnose why MuJoCo yaw response doesn't match linearized model."""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('mujoco_sim/robot_model.xml')
data = mujoco.MjData(model)

print("="*70)
print("YAW DYNAMICS DIAGNOSIS")
print("="*70)

# Get IDs
left_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left')
right_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right')
robot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot')

# Test: Apply differential torque and track yaw acceleration
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.25  # Set body height

# Apply constant differential torque
tau_diff = 0.05  # Differential torque
data.ctrl[left_act] = -tau_diff/2
data.ctrl[right_act] = tau_diff/2

# Step forward and measure yaw acceleration
yaw_data = []
time_data = []

for i in range(200):  # 0.2 seconds
    mujoco.mj_step(model, data)

    # Extract yaw from quaternion
    quat = data.qpos[3:7]
    yaw = 2 * np.arctan2(quat[3], quat[0])

    # Get yaw rate from freejoint velocity
    # Freejoint velocity: [vx, vy, vz, wx, wy, wz]
    yaw_rate = data.qvel[5]  # wz = yaw rate

    if i % 10 == 0:
        time_data.append(data.time)
        yaw_data.append((yaw, yaw_rate))

# Calculate yaw acceleration from finite differences
yaw_rates = [d[1] for d in yaw_data]
dt = time_data[1] - time_data[0] if len(time_data) > 1 else 0.001

if len(yaw_rates) > 1:
    yaw_accel_measured = (yaw_rates[-1] - yaw_rates[0]) / (time_data[-1] - time_data[0])
else:
    yaw_accel_measured = 0

# Expected from linearized model
yaw_accel_expected = 1200 * tau_diff

print(f"\nDifferential Torque Test:")
print(f"  Applied differential torque: {tau_diff} Nm")
print(f"  (τ_L={-tau_diff/2:.3f}, τ_R={tau_diff/2:.3f})")
print(f"\n  Expected yaw acceleration: {yaw_accel_expected:.1f} rad/s²")
print(f"  Measured yaw acceleration: {yaw_accel_measured:.1f} rad/s²")
print(f"  Ratio (measured/expected): {yaw_accel_measured/yaw_accel_expected:.4f}")
print(f"  Discrepancy factor: {yaw_accel_expected/yaw_accel_measured:.1f}x")

# Check MuJoCo model structure
print(f"\nMuJoCo Model Structure:")
print(f"  Total DOFs (nv): {model.nv}")
print(f"  Freejoint DOFs: 6 (3 trans + 3 rot)")
print(f"  Wheel joint DOFs: 2 (1 per wheel)")
print(f"  qvel indices: [0-5]=freejoint, [6]=wheel_r, [7]=wheel_l")

# Check what's happening with wheel velocities
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.25

data.ctrl[left_act] = -0.05
data.ctrl[right_act] = 0.05

for i in range(500):
    mujoco.mj_step(model, data)

# qvel for this model: [vx, vy, vz, wx, wy, wz, wheel_r_vel, wheel_l_vel]
wheel_r_vel = data.qvel[6]  # Right wheel joint velocity
wheel_l_vel = data.qvel[7]  # Left wheel joint velocity

print(f"\nWheel Response:")
print(f"  Left wheel angular velocity: {wheel_l_vel:.3f} rad/s")
print(f"  Right wheel angular velocity: {wheel_r_vel:.3f} rad/s")
print(f"  Wheel velocity difference: {wheel_r_vel - wheel_l_vel:.3f} rad/s")

# Calculate expected body yaw rate from wheel velocities
# For differential drive: yaw_rate = (v_R - v_L) / track_width
# where v_R = wheel_r_vel * wheel_radius
track_width = 0.3
wheel_radius = 0.05
expected_yaw_from_wheels = (wheel_r_vel - wheel_l_vel) * wheel_radius / track_width
actual_yaw_rate = data.qvel[5]

print(f"\nYaw Rate from Kinematics:")
print(f"  Expected from wheel velocities: {expected_yaw_from_wheels:.3f} rad/s")
print(f"  Actual body yaw rate: {actual_yaw_rate:.3f} rad/s")
if abs(expected_yaw_from_wheels) > 1e-6:
    print(f"  Ratio: {actual_yaw_rate/expected_yaw_from_wheels:.3f}")
else:
    print(f"  Ratio: N/A (wheels not spinning)")

# Check if robot is sliding vs rolling
body_vel_x = data.qvel[0]
expected_vel_from_wheels = (wheel_l_vel + wheel_r_vel) / 2 * wheel_radius

print(f"\nForward Motion Check:")
print(f"  Body forward velocity: {body_vel_x:.4f} m/s")
print(f"  Expected from wheel rolling: {expected_vel_from_wheels:.4f} m/s")
if abs(expected_vel_from_wheels) > 1e-6:
    print(f"  Ratio (slip indicator): {body_vel_x/expected_vel_from_wheels:.3f}")
else:
    print(f"  Ratio (slip indicator): N/A")

print("\n" + "="*70)
print("HYPOTHESIS:")
print("="*70)

ratio = yaw_accel_measured/yaw_accel_expected
if ratio < 0.01:
    print("⚠️  Yaw acceleration is <1% of expected!")
    print("\nPossible causes:")
    print("  1. Linearized model uses wrong dynamics formulation")
    print("  2. Wheel-ground contact prevents yaw motion (slip/friction)")
    print("  3. MuJoCo integrator/solver settings dampen yaw motion")
    print("  4. The freejoint + wheel joints create unexpected coupling")
    print("\nRecommendation: The linearized model CANNOT be used for yaw control")
    print("with this MuJoCo robot. Need to either:")
    print("  a) Fix the MuJoCo model physics")
    print("  b) Re-derive linearized dynamics from MuJoCo simulation data")
    print("  c) Use a different robot model that matches the assumptions")
elif ratio < 0.5:
    print(f"Yaw response is {ratio*100:.1f}% of expected - significant mismatch")
else:
    print(f"Yaw response is {ratio*100:.1f}% of expected - reasonable match")

print("="*70)

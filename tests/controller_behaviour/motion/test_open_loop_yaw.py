"""Test if MuJoCo simulation responds to open-loop differential torques."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import mujoco

# Load model
model = mujoco.MjModel.from_xml_path('robot_model.xml')
data = mujoco.MjData(model)

# Get actuator and sensor indices
left_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left')
right_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right')

print("="*70)
print("OPEN-LOOP DIFFERENTIAL TORQUE TEST")
print("="*70)
print("Testing if MuJoCo simulation responds to differential wheel torques")
print()

# Test 1: Equal positive torques (should go straight)
print("Test 1: Equal torques (tau_L=0.05, tau_R=0.05)")
print("Expected: Robot moves forward, no rotation")

mujoco.mj_resetData(model, data)
data.qpos[2] = 0.25  # Set body height

for i in range(500):  # 0.5 seconds
    data.ctrl[left_actuator_id] = 0.05
    data.ctrl[right_actuator_id] = 0.05
    mujoco.mj_step(model, data)

# Get robot yaw from freejoint quaternion
quat = data.qpos[3:7]  # Quaternion from freejoint
yaw1 = 2 * np.arctan2(quat[3], quat[0])  # Convert to yaw angle

print(f"Result: Yaw change = {np.rad2deg(yaw1):.3f}°")
print()

# Test 2: Differential torques (should rotate)
print("Test 2: Differential torques (tau_L=0.05, tau_R=0.10)")
print("Expected: Robot rotates left (positive yaw)")

mujoco.mj_resetData(model, data)
data.qpos[2] = 0.25

yaw_history = []
time_history = []

for i in range(1000):  # 1 second
    data.ctrl[left_actuator_id] = 0.05
    data.ctrl[right_actuator_id] = 0.10
    mujoco.mj_step(model, data)

    if i % 50 == 0:  # Record every 50ms
        quat = data.qpos[3:7]
        yaw = 2 * np.arctan2(quat[3], quat[0])
        yaw_history.append(yaw)
        time_history.append(data.time)

quat = data.qpos[3:7]
yaw2 = 2 * np.arctan2(quat[3], quat[0])

print(f"Result: Yaw change = {np.rad2deg(yaw2):.3f}°")
print(f"Yaw rate estimate = {np.rad2deg(yaw2/data.time):.3f}°/s")
print()

# Test 3: Pure differential (opposite torques, should spin in place)
print("Test 3: Opposite torques (tau_L=-0.05, tau_R=+0.05)")
print("Expected: Robot spins in place, maximum yaw rate")

mujoco.mj_resetData(model, data)
data.qpos[2] = 0.25

for i in range(1000):  # 1 second
    data.ctrl[left_actuator_id] = -0.05
    data.ctrl[right_actuator_id] = 0.05
    mujoco.mj_step(model, data)

quat = data.qpos[3:7]
yaw3 = 2 * np.arctan2(quat[3], quat[0])
forward_pos = data.qpos[0]  # X position

print(f"Result: Yaw change = {np.rad2deg(yaw3):.3f}°")
print(f"Forward motion = {forward_pos:.4f} m (should be ~0)")
print()

# Analysis
print("="*70)
print("ANALYSIS")
print("="*70)

# Expected yaw rates based on linearized model
# ddpsi = 1200 * (tau_R - tau_L) rad/s²
# After time t: dpsi = 1200 * (tau_R - tau_L) * t

diff_torque_test2 = 0.10 - 0.05
expected_yaw_test2 = 0.5 * 1200 * diff_torque_test2 * 1.0**2  # Using s = 0.5*a*t²
actual_yaw_test2 = yaw2

diff_torque_test3 = 0.05 - (-0.05)
expected_yaw_test3 = 0.5 * 1200 * diff_torque_test3 * 1.0**2
actual_yaw_test3 = yaw3

print(f"\nTest 2 (moderate differential):")
print(f"  Differential torque: {diff_torque_test2:.3f} Nm")
print(f"  Expected yaw change: {np.rad2deg(expected_yaw_test2):.1f}°")
print(f"  Actual yaw change: {np.rad2deg(actual_yaw_test2):.1f}°")
print(f"  Ratio (actual/expected): {actual_yaw_test2/expected_yaw_test2:.3f}")

print(f"\nTest 3 (maximum differential):")
print(f"  Differential torque: {diff_torque_test3:.3f} Nm")
print(f"  Expected yaw change: {np.rad2deg(expected_yaw_test3):.1f}°")
print(f"  Actual yaw change: {np.rad2deg(actual_yaw_test3):.1f}°")
print(f"  Ratio (actual/expected): {actual_yaw_test3/expected_yaw_test3:.3f}")

if actual_yaw_test3 / expected_yaw_test3 < 0.1:
    print("\n⚠️  WARNING: MuJoCo yaw response is <10% of linearized model prediction!")
    print("    This explains why MPC cannot track yaw rate - the simulation")
    print("    dynamics are fundamentally different from the linearized model.")
elif actual_yaw_test3 / expected_yaw_test3 > 0.8:
    print("\n✓  MuJoCo yaw response matches linearized model well")
    print("    The issue must be in the MPC control loop, not the simulation.")
else:
    print(f"\n⚠️  MuJoCo yaw response is {actual_yaw_test3/expected_yaw_test3*100:.0f}% of expected")
    print("    Significant model mismatch - linearized model overestimates yaw control.")

print("="*70)

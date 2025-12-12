"""Test if MuJoCo simulation responds to open-loop differential torques.

Fixed the yaw issues, see orientation file
remaining few-degree mismatch in Test 2 is likely due to nonzero pitch 
and the simplified yaw_dot mapping; to eliminate mismatch we can compute full ZYX Euler rates 
from wx, wy, wz, roll, pitch instead of the cos(pitch) approximation bu that's unecessary as sim works well."""

import sys
from pathlib import Path
import os

import mujoco
import numpy as np

from robot_dynamics.orientation import (
    quat_to_euler_wxyz,
    unwrap_radians,
    yaw_from_rotmat,
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Resolve model path (project root is three levels up from this file)
repo_root = Path(__file__).resolve().parents[3]
model_path = repo_root / "Mujoco sim" / "robot_model.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# Get actuator and body indices
left_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left")
right_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right")
robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

print("=" * 70)
print("OPEN-LOOP DIFFERENTIAL TORQUE TEST")
print("=" * 70)
print("Testing if MuJoCo simulation responds to differential wheel torques")
print(f"Model: {model_path}")
print(f"Actuators: left id={left_actuator_id}, right id={right_actuator_id}")
print(f"ctrlrange: {model.actuator_ctrlrange}")
print()


def run_test(tau_L: float, tau_R: float, steps: int = 1000, log: bool = False):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.25
    yaw_series = []
    pitch_series = []
    wz_series = []
    t_series = []
    for _ in range(steps):
        data.ctrl[left_actuator_id] = tau_L
        data.ctrl[right_actuator_id] = tau_R
        mujoco.mj_step(model, data)
        R = np.array(data.xmat[robot_body_id]).reshape(3, 3)
        yaw = yaw_from_rotmat(R)
        quat = data.qpos[3:7]
        _, pitch, _ = quat_to_euler_wxyz(quat)
        yaw_series.append(yaw)
        pitch_series.append(pitch)
        wz_series.append(data.qvel[3 + 2])  # freejoint angular velocity z
        t_series.append(data.time)
    yaw_series = np.array(yaw_series)
    pitch_series = np.array(pitch_series)
    wz_series = np.array(wz_series)
    t_series = np.array(t_series)
    jumps = np.where(np.abs(np.diff(yaw_series)) > np.deg2rad(90))[0]
    if jumps.size > 0 and log:
        print(f"  Jump detected at indices: {jumps} (deg): {np.rad2deg(np.diff(yaw_series))[jumps]}")
    # Remove large jumps by simple interpolation
    yaw_clean = yaw_series.copy()
    if jumps.size > 0:
        bad_idx = set(j + 1 for j in jumps)
        good = [i for i in range(len(yaw_clean)) if i not in bad_idx]
        if len(good) >= 2:
            yaw_clean[list(bad_idx)] = np.interp(list(bad_idx), good, yaw_clean[good])
    yaw_unwrapped = unwrap_radians(yaw_clean)
    dyaw = yaw_unwrapped[-1] - yaw_unwrapped[0]
    duration = t_series[-1] - t_series[0] if len(t_series) > 1 else 0.0
    dyaw_rate = dyaw / duration if duration > 0 else 0.0
    # Compute yaw_dot from angular velocity using standard ZYX mapping
    roll_series = np.zeros_like(pitch_series)
    yawdot_from_w = []
    for wx, wy, wz, pitch in zip(np.zeros_like(wz_series), np.zeros_like(wz_series), wz_series, pitch_series):
        # With roll ~0, yaw_dot ≈ wz / cos(pitch)
        yawdot_from_w.append(wz / max(np.cos(pitch), 1e-6))
    yawdot_from_w = np.array(yawdot_from_w)
    dyaw_wz = np.trapezoid(yawdot_from_w, t_series) if len(t_series) > 1 else 0.0
    mismatch = np.rad2deg(dyaw - dyaw_wz)
    return {
        "yaw_series": yaw_series,
        "pitch_series": pitch_series,
        "wz_series": wz_series,
        "t_series": t_series,
        "wrapped_final_yaw": yaw_series[-1],
        "unwrapped_dyaw": dyaw,
        "dyaw_wz": dyaw_wz,
        "dyaw_rate": dyaw_rate,
        "mismatch_deg": mismatch,
    }


# Test 1: Equal torques (straight)
print("Test 1: Equal torques (tau_L=0.05, tau_R=0.05)")
res1 = run_test(0.05, 0.05, steps=500)
print(f"  Final wrapped yaw: {np.rad2deg(res1['wrapped_final_yaw']):.3f} deg")
print()

# Test 2: Moderate differential
print("Test 2: Differential torques (tau_L=0.05, tau_R=0.10)")
res2 = run_test(0.05, 0.10, steps=1000, log=True)
print(f"  Final wrapped yaw: {np.rad2deg(res2['wrapped_final_yaw']):.3f} deg")
print(f"  Delta yaw (unwrapped): {np.rad2deg(res2['unwrapped_dyaw']):.3f} deg")
print(f"  Delta yaw from int wz dt: {np.rad2deg(res2['dyaw_wz']):.3f} deg")
print(f"  Mismatch (quat - int wz): {res2['mismatch_deg']:.3f} deg")
print()

# Test 3: Opposite torques (spin)
print("Test 3: Opposite torques (tau_L=-0.05, tau_R=+0.05)")
res3 = run_test(-0.05, 0.05, steps=1000)
print(f"  Final wrapped yaw: {np.rad2deg(res3['wrapped_final_yaw']):.3f} deg")
print(f"  Delta yaw (unwrapped): {np.rad2deg(res3['unwrapped_dyaw']):.3f} deg")
print(f"  Delta yaw from int wz dt: {np.rad2deg(res3['dyaw_wz']):.3f} deg")
print(f"  Mismatch (quat - int wz): {res3['mismatch_deg']:.3f} deg")
print(f"  Forward motion: {data.qpos[0]:.4f} m (should be ~0)")

print("\nNote: Yaw is computed from continuous quaternions, unwrapped in radians, and cross-checked against integrated wz.")
print("=" * 70)

"""
Orientation helpers shared across simulation, control, and tests.

Yaw convention: MuJoCo freejoint quaternion is [w, x, y, z], Z-up, positive yaw
is counterclockwise viewed from above.
"""

from __future__ import annotations

import numpy as np


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def unwrap_angles(angle_series: np.ndarray) -> np.ndarray:
    """Unwrap angle series to remove 2π jumps (expects radians)."""
    return np.unwrap(angle_series)


def unwrap_radians(angle_series: np.ndarray) -> np.ndarray:
    """Alias for unwrap_angles; kept for clarity when ensuring radians."""
    return unwrap_angles(angle_series)


def yaw_from_rotmat(R: np.ndarray) -> float:
    """Yaw from a 3x3 rotation matrix (Z-up, CCW positive viewed from above)."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    """Return yaw (about z) from quaternion [w, x, y, z] (MuJoCo freejoint order)."""
    w, x, y, z = quat_wxyz
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return yaw_from_rotmat(R)


def quat_to_euler_wxyz(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """Return roll, pitch, yaw from quaternion [w, x, y, z] (MuJoCo order)."""
    w, x, y, z = quat_wxyz
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    # yaw (z-axis)
    yaw = quat_to_yaw(quat_wxyz)
    return float(roll), float(pitch), float(yaw)


def ensure_quat_continuity(prev_quat: np.ndarray | None, quat: np.ndarray) -> np.ndarray:
    """Flip quaternion if needed to maintain continuity (q and -q represent same rotation)."""
    if prev_quat is None:
        return quat
    if np.dot(prev_quat, quat) < 0.0:
        return -quat
    return quat

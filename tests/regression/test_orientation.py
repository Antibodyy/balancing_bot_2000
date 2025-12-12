import numpy as np

from robot_dynamics.orientation import quat_to_yaw, unwrap_angles, wrap_to_pi, quat_to_euler_wxyz


def test_quat_to_yaw_pure_yaw():
    yaw = 0.7
    qw = np.cos(yaw / 2)
    qz = np.sin(yaw / 2)
    quat = np.array([qw, 0.0, 0.0, qz])
    assert np.isclose(quat_to_yaw(quat), yaw, atol=1e-9)


def test_quat_to_yaw_with_pitch():
    yaw = 0.9
    pitch = 0.3
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = 1.0
    sr = 0.0
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    quat = np.array([qw, qx, qy, qz])
    roll, pitch_out, yaw_out = quat_to_euler_wxyz(quat)
    assert np.isclose(yaw_out, yaw, atol=1e-9)
    assert np.isclose(pitch_out, pitch, atol=1e-9)
    assert np.isclose(roll, 0.0, atol=1e-9)


def test_unwrap_angles_crossing_pi():
    # Cross +pi to -pi should unwrap smoothly
    series = np.deg2rad(np.array([170, 179, -179, -170]))
    unwrapped = unwrap_angles(series)
    diffs = np.diff(unwrapped)
    assert np.all(np.abs(diffs) < np.deg2rad(30)), "unwrap produced large jump"


def test_unwrap_and_integrate_wz_consistency():
    # Synthetic yaw profile: accelerate then hold
    t = np.linspace(0, 1.0, 51)
    yaw_true = 0.5 * (t**2)  # 0 to 0.5 rad over 1s
    yaw_wrapped = wrap_to_pi(yaw_true)
    yaw_unwrapped = unwrap_angles(yaw_wrapped)
    wz = np.gradient(yaw_true, t)
    yaw_change = yaw_unwrapped[-1] - yaw_unwrapped[0]
    yaw_int = np.trapezoid(wz, t)
    assert np.isclose(yaw_change, yaw_true[-1], atol=1e-4)
    assert np.isclose(yaw_int, yaw_true[-1], atol=1e-3)


def test_quat_continuity_and_yaw_from_rotmat():
    # q and -q must produce same yaw; continuity helper should flip when needed
    yaw = 1.0
    qw = np.cos(yaw / 2)
    qz = np.sin(yaw / 2)
    q1 = np.array([qw, 0.0, 0.0, qz])
    q2 = -q1
    from robot_dynamics.orientation import yaw_from_rotmat, ensure_quat_continuity

    q2c = ensure_quat_continuity(q1, q2)
    assert np.allclose(q2c, q1)
    R = np.array(
        [
            [1 - 2 * (0**2 + qz**2), 2 * (0 * 0 - qz * qw), 2 * (0 * qz + 0 * qw)],
            [2 * (0 * 0 + qz * qw), 1 - 2 * (0**2 + qz**2), 2 * (0 * qz - 0 * qw)],
            [0, 0, 1],
        ]
    )
    assert np.isclose(yaw_from_rotmat(R), yaw, atol=1e-9)

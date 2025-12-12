# Dynamics of Two-Wheeled Self-Balancing Robot

## 1. System Parameters & Variables

### Constants
* $m$: Mass of the robot body (pendulum).
* $M$: Mass of the wheel axle + wheels.
* $J_w$: Rotational inertia of the wheels.
* $I_y$: Moment of inertia of the body about the pitch axis (y-axis).
* $I_z$: Moment of inertia of the entire system about the yaw axis (z-axis).
* $l$: Distance from the axle to the body center of mass (COM).
* $r$: Wheel radius.
* $d$: Wheel spacing (track width).
* $g$: Gravitational acceleration.

### State Variables ($q$)
* $x$: Horizontal position of the axle.
* $\theta$: Pitch angle of the body ($\theta=0$ is the unstable upright equilibrium).
* $\psi$: Yaw angle (heading).

### Inputs
* $\tau_R$: Torque applied to the right wheel.
* $\tau_L$: Torque applied to the left wheel.

---

## 2. Kinematics & Generalized Forces

**Differential Drive Relations**
The forward velocity $v$ and yaw rate $\dot{\psi}$ are related to wheel velocities:
$$
v = \frac{r}{2}(\omega_R + \omega_L) = \dot{x}
$$
$$
\dot{\psi} = \frac{r}{d}(\omega_R - \omega_L)
$$

**Generalized Forces**
Using the principle of virtual work, we define the generalized forces for translation ($F_{tr}$) and rotation ($\tau_\psi$):
$$
F_{tr} = \frac{\tau_R + \tau_L}{r}
$$
$$
\tau_\psi = \frac{d}{2r}(\tau_R - \tau_L)
$$

---

## 3. Lagrangian Formulation

### Kinetic Energy ($T$)
The total kinetic energy consists of:
1.  **Linear Translation**: Energy of the effective mass $M_{eff}$ (including wheel inertia).
2.  **Coupled Motion**: Energy from the body rotating and translating simultaneously.
3.  **Rotation**: Pure rotational energy of the body pitch and system yaw.

$$
T = \frac{1}{2} M_{eff} \dot{x}^{2} + m l \dot{x} \dot{\theta} \cos\theta + \frac{1}{2} (I_y + m l^{2}) \dot{\theta}^{2} + \frac{1}{2} I_z \dot{\psi}^{2}
$$

*Where the effective mass is defined as:*
$$
M_{eff} = M + m + \frac{2J_w}{r^2}
$$

### Potential Energy ($V$)
The potential energy is defined relative to the axle, maximized at the upright position ($\theta=0$):
$$
V = m g l \cos\theta
$$

### The Lagrangian ($L$)
$$
L = T - V
$$

---

## 4. Equations of Motion

We apply the Euler-Lagrange equation:
$$
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}}\right) - \frac{\partial L}{\partial q} = Q_i
$$

### I. Forward Motion ($x$)
*Generalized Force:* $Q_x = F_{tr}$

$$
M_{eff}\ddot{x} + ml\cos\theta\,\ddot{\theta} - ml\sin\theta\,\dot{\theta}^2 = F_{tr}
$$

### II. Pitch Dynamics ($\theta$)
*Generalized Force:* $Q_\theta = -r F_{tr} = -(\tau_R + \tau_L)$
*(Note: The reaction torque acts opposite to the wheel rotation)*

$$
(I_y + ml^2)\ddot{\theta} + ml\cos\theta\,\ddot{x} - mgl\sin\theta = -(\tau_R + \tau_L)
$$

### III. Yaw Dynamics ($\psi$)
*Generalized Force:* $Q_\psi = \tau_\psi$

$$
I_z \ddot{\psi} = \frac{d}{2r}(\tau_R - \tau_L)
$$

---

## 5. Final Coupled System

The complete nonlinear dynamic model is described by the following system of equations:

1.  **(Longitudinal Force Balance)**
    $$(M + m + \frac{2J_w}{r^2})\ddot{x} + ml \cos\theta \ddot{\theta} - ml \dot{\theta}^{2} \sin\theta = \frac{\tau_R + \tau_L}{r}$$

2.  **(Pitch Moment Balance)**
    $$(I_y + m l^{2}) \ddot{\theta} + ml \cos\theta \ddot{x} - m g l \sin\theta = -(\tau_R + \tau_L)$$

3.  **(Yaw Moment Balance)**
    $$\ddot{\psi} = \frac{d}{2r I_z}(\tau_R - \tau_L)$$

## Orientation / Yaw extraction

- MuJoCo freejoint quaternions are ordered `[w, x, y, z]`.
- Yaw (heading) uses a rotation-matrix path for robustness: see `yaw_from_rotmat` in `robot_dynamics/orientation.py`.
- Use `robot_dynamics/orientation.py` helpers:
  - `quat_to_yaw(quat_wxyz)` for yaw
  - `quat_to_euler_wxyz(quat_wxyz)` for roll, pitch, yaw
  - `unwrap_angles(series)` before differencing yaw to avoid ±π jumps
  - `wrap_to_pi(angle)` to normalize a single angle into [-π, π]

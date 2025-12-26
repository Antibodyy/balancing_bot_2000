from cmath import tau
import casadi as ca
import numpy as np
import mujoco
import mujoco.viewer
import scipy.linalg
import matplotlib.pyplot as plt

phi = np.deg2rad(0.0) 
data_log = {
    'time': [],
    'theta': [],
    'torque': [],
    'theta_target': [] 
}

def get_robot_dynamics():

    x = ca.SX.sym('x')
    theta = ca.SX.sym('theta')
    psi = ca.SX.sym('psi')
    dx = ca.SX.sym('dx')
    dtheta = ca.SX.sym('dtheta')
    dpsi = ca.SX.sym('dpsi')
    tau_L = ca.SX.sym('tau_L')
    tau_R = ca.SX.sym('tau_R')
    q = ca.vertcat(x, theta, psi)
    dq = ca.vertcat(dx, dtheta, dpsi)
    tau = ca.vertcat(tau_L, tau_R)

    M_b = 1.5  # Mass of body [kg]
    M_w = 0.5 # Mass of wheel [kg]
    L   = 0.2 # Length to COM [m]
    R   = 0.05 # Wheel radius [m]
    g   = 9.81 # Gravity [m/s^2]
    W   = 0.3  # Track width (distance between wheels in m)

    I_wy = 0.001 # Wheel inertias
    I_wz = 0.001 
    I_bx = 0.05 # Body inertias
    I_by = 0.05  
    I_bz = 0.01  

    M11 = M_b+ M_w + I_wy/(R**2)
    M12 = M_b *L * ca.cos(theta) #also equal to M21
    M22 = (M_b +M_w)*(L**2) + I_by
    M33 = I_wz +I_bx*(ca.sin(theta)**2) -I_bz*(ca.cos(theta)**2)
    M = ca.vertcat(
        ca.horzcat(M11, M12, 0),
        ca.horzcat(M12,M22, 0),
        ca.horzcat(0, 0,  M33)
    )

    C1 = -M_b * L * (dtheta**2) * ca.sin(theta)
    C2 = -0.5 * (I_bx - I_bz) * (dpsi**2) * ca.sin(2*theta)
    C3 = (I_bx - I_bz) * dtheta * dpsi * ca.sin(2*theta)

    C = ca.vertcat(C1, C2, C3)


    G1 = (M_b + M_w) * g * ca.sin(phi)
    G2 = -M_b * g * L * ca.sin(theta + phi)
    G3 = 0

    G = ca.vertcat(G1, G2, G3)

    tau_sum = tau_L + tau_R
    tau_diff = tau_R - tau_L
    F1 = (1/R) * tau_sum
    F2 = -tau_sum  
    F3 = (W/(2*R))*tau_diff

    F_u = ca.vertcat(F1, F2, F3)

    rhs = F_u - C - G
    q_dd = ca.solve(M, rhs)
    x_state = ca.vertcat(q, dq)
    x_dot_vec = ca.vertcat(dq, q_dd)

    f = ca.Function('robot_dynamics', [x_state, tau], [x_dot_vec], ['x', 'u'], ['x_dot'])
    return f, x_state, tau

x0 = np.array([0, -phi, 0, 0, 0, 0])
u0 = np.array([0.25, 0.25])
theta_eq = -phi
x_eq_val = np.array([0, theta_eq, 0, 0, 0, 0])
u_eq_val = np.zeros(2)
robot_dynamics, sym_x, sym_u = get_robot_dynamics()
jac_A = ca.jacobian(robot_dynamics(sym_x, sym_u), sym_x)
jac_B = ca.jacobian(robot_dynamics(sym_x, sym_u), sym_u)
linearizer = ca.Function('linearizer', [sym_x, sym_u], [jac_A, jac_B])
A_res, B_res = linearizer(x_eq_val, u_eq_val)
A_num = np.array(A_res)
B_num = np.array(B_res)
Q = np.diag([1.0, 500.0, 1.0, 0.1, 10.0, 0.1]) 
R = np.eye(2) * 0.1
P = scipy.linalg.solve_continuous_are(A_num, B_num, Q, R)
K = np.linalg.inv(R)@B_num.T@P
print(K)

def controller_callback(model, data):
    x_pos = data.qpos[0]
    theta = data.qpos[1]
    x_vel = data.qvel[0]
    d_theta = data.qvel[1]
    x_curr = np.array([x_pos, theta, 0.0, x_vel, d_theta, 0.0])
    u = -K @ (x_curr-x_eq_val)
    if abs(theta) > 0.05 and int(data.time * 100) % 10 == 0:
        print(f"Tilt: {theta:.3f} rad and LQR Reaction Torque: {u[0]:.3f} N")
    data.ctrl[0] = u[0]
    data.ctrl[1] = u[1]
    data_log['time'].append(data.time)
    data_log['theta'].append(theta)
    data_log['torque'].append(u[0]) # Storing left wheel torque
    data_log['theta_target'].append(theta_eq)

def main():
    model = mujoco.MjModel.from_xml_path('Aryan/Mujoco sim/robot_model.xml')
    data = mujoco.MjData(model)
    data.qpos[1] = theta_eq
    mujoco.set_mjcb_control(controller_callback)
    mujoco.viewer.launch(model, data)
    time_arr = np.array(data_log['time'])
    theta_arr = np.array(data_log['theta'])
    torque_arr = np.array(data_log['torque'])
    target_arr = np.array(data_log['theta_target'])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    ax1.plot(time_arr, theta_arr, label='Actual Pitch', color='blue')
    ax1.plot(time_arr, target_arr, label='Target (Vertical)', color='red', linestyle='--')
    ax1.set_ylabel('Pitch Angle (rad)')
    ax1.set_title('Robot Balancing Performance')
    ax1.legend()
    ax1.grid(True)
        
    ax2.plot(time_arr, torque_arr, label='Motor Torque (Left)', color='green')
    ax2.set_ylabel('Torque (Nm)')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
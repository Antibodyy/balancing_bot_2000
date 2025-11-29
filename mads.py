from cmath import tau
import casadi as ca
import numpy as np
import mujoco
import mujoco.viewer
import scipy.linalg

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
    phi = np.deg2rad(15.0)  # Slope angle [rad]
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

x0 = np.array([0, 0.1, 0, 0, 0, 0])
u0 = np.array([0.25, 0.25])
robot_dynamics, sym_x, sym_u = get_robot_dynamics()
jac_A = ca.jacobian(robot_dynamics(sym_x, sym_u), sym_x)
jac_B = ca.jacobian(robot_dynamics(sym_x, sym_u), sym_u)
A_func = ca.Function('A_func', [sym_x, sym_u], [jac_A])
B_func = ca.Function('B_func', [sym_x, sym_u], [jac_B])
A_num = np.array(A_func(np.zeros(6), np.zeros(2)))
B_num = np.array(B_func(np.zeros(6), np.zeros(2)))
Q = np.diag([1.0, 500.0, 1.0, 0.1, 10.0, 0.1]) 
R = np.eye(2) * 0.1
P = scipy.linalg.solve_continuous_are(A_num, B_num, Q, R)
K = np.linalg.inv(R) @ B_num.T @ P
print(f"LQR Gain Matrix Calculated. Shape: {K.shape}")
print(K)

def controller_callback(model, data):
    x_pos = data.qpos[0]
    theta = data.qpos[1]
    x_vel = data.qvel[0]
    d_theta = data.qvel[1]
    x_curr = np.array([x_pos, theta, 0.0, x_vel, d_theta, 0.0])
    u = -K @ x_curr
    if abs(theta) > 0.05 and int(data.time * 100) % 10 == 0:
        print(f"Tilt: {theta:.3f} rad | LQR Reaction Torque: {u[0]:.3f} N")
    data.ctrl[0] = u[0]
    data.ctrl[1] = u[1]

def main():
    model = mujoco.MjModel.from_xml_path('MEC231A/Final_project/balancing_bot_2000/robot_model.xml')
    data = mujoco.MjData(model)
    mujoco.set_mjcb_control(controller_callback)
    mujoco.viewer.launch(model, data)
if __name__ == "__main__":
    main()

# mjcf_xml = """
# <mujoco model="balance_bot">
#   <option timestep="0.005" gravity="0 0 -9.81"/>

#   <visual>
#     <rgba haze="0.15 0.25 0.35 1"/>
#     <quality shadowsize="2048"/>
#   </visual>

#   <worldbody>
#     <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    
#     <geom name="floor" type="plane" size="0 15 0.05" rgba=".8 .9 .8 1"/>

#     <body name="robot" pos="0 0 0.05">
      
#       <joint name="slide_x" type="slide" axis="1 0 0" damping="0.1"/>
      
#       <joint name="pitch" type="hinge" axis="0 1 0" pos="0 0 0"/>
      
#       <geom type="box" size="0.05 0.05 0.2" pos="0 0 0.2" rgba="0.9 0.5 0.1 1" mass="10.0"/>

#       <body name="wheel_r" pos="0 -0.15 0">
#         <joint name="motor_r" axis="0 1 0"/>
#         <geom type="cylinder" size="0.05 0.02" rgba="0.1 0.1 0.1 1" mass="0.5" fromto="0 0 0 0 0.02 0"/>
#       </body>

#       <body name="wheel_l" pos="0 0.15 0">
#         <joint name="motor_l" axis="0 1 0"/>
#         <geom type="cylinder" size="0.05 0.02" rgba="0.1 0.1 0.1 1" mass="0.5" fromto="0 0 0 0 -0.02 0"/>
#       </body>
#     </body>
#   </worldbody>

#   <actuator>
#     <motor name="left" joint="motor_l" gear="1"/>
#     <motor name="right" joint="motor_r" gear="1"/>
#   </actuator>
# </mujoco>
# """


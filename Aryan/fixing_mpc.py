import mujoco
import mujoco.viewer
import numpy as np
import casadi as ca
import time

# ==========================================
# 1. ROBOT XML (Corrected Wheels)
# ==========================================
robot_xml = """
<mujoco model="balance_bot">
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba=".8 .9 .8 1"/>
    
    <body name="body" pos="0 0 0.25">
      <joint name="root" type="free"/>
      <geom type="box" size="0.05 0.1 0.2" mass="1.5" rgba="0.8 0.2 0.2 1"/>
      
      <body name="left_wheel" pos="0 0.12 -0.2">
        <joint name="left_wheel_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" mass="0.5" rgba="0.2 0.2 0.8 1" euler="90 0 0"/>
      </body>

      <body name="right_wheel" pos="0 -0.12 -0.2">
        <joint name="right_wheel_joint" type="hinge" axis="0 1 0"/>
        <geom type="cylinder" size="0.05 0.02" mass="0.5" rgba="0.2 0.2 0.8 1" euler="90 0 0"/>
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="left_motor" joint="left_wheel_joint" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
    <motor name="right_motor" joint="right_wheel_joint" gear="1" ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""

# ==========================================
# 2. DYNAMICS & MPC CLASS (Longer Horizon)
# ==========================================
def get_robot_dynamics_discrete(dt):
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
    state = ca.vertcat(q, dq)

    # Physics Constants
    M_b, M_w = 1.5, 0.5
    L, R, W = 0.2, 0.05, 0.24
    g = 9.81

    # --- DYNAMICS ---
    # 1. Linear: Force = Torque / Radius
    # F = ma => a = F/m
    acc_x = (tau_L/R + tau_R/R) / (M_b + 2*M_w)
    
    # 2. Angular (Pitch): 
    # Critical interaction: To pitch BACK (up), you must accelerate FORWARD.
    # We model this as the torque reacting against the body.
    # Sign: Positive Theta = falling forward. 
    #       Gravity increases theta (+). 
    #       Positive Torque (driving forward) pushes body backward (-).
    acc_theta = (g/L)*ca.sin(theta) - (tau_L + tau_R)/(M_b * L**2) 
    
    # 3. Yaw
    acc_psi = (tau_R - tau_L) * W 

    rhs_ode = ca.vertcat(dx, dtheta, dpsi, acc_x, acc_theta, acc_psi)
    
    ode_dict = {'x': state, 'p': tau, 'ode': rhs_ode}
    opts = {'t0': 0, 'tf': dt}
    return ca.integrator('F_int', 'rk', ode_dict, opts)

class RobotMPC:
    def __init__(self, N=20, dt_mpc=0.05):
        """
        N=20, dt=0.05 => 1.0 second prediction horizon.
        This allows the robot to "see" the fall coming.
        """
        self.opti = ca.Opti()
        self.X = self.opti.variable(6, N + 1)
        self.U = self.opti.variable(2, N)
        self.x0 = self.opti.parameter(6)
        
        # Get dynamics with the MPC timestep (not physics timestep)
        F = get_robot_dynamics_discrete(dt_mpc)
        
        # --- TUNING WEIGHTS ---
        # Prioritize Theta (200.0) heavily over Position X (1.0).
        # If we penalize X too much, it won't drive "under" the fall to catch itself.
        Q = np.diag([1.0, 200.0, 1.0, 5.0, 10.0, 1.0]) 
        #           [x,   th,   psi, dx,  dth,  dpsi]
        
        R = np.diag([0.1, 0.1]) # Cheap control
        
        self.opti.subject_to(self.X[:, 0] == self.x0)
        
        for k in range(N):
            res = F(x0=self.X[:, k], p=self.U[:, k])
            self.opti.subject_to(self.X[:, k+1] == res['xf'])
            
            # Allow higher torque for recovery
            self.opti.subject_to(self.opti.bounded(-5, self.U[:, k], 5))
            
            # Cost
            curr_x = self.X[:, k]
            cost = ca.mtimes([curr_x.T, Q, curr_x]) + ca.mtimes([self.U[:, k].T, R, self.U[:, k]])
            self.opti.minimize(cost)
            
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.max_iter': 15}
        self.opti.solver('ipopt', opts)

    def solve(self, state):
        self.opti.set_value(self.x0, state)
        
        # Warm start: Use previous solution to speed up next step
        # (Simple version: just initialize with 0, robust enough for this)
        self.opti.set_initial(self.U, 0.0) 
        
        try:
            sol = self.opti.solve()
            return sol.value(self.U[:, 0])
        except:
            # If solver fails, return 0 (or a safe backup controller)
            # print("Solver failed")
            return np.zeros(2)

# ==========================================
# 3. MAIN SIMULATION LOOP
# ==========================================
def main():
    model = mujoco.MjModel.from_xml_string(robot_xml)
    data = mujoco.MjData(model)
    
    # We use a larger dt for MPC (0.05) than physics (0.01)
    mpc = RobotMPC(N=20, dt_mpc=0.05)
    
    # Helper to throttle MPC calls
    mpc_counter = 0
    mpc_skip = 5 # Run MPC every 5 physics steps (0.05s / 0.01s = 5)
    last_ctrl = np.zeros(2)

    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # --- State Estimation ---
            q_x = data.qpos[0]
            
            # Quaternion to Pitch (Theta)
            qw, qx, qy, qz = data.qpos[3:7]
            sin_p = 2 * (qw * qy - qz * qx)
            if np.abs(sin_p) >= 1: theta = np.sign(sin_p) * (np.pi / 2)
            else: theta = np.arcsin(sin_p)
            
            # Fix: If theta is > 45 deg (0.78 rad), it has likely fallen. 
            # Reset or stop control to prevent crazy spinning.
            if abs(theta) > 0.8:
                last_ctrl = np.zeros(2)
            elif mpc_counter % mpc_skip == 0:
                # --- Run MPC at 20Hz ---
                # Check angular velocity index (4 for free joint Y-axis)
                dtheta = data.qvel[4] 
                state = np.array([q_x, theta, 0, data.qvel[0], dtheta, 0])
                
                last_ctrl = mpc.solve(state)
            
            mpc_counter += 1
            
            # --- Apply Control ---
            data.ctrl[0] = last_ctrl[0]
            data.ctrl[1] = last_ctrl[1]
            
            mujoco.mj_step(model, data)
            viewer.sync()

            # Real-time sync
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

if __name__ == "__main__":
    main()
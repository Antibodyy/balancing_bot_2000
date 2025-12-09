from cmath import tau
import casadi as ca
import numpy as np
import mujoco
import mujoco.viewer
import scipy.linalg
import matplotlib.pyplot as plt

phi = np.deg2rad(35.0) 
data_log = {
    'time': [],
    'theta': [],
    'torque': [],
    'theta_target': [] 
}

def main():
    model = mujoco.MjModel.from_xml_path('robot_model.xml')
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
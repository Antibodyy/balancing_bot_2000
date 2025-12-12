import mujoco
import mujoco.viewer
import numpy as np


def main():
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path('Mujoco sim/robot_model.xml')
    data = mujoco.MjData(model)

    # Set initial pitch angle to vertical (0 degrees)
    data.qpos[1] = 0.0  # pitch angle

    # Launch the interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Run simulation
        while viewer.is_running():
            # Step the simulation
            mujoco.mj_step(model, data)

            # Sync the viewer
            viewer.sync()


if __name__ == "__main__":
    main()

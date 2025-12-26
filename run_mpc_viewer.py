"""
Simple script to run the original MPC simulation with MuJoCo viewer.
Run from the stable-mpc-diego directory.
"""
import numpy as np
from simulation import SimulationConfig, MPCSimulation
from mpc import ReferenceCommand, ReferenceMode

# Configuration
config = SimulationConfig(
    model_path='robot_model.xml',
    robot_params_path='config/simulation/robot_params.yaml',
    mpc_params_path='config/simulation/mpc_params.yaml',
    estimator_params_path='config/simulation/estimator_params.yaml',
    duration_s=10.0,
    render=True  # Enable MuJoCo viewer
)

# Create simulation
print("Creating MPC simulation...")
sim = MPCSimulation(config)

# Run with viewer
initial_pitch_deg = 3.0
initial_pitch_rad = np.deg2rad(initial_pitch_deg)

print(f"\nStarting simulation with {initial_pitch_deg}° initial pitch perturbation")
print("MuJoCo Viewer Controls:")
print("  Space = pause/resume")
print("  Backspace = reset")
print("  Mouse drag = rotate view")
print("  Scroll = zoom")
print("\nClose viewer window to stop simulation...\n")

result = sim.run_with_viewer(
    initial_pitch_rad=initial_pitch_rad,
    reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE)
)

# Print results
print("\n" + "="*60)
print("Simulation Results")
print("="*60)
print(f"  Success: {result.success}")
#print(f"  Duration: {result.time_s[0]:.2f}s")
print(f"  Mean solve time: {result.mean_solve_time_ms:.2f} ms")
print(f"  Max solve time: {result.max_solve_time_ms:.2f} ms")

if len(result.state_history) > 0:
    max_pitch = np.rad2deg(np.max(np.abs(result.state_history[:, 1])))
    final_pitch = np.rad2deg(result.state_history[-1, 1])
    final_position = result.state_history[-1, 0]
    print(f"  Max pitch: {max_pitch:.3f}°")
    print(f"  Final pitch: {final_pitch:.3f}°")
    print(f"  Final position: {final_position:.3f} m")
print("="*60)

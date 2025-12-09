"""Verify velocity fix is working in viewer mode."""

import numpy as np
from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode

# Track velocity comparison
velocity_data = []

def track_velocity(sim, data):
    """Track velocity during simulation."""
    true_v = data.qvel[sim.SLIDE_X_JOINT]
    est_v = sim._controller._state_estimator._state if hasattr(sim._controller._state_estimator, '_state') else 0.0
    velocity_data.append((data.time, true_v, est_v))

# Create simulation
config = SimulationConfig(
    model_path='robot_model.xml',
    robot_params_path='config/robot_params.yaml',
    mpc_params_path='config/mpc_params.yaml',
    estimator_params_path='config/estimator_params.yaml',
    duration_s=2.0,
)

sim = MPCSimulation(config)

print("Running simulation with viewer for 2 seconds...")
print("Close the viewer window to see velocity analysis.\n")

# Check configuration
print(f"Controller use_simulation_velocity: {sim._controller._use_simulation_velocity}")
print(f"Controller _true_ground_velocity (initial): {sim._controller._true_ground_velocity}")

result = sim.run_with_viewer(
    duration_s=2.0,
    initial_pitch_rad=np.deg2rad(3.0),
    reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
)

# Analysis
print("\n" + "=" * 60)
print("VELOCITY ANALYSIS")
print("=" * 60)

if len(result.state_history) > 0:
    print(f"\nVelocity Comparison (first 10 steps):")
    print(f"{'Step':>4} {'Time':>6} {'True_v':>10} {'Est_v':>10} {'Error':>10} {'Pitch':>10}")
    print(f"{'':>4} {'(s)':>6} {'(m/s)':>10} {'(m/s)':>10} {'(m/s)':>10} {'(deg)':>10}")
    print("-" * 70)

    for i in range(min(10, len(result.time_s))):
        true_v = result.state_history[i, 3]
        est_v = result.state_estimate_history[i, 3]
        error = est_v - true_v
        pitch = np.rad2deg(result.state_history[i, 1])

        print(f"{i:4d} {result.time_s[i]:6.3f} {true_v:10.4f} {est_v:10.4f} {error:10.4f} {pitch:10.4f}")

    # Overall statistics
    velocity_errors = result.state_estimate_history[:, 3] - result.state_history[:, 3]
    print(f"\nVelocity Error Statistics:")
    print(f"  Mean absolute error: {np.mean(np.abs(velocity_errors)):.4f} m/s")
    print(f"  Max absolute error: {np.max(np.abs(velocity_errors)):.4f} m/s")
    print(f"  RMS error: {np.sqrt(np.mean(velocity_errors**2)):.4f} m/s")

print(f"\nSimulation Result:")
print(f"  Success: {result.success}")
print(f"  Duration: {result.time_s[-1]:.3f}s" if len(result.time_s) > 0 else "  Duration: 0s")
if len(result.state_history) > 0:
    print(f"  Max pitch: {np.rad2deg(np.max(np.abs(result.state_history[:, 1]))):.2f}°")
    print(f"  Final pitch: {np.rad2deg(result.state_history[-1, 1]):.2f}°")

"""Compare behavior with and without velocity estimation fix."""

import numpy as np
import matplotlib.pyplot as plt
from simulation import MPCSimulation, SimulationConfig
from robot_dynamics.parameters import PITCH_INDEX, VELOCITY_INDEX
from mpc import ReferenceCommand, ReferenceMode

config = SimulationConfig(
    model_path='robot_model.xml',
    robot_params_path='config/robot_params.yaml',
    mpc_params_path='config/mpc_params.yaml',
    estimator_params_path='config/estimator_params.yaml',
)

initial_pitch = 0.05  # ~3 degrees
duration = 2.0

print("="*80)
print("COMPARING: BROKEN vs FIXED VELOCITY ESTIMATION")
print("="*80)

# Test 1: Current (broken) implementation
print("\n1. Running with BROKEN velocity estimation...")
sim_broken = MPCSimulation(config)
result_broken = sim_broken.run(
    duration_s=duration,
    initial_pitch_rad=initial_pitch,
)

print(f"   Result: {'SUCCESS' if result_broken.success else 'FAILED'}")
if not result_broken.success:
    print(f"   Fell at t={result_broken.time_s[-1]:.3f}s")

# Test 2: Fixed implementation
print("\n2. Running with FIXED velocity estimation...")
sim_fixed = MPCSimulation(config)

# Patch the state estimation
original_estimate = sim_fixed._controller._estimate_state

def patched_estimate(sensor_data):
    state = original_estimate(sensor_data)
    # Use true MuJoCo velocity instead of encoder-based
    state[VELOCITY_INDEX] = sim_fixed._data.qvel[sim_fixed.SLIDE_X_JOINT]
    return state

sim_fixed._controller._estimate_state = patched_estimate

result_fixed = sim_fixed.run(
    duration_s=duration,
    initial_pitch_rad=initial_pitch,
)

print(f"   Result: {'SUCCESS' if result_fixed.success else 'FAILED'}")
if result_fixed.success:
    final_pitch = np.rad2deg(result_fixed.state_history[-1, PITCH_INDEX])
    max_pitch = np.rad2deg(np.max(np.abs(result_fixed.state_history[:, PITCH_INDEX])))
    print(f"   Final pitch: {final_pitch:.2f}°")
    print(f"   Max pitch: {max_pitch:.2f}°")

# Comparison plot
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle(f'Comparison: Broken vs Fixed Velocity Estimation\n'
             f'Initial Perturbation: {np.rad2deg(initial_pitch):.2f}°',
             fontsize=14, fontweight='bold')

# Broken - Pitch
ax = axes[0, 0]
ax.plot(result_broken.time_s, np.rad2deg(result_broken.state_history[:, PITCH_INDEX]),
        'r-', linewidth=2, label='Broken')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axhline(y=30, color='r', linestyle='--', alpha=0.3)
ax.axhline(y=-30, color='r', linestyle='--', alpha=0.3)
ax.set_ylabel('Pitch (deg)')
ax.set_title('BROKEN: Pitch Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Fixed - Pitch
ax = axes[0, 1]
ax.plot(result_fixed.time_s, np.rad2deg(result_fixed.state_history[:, PITCH_INDEX]),
        'g-', linewidth=2, label='Fixed')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axhline(y=30, color='r', linestyle='--', alpha=0.3)
ax.axhline(y=-30, color='r', linestyle='--', alpha=0.3)
ax.set_ylabel('Pitch (deg)')
ax.set_title('FIXED: Pitch Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Broken - Velocity
ax = axes[1, 0]
ax.plot(result_broken.time_s, result_broken.state_history[:, VELOCITY_INDEX],
        'b-', linewidth=2, label='True')
ax.plot(result_broken.time_s, result_broken.state_estimate_history[:, VELOCITY_INDEX],
        'r--', linewidth=2, label='Estimated')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('BROKEN: Velocity (True vs Estimated)')
ax.legend()
ax.grid(True, alpha=0.3)

# Fixed - Velocity
ax = axes[1, 1]
ax.plot(result_fixed.time_s, result_fixed.state_history[:, VELOCITY_INDEX],
        'b-', linewidth=2, label='True')
ax.plot(result_fixed.time_s, result_fixed.state_estimate_history[:, VELOCITY_INDEX],
        'g--', linewidth=2, label='Estimated (Fixed)')
ax.set_ylabel('Velocity (m/s)')
ax.set_title('FIXED: Velocity (True vs Estimated)')
ax.legend()
ax.grid(True, alpha=0.3)

# Broken - Control
ax = axes[2, 0]
total_torque_broken = result_broken.control_history[:, 0] + result_broken.control_history[:, 1]
ax.plot(result_broken.time_s, total_torque_broken, 'r-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Total Torque (N⋅m)')
ax.set_title('BROKEN: Control Response')
ax.grid(True, alpha=0.3)

# Fixed - Control
ax = axes[2, 1]
total_torque_fixed = result_fixed.control_history[:, 0] + result_fixed.control_history[:, 1]
ax.plot(result_fixed.time_s, total_torque_fixed, 'g-', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Total Torque (N⋅m)')
ax.set_title('FIXED: Control Response')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_output/broken_vs_fixed_comparison.png', dpi=150)
print(f"\n📊 Comparison plot saved to: debug_output/broken_vs_fixed_comparison.png")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print("\nBROKEN Implementation:")
print(f"  Survived: {result_broken.success}")
print(f"  Duration: {result_broken.time_s[-1]:.3f}s / {duration:.1f}s")
max_vel_error_broken = np.max(np.abs(
    result_broken.state_estimate_history[:, VELOCITY_INDEX] -
    result_broken.state_history[:, VELOCITY_INDEX]
))
print(f"  Max velocity error: {max_vel_error_broken:.3f} m/s")

print("\nFIXED Implementation:")
print(f"  Survived: {result_fixed.success}")
print(f"  Duration: {result_fixed.time_s[-1]:.3f}s / {duration:.1f}s")
max_vel_error_fixed = np.max(np.abs(
    result_fixed.state_estimate_history[:, VELOCITY_INDEX] -
    result_fixed.state_history[:, VELOCITY_INDEX]
))
print(f"  Max velocity error: {max_vel_error_fixed:.3f} m/s")

print("\n" + "="*80)
plt.show()

"""Quick test: pure yaw rate tracking without forward velocity."""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from simulation import SimulationConfig
from debug import MPCDiagnostics
from mpc import ReferenceCommand, ReferenceMode

config = SimulationConfig(
    model_path='mujoco_sim/robot_model.xml',
    robot_params_path='config/simulation/robot_params.yaml',
    mpc_params_path='config/simulation/mpc_params.yaml',
    estimator_params_path='config/simulation/estimator_params.yaml',
    duration_s=5.0,
)

diag = MPCDiagnostics(config)

# Test: pure yaw rate, zero forward velocity
target_yaw_rate = 0.3  # rad/s

def reference_callback(time_s: float) -> ReferenceCommand:
    return ReferenceCommand(
        mode=ReferenceMode.VELOCITY,
        velocity_mps=0.0,  # NO forward velocity
        yaw_rate_radps=target_yaw_rate,  # ONLY yaw rate
    )

print(f"\nTesting PURE yaw rate: {target_yaw_rate} rad/s ({np.rad2deg(target_yaw_rate):.1f}°/s)")
print("Forward velocity: 0.0 m/s\n")

result, summary = diag.run_with_diagnostics(
    duration_s=5.0,
    initial_pitch_rad=0.0,
    reference_command_callback=reference_callback,
    verbose=True,
)

# Analyze yaw rate tracking
yaw_rates = result.state_history[:, 5]
mean_yaw_rate = np.mean(yaw_rates[50:])  # Skip settling
yaw_rate_error = abs(mean_yaw_rate - target_yaw_rate)

print(f"\n{'='*60}")
print(f"PURE YAW RATE TEST RESULTS")
print(f"{'='*60}")
print(f"Target yaw rate: {target_yaw_rate:.3f} rad/s ({np.rad2deg(target_yaw_rate):.1f}°/s)")
print(f"Actual yaw rate: {mean_yaw_rate:.3f} rad/s ({np.rad2deg(mean_yaw_rate):.1f}°/s)")
print(f"Error: {yaw_rate_error:.4f} rad/s ({yaw_rate_error/target_yaw_rate*100:.1f}%)")
print(f"{'='*60}\n")

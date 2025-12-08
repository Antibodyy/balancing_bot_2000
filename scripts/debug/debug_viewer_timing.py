"""Debug viewer mode timing issues."""

import numpy as np
import time
from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode

# Track control callback invocations
callback_times = []
mpc_run_times = []
mpc_solve_times = []

original_step = None

def wrap_controller_step(original_method):
    """Wrapper to track MPC timing."""
    def wrapped(sensor_data, reference_command):
        start = time.perf_counter()
        result = original_method(sensor_data, reference_command)
        elapsed = time.perf_counter() - start
        mpc_run_times.append(elapsed * 1000)  # Convert to ms
        mpc_solve_times.append(result.mpc_solution.solve_time_s * 1000)
        return result
    return wrapped

# Create simulation
config = SimulationConfig(
    model_path='robot_model.xml',
    robot_params_path='config/robot_params.yaml',
    mpc_params_path='config/mpc_params.yaml',
    estimator_params_path='config/estimator_params.yaml',
    duration_s=2.0,
)

sim = MPCSimulation(config)

# Wrap the controller step method
original_step = sim._controller.step
sim._controller.step = wrap_controller_step(original_step)

print("Running simulation with viewer for 2 seconds...")
print("Close the viewer window to see timing analysis.\n")

result = sim.run_with_viewer(
    duration_s=2.0,
    initial_pitch_rad=np.deg2rad(3.0),
    reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE),
)

# Restore original method
sim._controller.step = original_step

# Analysis
print("\n" + "=" * 60)
print("TIMING ANALYSIS")
print("=" * 60)

print(f"\nMPC Controller Invocations: {len(mpc_run_times)}")
print(f"Expected (at 50Hz for 2s): ~100")

if len(mpc_run_times) > 0:
    print(f"\nTotal MPC step time (includes all overhead):")
    print(f"  Mean: {np.mean(mpc_run_times):.1f}ms")
    print(f"  Median: {np.median(mpc_run_times):.1f}ms")
    print(f"  Max: {np.max(mpc_run_times):.1f}ms")
    print(f"  Min: {np.min(mpc_run_times):.1f}ms")

    print(f"\nJust MPC solve time:")
    print(f"  Mean: {np.mean(mpc_solve_times):.1f}ms")
    print(f"  Median: {np.median(mpc_solve_times):.1f}ms")
    print(f"  Max: {np.max(mpc_solve_times):.1f}ms")
    print(f"  Min: {np.min(mpc_solve_times):.1f}ms")

    overhead = np.array(mpc_run_times) - np.array(mpc_solve_times)
    print(f"\nOverhead (non-solve time):")
    print(f"  Mean: {np.mean(overhead):.1f}ms")
    print(f"  Max: {np.max(overhead):.1f}ms")

print(f"\nSimulation Result:")
print(f"  Success: {result.success}")
print(f"  Steps recorded: {len(result.time_s)}")

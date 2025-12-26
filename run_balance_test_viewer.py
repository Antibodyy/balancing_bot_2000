"""
Run the balance test with small perturbation in MuJoCo viewer.

This recreates the test_balance_with_small_perturbation test but shows
the simulation in the interactive viewer instead of running headless.
"""
import numpy as np
from simulation import MPCSimulation, SimulationConfig
from mpc import ReferenceCommand, ReferenceMode

# Create simulation with default config (same as test fixture)
config = SimulationConfig(
    model_path='robot_model.xml',
    robot_params_path='config/simulation/robot_params.yaml',
    mpc_params_path='config/simulation/mpc_params.yaml',
    estimator_params_path='config/simulation/estimator_params.yaml',
    duration_s=10.0,  # Longer duration for viewer mode
    render=True  # Enable viewer
)

simulation = MPCSimulation(config)

print("\n" + "="*70)
print("BALANCE TEST WITH SMALL PERTURBATION (2° forward tilt)")
print("="*70)
print("\nTest scenario:")
print("  Initial pitch: 2.0 degrees forward")
print("  Reference: BALANCE (stabilize upright)")
print("  Duration: Runs until you close viewer or robot falls")
print("\nStarting MuJoCo viewer...")
print("Close the viewer window to see results.\n")

# Run with viewer (same initial condition as test)
result = simulation.run_with_viewer(
    initial_pitch_rad=np.deg2rad(2),  # 2 degree forward tilt
    reference_command=ReferenceCommand(mode=ReferenceMode.BALANCE)
)

# Print results (same checks as test)
print("\n" + "="*70)
print("SIMULATION RESULTS")
print("="*70)
print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")
print(f"Duration: {result.total_duration_s:.2f}s")
print(f"Mean solve time: {result.mean_solve_time_ms:.2f}ms")
print(f"Max solve time: {result.max_solve_time_ms:.2f}ms")
print(f"Deadline violations: {result.deadline_violations}")

if len(result.state_history) > 0:
    final_pitch = np.rad2deg(result.state_history[-1, 1])
    max_pitch = np.rad2deg(np.max(np.abs(result.state_history[:, 1])))
    final_position = result.state_history[-1, 0]

    print(f"\nFinal State:")
    print(f"  Pitch: {final_pitch:.4f}°")
    print(f"  Max pitch: {max_pitch:.4f}°")
    print(f"  Position: {final_position:.4f} m")

    # Test assertions (from original test)
    print(f"\nTest Checks:")
    print(f"  ✓ Simulation succeeded: {result.success}")
    print(f"  ✓ Final pitch < 0.1°: {abs(final_pitch) < 0.1} ({abs(final_pitch):.4f}°)")
    print(f"  ✓ Robot stayed near origin: {abs(final_position) < 0.5} ({abs(final_position):.4f} m)")
    print(f"  ✓ Solve time < 15ms: {result.mean_solve_time_ms < 15} ({result.mean_solve_time_ms:.2f} ms)")

print("\n" + "="*70)

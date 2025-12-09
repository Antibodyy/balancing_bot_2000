"""Quick diagnostic check - minimal output."""

from simulation import SimulationConfig
from debug import MPCDiagnostics

config = SimulationConfig(
    model_path='robot_model.xml',
    robot_params_path='config/robot_params.yaml',
    mpc_params_path='config/mpc_params.yaml',
    estimator_params_path='config/estimator_params.yaml',
)

diag = MPCDiagnostics(config)

print("Running quick diagnostic check...")
print("Initial perturbation: ~3 degrees")
print("Duration: 1 second\n")

# Quick test
result, summary = diag.run_with_diagnostics(
    duration_s=1.0,
    initial_pitch_rad=0.05,
    verbose=False,  # No step-by-step trace
)

# Just print summary
summary.print_summary()

# Generate one comprehensive plot
print("\nGenerating diagnostic plots...")
diag.plot_all(result, save_dir="debug_output/quick_check", show=True)
print("\nPlots saved to debug_output/quick_check/")

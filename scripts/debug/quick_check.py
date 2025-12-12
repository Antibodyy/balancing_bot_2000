"""Quick diagnostic check - minimal output."""

import sys
from pathlib import Path

# Ensure project root and debug are importable
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from simulation import SimulationConfig
import importlib.util

# Load MPCDiagnostics explicitly from debug/mpc_diagnostics.py
mpc_diag_path = (project_root / "debug" / "mpc_diagnostics.py").resolve()
spec = importlib.util.spec_from_file_location("debug.mpc_diagnostics", mpc_diag_path)
mpc_diagnostics = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mpc_diagnostics)
MPCDiagnostics = mpc_diagnostics.MPCDiagnostics

config = SimulationConfig(
    model_path='Mujoco sim/robot_model.xml',
    robot_params_path='config/simulation/robot_params.yaml',
    mpc_params_path='config/simulation/mpc_params.yaml',
    estimator_params_path='config/simulation/estimator_params.yaml',
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
diag.plot_all(result, save_dir="test_and_debug_output/quick_check", show=True)
print("\nPlots saved to test_and_debug_output/quick_check/")

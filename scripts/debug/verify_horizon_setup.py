"""Diagnostic script to verify MPC horizon configuration.

Verifies that:
1. Terminal cost is properly disabled/enabled
2. Horizon length is correctly set
3. Solver is using expected configuration
"""

import sys
from pathlib import Path
import tempfile
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import yaml
from simulation import SimulationConfig, MPCSimulation
from debug import MPCDiagnostics

print("="*80)
print("MPC HORIZON CONFIGURATION VERIFICATION")
print("="*80)

# Test with different horizon lengths
test_horizons = [10, 40, 160]

for horizon in test_horizons:
    print(f"\n{'='*60}")
    print(f"Testing Horizon: {horizon} steps")
    print(f"{'='*60}")

    # Create temp config with NO terminal cost
    with open('config/simulation/mpc_params.yaml', 'r') as f:
        mpc_params = yaml.safe_load(f)

    mpc_params['prediction_horizon_steps'] = horizon
    mpc_params['use_terminal_cost_dare'] = False
    mpc_params['terminal_cost_scale'] = 0.001

    temp_dir = Path(tempfile.mkdtemp(prefix="verify_horizon_"))
    temp_path = temp_dir / f'verify_h{horizon}.yaml'
    with open(temp_path, 'w') as f:
        yaml.dump(mpc_params, f)

    # Create simulation
    config = SimulationConfig(
        model_path='Mujoco sim/robot_model.xml',
        robot_params_path='config/simulation/robot_params.yaml',
        mpc_params_path=temp_path,
        estimator_params_path='config/simulation/estimator_params.yaml',
    )

    diag = MPCDiagnostics(config)

    # Check what was actually loaded
    print(f"\n  Loaded Configuration:")
    print(f"    Horizon: {diag.mpc_config.prediction_horizon_steps} steps")
    print(f"    use_terminal_cost_dare: {diag.mpc_config.use_terminal_cost_dare}")
    print(f"    terminal_cost_scale: {diag.mpc_config.terminal_cost_scale}")

    # Access the MPC solver (need to create controller first)
    sim = diag.simulation
    controller = sim._build_controller()
    solver = controller._mpc_solver

    print(f"\n  MPC Solver Properties:")
    print(f"    Actual horizon: {solver.prediction_horizon_steps}")

    # Check terminal cost matrix
    P = solver._terminal_cost
    Q = solver._state_cost

    print(f"\n  Cost Matrices:")
    print(f"    Q (state cost) norm: {np.linalg.norm(Q):.6f}")
    print(f"    P (terminal cost) norm: {np.linalg.norm(P):.6f}")
    print(f"    P/Q ratio: {np.linalg.norm(P) / np.linalg.norm(Q):.6f}")

    # Check if P is approximately scale * Q (what it should be if DARE is disabled)
    # or if P is much larger (DARE solution)
    if np.linalg.norm(P) > 2 * np.linalg.norm(Q):
        print(f"    ⚠️  WARNING: P >> Q, DARE solution likely being used!")
        print(f"    ⚠️  Terminal cost NOT disabled properly!")
    elif np.linalg.norm(P) < 0.01 * np.linalg.norm(Q):
        print(f"    ✓ Terminal cost appears to be scaled down significantly")
    else:
        print(f"    ? Unclear if terminal cost is properly disabled")

    # Check solver type
    print(f"\n  Solver Configuration:")
    print(f"    Requested solver: {diag.mpc_config.solver_name}")
    print(f"    ⚠️  NOTE: Code HARDCODES IPOPT regardless of config!")

    import os
    os.remove(temp_path)

print("\n" + "="*80)
print("SUMMARY OF FINDINGS:")
print("="*80)
print("\n1. Terminal Cost Issue:")
print("   The code ALWAYS computes P via DARE (lines 205-209 in mpc_simulation.py)")
print("   The 'use_terminal_cost_dare' flag is IGNORED!")
print("   It only applies 'terminal_cost_scale' to reduce P")
print("\n2. Solver Issue:")
print("   The code HARDCODES 'ipopt' solver (line 285 in linear_mpc_solver.py)")
print("   The 'solver_name' config parameter is IGNORED!")
print("\n3. Impact on Results:")
print("   With terminal_cost_scale=0.001, you get 0.001*P_dare")
print("   This is small but NOT zero - still influences optimization!")
print("   Longer horizons may have numerical issues with IPOPT")
print("="*80)

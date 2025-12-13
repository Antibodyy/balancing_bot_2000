"""Extract linearized dynamics from MuJoCo and compare."""

import numpy as np
import mujoco
from robot_dynamics import RobotParameters, linearize_at_equilibrium, compute_equilibrium_state

# Load
model = mujoco.MjModel.from_xml_path('mujoco_sim/robot_model.xml')
data = mujoco.MjData(model)
params = RobotParameters.from_yaml('config/simulation/robot_params.yaml')

# Set to equilibrium
data.qpos[:] = 0
data.qvel[:] = 0
data.ctrl[:] = 0

# Compute MuJoCo's mass matrix
mujoco.mj_forward(model, data)
M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M, data.qM)

print("=" * 70)
print("MUJOCO MASS MATRIX")
print("=" * 70)
print("\nFull mass matrix M (4x4 for [x, pitch, wheel_r, wheel_l]):")
print(M)

# Extract the 2x2 submatrix for [x, pitch]
M_xtheta = M[:2, :2]
print("\nRelevant submatrix for [x, pitch]:")
print(M_xtheta)
print(f"\n  M[0,0] (M_eff for x):     {M_xtheta[0,0]:.6f} kg")
print(f"  M[0,1] (coupling):         {M_xtheta[0,1]:.6f}")
print(f"  M[1,0] (coupling):         {M_xtheta[1,0]:.6f}")
print(f"  M[1,1] (I_eff for pitch): {M_xtheta[1,1]:.6f} kg⋅m²")

# Compare with our parameters
print("\n" + "=" * 70)
print("OUR PARAMETERS")
print("=" * 70)
m = params.body_mass_kg
l = params.com_distance_m
print(f"\n  M_eff (from params): {params.effective_mass_kg:.6f} kg")
print(f"  Coupling (m*l):      {m*l:.6f}")
print(f"  I_eff (from params): {params.effective_pitch_inertia_kg_m2:.6f} kg⋅m²")

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"\nM_eff:")
print(f"  MuJoCo:  {M_xtheta[0,0]:.6f}")
print(f"  Ours:    {params.effective_mass_kg:.6f}")
print(f"  Ratio:   {M_xtheta[0,0]/params.effective_mass_kg:.3f}x")

print(f"\nCoupling (m*l):")
print(f"  MuJoCo:  {M_xtheta[0,1]:.6f}")
print(f"  Ours:    {m*l:.6f}")
print(f"  Ratio:   {M_xtheta[0,1]/(m*l):.3f}x")

print(f"\nI_eff:")
print(f"  MuJoCo:  {M_xtheta[1,1]:.6f}")
print(f"  Ours:    {params.effective_pitch_inertia_kg_m2:.6f}")
print(f"  Ratio:   {M_xtheta[1,1]/params.effective_pitch_inertia_kg_m2:.3f}x")

if abs(M_xtheta[1,1]/params.effective_pitch_inertia_kg_m2 - 1.0) > 0.1:
    print(f"\n⚠️  PITCH INERTIA MISMATCH > 10%!")
    print(f"This explains why the dynamics don't match.")

# Now let's compute what I_eff should be to match MuJoCo
I_body_needed = M_xtheta[1,1] - m * l**2
print(f"\n" + "=" * 70)
print("CORRECTION NEEDED")
print("=" * 70)
print(f"\nTo match MuJoCo, we need:")
print(f"  I_body = M_mujoco[1,1] - m*l²")
print(f"  I_body = {M_xtheta[1,1]:.6f} - {m}*{l}**2")
print(f"  I_body = {I_body_needed:.6f} kg⋅m²")
print(f"\nCurrent value: {params.body_pitch_inertia_kg_m2:.6f} kg⋅m²")
print(f"Difference: {I_body_needed - params.body_pitch_inertia_kg_m2:.6f} kg⋅m²")

print("\n" + "=" * 70)

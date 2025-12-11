"""
LQR using the SAME dynamics and parameters as the MPC.

State order (identical to MPC):
    x = [x, theta, psi, dx, dtheta, dpsi]
    u = [tau_L, tau_R]

Linearization point:
    Uses compute_equilibrium_state(params, desired_velocity_mps=0.0).
    This respects the chosen YAML's `ground_slope_rad`.

Default:
    Continuous-time LQR pip (CARE). To compare with MPC's ZOH-discrete model,
    set MPC_SAMPLING_PERIOD_S to your Ts for discrete-time LQR (DARE).
"""

from __future__ import annotations

import os
import numpy as np
import scipy.linalg
import casadi as ca
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Shared model + params with MPC
from robot_dynamics.parameters import (
    RobotParameters,
    STATE_DIMENSION,
    CONTROL_DIMENSION,
)
from robot_dynamics.continuous_dynamics import (
    build_dynamics_model,
    compute_equilibrium_state,
)

# --- Discretizer import name varies across branches; handle both ---
try:
    from robot_dynamics.discretization import (
        discretize_linear_system as _discretize,
        DiscreteDynamics,
    )
except ImportError:
    from robot_dynamics.discretization import (
        discretize_linear_dynamics as _discretize,
        DiscreteDynamics,
    )

# Config

# Candidate locations for the robot params YAML
PARAMS_CANDIDATES = [
    "config/robot_params.yaml",
    "config/simulation/robot_params.yaml",
    "config/hardware/robot_params.yaml",
]

# LQR weights (tune as desired; these are not part of "dynamics")
Q_diag = np.array([1.0, 500.0, 1.0, 0.1, 10.0, 0.1], dtype=float)
R_diag = np.array([0.1, 0.1], dtype=float)

# Set to your MPC sampling time (e.g., 0.02) to run DISCRETE LQR; None = CONTINUOUS
MPC_SAMPLING_PERIOD_S: float | None = 0.02  # if on older Python, use Optional[float] = None

# Helpers

def resolve_params_yaml() -> str:
    """Return the first existing params YAML from candidates, or raise with guidance."""
    for p in PARAMS_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find a robot params YAML in any of these paths:\n"
        + "\n".join(f"  - {p}" for p in PARAMS_CANDIDATES)
        + f"\n\nCWD: {os.getcwd()}\n"
        "Fix by either:\n"
        "  • copying/creating config/robot_params.yaml, or\n"
        "  • placing your file in config/simulation/ or config/hardware/, or\n"
        "  • editing PARAMS_CANDIDATES in this script to the correct path."
    )

# Build linear model A,B that MATCH MPC

def linearize_shared_dynamics(params: RobotParameters):
    """Return continuous-time (A,B) and equilibrium (x_eq, u_eq) identical to MPC's source."""
    # Same CasADi dynamics function MPC uses: xdot = f(x,u; params)
    f = build_dynamics_model(params)  # casadi.Function

    # Symbolic variables (match MPC dimensions)
    x_sym = ca.SX.sym("x", STATE_DIMENSION)
    u_sym = ca.SX.sym("u", CONTROL_DIMENSION)

    # Jacobians
    xdot_sym = f(x_sym, u_sym)
    A_sym = ca.jacobian(xdot_sym, x_sym)
    B_sym = ca.jacobian(xdot_sym, u_sym)
    jac_fun = ca.Function("jac_fun", [x_sym, u_sym], [A_sym, B_sym])

    # Linearization point from params (includes ground_slope_rad)
    x_eq, u_eq = compute_equilibrium_state(params, desired_velocity_mps=0.0)

    # Evaluate to NumPy
    A_num, B_num = jac_fun(x_eq, u_eq)
    A = np.array(A_num)
    B = np.array(B_num)
    return A, B, x_eq, u_eq

# LQR solvers

def solve_continuous_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Continuous-time LQR via CARE."""
    P = scipy.linalg.solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K, P

def solve_discrete_lqr(A_d: np.ndarray, B_d: np.ndarray, Q: np.ndarray, R: np.ndarray):
    """Discrete-time LQR via DARE."""
    P = scipy.linalg.solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.inv(B_d.T @ P @ B_d + R) @ (B_d.T @ P @ A_d)
    return K, P



def main():
    # Locate and load params
    params_yaml = resolve_params_yaml()
    params = RobotParameters.from_yaml(params_yaml)

    # Get (A,B) and equilibrium from the shared model
    A, B, x_eq, u_eq = linearize_shared_dynamics(params)

    # Build Q,R
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)

    if MPC_SAMPLING_PERIOD_S is None:
        # Continuous-time LQR
        K, P = solve_continuous_lqr(A, B, Q, R)
        mode = "continuous"
    else:
        # Discrete-time LQR using the SAME ZOH discretizer as MPC
        disc: DiscreteDynamics = _discretize(A, B, MPC_SAMPLING_PERIOD_S)
        A_d = disc.state_matrix_discrete
        B_d = disc.control_matrix_discrete
        K, P = solve_discrete_lqr(A_d, B_d, Q, R)
        mode = f"discrete (Ts={MPC_SAMPLING_PERIOD_S:.6f}s)"

    # results display
    np.set_printoptions(precision=5, suppress=True)
    print("=== LQR using MPC-shared dynamics ===")
    print(f"Params YAML: {params_yaml}")
    print(f"Mode: {mode}")
    print("A shape:", A.shape, "B shape:", B.shape)
    print("x_eq:", x_eq)
    print("u_eq:", u_eq)
    print("K (state feedback gain):\n", K)

    return {"A": A, "B": B, "x_eq": x_eq, "u_eq": u_eq, "K": K, "Q": Q, "R": R}

if __name__ == "__main__":
    main()

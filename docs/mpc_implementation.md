# MPC Implementation for Self-Balancing Robot

This document describes the Model Predictive Control (MPC) implementation for a two-wheel self-balancing robot (Pololu Balboa 32U4 + Raspberry Pi 4).

## Overview

The implementation provides a complete control pipeline from sensor measurements to wheel torque commands:

```
Sensors → State Estimation → Reference Generation → MPC Solver → Control Output
```

**Key Specifications:**
- Control rate: 50 Hz (20ms period)
- MPC solve time: ~8-12ms (warm-started), ~200ms (cold start)
- State dimension: 6 (position, pitch, yaw, velocity, pitch rate, yaw rate)
- Control dimension: 2 (left/right wheel torques)
- Prediction horizon: 20 steps (400ms lookahead)

## Architecture

### Module Structure

```
balancing_bot_2000/
├── mpc/                          # MPC core module
│   ├── __init__.py               # Public API exports
│   ├── config.py                 # MPCConfig dataclass
│   ├── cost_matrices.py          # Q, R, P matrix construction
│   ├── constraints.py            # State/input bounds
│   ├── linear_mpc_solver.py      # CasADi Opti solver
│   ├── reference_generator.py    # Trajectory generation
│   └── _internal/
│       └── validation.py         # Input validation utilities
│
├── state_estimation/             # State estimation module
│   ├── __init__.py
│   ├── config.py                 # EstimatorConfig dataclass
│   ├── complementary_filter.py   # Pitch estimator
│   └── _internal/
│       └── imu_fusion.py         # Sensor fusion math
│
├── control_pipeline/             # Control orchestration
│   ├── __init__.py
│   ├── controller.py             # BalanceController class
│   └── timing.py                 # Performance monitoring
│
├── simulation/                   # MuJoCo validation (optional)
│   ├── __init__.py
│   └── mpc_simulation.py         # Simulation harness
│
├── robot_dynamics/               # Dynamics (pre-existing)
│   ├── continuous_dynamics.py    # CasADi symbolic dynamics
│   ├── linearization.py          # Jacobian computation
│   ├── discretization.py         # ZOH discretization
│   └── parameters.py             # Physical parameters
│
└── config/                       # Configuration files
    ├── robot_params.yaml         # Physical parameters
    ├── mpc_params.yaml           # MPC tuning
    └── estimator_params.yaml     # Estimator settings
```

## MPC Formulation

### Optimization Problem

The MPC solves a quadratic program (QP) at each control step:

```
minimize   Σ_{k=0}^{N-1} [(x_k - x_ref)ᵀ Q (x_k - x_ref) + u_kᵀ R u_k]
           + (x_N - x_ref)ᵀ P (x_N - x_ref)

subject to x_{k+1} = A_d x_k + B_d u_k    (discrete dynamics)
           x_0 = x_current                 (initial condition)
           x_min ≤ x_k ≤ x_max            (state bounds)
           u_min ≤ u_k ≤ u_max            (control bounds)
```

Where:
- **N = 20**: Prediction horizon steps
- **Q**: State cost matrix (diagonal: [1, 100, 10, 1, 10, 1])
- **R**: Control cost matrix (diagonal: [0.1, 0.1])
- **P**: Terminal cost from Discrete Algebraic Riccati Equation (DARE)

### State Vector

```
x = [x, θ, ψ, ẋ, θ̇, ψ̇]ᵀ
```

| Index | Variable | Description | Unit |
|-------|----------|-------------|------|
| 0 | x | Forward position | m |
| 1 | θ | Pitch angle (from vertical) | rad |
| 2 | ψ | Yaw/heading angle | rad |
| 3 | ẋ | Forward velocity | m/s |
| 4 | θ̇ | Pitch rate | rad/s |
| 5 | ψ̇ | Yaw rate | rad/s |

### Control Vector

```
u = [τ_L, τ_R]ᵀ
```

| Index | Variable | Description | Limit |
|-------|----------|-------------|-------|
| 0 | τ_L | Left wheel torque | ±0.25 N·m |
| 1 | τ_R | Right wheel torque | ±0.25 N·m |

### Constraints

**State Constraints:**
- Pitch: |θ| ≤ 30° (0.524 rad)
- Pitch rate: |θ̇| ≤ 5 rad/s

**Input Constraints:**
- Torque: |τ| ≤ 0.25 N·m (motor stall limit)

## Key Components

### LinearMPCSolver

The main MPC solver uses CasADi's Opti interface with IPOPT backend:

```python
from mpc import LinearMPCSolver, MPCConfig

solver = LinearMPCSolver(
    prediction_horizon_steps=20,
    discrete_dynamics=discrete,
    state_cost=Q,
    control_cost=R,
    terminal_cost=P,
    state_constraints=state_constraints,
    input_constraints=input_constraints,
)

solution = solver.solve(current_state, reference_trajectory)
torque = solution.optimal_control  # [τ_L, τ_R]
```

**Features:**
- Warm-starting for fast subsequent solves
- Time-varying reference trajectory support
- Full predicted trajectory output for diagnostics

### ComplementaryFilter

Estimates pitch angle from IMU data:

```python
from state_estimation import ComplementaryFilter

filter = ComplementaryFilter(
    time_constant_s=0.1,
    sampling_period_s=0.02,
)

pitch = filter.update(imu_reading)
```

**Fusion Formula:**
```
θ_est = α·θ_gyro + (1-α)·θ_accel
α = τ / (τ + Ts)
```

Where τ is the time constant (how much to trust gyroscope vs accelerometer).

### ReferenceGenerator

Generates reference trajectories for different modes:

```python
from mpc import ReferenceGenerator, ReferenceCommand, ReferenceMode

ref_gen = ReferenceGenerator(
    sampling_period_s=0.02,
    prediction_horizon_steps=20,
)

# Balance mode (stay upright stationary)
ref = ref_gen.generate(ReferenceCommand(mode=ReferenceMode.BALANCE))

# Velocity tracking
ref = ref_gen.generate(ReferenceCommand(
    mode=ReferenceMode.VELOCITY,
    velocity_mps=0.2,
    yaw_rate_radps=0.1,
))

# Position tracking
ref = ref_gen.generate(
    ReferenceCommand(
        mode=ReferenceMode.POSITION,
        target_position_m=1.0,
        target_heading_rad=0.0,
    ),
    current_state=state,
)
```

### BalanceController

Main orchestrator that ties everything together:

```python
from control_pipeline import BalanceController, SensorData

controller = BalanceController(
    mpc_solver=solver,
    state_estimator=estimator,
    reference_generator=ref_gen,
    sampling_period_s=0.02,
    wheel_radius_m=0.05,
    track_width_m=0.3,
)

output = controller.step(sensor_data, reference_command)
# output.torque_left_nm, output.torque_right_nm
```

## Configuration

### MPC Parameters (`config/simulation/mpc_params.yaml`)

```yaml
prediction_horizon_steps: 20
sampling_period_s: 0.02

state_cost_diagonal: [1.0, 100.0, 10.0, 1.0, 10.0, 1.0]
control_cost_diagonal: [0.1, 0.1]
use_terminal_cost_dare: true

pitch_limit_rad: 0.524      # 30°
pitch_rate_limit_radps: 5.0
control_limit_nm: 0.25

solver_name: 'osqp'         # Note: Actually uses IPOPT
warm_start_enabled: true
```

### Estimator Parameters (`config/simulation/estimator_params.yaml`)

```yaml
complementary_filter_time_constant_s: 0.1
sampling_period_s: 0.02
```

## Performance

### Timing Results

| Metric | Value |
|--------|-------|
| First solve (cold) | ~200ms |
| Subsequent solves (warm) | ~8-12ms |
| State estimation | <0.1ms |
| Reference generation | <0.1ms |
| Total loop time | ~10-15ms |

### Test Coverage

| Test File | Tests | Status |
|-----------|-------|--------|
| test_cost_matrices.py | 10 | Passing |
| test_constraints.py | 13 | Passing |
| test_reference_generator.py | 17 | Passing |
| test_state_estimation.py | 21 | Passing |
| test_mpc_solver.py | 16 | Passing |
| test_control_pipeline.py | 13 | Passing |
| test_parameters.py | 7 | Passing |
| test_dynamics.py | 24 | Passing |
| test_mpc_simulation.py | 17 | 14 pass, 3 skip* |
| **Total** | **138** | **135 pass, 3 skip** |

*Perturbation tests skip if MPC solve time is too slow for recovery on current hardware.

## MuJoCo Simulation

When MuJoCo is installed, the simulation module provides validation:

```python
from simulation import MPCSimulation, SimulationConfig

sim = MPCSimulation(SimulationConfig(
    model_path='Mujoco sim/robot_model.xml',
    duration_s=30.0,
))

# Run headless simulation
result = sim.run(
    initial_pitch_rad=np.deg2rad(5),
)

# Run with interactive viewer
result = sim.run_with_viewer()

# Print results
print(f"Success: {result.success}")
print(f"Mean solve time: {result.mean_solve_time_ms:.1f}ms")
print(f"Max solve time: {result.max_solve_time_ms:.1f}ms")
```

### Validation Tests

1. **Balance Stabilization**: Hold upright for 30+ seconds
2. **Disturbance Rejection**: Recover from 5N push
3. **Trajectory Tracking**: Track velocity reference
4. **Timing Verification**: Solve time < 15ms

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| MPC Type | Linear MPC | QP solves in 1-15ms (vs 10-50ms for NLP) |
| Solver | CasADi Opti + IPOPT | Robust, warm-start capable, open-source |
| Terminal Cost | scipy DARE | Optimal infinite-horizon approximation |
| State Estimation | Complementary filter | Simple, <0.1ms, sufficient for pitch |
| Linearization | At equilibrium | Re-linearize possible for large deviations |

## Future Work

1. **Extended Kalman Filter (EKF)**: Full 6-state estimation with encoder fusion
2. **Hardware Interface**: USB/UART communication with Balboa 32U4
3. **Successive Linearization**: Re-linearize at current state for large deviations
4. **Real-time Scheduling**: Priority threads, deadline management


## References

1. Rawlings, J. B., & Mayne, D. Q. (2009). Model Predictive Control: Theory and Design
2. CasADi Documentation: https://web.casadi.org/docs/
3. MuJoCo Documentation: https://mujoco.readthedocs.io/

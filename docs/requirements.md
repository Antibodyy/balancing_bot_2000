# Requirements Document: MPC-Based Self-Balancing Robot

## Project Overview
Development of Model Predictive Control (MPC) for a two-wheel self-balancing robot with independent motor actuation. The system combines dynamic balance stabilization with trajectory tracking and heading control.

**Course**: ME C231A/EE C220B - Model Predictive Control  
**Weight**: 20% of final grade  
**Hardware Platform**: Pololu Balboa 32U4 + Raspberry Pi 4 (4GB)

---

## 1. Hardware Specifications

### 1.1 Primary Hardware
- **Robot Platform**: Pololu Balboa 32U4 Balancing Robot Kit (Product #3575)
  - Onboard ATmega32U4 microcontroller
  - Two 50:1 HP micro metal gearmotors with encoders
  - LSM6DS33 6-axis IMU (accelerometer + gyroscope)
  - 80x10mm wheels
  - TB6612FNG dual motor driver
  
- **Computation Unit**: Raspberry Pi 4 Model B (4GB RAM)
  - Quad-core ARM Cortex-A72 @ 1.5 GHz
  - Communication via USB or UART with Balboa 32U4

### 1.2 Sensor Suite
- **Inertial Measurement**: LSM6DS33 IMU
  - 3-axis accelerometer (±2g, ±4g, ±8g, ±16g)
  - 3-axis gyroscope (±125°/s to ±2000°/s)
  - I²C interface to 32U4
  
- **Wheel Encoders**: Magnetic encoders on both motors
  - 12 counts per revolution of motor shaft
  - 600 counts per revolution of wheel (with 75:1 gearing)
  - Quadrature encoding for direction sensing

### 1.3 Actuation
- **Motors**: Two independently controlled DC motors
  - Voltage: 6V nominal (max 6V)
  - Speed: ~150 RPM at 6V (no load)
  - Torque: ~35 oz-in (0.25 N·m) at 6V stall
  - PWM control via TB6612FNG driver

---

## 2. System Architecture

### 2.1 Hardware Architecture
```
┌─────────────────────────────────────┐
│      Raspberry Pi 4                 │
│  - MPC optimization (high-level)    │
│  - State estimation                 │
│  - Trajectory planning              │
└──────────────┬──────────────────────┘
               │ USB/UART (bidirectional)
               │ ↓ Control commands (u₁, u₂)
               │ ↑ Sensor data (IMU, encoders)
               ▼
┌─────────────────────────────────────┐
│      Balboa 32U4                    │
│  - Low-level motor control          │
│  - Encoder reading                  │
│  - IMU data acquisition             │
│  - Safety monitoring                │
└─────────────────────────────────────┘
```

**Assumption**: Communication latency between Pi and 32U4 is <10ms for real-time control at 50-100 Hz.

### 2.2 Software Architecture
- **High-Level Controller** (Raspberry Pi):
  - Python 3.9+S
  - MPC solver: CasADi
  - State estimator: EKF or complementary filter
  
- **Low-Level Controller** (32U4):
  - Arduino-compatible C/C++
  - Motor PWM generation
  - Encoder interrupt handling
  - IMU data filtering

---

## 3. Control System Requirements

### 3.1 State Space Model
The robot is modeled as a coupled system with:

**States** (n =  6):
- θ: pitch angle (body tilt)
- θ̇_dot: pitch angular velocity
- x, y: position in ground frame
- ψ: heading angle
- ψ_dot: yaw rate

**Control Inputs** (m = 2):
- u_1: left motor voltage/torque
- u_2: right motor voltage/torque

**Dynamics**: Nonlinear coupled differential equations
- Inverted pendulum dynamics for pitch (θ, θ̇)
- Differential drive kinematics for (x, y, ψ)
- No-slip wheel constraints

**Assumption**: Quasi-static approximation for horizontal motion; dynamic coupling through pitch dynamics.

### 3.2 MPC Formulation

**Objective**: Minimize quadratic cost function
```
J = Σ(k=0 to N-1) [||x(k) - x_ref(k)||²_Q + ||u(k)||²_R] + ||x(N) - x_ref(N)||²_P
```

**Constraints**:
- State constraints:
  - |θ| ≤ θ_max (e.g., ±30°)
  - |θ̇| ≤ θ̇_max
  
- Input constraints:
  - |u₁|, |u₂| ≤ u_max (voltage/torque limits)
  - Optional: Δu constraints for smoothness
  
- Safety constraints:
  - Emergency stop if |θ| > θ_critical (e.g., 45°)

**Prediction Horizon**: N = 10-30 steps  
**Sampling Time**: T_s = 65 ms (≈16 Hz control loop)

**Assumption**: Linearized model around vertical equilibrium (θ = 0) or successive linearization for nonlinear MPC.

### 3.3 Control Objectives

**Primary Objectives**:
1. **Balance Stabilization**: Maintain |θ| < 5° during operation
2. **Position Tracking**: Follow desired (x, y) trajectory with error < 10 cm
3. **Heading Control**: Track desired heading ψ with error < 5°

**Secondary Objectives**:
4. Smooth control inputs (minimize Δu)
5. Energy efficiency
6. Disturbance rejection (pushes, ground irregularities)

### 3.4 Performance Metrics
- **Settling Time**: <2 seconds after disturbance
- **Steady-State Error**: <2° for pitch, <5 cm for position
- **Control Frequency**: ≥16 Hz sustained
- **Computation Time**: MPC solve time <15 ms per iteration

---

## 4. Software Requirements

### 4.1 Development Environment
- **Raspberry Pi**:
  - OS: Raspberry Pi OS (Debian-based)
  - Python 3.9+ with numpy, scipy, matplotlib
  - MPC solver: CasADi
  - 
  ommunication: pySerial or USB HID
  
- **Balboa 32U4**:
  - Arduino IDE 1.8.x or 2.x
  - Pololu libraries: Balboa32U4, LSM6, Encoder

### 4.2 Key Libraries and Tools
| Component | Tool/Library | Purpose |
|-----------|-------------|---------|
| MPC Solver | CasADi | Real-time optimization |
| Symbolic Math | CasADi | Model derivatives, code generation |
| State Estimation | FilterPy, custom EKF | Sensor fusion |
| Communication | pySerial | Pi ↔ 32U4 interface |
| Visualization | matplotlib, real-time plotting | Debugging and analysis |
| Motor Control | Arduino PWM | Low-level actuation |

### 4.3 Data Logging
- Log states, controls, computation time at control frequency
- Export to CSV for offline analysis
- Real-time plotting for debugging

---

## 5. Project Deliverables

### 5.1 Documentation
1. **Mathematical Model**: 
   - Nonlinear state-space equations
   - Linearization approach
   - Parameter identification results

2. **MPC Formulation**:
   - Cost function design
   - Constraint specification
   - Horizon selection rationale

3. **Implementation Report**:
   - System architecture
   - Software components
   - Calibration procedures

### 5.2 Code Deliverables
- MPC controller (Python)
- Low-level firmware (C/C++ for 32U4)
- State estimator
- Communication protocol
- Simulation code (optional)

### 5.3 Experimental Results
- Balance stabilization tests
- Trajectory tracking demonstrations
- Disturbance rejection experiments
- Performance metrics vs. requirements
- Video documentation

## 6. Safety and Risk Management

### 6.1 Safety Features
- **Emergency Stop**: Hardware button on 32U4
- **Tilt Threshold**: Automatic motor cutoff if |θ| > 45°
- **Communication Timeout**: Stop motors if no command for >100ms
- **Battery Monitoring**: Low voltage cutoff

### 6.2 Risk Mitigation
- **Fall Protection**: Soft foam padding on robot edges
- **Testing Area**: Clear, flat surface away from obstacles
- **Incremental Testing**: Start with small perturbations
- **Code Validation**: Extensive simulation before hardware tests

---

## 7. Assumptions and Limitations

1. Battery voltage remains approximately constant during tests
2. IMU and encoder measurements are sufficiently accurate after filtering
3. Communication latency is negligible compared to control period
4. Motor dynamics are fast relative to body dynamics
5. Small angle approximation valid for linearization (|θ| < 20°)

## 8. Success Criteria

The project is considered successful if:
1. ✓ Robot maintains balance autonomously for ≥30 seconds
2. ✓ Recovers from moderate manual pushes (≤5 N)
3. ✓ Tracks spline trajectory within ±10 cm
4. ✓ MPC runs at ≥16 Hz on Raspberry Pi
5. ✓ All safety features function correctly
6. ✓ Complete documentation and code submitted

---

## 9. References

- Pololu Balboa 32U4 User's Guide: https://www.pololu.com/docs/0J70
- Course materials: MPC textbook (Borrelli et al.)
- acados documentation: https://docs.acados.org/
- Related work: Self-balancing robots with MPC (literature survey)

---

**Document Version**: 1.0  
**Last Updated**: December 2024  
**Status**: Initial Requirements

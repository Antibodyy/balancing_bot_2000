## Testing Guide

Comprehensive guide for testing and debugging the balancing bot MPC controller.

## Quick Start

### Setup

**Important:** All scripts require the project root in PYTHONPATH. Either:

```bash
# Option 1: Set for each command
PYTHONPATH=. python3 script_name.py

# Option 2: Export for your session
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Running Tests

```bash
# All unit tests (pytest doesn't need PYTHONPATH)
python3 -m pytest tests/

# Specific test category
python3 -m pytest tests/test_dynamics.py

# Regression tests
python3 -m pytest tests/regression/

# With coverage
python3 -m pytest --cov=. tests/
```

### Quick Diagnostics

```bash
# Fastest diagnostic check (1 second simulation)
PYTHONPATH=. python3 scripts/debug/quick_check.py

# Test multiple perturbation sizes
PYTHONPATH=. python3 scripts/debug/debug_perturbation.py

# Visual confirmation with MuJoCo viewer
PYTHONPATH=. python3 scripts/debug/debug_perturbation.py --viewer --perturbation 3
```

## Test Categories

### Unit Tests (tests/)

**Purpose:** Verify individual components work correctly in isolation.

**Test files:**
- `test_parameters.py` - Parameter loading and validation
- `test_dynamics.py` - Continuous dynamics equations
- `test_linearization.py` - Linearization at equilibrium
- `test_discretization.py` - Discretization accuracy
- `test_cost_matrices.py` - Q, R, P matrix generation
- `test_constraints.py` - Control and state constraints
- `test_mpc_solver.py` - QP solver correctness
- `test_state_estimation.py` - Complementary filter
- `test_reference_generator.py` - Reference trajectory generation
- `test_control_pipeline.py` - Full control loop
- `test_mpc_simulation.py` - Simulation integration (requires MuJoCo)

**Success criteria:**
- All assertions pass
- No runtime errors

**Test files:**
- `test_parameters.py` - Parameter loading and validation
- `test_dynamics.py` - 

### Regression Tests (tests/regression/)

**Purpose:** Verify the system maintains correct behavior after code changes.

**When to run:**
- Before merging PRs
- After significant changes
- Weekly regression checks

**Test files:**

#### test_perturbation_recovery.py
Tests recovery from various perturbation magnitudes (0°, 0.5°, 1°, 2°, 3°).

**Verifies:**
- Robot survives 2+ seconds
- Velocity estimation error < 0.01 m/s
- Final pitch < 10°
- Control saturation < 50%

#### test_equilibrium_stability.py
Tests long-duration stability (10 seconds) from various initial conditions.

**Verifies:**
- Pitch drift < 1° over 10 seconds
- Mean solve time < 20ms
- Zero deadline violations
- Velocity estimation accuracy maintained

#### test_dynamics_validation.py
Validates dynamics model matches MuJoCo (requires MuJoCo installed).

**Verifies:**
- Mass matrix accuracy < 1% error
- Geometry parameters match
- Equilibrium computation correct
- Linearization has expected stability properties

### Debug Scripts (scripts/debug/)

**Purpose:** Interactive diagnostic tools for troubleshooting control issues.

**When to use:** Investigating failures, tuning controller, analyzing behavior

#### quick_check.py
- **Duration:** 1 second
- **Use when:** Quick sanity check after code changes
- **Output:** 4 diagnostic plots in `test_and_debug_output/quick_check/`

```bash
PYTHONPATH=. python3 scripts/debug/quick_check.py
```

#### debug_perturbation.py
- **Duration:** 2 seconds per test (or 10s with --viewer)
- **Use when:** Analyzing recovery behavior
- **Options:**
  - `--viewer` - Show MuJoCo visualization
  - `--perturbation DEGREES` - Test specific angle

```bash
# Headless batch test
PYTHONPATH=. python3 scripts/debug/debug_perturbation.py

# Interactive single test
PYTHONPATH=. python3 scripts/debug/debug_perturbation.py --viewer --perturbation 3
```

#### debug_force_disturbance.py
- **Tests:** External force rejection (1N, 5N, 10N)
- **Use when:** Testing robustness to external pushes
- **Output:** Timeline plots showing disturbance response

```bash
PYTHONPATH=. python3 scripts/debug/debug_force_disturbance.py
```

#### debug_velocity_viewer.py
- **Use when:** Debugging state estimation issues
- **Shows:** Real-time velocity estimation monitoring

```bash
PYTHONPATH=. python3 scripts/debug/debug_velocity_viewer.py
```

#### debug_viewer_timing.py
- **Use when:** Investigating solver performance issues
- **Shows:** MPC performance profiling with visualization

```bash
PYTHONPATH=. python3 scripts/debug/debug_viewer_timing.py
```

#### debug_compare_fix.py
- **Use when:** Demonstrating before/after comparison
- **Shows:** Side-by-side comparison of two approaches

```bash
PYTHONPATH=. python3 scripts/debug/debug_compare_fix.py
```

### Validation Scripts (scripts/validation/)

**Purpose:** Verify analytical model matches MuJoCo simulation.

**When to use:**
- After parameter changes
- After dynamics model updates
- When investigating model mismatch

#### check_dynamics_match.py
Compares robot parameters to MuJoCo model geometry.

**Checks:**
- Mass values (body, wheels)
- Inertia tensors
- Geometric dimensions

```bash
PYTHONPATH=. python3 scripts/validation/check_dynamics_match.py
```

#### compare_mujoco_dynamics.py
Numerical comparison of state derivatives.

**Process:**
1. Set identical state in both systems
2. Apply identical control
3. Compare resulting accelerations

```bash
PYTHONPATH=. python3 scripts/validation/compare_mujoco_dynamics.py
```

#### extract_mujoco_matrices.py
Extracts mass matrix and dynamics from MuJoCo.

**Use when:** Need ground truth for validation

```bash
PYTHONPATH=. python3 scripts/validation/extract_mujoco_matrices.py
```

### Viewer Utilities (scripts/viewer/)

**Purpose:** Visual inspection tools.

#### view_robot.py
Display robot model in MuJoCo viewer.

**Use when:** Verifying URDF/XML model geometry

```bash
PYTHONPATH=. python3 scripts/viewer/view_robot.py
```

## Understanding Diagnostic Outputs

### Plot Types

#### state_comparison.png (6 panels)
Six-panel state comparison between true and estimated states.

**Key panels:**
- Panel 1: Position tracking
- Panel 2: Pitch tracking
- Panel 3: Yaw tracking
- Panel 4: Velocity ⭐ (blue=true, red=estimated)
- Panel 5: Pitch rate
- Panel 6: Yaw rate

**What to check:**
- Estimated states should track true states closely
- Large divergence indicates state estimation issues

#### control_analysis.png (4 panels)
Control command analysis.

**Panels:**
- Panel 1-2: Left and right motor torques
- Panel 3: Total torque
- Panel 4: Saturation indicator ⭐

**What to check:**
- High saturation (>80%) indicates underactuation
- Control should be smooth, not erratic

#### closed_loop_diagnosis.png (6 panels) ⭐ MOST USEFUL
Comprehensive closed-loop diagnostic.

**Panel 1:** Pitch tracking (true vs estimated)
**Panel 2:** Pitch rate tracking
**Panel 4:** Control direction scatter ⭐ CRITICAL
- Should show positive slope (inverted pendulum)
- Negative pitch → Negative torque
- Positive pitch → Positive torque

**Panel 5:** MPC solve time
- Should be < 20ms for 50Hz control
- Spikes indicate solver struggling

**Panel 6:** Prediction error
- One-step ahead error
- Should be < 0.1 rad (< 100 mrad)

#### prediction_accuracy.png
MPC prediction accuracy visualization.

- Blue = Actual trajectory
- Red = MPC predictions
- Predictions should follow actual closely

### Console Output

```
Step   Time   True θ      Est θ      τ_L      τ_R    τ_total    Solve
   0  0.000     2.8648     2.8648   0.1861   0.1861   0.3722     12.3
   1  0.020     2.7935     2.7577   0.1883   0.1883   0.3765      8.5
```

**Check for:**
- Sign consistency: Positive pitch → Positive torque
- Solve time: Should be < 20ms for 50Hz control
- Estimation accuracy: Est θ should track True θ

### Diagnostic Summary

```
============================================================
DIAGNOSTIC SUMMARY
============================================================

State Estimation:
  Max pitch error:      0.125 deg
  Mean pitch error:     0.058 deg
  Max pitch rate error: 2.345 deg/s

Control:
  Mean total torque:    0.3845 N⋅m
  Saturation fraction:  45.2%       # >80% indicates limits too low

Prediction Accuracy:
  Mean 1-step error:    12.345 mrad  # Should be < 50 mrad
  Max 1-step error:     45.678 mrad

Timing:
  Mean solve time:      9.2 ms       # Should be < 20ms
  Max solve time:       15.3 ms

Outcome:
  Result: SUCCESS (2.000s)
```

## Common Failure Patterns

### Pattern 1: Control Gains Too High

**Symptoms:**
- Oscillates wildly with growing amplitude
- High saturation (>90%)
- Overshoots equilibrium repeatedly

**Fix:** Reduce Q matrix weights in `config/simulation/mpc_params.yaml`

### Pattern 2: Model Mismatch

**Symptoms:**
- Predictions diverge from actual
- Large 1-step errors (>100 mrad)
- Control seems reasonable but doesn't work

**Fix:**
- Run `PYTHONPATH=. python3 scripts/validation/check_dynamics_match.py`
- Verify parameters match MuJoCo
- Check linearization accuracy

### Pattern 3: State Estimation Issues

**Symptoms:**
- Estimated states diverge from true states
- Control direction inconsistent
- Panel 4 in closed_loop_diagnosis shows scatter

**Fix:**
- Check complementary filter gains in `config/simulation/estimator_params.yaml`
- Verify sensor data is correct
- Run `python3 -m pytest tests/test_state_estimation.py`

### Pattern 4: Performance Issues

**Symptoms:**
- Solve time > 20ms frequently
- Deadline violations
- Falls in viewer mode but works headless

**Fix:**
- Reduce horizon in `config/simulation/mpc_params.yaml`
- Check for warm-start issues
- Optimize solver settings

### Pattern 5: Works But Needs Tuning

**Symptoms:**
- Oscillates but doesn't fall
- Settles slowly
- Low saturation (<30%)

**Fix:**
- Tune Q/R weights for faster response
- Increase control authority if saturation is low

## Troubleshooting Workflow

1. **Quick sanity check**
   ```bash
   PYTHONPATH=. python3 scripts/debug/quick_check.py
   ```

2. **Check diagnostic plots**
   - Open `test_and_debug_output/quick_check/closed_loop_diagnosis.png`
   - Verify control direction (Panel 4)
   - Check solve times (Panel 5)
   - Check prediction error (Panel 6)

3. **If control direction is wrong**
   - Verify dynamics signs in `robot_dynamics/`
   - Check control sign in `control_pipeline/controller.py`
   - Run `python3 -m pytest tests/test_mpc_solver.py`

4. **If model mismatch suspected**
   ```bash
   PYTHONPATH=. python3 scripts/validation/check_dynamics_match.py
   PYTHONPATH=. python3 scripts/validation/compare_mujoco_dynamics.py
   ```

5. **If performance issues**
   ```bash
   PYTHONPATH=. python3 scripts/debug/debug_viewer_timing.py
   ```

6. **If state estimation issues**
   - Check `state_comparison.png` for divergence
   - Run `python3 -m pytest tests/test_state_estimation.py`
   - Tune estimator parameters

## Development Workflow

### Before Committing
```bash
# Run unit tests
python3 -m pytest tests/

# Quick diagnostic check
PYTHONPATH=. python3 scripts/debug/quick_check.py
```

### Before Merging PR
```bash
# Full test suite
python3 -m pytest tests/ tests/regression/

# Full diagnostic suite
PYTHONPATH=. python3 scripts/debug/debug_perturbation.py
```

### After Dynamics Changes
```bash
# Verify model matches MuJoCo
PYTHONPATH=. python3 scripts/validation/check_dynamics_match.py

# Run dynamics tests
python3 -m pytest tests/test_dynamics.py tests/test_linearization.py
```

### Tuning Controller
1. Run `PYTHONPATH=. python3 scripts/debug/debug_perturbation.py --viewer`
2. Observe behavior visually
3. Modify `config/simulation/mpc_params.yaml`
4. Repeat until satisfactory
5. Run regression tests to verify

### Investigating Failures
1. Run `PYTHONPATH=. python3 scripts/debug/quick_check.py`
2. Check plots in `test_and_debug_output/quick_check/`
3. Identify failure mode from plots
4. Run targeted script (velocity_viewer, force_disturbance, etc.)
5. Fix issue
6. Verify with regression tests

## CI/CD Integration

### Recommended Pipeline

```yaml
test:
  stages:
    - unit_tests
    - regression_tests

  unit_tests:
    script:
      - python3 -m pytest tests/ --cov=.
    timeout: 5 minutes

  regression_tests:
    script:
      - python3 -m pytest tests/regression/
    timeout: 10 minutes
```

### Test Organization

- Unit tests should run quickly (< 60s total)
- Regression tests can be longer (< 10 minutes)
- Validation tests require MuJoCo and can be optional

## Files and Directories

### Core Test Files
```
tests/
├── test_parameters.py          # Parameter validation
├── test_dynamics.py            # Dynamics equations
├── test_linearization.py       # Linearization accuracy
├── test_discretization.py      # Discretization methods
├── test_cost_matrices.py       # MPC cost matrices
├── test_constraints.py         # State/control constraints
├── test_mpc_solver.py          # QP solver
├── test_state_estimation.py    # Complementary filter
├── test_reference_generator.py # Reference trajectories
├── test_control_pipeline.py    # Control loop integration
├── test_mpc_simulation.py      # MuJoCo simulation
└── regression/
    ├── test_perturbation_recovery.py   # Recovery tests
    ├── test_equilibrium_stability.py   # Long-term stability
    └── test_dynamics_validation.py     # Model validation
```

### Debug Infrastructure
```
debug/
├── __init__.py              # Package exports
├── mpc_diagnostics.py       # MPCDiagnostics class
└── plotting.py              # Plotting functions

test_and_debug_output/               # Generated plots (gitignored)
└── .gitignore
```

### Scripts
```
scripts/
├── debug/
│   ├── quick_check.py              # Fast diagnostic
│   ├── debug_perturbation.py       # Perturbation analysis
│   ├── debug_force_disturbance.py  # Force rejection
│   ├── debug_velocity_viewer.py    # Velocity monitoring
│   ├── debug_viewer_timing.py      # Performance profiling
│   └── debug_compare_fix.py        # Comparison tool
├── validation/
│   ├── check_dynamics_match.py     # Parameter validation
│   ├── compare_mujoco_dynamics.py  # Dynamics comparison
│   └── extract_mujoco_matrices.py  # Matrix extraction
└── viewer/
    └── view_robot.py               # Model visualization
```

## Tips and Best Practices

### Writing Tests
- Keep unit tests focused and fast
- Use fixtures for common setup
- Parametrize tests for multiple scenarios
- Add descriptive assertion messages

### Debugging
- Always start with `quick_check.py`
- Use viewer mode sparingly (slower)
- Save plots for later analysis
- Document findings in comments

### Performance
- Profile before optimizing
- Check solve times in diagnostic summary
- Warm-starting improves performance significantly
- Viewer mode is slower than headless

### Tuning
- Make one change at a time
- Document parameter changes
- Run regression tests after tuning
- Keep a log of what works

## Additional Resources

- [Dynamics Model](dynamics.md) - Mathematical model documentation
- [Style Guide](style_guide.md) - Code style conventions
- [Requirements](requirements.md) - Dependencies and setup

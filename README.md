# balancing_bot_2000

Balance the bot in 2000 ways (jk. One's plenty)

Two-wheeled self-balancing robot controlled with Model Predictive Control (MPC).

## Quick Start

### Setup

All Python commands need the project root in PYTHONPATH:

```bash
# Option 1: Set PYTHONPATH for each command
PYTHONPATH=. python3 script_name.py

# Option 2: Export PYTHONPATH for your session
export PYTHONPATH=/home/diego/balancing_bot_2000:$PYTHONPATH
```

### Running Tests

```bash
# All unit tests
python3 -m pytest tests/

# Regression tests
python3 -m pytest tests/regression/

# Quick diagnostic check
PYTHONPATH=. python3 scripts/debug/quick_check.py
```

### Development

```bash
# Run simulation with visualization
PYTHONPATH=. python3 "mujoco_sim/run_simulation.py"

# Validate dynamics model
PYTHONPATH=. python3 scripts/validation/check_dynamics_match.py

# View robot model
PYTHONPATH=. python3 scripts/viewer/view_robot.py
```

## Directory Structure (update this!!)

```
balancing_bot_2000/
??? README.md
??? requirements.txt
??? MUJOCO_LOG.TXT
??? config/                 # Configuration files (YAML)
?   ??? hardware/
?   ??? simulation/
??? control_pipeline/       # MPC controller implementation
??? debug/                  # Diagnostic tools and plotting
??? test_and_debug_output/           # Raw debug output and logged data
??? docs/                   # Documentation
?   ??? TESTING_GUIDE.md
?   ??? dynamics.md
?   ??? llm.md
?   ??? requirements.md
?   ??? style_guide.md
??? firmware/               # Firmware and microcontroller code
?   ??? balboa_interface/
??? hardware/               # Hardware interface modules and drivers
?   ??? run_robot.py
?   ??? __init__.py
?   ??? config_loader.py
?   ??? hardware_controller.py
?   ??? i2c_interface.py
?   ??? serial_interface.py
??? LQR/                    # Legacy LQR controller
?   ??? LQR.py
??? mpc/                    # MPC solver and components
??? robot_dynamics/         # Dynamics equations and linearization
??? scripts/                # Utility scripts
?   ??? debug/
?   ??? validation/
?   ??? viewer/
??? simulation/             # MuJoCo simulation and helpers
??? state_estimation/       # IMU fusion and filters
??? tests/                  # Test suite
?   ??? regression/
?   ??? controller_behaviour/
??? mujoco_sim/             # MuJoCo models and run_simulation entrypoint
?   ??? robot_model.xml
?   ??? run_simulation.py
??? viewer/                 # Lightweight viewer utilities
```

## Testing

See [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for comprehensive testing documentation.

### Test Categories

- **Unit Tests** (`tests/`) - Individual component verification
- **Regression Tests** (`tests/regression/`) - System-level behavior verification
- **Debug Scripts** (`scripts/debug/`) - Interactive diagnostic tools
- **Validation Scripts** (`scripts/validation/`) - Model consistency checks

## Documentation

- [Project Report](docs/project_report.tex) - Findings summary and open items
- [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) - Testing and debugging guide
- [dynamics.md](docs/dynamics.md) - Mathematical model documentation
- [style_guide.md](docs/style_guide.md) - Code style conventions
- [requirements.md](docs/requirements.md) - Dependencies and setup

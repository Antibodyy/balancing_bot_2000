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
PYTHONPATH=. python3 mads.py

# Validate dynamics model
PYTHONPATH=. python3 scripts/validation/check_dynamics_match.py

# View robot model
PYTHONPATH=. python3 scripts/viewer/view_robot.py
```

## Directory Structure (update this!!)

```
balancing_bot_2000/
├── README.md
├── requirements.txt
├── MUJOCO_LOG.TXT
├── mads.py
├── run_robot.py
├── run_simulation.py
├── config/                 # Configuration files (YAML)
│   ├── hardware/
│   └── simulation/
├── control_pipeline/       # MPC controller implementation
├── debug/                  # Diagnostic tools and plotting
├── debug_output/           # Raw debug output and logged data
├── docs/                   # Documentation
│   ├── TESTING_GUIDE.md
│   ├── dynamics.md
│   ├── llm.md
│   ├── requirements.md
│   └── style_guide.md
├── firmware/               # Firmware and microcontroller code
│   └── balboa_interface/
├── hardware/               # Hardware interface modules and drivers
│   ├── __init__.py
│   ├── config_loader.py
│   ├── hardware_controller.py
│   ├── i2c_interface.py
│   └── serial_interface.py
├── mpc/                    # MPC solver and components
├── robot_dynamics/         # Dynamics equations and linearization
├── scripts/                # Utility scripts
│   ├── debug/
│   ├── validation/
│   └── viewer/
├── simulation/             # MuJoCo simulation and helpers
├── state_estimation/       # IMU fusion and filters
├── tests/                  # Test suite
│   ├── regression/
│   └── test_*.py
└── viewer/                 # Lightweight viewer utilities
```

## Testing

See [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) for comprehensive testing documentation.

### Test Categories

- **Unit Tests** (`tests/`) - Individual component verification
- **Regression Tests** (`tests/regression/`) - System-level behavior verification
- **Debug Scripts** (`scripts/debug/`) - Interactive diagnostic tools
- **Validation Scripts** (`scripts/validation/`) - Model consistency checks

## Documentation

- [TESTING_GUIDE.md](docs/TESTING_GUIDE.md) - Testing and debugging guide
- [dynamics.md](docs/dynamics.md) - Mathematical model documentation
- [style_guide.md](docs/style_guide.md) - Code style conventions
- [requirements.md](docs/requirements.md) - Dependencies and setup

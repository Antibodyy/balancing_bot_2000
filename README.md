# balancing_bot_2000

Balance the bot in 2000 ways (jk. One's plenty)

Two-wheeled self-balancing robot controlled with Model Predictive Control (MPC).

## Quick Start

### Running Tests

```bash
# All unit tests
pytest tests/

# Regression tests
pytest tests/regression/

# Quick diagnostic check
python scripts/debug/quick_check.py
```

### Development

```bash
# Run simulation with visualization
python mads.py

# Validate dynamics model
python scripts/validation/check_dynamics_match.py

# View robot model
python scripts/viewer/view_robot.py
```

## Directory Structure

```
balancing_bot_2000/
├── config/                 # Configuration files (YAML)
├── control_pipeline/       # MPC controller implementation
├── debug/                  # Diagnostic tools and plotting
├── docs/                   # Documentation
│   ├── TESTING_GUIDE.md    # Comprehensive testing guide
│   ├── dynamics.md         # Mathematical model
│   ├── style_guide.md      # Code style conventions
│   ├── requirements.md     # Dependencies
│   └── llm.md              # LLM interaction guidelines
├── mpc/                    # MPC solver and components
├── robot_dynamics/         # Dynamics equations
├── scripts/                # Utility scripts
│   ├── debug/              # Debug and diagnostic scripts
│   ├── validation/         # Model validation scripts
│   └── viewer/             # Visualization utilities
├── simulation/             # MuJoCo simulation
└── tests/                  # Test suite
    ├── regression/         # Regression tests
    └── test_*.py           # Unit tests
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

"""Hardware validation test suite for Phase 1.

This package contains test scripts to validate the hardware interface:
- Serial communication with Arduino
- IMU data quality
- Encoder position tracking
- Motor open-loop control

See README.md for usage instructions.
"""

import pytest

pytest.skip("Hardware-in-the-loop tests require the Balboa hardware stack", allow_module_level=True)

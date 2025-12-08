# Hardware Validation Tests

This directory contains Raspberry Pi test scripts for Phase 1 hardware validation.

## Test Suite Overview

| Test | Type | Duration | Description |
|------|------|----------|-------------|
| 1. Serial Communication | Automated | 5s | Verify 200 Hz packet rate from Arduino |
| 2. IMU Data | Automated | 5s | Check accelerometer (~9.8 m/s²) and gyro (~0 rad/s) |
| 3. Encoder Tracking | Interactive | ~30s | Verify encoder counts when wheels rotated |
| 4. Motor Open-Loop | Interactive | ~10s | Test motor response to torque commands |

## Prerequisites

1. **Arduino firmware uploaded** to Balboa 32U4:
   - See [`firmware/README.md`](../../firmware/README.md) for upload instructions

2. **Raspberry Pi setup**:
   ```bash
   # Install dependencies
   pip install pyserial numpy

   # Verify serial port
   ls /dev/tty* | grep ACM
   # Should show /dev/ttyACM0 (or similar)

   # Add user to dialout group (for serial port access)
   sudo usermod -a -G dialout $USER
   # Log out and back in for group change to take effect
   ```

3. **Hardware connections**:
   - Raspberry Pi connected to Balboa 32U4 via USB
   - Balboa powered (batteries or USB)
   - IMU, encoders, and motors properly connected

## Running Tests

### Quick Start: Run All Tests
```bash
python tests/hardware/run_all_tests.py --port /dev/ttyACM0
```

This runs all 4 tests in sequence and provides a summary.

### Run Individual Tests

**Test 1: Serial Communication** (Automated)
```bash
python tests/hardware/test_serial_communication.py --port /dev/ttyACM0 --duration 5
```
- Keep robot stationary
- Verifies packets arrive at ~200 Hz
- Expected: 180-220 Hz (±10% tolerance)

**Test 2: IMU Data** (Automated)
```bash
python tests/hardware/test_imu_data.py --port /dev/ttyACM0 --duration 5
```
- Keep robot stationary and level
- Verifies accelerometer reads ~9.8 m/s² (gravity)
- Verifies gyroscope near zero (no rotation)

**Test 3: Encoder Tracking** (Interactive)
```bash
python tests/hardware/test_encoders.py --port /dev/ttyACM0
```
- Follow on-screen prompts
- Manually rotate left wheel forward
- Manually rotate right wheel forward
- Verifies encoder positions update correctly

**Test 4: Motor Open-Loop** (Interactive)
```bash
python tests/hardware/test_motor_open_loop.py --port /dev/ttyACM0
```
- ⚠️ **WARNING: Wheels will spin!**
- Place robot on test stand or elevate wheels
- Tests both wheels in both directions
- Verifies motors respond to torque commands

## Expected Results

All tests should **PASS** before proceeding to Phase 2 (MPC deployment).

### Success Criteria

✓ **Serial Communication**: Packet rate 180-220 Hz
✓ **IMU Data**: Accel magnitude 9.0-10.5 m/s², gyro bias <0.1 rad/s
✓ **Encoders**: Positions update when wheels rotated, noise <0.01 rad
✓ **Motors**: Wheels spin in correct direction for torque commands

## Troubleshooting

### Serial Connection Issues
```
Error: Failed to connect to Arduino
```
**Solutions:**
- Verify Arduino is connected: `ls /dev/ttyACM*`
- Check user has serial permissions: `groups` (should include `dialout`)
- Try different port: `--port /dev/ttyUSB0`
- Reset Arduino (press reset button)

### Low Packet Rate
```
✗ TEST FAILED - Packet rate is outside acceptable range
  Expected: 200 ± 10% Hz
  Measured: 50.2 Hz
```
**Solutions:**
- Check baud rate matches firmware (115200)
- Verify USB cable quality (try different cable)
- Check for USB power issues (try powered hub)

### IMU Initialization Failed
```
LED blinking red rapidly on Arduino
```
**Solutions:**
- Check I2C connections to LSM6DS33
- Verify LSM6 library is installed in Arduino IDE
- Re-upload firmware

### Encoders Not Updating
```
✗ Left wheel moved forward (FAIL - expected positive change)
```
**Solutions:**
- Check encoder connections
- Verify wheel rotated forward (not backward)
- Check `WHEEL_COUNTS_PER_REV` in firmware (should be 600)

### Motors Not Spinning
```
✗ Left wheel moves forward with positive torque (FAIL)
```
**Solutions:**
- Check motor connections (swap if reversed)
- Verify motor driver is working
- Increase test torque in script
- Check battery voltage (low battery = weak motors)

## Safety Notes

⚠️ **Motor Tests**:
- Always use test stand or elevate wheels
- Keep hands clear of spinning wheels
- Have emergency stop (Ctrl+C) ready

⚠️ **Power**:
- Don't run motors from USB power alone (use batteries)
- Check battery voltage before tests

## Next Steps

Once all tests pass:
1. ✓ Phase 1 complete!
2. Proceed to **Phase 2: MPC Deployment & Balancing**
3. See main implementation plan for next steps

## Test Output Examples

### Successful Test
```
==============================================================
TEST 1: Serial Communication
==============================================================
Port:     /dev/ttyACM0
Baudrate: 115200
Duration: 5s
--------------------------------------------------------------
Connecting to Arduino...
✓ Connected

Collecting packets for 5s...

==============================================================
RESULTS
==============================================================
Packets received:     1024
Duration:             5.02s
Average rate:         204.0 Hz
Expected rate:        200 Hz

==============================================================
✓ TEST PASSED - Packet rate is acceptable
```

### Failed Test
```
==============================================================
VALIDATION
==============================================================
✓ Accel magnitude: 9.78 m/s² (PASS)
✓ Accel noise: 0.042 m/s² (PASS)
✗ Gyro bias: 0.234 rad/s (FAIL - expected <0.1)

==============================================================
✗ TEST FAILED (2/3 checks passed)
```

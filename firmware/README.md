# Balboa 32U4 Firmware

This directory contains the Arduino firmware for the Pololu Balboa 32U4 balancing robot.

## Requirements

### Hardware
- Pololu Balboa 32U4 Balancing Robot Kit
- Raspberry Pi 4 (connected via USB)

### Software
- Arduino IDE 1.8.x or 2.x
- Pololu Balboa32U4 library
- Pololu LSM6 library

## Installation

1. **Install Arduino IDE**:
   - Download from https://www.arduino.cc/en/software

2. **Install Pololu Libraries**:
   - Open Arduino IDE
   - Go to Sketch → Include Library → Manage Libraries
   - Search for and install:
     - "Balboa32U4" by Pololu
     - "LSM6" by Pololu

3. **Open Firmware**:
   - Open `firmware/balboa_interface/balboa_interface.ino` in Arduino IDE

4. **Select Board**:
   - Tools → Board → Arduino Leonardo (Balboa uses same bootloader)
   - Tools → Port → Select your Balboa (usually /dev/ttyACM0 on Linux)

5. **Upload**:
   - Click Upload button or Sketch → Upload
   - Wait for "Done uploading" message

## Verification

After uploading, the robot should:
1. Blink red LED rapidly if IMU initialization fails (check I2C connections)
2. Blink green LED once on successful startup
3. Be ready to receive commands from Raspberry Pi

## Communication Protocol

The firmware uses binary serial communication at 115200 baud:

### Control Packet (Pi → Arduino, 10 bytes)
```
[Header=0xAA][Left Torque: float32][Right Torque: float32][Checksum: uint8]
```

### Sensor Packet (Arduino → Pi, 45 bytes)
```
[Header=0xBB][Timestamp: uint32][Accel XYZ: 3×float32]
[Gyro XYZ: 3×float32][Encoder L/R: 2×float32][Checksum: uint8]
```

Sensor packets are sent at 200 Hz.

## Calibration Notes

The firmware uses a simple linear torque-to-PWM mapping:
```
PWM_speed = torque_nm * TORQUE_TO_SPEED_GAIN
```

The `TORQUE_TO_SPEED_GAIN` constant (line 34) should be calibrated during system identification for optimal performance. The default value of ~1200 is an initial estimate.

## Troubleshooting

**LED Indicators**:
- Rapid red blink: IMU initialization failed
- Green blink on startup: Firmware ready
- Yellow blink: Valid control packet received

**Common Issues**:
- Upload fails: Check board selection (Arduino Leonardo) and port
- IMU error: Verify I2C connections, ensure LSM6 library is installed
- No serial communication: Check USB cable, verify baud rate (115200)

## Modifications

If you modify the firmware:
1. Keep the packet format synchronized with `hardware/serial_interface.py`
2. Update checksum computation if packet structure changes
3. Test communication before deploying to robot

## See Also

- [Hardware Setup Guide](../docs/hardware_setup.md) (to be created)
- [Serial Interface](../hardware/serial_interface.py) - Python counterpart
- [Pololu Balboa Documentation](https://www.pololu.com/docs/0J70)

/**
 * Balboa 32U4 Interface for MPC Balancing Robot - USB Serial
 *
 * Communicates with Raspberry Pi over USB Serial for:
 * - Sending sensor data (IMU, encoders, battery)
 * - Receiving motor commands
 *
 * Author: MPC Balancing Robot Project
 * Hardware: Pololu Balboa 32U4
 */

#include <Balboa32U4.h>
#include <LSM6.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// Communication settings
const uint32_t SERIAL_BAUD = 115200;
const uint8_t DATA_UPDATE_RATE_HZ = 200;
const uint32_t DATA_UPDATE_PERIOD_US = 1000000 / DATA_UPDATE_RATE_HZ;

// IMU configuration
LSM6 imu;

// Encoder configuration
const float ENCODER_COUNTS_PER_REV = 12.0;
const float GEARING_RATIO = 75.81;  // Balboa 32U4 with 75:1 motors
const float WHEEL_COUNTS_PER_REV = ENCODER_COUNTS_PER_REV * GEARING_RATIO;

// Motor configuration
const int16_t MOTOR_MAX_SPEED = 300;  // TODO: check this
const int16_t MOTOR_MIN_SPEED = -300;

// ============================================================================
// HARDWARE OBJECTS
// ============================================================================

Balboa32U4Motors motors;
Balboa32U4Encoders encoders;
Balboa32U4ButtonA buttonA;

// ============================================================================
// PROTOCOL DEFINITIONS
// ============================================================================

// Packet start bytes for reliable framing
const uint8_t PACKET_START_1 = 0xAA;
const uint8_t PACKET_START_2 = 0x55;

// Message types
const uint8_t MSG_SENSOR_DATA = 0x01;
const uint8_t MSG_MOTOR_CMD = 0x02;
const uint8_t MSG_STATUS = 0x03;
const uint8_t MSG_ACK = 0x0A;

// Sensor data packet structure (32 bytes)
struct __attribute__((packed)) SensorDataPacket {
  uint8_t start1;          // 0xAA
  uint8_t start2;          // 0x55
  uint8_t msg_type;        // MSG_SENSOR_DATA
  uint8_t length;          // Payload length
  uint32_t timestamp_us;   // Microsecond timestamp
  int16_t accel_x;         // Raw accelerometer X
  int16_t accel_y;         // Raw accelerometer Y
  int16_t accel_z;         // Raw accelerometer Z
  int16_t gyro_x;          // Raw gyroscope X
  int16_t gyro_y;          // Raw gyroscope Y
  int16_t gyro_z;          // Raw gyroscope Z
  int16_t encoder_left;    // Left encoder counts
  int16_t encoder_right;   // Right encoder counts
  uint16_t battery_mv;     // Battery voltage in millivolts
  uint8_t checksum;        // Simple checksum
};

// Motor command packet structure (8 bytes)
struct __attribute__((packed)) MotorCmdPacket {
  uint8_t start1;          // 0xAA
  uint8_t start2;          // 0x55
  uint8_t msg_type;        // MSG_MOTOR_CMD
  uint8_t length;          // Payload length
  int16_t motor_left;      // Left motor speed (-300 to 300)
  int16_t motor_right;     // Right motor speed (-300 to 300)
  uint8_t checksum;        // Simple checksum
};

// ============================================================================
// GLOBAL STATE
// ============================================================================

SensorDataPacket sensor_packet;
bool debug_mode = false;
unsigned long last_button_check_ms = 0;

// Motor command state
int16_t motor_left_cmd = 0;
int16_t motor_right_cmd = 0;
unsigned long last_motor_cmd_ms = 0;
const unsigned long MOTOR_TIMEOUT_MS = 500;  // Stop motors if no command for 500ms

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

// Calculate simple checksum (XOR of all bytes except checksum field)
uint8_t calculateChecksum(uint8_t* data, uint8_t length) {
  uint8_t checksum = 0;
  for (uint8_t i = 0; i < length; i++) {
    checksum ^= data[i];
  }
  return checksum;
}

// Clamp motor speed to valid range
int16_t clampMotorSpeed(int16_t speed) {
  if (speed > MOTOR_MAX_SPEED) return MOTOR_MAX_SPEED;
  if (speed < MOTOR_MIN_SPEED) return MOTOR_MIN_SPEED;
  return speed;
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Initialize USB Serial
  Serial.begin(SERIAL_BAUD);

  // Wait for USB serial to be ready (timeout after 3 seconds)
  unsigned long start = millis();
  while (!Serial && (millis() - start < 3000)) {
    delay(10);
  }

  // Initialize IMU
  if (!imu.init()) {
    // Flash red LED to indicate error
    while (1) {
      ledRed(1);
      delay(100);
      ledRed(0);
      delay(100);
    }
  }

  // Configure IMU for optimal performance
  // Accel: 208 Hz ODR, ±4g full scale
  imu.writeReg(LSM6::CTRL1_XL, 0b01011000);  // 0x58
  // Gyro: 208 Hz ODR, ±500 dps full scale
  imu.writeReg(LSM6::CTRL2_G, 0b01010100);   // 0x54
  // Enable auto-increment for multi-byte reads
  imu.writeReg(LSM6::CTRL3_C, 0b00000100);   // 0x04

  delay(100);  // Let IMU stabilize

  // Initialize encoders
  encoders.getCountsAndResetLeft();
  encoders.getCountsAndResetRight();

  // Initialize motors (stopped)
  motors.setSpeeds(0, 0);

  // Initialize sensor packet structure
  sensor_packet.start1 = PACKET_START_1;
  sensor_packet.start2 = PACKET_START_2;
  sensor_packet.msg_type = MSG_SENSOR_DATA;
  sensor_packet.length = sizeof(SensorDataPacket) - 5;  // Exclude header + checksum

  delay(500);

  // Indicate ready
  ledGreen(1);
  delay(500);
  ledGreen(0);

  Serial.println("READY");  // Signal to RPi that we're ready
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  unsigned long current_time_us = micros();
  unsigned long current_time_ms = millis();

  // Check button A to toggle debug mode
  if (current_time_ms - last_button_check_ms >= 200) {
    last_button_check_ms = current_time_ms;
    if (buttonA.getSingleDebouncedPress()) {
      debug_mode = !debug_mode;
      if (debug_mode) {
        Serial.println("DEBUG_ON");
      } else {
        Serial.println("DEBUG_OFF");
      }
    }
  }

  // Process incoming serial commands
  processSerialCommands();

  // Safety: Stop motors if no command received within timeout
  if (current_time_ms - last_motor_cmd_ms > MOTOR_TIMEOUT_MS) {
    if (motor_left_cmd != 0 || motor_right_cmd != 0) {
      motor_left_cmd = 0;
      motor_right_cmd = 0;
      motors.setSpeeds(0, 0);
    }
  }

  // Send sensor data at fixed rate
  static unsigned long last_data_update_us = 0;
  if (current_time_us - last_data_update_us >= DATA_UPDATE_PERIOD_US) {
    last_data_update_us = current_time_us;

    sendSensorData();
  }
}

// ============================================================================
// COMMUNICATION FUNCTIONS
// ============================================================================

void sendSensorData() {
  // Read sensors
  imu.read();

  // Fill sensor packet
  sensor_packet.timestamp_us = micros();
  sensor_packet.accel_x = imu.a.x;
  sensor_packet.accel_y = imu.a.y;
  sensor_packet.accel_z = imu.a.z;
  sensor_packet.gyro_x = imu.g.x;
  sensor_packet.gyro_y = imu.g.y;
  sensor_packet.gyro_z = imu.g.z;
  sensor_packet.encoder_left = encoders.getCountsLeft();
  sensor_packet.encoder_right = encoders.getCountsRight();
  sensor_packet.battery_mv = readBatteryMillivolts();

  // Calculate checksum
  sensor_packet.checksum = calculateChecksum((uint8_t*)&sensor_packet,
                                              sizeof(SensorDataPacket) - 1);

  // Send packet
  Serial.write((uint8_t*)&sensor_packet, sizeof(SensorDataPacket));

  // Debug output (if enabled)
  if (debug_mode) {
    static unsigned long last_debug_ms = 0;
    unsigned long now = millis();
    if (now - last_debug_ms >= 500) {  // Print every 500ms
      last_debug_ms = now;
      Serial.print("DBG: IMU(");
      Serial.print(sensor_packet.accel_x); Serial.print(",");
      Serial.print(sensor_packet.accel_y); Serial.print(",");
      Serial.print(sensor_packet.accel_z); Serial.print(") ");
      Serial.print("ENC(");
      Serial.print(sensor_packet.encoder_left); Serial.print(",");
      Serial.print(sensor_packet.encoder_right); Serial.print(") ");
      Serial.print("BAT:");
      Serial.print(sensor_packet.battery_mv);
      Serial.println("mV");
    }
  }
}

void processSerialCommands() {
  // Check if we have enough bytes for a motor command packet
  if (Serial.available() >= sizeof(MotorCmdPacket)) {
    // Try to find packet start
    if (Serial.peek() == PACKET_START_1) {
      MotorCmdPacket cmd_packet;
      Serial.readBytes((uint8_t*)&cmd_packet, sizeof(MotorCmdPacket));

      // Verify packet structure
      if (cmd_packet.start1 == PACKET_START_1 &&
          cmd_packet.start2 == PACKET_START_2 &&
          cmd_packet.msg_type == MSG_MOTOR_CMD) {

        // Verify checksum
        uint8_t checksum = calculateChecksum((uint8_t*)&cmd_packet,
                                               sizeof(MotorCmdPacket) - 1);
        if (checksum == cmd_packet.checksum) {
          // Valid command - apply motor speeds
          motor_left_cmd = clampMotorSpeed(cmd_packet.motor_left);
          motor_right_cmd = clampMotorSpeed(cmd_packet.motor_right);
          motors.setSpeeds(motor_left_cmd, motor_right_cmd);
          last_motor_cmd_ms = millis();

          if (debug_mode) {
            Serial.print("CMD: L=");
            Serial.print(motor_left_cmd);
            Serial.print(" R=");
            Serial.println(motor_right_cmd);
          }
        } else {
          // Checksum error
          if (debug_mode) {
            Serial.println("ERR: Checksum");
          }
        }
      }
    } else {
      // Invalid start byte - discard one byte and try again
      Serial.read();
    }
  }

  // Handle text commands for debugging
  if (Serial.available() > 0) {
    char c = Serial.peek();
    // Check if this looks like a text command (printable ASCII)
    if (c >= 'A' && c <= 'z') {
      String cmd = Serial.readStringUntil('\n');
      cmd.trim();

      if (cmd == "PING") {
        Serial.println("PONG");
      } else if (cmd == "STATUS") {
        Serial.print("Motors: L=");
        Serial.print(motor_left_cmd);
        Serial.print(" R=");
        Serial.println(motor_right_cmd);
        Serial.print("Battery: ");
        Serial.print(readBatteryMillivolts());
        Serial.println("mV");
      } else if (cmd == "STOP") {
        motor_left_cmd = 0;
        motor_right_cmd = 0;
        motors.setSpeeds(0, 0);
        Serial.println("OK");
      } else if (cmd.startsWith("MOTOR")) {
        // Format: MOTOR <left> <right>
        int idx1 = cmd.indexOf(' ');
        int idx2 = cmd.indexOf(' ', idx1 + 1);
        if (idx1 > 0 && idx2 > 0) {
          int16_t left = cmd.substring(idx1 + 1, idx2).toInt();
          int16_t right = cmd.substring(idx2 + 1).toInt();
          motor_left_cmd = clampMotorSpeed(left);
          motor_right_cmd = clampMotorSpeed(right);
          motors.setSpeeds(motor_left_cmd, motor_right_cmd);
          last_motor_cmd_ms = millis();
          Serial.println("OK");
        }
      }
    }
  }
}

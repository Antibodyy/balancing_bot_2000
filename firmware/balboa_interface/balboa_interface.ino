/**
 * Balboa 32U4 Interface for MPC Balancing Robot - I2C Slave Version
 *
 * This firmware provides low-level hardware interface for the Raspberry Pi:
 * - Motor control via torque commands
 * - Encoder position tracking
 * - I2C slave communication with Raspberry Pi
 *
 * Note: IMU sensors are read directly by Raspberry Pi (I2C master).
 * The Balboa only handles motors and encoders.
 *
 * Uses Pololu RPI Slave Arduino Library:
 * https://github.com/pololu/pololu-rpi-slave-arduino-library
 *
 * Author: MPC Balancing Robot Project
 * Hardware: Pololu Balboa 32U4
 */

#include <Balboa32U4.h>
#include <PololuRPiSlave.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// I2C Slave Address
const uint8_t I2C_SLAVE_ADDRESS = 0x20;

// Encoder configuration
const float ENCODER_COUNTS_PER_REV = 12.0;
const float GEARING_RATIO = 50.0;
const float WHEEL_COUNTS_PER_REV = ENCODER_COUNTS_PER_REV * GEARING_RATIO;

// Motor configuration
const float MOTOR_MAX_TORQUE_NM = 0.25;
const int16_t MOTOR_MAX_SPEED = 300;
const float TORQUE_TO_SPEED_GAIN = (float)MOTOR_MAX_SPEED / MOTOR_MAX_TORQUE_NM;

// Update rate
const unsigned long UPDATE_PERIOD_US = 5000;  // 200 Hz

// ============================================================================
// I2C DATA STRUCTURE
// ============================================================================

/**
 * I2C Buffer Structure
 *
 * Sensor Data (Read by Raspberry Pi):
 * - uint32: timestamp_us (4 bytes)
 * - float32[2]: encoder L/R (8 bytes)
 * Total: 12 bytes
 *
 * Motor Commands (Written by Raspberry Pi):
 * - float32: torque_left_nm (4 bytes)
 * - float32: torque_right_nm (4 bytes)
 * Total: 8 bytes
 */
struct I2CData {
  // Sensor data (read by RPi) - offset 0
  uint32_t timestamp_us;
  float encoder_left_rad;
  float encoder_right_rad;

  // Motor commands (written by RPi) - offset 12
  float torque_left_nm;
  float torque_right_nm;
};

// ============================================================================
// HARDWARE OBJECTS
// ============================================================================

Balboa32U4Motors motors;
Balboa32U4Encoders encoders;
Balboa32U4ButtonA buttonA;
PololuRPiSlave<I2CData, 5> rpiSlave;

// ============================================================================
// GLOBAL STATE
// ============================================================================

// Encoder data (in radians)
float encoder_left_rad = 0.0;
float encoder_right_rad = 0.0;

// Motor commands
float torque_left_nm = 0.0;
float torque_right_nm = 0.0;

// Timing
unsigned long last_update_time_us = 0;

// Debug mode
bool debug_mode = false;
unsigned long last_button_check_ms = 0;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Initialize USB Serial for debugging
  Serial.begin(115200);

  // Initialize I2C slave for Raspberry Pi communication
  rpiSlave.init(I2C_SLAVE_ADDRESS);

  // Initialize encoders
  encoders.getCountsAndResetLeft();
  encoders.getCountsAndResetRight();

  // Initialize motors (stopped)
  motors.setSpeeds(0, 0);

  // Wait for initialization
  delay(500);

  // Indicate ready
  ledGreen(1);
  delay(500);
  ledGreen(0);

  Serial.println("Balboa 32U4 I2C Slave Ready");
  Serial.print("I2C Slave Address: 0x");
  Serial.println(I2C_SLAVE_ADDRESS, HEX);
  Serial.println("Note: IMU read by Raspberry Pi");
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
        Serial.println("\n=== DEBUG MODE ENABLED ===");
      } else {
        Serial.println("\n=== DEBUG MODE DISABLED ===");
      }
    }
  }

  // Update sensors and motors at fixed rate
  if (current_time_us - last_update_time_us >= UPDATE_PERIOD_US) {
    last_update_time_us = current_time_us;

    readEncoders();
    updateI2CBuffer();

    if (debug_mode) {
      printDebugInfo();
    }
  }

  // Handle I2C communication with Raspberry Pi
  rpiSlave.updateBuffer();

  // Read motor commands from I2C buffer
  readMotorCommands();

  // Apply motor commands
  applyMotorCommands();
}

// ============================================================================
// ENCODER FUNCTIONS
// ============================================================================

void readEncoders() {
  // Read encoder counts
  int16_t counts_left = encoders.getCountsLeft();
  int16_t counts_right = encoders.getCountsRight();

  // Convert to radians
  encoder_left_rad = (counts_left / WHEEL_COUNTS_PER_REV) * TWO_PI;
  encoder_right_rad = (counts_right / WHEEL_COUNTS_PER_REV) * TWO_PI;
}

// ============================================================================
// MOTOR FUNCTIONS
// ============================================================================

void readMotorCommands() {
  /**
   * Read motor commands from I2C buffer (written by Raspberry Pi)
   */
  torque_left_nm = rpiSlave.buffer.torque_left_nm;
  torque_right_nm = rpiSlave.buffer.torque_right_nm;
}

void applyMotorCommands() {
  // Convert torque (N⋅m) to motor speed (PWM)
  int16_t speed_left = torqueToSpeed(torque_left_nm);
  int16_t speed_right = torqueToSpeed(torque_right_nm);

  // Apply to motors
  motors.setSpeeds(speed_left, speed_right);
}

int16_t torqueToSpeed(float torque_nm) {
  // Linear mapping: speed = K * torque
  float speed = torque_nm * TORQUE_TO_SPEED_GAIN;

  // Clamp to motor limits
  if (speed > MOTOR_MAX_SPEED) speed = MOTOR_MAX_SPEED;
  if (speed < -MOTOR_MAX_SPEED) speed = -MOTOR_MAX_SPEED;

  return (int16_t)speed;
}

// ============================================================================
// I2C COMMUNICATION
// ============================================================================

void updateI2CBuffer() {
  /**
   * Update I2C buffer with latest sensor data for Raspberry Pi to read
   */
  rpiSlave.buffer.timestamp_us = micros();
  rpiSlave.buffer.encoder_left_rad = encoder_left_rad;
  rpiSlave.buffer.encoder_right_rad = encoder_right_rad;

  // Note: Motor commands are written by RPi, so we don't update them here
}

// ============================================================================
// DEBUG FUNCTIONS
// ============================================================================

void printDebugInfo() {
  static unsigned long last_print_ms = 0;
  unsigned long now = millis();

  // Print at 10 Hz
  if (now - last_print_ms >= 100) {
    last_print_ms = now;

    Serial.println("--- Sensor Data ---");
    Serial.print("Timestamp: ");
    Serial.print(micros());
    Serial.println(" us");

    Serial.print("Encoders (rad): L=");
    Serial.print(encoder_left_rad, 4);
    Serial.print(" R=");
    Serial.println(encoder_right_rad, 4);

    Serial.print("Motors (Nm):    L=");
    Serial.print(torque_left_nm, 4);
    Serial.print(" R=");
    Serial.println(torque_right_nm, 4);

    Serial.println();
  }
}

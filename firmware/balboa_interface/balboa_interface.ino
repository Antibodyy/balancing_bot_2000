/**
 * Balboa 32U4 Interface for MPC Balancing Robot - I2C Slave with Wire Library
 *
 * Uses standard Arduino Wire library for I2C slave communication.
 * This replaces PololuRPiSlave which has compatibility issues.
 *
 * Author: MPC Balancing Robot Project
 * Hardware: Pololu Balboa 32U4
 */

#include <Balboa32U4.h>
#include <Wire.h>

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
// I2C DATA BUFFER
// ============================================================================

// Total buffer: 20 bytes
// Sensor data (read by RPi): 12 bytes
// Motor commands (write by RPi): 8 bytes
uint8_t i2c_buffer[20];

// Buffer offsets
const uint8_t OFFSET_TIMESTAMP = 0;     // uint32 (4 bytes)
const uint8_t OFFSET_ENCODER_L = 4;     // float32 (4 bytes)
const uint8_t OFFSET_ENCODER_R = 8;     // float32 (4 bytes)
const uint8_t OFFSET_MOTOR_L = 12;      // float32 (4 bytes)
const uint8_t OFFSET_MOTOR_R = 16;      // float32 (4 bytes)

// Buffer state
volatile uint8_t i2c_read_index = 0;
volatile uint8_t i2c_write_index = 0;
volatile bool i2c_write_mode = false;

// ============================================================================
// HARDWARE OBJECTS
// ============================================================================

Balboa32U4Motors motors;
Balboa32U4Encoders encoders;
Balboa32U4ButtonA buttonA;

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
// I2C INTERRUPT HANDLERS
// ============================================================================

// Called when RPi sends data (motor commands)
void receiveEvent(int byte_count) {
  if (byte_count == 0) return;

  // First byte is the register/offset address
  uint8_t offset = Wire.read();
  i2c_write_index = offset;
  i2c_write_mode = true;

  // Read remaining bytes into buffer
  while (Wire.available() && i2c_write_index < 20) {
    i2c_buffer[i2c_write_index++] = Wire.read();
  }

  // Update motor commands from buffer
  if (i2c_write_index > OFFSET_MOTOR_L) {
    memcpy(&torque_left_nm, &i2c_buffer[OFFSET_MOTOR_L], sizeof(float));
  }
  if (i2c_write_index > OFFSET_MOTOR_R) {
    memcpy(&torque_right_nm, &i2c_buffer[OFFSET_MOTOR_R], sizeof(float));
  }
}

// Called when RPi requests data (sensor readings)
void requestEvent() {
  if (i2c_write_mode) {
    // First request after write - send nothing, just set read index
    i2c_read_index = i2c_write_index;
    i2c_write_mode = false;
    return;
  }

  // Send one byte
  if (i2c_read_index < 20) {
    Wire.write(i2c_buffer[i2c_read_index++]);
  } else {
    Wire.write(0);  // Send zero if out of bounds
  }
}

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Initialize USB Serial for debugging
  Serial.begin(115200);

  // Initialize I2C slave
  Wire.begin(I2C_SLAVE_ADDRESS);
  Wire.onReceive(receiveEvent);
  Wire.onRequest(requestEvent);

  // Initialize encoders
  encoders.getCountsAndResetLeft();
  encoders.getCountsAndResetRight();

  // Initialize motors (stopped)
  motors.setSpeeds(0, 0);

  // Initialize buffer to zeros
  memset(i2c_buffer, 0, sizeof(i2c_buffer));

  // Wait for initialization
  delay(500);

  // Indicate ready
  ledGreen(1);
  delay(500);
  ledGreen(0);

  Serial.println("Balboa 32U4 I2C Slave Ready (Wire Library)");
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
  uint32_t timestamp_us = micros();

  // Copy data into buffer (little-endian)
  memcpy(&i2c_buffer[OFFSET_TIMESTAMP], &timestamp_us, sizeof(uint32_t));
  memcpy(&i2c_buffer[OFFSET_ENCODER_L], &encoder_left_rad, sizeof(float));
  memcpy(&i2c_buffer[OFFSET_ENCODER_R], &encoder_right_rad, sizeof(float));
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

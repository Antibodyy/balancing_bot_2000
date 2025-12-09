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
#include <LSM6.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// I2C Slave Address
const uint8_t I2C_SLAVE_ADDRESS = 0x20;

// IMU configuration
const uint8_t IMU_UPDATE_RATE_HZ = 200;
const unsigned long IMU_UPDATE_PERIOD_US = 1000000 / IMU_UPDATE_RATE_HZ;

// Encoder configuration (TEMPORARILY DISABLED - IMU ONLY)
// const float ENCODER_COUNTS_PER_REV = 12.0;
// const float GEARING_RATIO = 50.0;
// const float WHEEL_COUNTS_PER_REV = ENCODER_COUNTS_PER_REV * GEARING_RATIO;

// Motor configuration (TEMPORARILY DISABLED - IMU ONLY)
// const float MOTOR_MAX_TORQUE_NM = 0.25;
// const int16_t MOTOR_MAX_SPEED = 300;
// const float TORQUE_TO_SPEED_GAIN = (float)MOTOR_MAX_SPEED / MOTOR_MAX_TORQUE_NM;

// ============================================================================
// I2C DATA BUFFER - IMU ONLY
// ============================================================================

// Buffer layout: 16 bytes total
// - timestamp: uint32 (4 bytes)
// - accel: 3x int16 (6 bytes)
// - gyro: 3x int16 (6 bytes)
uint8_t i2c_buffer[16];

// Buffer offsets
const uint8_t OFFSET_TIMESTAMP = 0;   // uint32 (4 bytes)
const uint8_t OFFSET_ACCEL_X = 4;     // int16 (2 bytes)
const uint8_t OFFSET_ACCEL_Y = 6;     // int16 (2 bytes)
const uint8_t OFFSET_ACCEL_Z = 8;     // int16 (2 bytes)
const uint8_t OFFSET_GYRO_X = 10;     // int16 (2 bytes)
const uint8_t OFFSET_GYRO_Y = 12;     // int16 (2 bytes)
const uint8_t OFFSET_GYRO_Z = 14;     // int16 (2 bytes)
const uint8_t BUFFER_SIZE = 16;

// Buffer state for I2C slave
volatile uint8_t i2c_read_index = 0;

// ============================================================================
// HARDWARE OBJECTS
// ============================================================================

LSM6 imu;
// Balboa32U4Motors motors;          // DISABLED - IMU only
// Balboa32U4Encoders encoders;       // DISABLED - IMU only
Balboa32U4ButtonA buttonA;

// ============================================================================
// GLOBAL STATE
// ============================================================================

// Encoder data (DISABLED - IMU only)
// float encoder_left_rad = 0.0;
// float encoder_right_rad = 0.0;

// Motor commands (DISABLED - IMU only)
// float torque_left_nm = 0.0;
// float torque_right_nm = 0.0;

// Debug mode
bool debug_mode = false;
unsigned long last_button_check_ms = 0;

// ============================================================================
// I2C INTERRUPT HANDLERS
// ============================================================================

// Called when RPi sends data (register address for read)
void receiveEvent(int byte_count) {
  if (byte_count == 0) return;

  // Read the register/offset address
  uint8_t offset = Wire.read();

  // Validate offset
  if (offset >= BUFFER_SIZE) {
    offset = 0;  // Default to start if invalid
  }

  // Set read pointer for subsequent requestEvent() calls
  i2c_read_index = offset;

  // Discard any remaining bytes (shouldn't be any for read setup)
  while (Wire.available()) {
    Wire.read();
  }
}

// Called when RPi requests data - CALLED ONCE PER BYTE
void requestEvent() {
  // Send current byte
  if (i2c_read_index < BUFFER_SIZE) {
    Wire.write(i2c_buffer[i2c_read_index]);
    i2c_read_index++;
  } else {
    // Out of bounds - send zero
    Wire.write(0);
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

  // Initialize IMU
  Serial.println("Initializing IMU...");
  if (!imu.init()) {
    Serial.println("ERROR: Failed to detect LSM6DS33!");
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

  Serial.println("IMU initialized: LSM6DS33");
  Serial.println("  Accel: ±4g, 208 Hz");
  Serial.println("  Gyro: ±500 dps, 208 Hz");

  // Initialize encoders (DISABLED - IMU only)
  // encoders.getCountsAndResetLeft();
  // encoders.getCountsAndResetRight();

  // Initialize motors (DISABLED - IMU only)
  // motors.setSpeeds(0, 0);

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
  Serial.println("Note: IMU read by Arduino, sent to RPi via I2C");
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

  // Update IMU at fixed rate (200 Hz)
  static unsigned long last_imu_update_us = 0;
  if (current_time_us - last_imu_update_us >= IMU_UPDATE_PERIOD_US) {
    last_imu_update_us = current_time_us;

    // Read IMU (accelerometer + gyroscope)
    imu.read();
    updateI2CBuffer();

    if (debug_mode) {
      printDebugInfo();
    }
  }
}

// ============================================================================
// ENCODER FUNCTIONS (DISABLED - IMU ONLY)
// ============================================================================

// void readEncoders() {
//   // Read encoder counts
//   int16_t counts_left = encoders.getCountsLeft();
//   int16_t counts_right = encoders.getCountsRight();
//
//   // Convert to radians
//   encoder_left_rad = (counts_left / WHEEL_COUNTS_PER_REV) * TWO_PI;
//   encoder_right_rad = (counts_right / WHEEL_COUNTS_PER_REV) * TWO_PI;
// }

// ============================================================================
// MOTOR FUNCTIONS (DISABLED - IMU ONLY)
// ============================================================================

// void applyMotorCommands() {
//   // Convert torque (N⋅m) to motor speed (PWM)
//   int16_t speed_left = torqueToSpeed(torque_left_nm);
//   int16_t speed_right = torqueToSpeed(torque_right_nm);
//
//   // Apply to motors
//   motors.setSpeeds(speed_left, speed_right);
// }
//
// int16_t torqueToSpeed(float torque_nm) {
//   // Linear mapping: speed = K * torque
//   float speed = torque_nm * TORQUE_TO_SPEED_GAIN;
//
//   // Clamp to motor limits
//   if (speed > MOTOR_MAX_SPEED) speed = MOTOR_MAX_SPEED;
//   if (speed < -MOTOR_MAX_SPEED) speed = -MOTOR_MAX_SPEED;
//
//   return (int16_t)speed;
// }

// ============================================================================
// I2C COMMUNICATION
// ============================================================================

void updateI2CBuffer() {
  /**
   * Update I2C buffer with latest IMU data for Raspberry Pi to read
   *
   * CRITICAL: Use noInterrupts() to prevent race condition where RPi
   * reads buffer while we're updating it mid-write.
   */

  uint32_t timestamp_us = micros();

  // Disable interrupts during buffer update to ensure atomic write
  noInterrupts();

  // Copy timestamp (4 bytes)
  memcpy(&i2c_buffer[OFFSET_TIMESTAMP], &timestamp_us, sizeof(uint32_t));

  // Copy accelerometer data (6 bytes) - raw int16 values
  memcpy(&i2c_buffer[OFFSET_ACCEL_X], &imu.a.x, sizeof(int16_t));
  memcpy(&i2c_buffer[OFFSET_ACCEL_Y], &imu.a.y, sizeof(int16_t));
  memcpy(&i2c_buffer[OFFSET_ACCEL_Z], &imu.a.z, sizeof(int16_t));

  // Copy gyroscope data (6 bytes) - raw int16 values
  memcpy(&i2c_buffer[OFFSET_GYRO_X], &imu.g.x, sizeof(int16_t));
  memcpy(&i2c_buffer[OFFSET_GYRO_Y], &imu.g.y, sizeof(int16_t));
  memcpy(&i2c_buffer[OFFSET_GYRO_Z], &imu.g.z, sizeof(int16_t));

  // Re-enable interrupts
  interrupts();
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

    Serial.println("--- IMU Data ---");
    Serial.print("Timestamp: ");
    Serial.print(micros());
    Serial.println(" us");

    Serial.print("Accel (raw): X=");
    Serial.print(imu.a.x);
    Serial.print(" Y=");
    Serial.print(imu.a.y);
    Serial.print(" Z=");
    Serial.println(imu.a.z);

    Serial.print("Gyro (raw):  X=");
    Serial.print(imu.g.x);
    Serial.print(" Y=");
    Serial.print(imu.g.y);
    Serial.print(" Z=");
    Serial.println(imu.g.z);

    // DEBUG: Print I2C buffer contents
    Serial.print("I2C Buffer[0-15]: ");
    for (int i = 0; i < 16; i++) {
      if (i2c_buffer[i] < 0x10) Serial.print("0");
      Serial.print(i2c_buffer[i], HEX);
      Serial.print(" ");
    }
    Serial.println();
    Serial.print("I2C read_index: ");
    Serial.println(i2c_read_index);

    Serial.println();
  }
}

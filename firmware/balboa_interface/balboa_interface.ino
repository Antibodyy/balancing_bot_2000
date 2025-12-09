/**
 * Balboa 32U4 Interface for MPC Balancing Robot
 *
 * This firmware provides low-level hardware interface for the Raspberry Pi:
 * - Motor control via torque commands
 * - IMU data acquisition (LSM6DS33)
 * - Encoder position tracking
 * - Binary serial communication
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

// Serial communication
const unsigned long BAUD_RATE = 115200;
const byte CONTROL_PACKET_HEADER = 0xAA;
const byte SENSOR_PACKET_HEADER = 0xBB;

// IMU configuration
const int IMU_SAMPLE_RATE_HZ = 200;
const unsigned long IMU_SAMPLE_PERIOD_US = 1000000UL / IMU_SAMPLE_RATE_HZ;

// Encoder configuration
const float ENCODER_COUNTS_PER_REV = 12.0;  // Motor shaft CPR
const float GEARING_RATIO = 50.0;            // 50:1 gearbox
const float WHEEL_COUNTS_PER_REV = ENCODER_COUNTS_PER_REV * GEARING_RATIO;  // 600

// Motor configuration
const float MOTOR_MAX_TORQUE_NM = 0.25;     // Maximum motor torque (N⋅m)
const int16_t MOTOR_MAX_SPEED = 300;        // Maximum motor PWM speed

// Calibration - torque to motor speed mapping (estimated)
// These values should be characterized during system identification
// For now, using linear approximation: speed = K_torque * torque_nm
const float TORQUE_TO_SPEED_GAIN = (float)MOTOR_MAX_SPEED / MOTOR_MAX_TORQUE_NM;  // ~1200

// ============================================================================
// HARDWARE OBJECTS
// ============================================================================

Balboa32U4Motors motors;
Balboa32U4Encoders encoders;
Balboa32U4ButtonA buttonA;
LSM6 imu;

// ============================================================================
// GLOBAL STATE
// ============================================================================

// IMU data (converted to SI units)
float accel_x_mps2 = 0.0;
float accel_y_mps2 = 0.0;
float accel_z_mps2 = 0.0;
float gyro_x_radps = 0.0;
float gyro_y_radps = 0.0;
float gyro_z_radps = 0.0;

// Encoder data (in radians, accumulated from deltas to avoid int16 overflow)
float encoder_left_rad = 0.0;
float encoder_right_rad = 0.0;
int16_t encoder_left_counts_prev = 0;
int16_t encoder_right_counts_prev = 0;

// Motor commands
float torque_left_nm = 0.0;
float torque_right_nm = 0.0;

// Timing
unsigned long last_imu_time_us = 0;
unsigned long last_packet_time_ms = 0;

// Debug mode
bool debug_mode = false;
unsigned long last_button_check_ms = 0;

// ============================================================================
// SETUP
// ============================================================================

void setup() {
  // Initialize serial communication
  Serial.begin(BAUD_RATE);

  // Initialize I2C
  Wire.begin();

  // Initialize IMU
  if (!imu.init()) {
    // IMU initialization failed - blink LED rapidly
    while (1) {
      ledRed(1);
      delay(100);
      ledRed(0);
      delay(100);
    }
  }

  // Configure IMU
  imu.enableDefault();
  // Configure accelerometer: ±4g, 208 Hz
  imu.writeReg(LSM6::CTRL1_XL, 0b01010100);
  // Configure gyroscope: ±500 dps, 208 Hz
  imu.writeReg(LSM6::CTRL2_G, 0b01010100);

  // Initialize encoders
  encoders.getCountsAndResetLeft();
  encoders.getCountsAndResetRight();

  // Initialize encoder tracking (should be 0 after reset)
  encoder_left_counts_prev = encoders.getCountsLeft();
  encoder_right_counts_prev = encoders.getCountsRight();

  // Initialize motors (stopped)
  motors.setSpeeds(0, 0);

  // Wait for serial connection (optional)
  delay(500);

  // Indicate ready
  ledGreen(1);
  delay(500);
  ledGreen(0);
}

// ============================================================================
// MAIN LOOP
// ============================================================================

void loop() {
  unsigned long current_time_us = micros();
  unsigned long current_time_ms = millis();

  // Check button A to toggle debug mode (debounced)
  if (current_time_ms - last_button_check_ms >= 200) {  // Check every 200ms
    last_button_check_ms = current_time_ms;
    if (buttonA.getSingleDebouncedPress()) {
      debug_mode = !debug_mode;
      if (debug_mode) {
        Serial.println("\n\n=== DEBUG MODE ENABLED ===");
        Serial.println("Human-readable output active");
        Serial.println("Press button A again to return to binary mode");
        Serial.println("===========================\n");
      } else {
        Serial.println("\n=== BINARY MODE ENABLED ===\n");
        delay(100);  // Let message send before switching to binary
      }
    }
  }

  // Read IMU at fixed rate
  if (current_time_us - last_imu_time_us >= IMU_SAMPLE_PERIOD_US) {
    last_imu_time_us = current_time_us;
    readIMU();
    readEncoders();
    sendSensorPacket();
  }

  // Check for incoming control packets (only in binary mode)
  if (!debug_mode && Serial.available() >= 10) {  // Control packet size
    receiveControlPacket();
  }

  // Apply motor commands
  applyMotorCommands();
}

// ============================================================================
// IMU FUNCTIONS
// ============================================================================

void readIMU() {
  imu.read();

  // Convert accelerometer to m/s²
  // LSM6DS33 ±4g mode: 1 LSB = 0.122 mg
  // Conversion: raw * 0.122e-3 * 9.81 m/s²
  const float accel_scale = 0.122e-3 * 9.81;
  accel_x_mps2 = imu.a.x * accel_scale;
  accel_y_mps2 = imu.a.y * accel_scale;
  accel_z_mps2 = imu.a.z * accel_scale;

  // Convert gyroscope to rad/s
  // LSM6DS33 ±500 dps mode: 1 LSB = 17.50 mdps
  // Conversion: raw * 17.50e-3 * (π/180) rad/s
  const float gyro_scale = 17.50e-3 * (PI / 180.0);
  gyro_x_radps = imu.g.x * gyro_scale;
  gyro_y_radps = imu.g.y * gyro_scale;
  gyro_z_radps = imu.g.z * gyro_scale;
}

// ============================================================================
// ENCODER FUNCTIONS
// ============================================================================

void readEncoders() {
  // Read encoder counts (these are cumulative and will overflow int16_t)
  int16_t counts_left = encoders.getCountsLeft();
  int16_t counts_right = encoders.getCountsRight();

  // Compute delta since last read (handles overflow correctly)
  int16_t delta_left = counts_left - encoder_left_counts_prev;
  int16_t delta_right = counts_right - encoder_right_counts_prev;

  // Update previous counts
  encoder_left_counts_prev = counts_left;
  encoder_right_counts_prev = counts_right;

  // Accumulate deltas in radians (float has much larger range than int16)
  // One wheel revolution = 2π radians = WHEEL_COUNTS_PER_REV counts
  encoder_left_rad += ((float)delta_left / WHEEL_COUNTS_PER_REV) * TWO_PI;
  encoder_right_rad += ((float)delta_right / WHEEL_COUNTS_PER_REV) * TWO_PI;
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
  // Simple linear mapping: speed = K * torque
  // Note: This should be calibrated during system identification!
  float speed = torque_nm * TORQUE_TO_SPEED_GAIN;

  // Clamp to motor limits
  if (speed > MOTOR_MAX_SPEED) speed = MOTOR_MAX_SPEED;
  if (speed < -MOTOR_MAX_SPEED) speed = -MOTOR_MAX_SPEED;

  return (int16_t)speed;
}

// ============================================================================
// SERIAL COMMUNICATION
// ============================================================================

void sendSensorPacket() {
  /**
   * Sensor packet format (45 bytes):
   * - uint8: header (0xBB)
   * - uint32: timestamp (microseconds)
   * - float32[3]: accelerometer XYZ (m/s²)
   * - float32[3]: gyroscope XYZ (rad/s)
   * - float32[2]: encoder L/R (radians)
   * - uint8: checksum
   */

  // Debug mode: print human-readable data
  if (debug_mode) {
    static unsigned long last_debug_print_ms = 0;
    unsigned long now = millis();

    // Print at 10 Hz in debug mode (not 200 Hz - too fast to read!)
    if (now - last_debug_print_ms >= 100) {
      last_debug_print_ms = now;

      Serial.println("--- Sensor Data ---");
      Serial.print("Timestamp: "); Serial.print(micros()); Serial.println(" us");
      Serial.print("Accel (m/s²):  X="); Serial.print(accel_x_mps2, 4);
      Serial.print(" Y="); Serial.print(accel_y_mps2, 4);
      Serial.print(" Z="); Serial.print(accel_z_mps2, 4);
      Serial.print(" |Mag|="); Serial.println(sqrt(accel_x_mps2*accel_x_mps2 +
                                                     accel_y_mps2*accel_y_mps2 +
                                                     accel_z_mps2*accel_z_mps2), 4);
      Serial.print("Gyro (rad/s):  X="); Serial.print(gyro_x_radps, 4);
      Serial.print(" Y="); Serial.print(gyro_y_radps, 4);
      Serial.print(" Z="); Serial.println(gyro_z_radps, 4);
      Serial.print("Encoders (rad): L="); Serial.print(encoder_left_rad, 4);
      Serial.print(" R="); Serial.println(encoder_right_rad, 4);
      Serial.print("Motors (Nm):    L="); Serial.print(torque_left_nm, 4);
      Serial.print(" R="); Serial.println(torque_right_nm, 4);
      Serial.println();
    }
    return;  // Don't send binary packet in debug mode
  }

  // Binary mode: send packet as before
  byte packet[38];  // 1 + 4 + 12 + 12 + 8 + 1 = 38 bytes
  int idx = 0;

  // Header
  packet[idx++] = SENSOR_PACKET_HEADER;

  // Timestamp (microseconds)
  unsigned long timestamp = micros();
  memcpy(&packet[idx], &timestamp, sizeof(timestamp));
  idx += sizeof(timestamp);

  // Accelerometer
  memcpy(&packet[idx], &accel_x_mps2, sizeof(float));
  idx += sizeof(float);
  memcpy(&packet[idx], &accel_y_mps2, sizeof(float));
  idx += sizeof(float);
  memcpy(&packet[idx], &accel_z_mps2, sizeof(float));
  idx += sizeof(float);

  // Gyroscope
  memcpy(&packet[idx], &gyro_x_radps, sizeof(float));
  idx += sizeof(float);
  memcpy(&packet[idx], &gyro_y_radps, sizeof(float));
  idx += sizeof(float);
  memcpy(&packet[idx], &gyro_z_radps, sizeof(float));
  idx += sizeof(float);

  // Encoders
  memcpy(&packet[idx], &encoder_left_rad, sizeof(float));
  idx += sizeof(float);
  memcpy(&packet[idx], &encoder_right_rad, sizeof(float));
  idx += sizeof(float);

  // Checksum (XOR of all bytes)
  byte checksum = 0;
  for (int i = 0; i < idx; i++) {
    checksum ^= packet[i];
  }
  packet[idx++] = checksum;

  // Send packet
  Serial.write(packet, idx);
}

void receiveControlPacket() {
  /**
   * Control packet format (10 bytes):
   * - uint8: header (0xAA)
   * - float32: left torque (N⋅m)
   * - float32: right torque (N⋅m)
   * - uint8: checksum
   */

  byte packet[10];

  // Read packet
  if (Serial.readBytes(packet, 10) != 10) {
    return;  // Incomplete packet
  }

  // Verify header
  if (packet[0] != CONTROL_PACKET_HEADER) {
    // Not a valid control packet - flush and return
    while (Serial.available() > 0) {
      Serial.read();
    }
    return;
  }

  // Verify checksum
  byte checksum_received = packet[9];
  byte checksum_computed = 0;
  for (int i = 0; i < 9; i++) {
    checksum_computed ^= packet[i];
  }

  if (checksum_received != checksum_computed) {
    return;  // Checksum mismatch
  }

  // Parse torque commands
  memcpy(&torque_left_nm, &packet[1], sizeof(float));
  memcpy(&torque_right_nm, &packet[5], sizeof(float));

  // Update last packet time for watchdog
  last_packet_time_ms = millis();

  // Note: LED blink removed to avoid blocking delay during motor control
  // The 1ms delay was causing encoder counts to be missed during high-frequency commands
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

byte computeChecksum(const byte* data, size_t length) {
  byte checksum = 0;
  for (size_t i = 0; i < length; i++) {
    checksum ^= data[i];
  }
  return checksum;
}

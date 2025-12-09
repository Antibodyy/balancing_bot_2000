#!/usr/bin/env python3
"""Debug binary packet communication with Arduino.

This script shows raw bytes, checksums, and packet structure to help
diagnose communication issues.

Usage:
    python tests/hardware/debug_binary_packets.py --port /dev/ttyACM0
"""

import argparse
import sys
import struct
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import serial
except ImportError:
    print("Error: pyserial not installed")
    print("Install with: pip install pyserial")
    sys.exit(1)


SENSOR_PACKET_HEADER = 0xBB
SENSOR_PACKET_SIZE = 38  # 1 + 4 + 12 + 12 + 8 + 1 = 38 bytes


def compute_checksum(data: bytes) -> int:
    """Compute XOR checksum."""
    checksum = 0
    for byte in data:
        checksum ^= byte
    return checksum & 0xFF


def debug_packets(port: str, count: int = 10):
    """Read and analyze packets with detailed debugging.

    Args:
        port: Serial port
        count: Number of packets to analyze
    """
    print("=" * 70)
    print("BINARY PACKET DEBUGGER")
    print("=" * 70)
    print(f"Port: {port}")
    print(f"Expected packet size: {SENSOR_PACKET_SIZE} bytes")
    print(f"Expected header: 0x{SENSOR_PACKET_HEADER:02X}")
    print()
    print("Looking for packets...")
    print("-" * 70)

    try:
        ser = serial.Serial(port, 115200, timeout=1)
        # Give Arduino time to start
        import time
        time.sleep(2)
        ser.reset_input_buffer()

        packets_found = 0
        bytes_read = 0

        while packets_found < count:
            # Search for header
            while True:
                byte = ser.read(1)
                if len(byte) == 0:
                    print(f"\nTimeout waiting for data (read {bytes_read} bytes total)")
                    return
                bytes_read += 1

                if byte[0] == SENSOR_PACKET_HEADER:
                    print(f"\n[Packet #{packets_found + 1}] Found header at byte {bytes_read}")
                    break

            # Read rest of packet
            rest_of_packet = ser.read(SENSOR_PACKET_SIZE - 1)
            if len(rest_of_packet) != SENSOR_PACKET_SIZE - 1:
                print(f"  ✗ Incomplete packet (got {len(rest_of_packet) + 1}/{SENSOR_PACKET_SIZE} bytes)")
                continue

            full_packet = byte + rest_of_packet

            # Show hex dump (first 20 bytes and last 5 bytes)
            print(f"  Hex dump (first 20 bytes):")
            print(f"    {' '.join(f'{b:02X}' for b in full_packet[:20])}")
            print(f"  Hex dump (last 5 bytes):")
            print(f"    ... {' '.join(f'{b:02X}' for b in full_packet[-5:])}")

            # Verify checksum
            checksum_received = full_packet[-1]
            checksum_computed = compute_checksum(full_packet[:-1])

            print(f"  Checksum: received=0x{checksum_received:02X}, computed=0x{checksum_computed:02X}", end="")
            if checksum_received == checksum_computed:
                print(" ✓ MATCH")
            else:
                print(" ✗ MISMATCH")

            # Parse packet structure
            try:
                # Unpack: header(1) + timestamp(4) + 6 floats(24) + 2 floats(8) + checksum(1) = 38... wait
                # Let me recalculate: 1 + 4 + 3*4 + 3*4 + 2*4 + 1 = 1 + 4 + 12 + 12 + 8 + 1 = 38
                # But we said 45 bytes in the firmware comment... let me check

                # Actually from firmware:
                # header(1) + timestamp(4) + accel(3*4=12) + gyro(3*4=12) + encoder(2*4=8) + checksum(1) = 38
                # But SENSOR_PACKET_SIZE = 45... there's a mismatch!

                print(f"  Unpacking data (assuming little-endian)...")

                idx = 1  # Skip header
                timestamp = struct.unpack('<I', full_packet[idx:idx+4])[0]
                idx += 4

                accel = struct.unpack('<fff', full_packet[idx:idx+12])
                idx += 12

                gyro = struct.unpack('<fff', full_packet[idx:idx+12])
                idx += 12

                encoders = struct.unpack('<ff', full_packet[idx:idx+8])
                idx += 8

                print(f"    Timestamp: {timestamp} us")
                print(f"    Accel:     X={accel[0]:8.4f} Y={accel[1]:8.4f} Z={accel[2]:8.4f} m/s²")
                print(f"    Gyro:      X={gyro[0]:8.4f} Y={gyro[1]:8.4f} Z={gyro[2]:8.4f} rad/s")
                print(f"    Encoders:  L={encoders[0]:8.4f} R={encoders[1]:8.4f} rad")

                # Sanity checks
                accel_mag = (accel[0]**2 + accel[1]**2 + accel[2]**2)**0.5
                print(f"    Accel magnitude: {accel_mag:.4f} m/s² (expect ~9.8)")

                if 8.0 < accel_mag < 11.0:
                    print("    ✓ Accelerometer magnitude looks reasonable")
                else:
                    print("    ⚠ Accelerometer magnitude seems wrong!")

            except struct.error as e:
                print(f"    ✗ Error unpacking: {e}")

            packets_found += 1

        print("\n" + "=" * 70)
        print(f"Analyzed {packets_found} packets")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'ser' in locals():
            ser.close()


def main():
    parser = argparse.ArgumentParser(description='Debug binary packet communication')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                       help='Serial port (default: /dev/ttyACM0)')
    parser.add_argument('--count', type=int, default=10,
                       help='Number of packets to analyze (default: 10)')
    args = parser.parse_args()

    print("\n⚠️  IMPORTANT: Make sure Arduino is in BINARY mode (not debug mode)")
    print("   Press button A if you see human-readable text in serial monitor\n")

    input("Press Enter to start debugging...")

    debug_packets(args.port, args.count)


if __name__ == '__main__':
    main()

"""Hardware controller wrapper for MPC balancing robot.

This module provides the main control loop that integrates:
- I2C communication with Arduino
- MPC controller from simulation
- Real-time control loop timing
"""

import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from control_pipeline import BalanceController, ControlOutput
from mpc import ReferenceCommand, ReferenceMode
from .i2c_interface import BalboaI2CInterface


@dataclass
class ControlLoopStats:
    """Statistics from control loop execution.

    Attributes:
        iterations: Total control loop iterations
        avg_loop_time_ms: Average loop time (milliseconds)
        max_loop_time_ms: Maximum loop time (milliseconds)
        avg_solve_time_ms: Average MPC solve time (milliseconds)
        max_solve_time_ms: Maximum MPC solve time (milliseconds)
        timing_violations: Count of iterations exceeding target period
    """
    iterations: int = 0
    avg_loop_time_ms: float = 0.0
    max_loop_time_ms: float = 0.0
    avg_solve_time_ms: float = 0.0
    max_solve_time_ms: float = 0.0
    timing_violations: int = 0


class HardwareBalanceController:
    """Hardware wrapper around simulation BalanceController.

    This class manages the real-time control loop:
    1. Read sensor data from Arduino via I2C
    2. Feed to MPC controller (from simulation)
    3. Send torque commands back to Arduino
    4. Monitor timing and log data

    Example:
        >>> from hardware import load_hardware_mpc, BalboaI2CInterface
        >>> from hardware import HardwareBalanceController
        >>> from mpc import ReferenceCommand, ReferenceMode
        >>>
        >>> controller = load_hardware_mpc()
        >>> i2c = BalboaI2CInterface(bus=1, address=0x20)
        >>> hw_controller = HardwareBalanceController(i2c, controller)
        >>>
        >>> reference = ReferenceCommand(mode=ReferenceMode.BALANCE)
        >>> hw_controller.run_control_loop(reference, duration_s=60.0)
    """

    def __init__(self,
                 i2c_interface: BalboaI2CInterface,
                 balance_controller: BalanceController,
                 target_frequency_hz: float = 50.0,
                 enable_logging: bool = True):
        """Initialize hardware controller.

        Args:
            i2c_interface: I2C connection to Arduino
            balance_controller: MPC controller (from simulation code!)
            target_frequency_hz: Target control loop frequency (default 50 Hz)
            enable_logging: Enable data logging to console
        """
        self.i2c = i2c_interface
        self.controller = balance_controller
        self.target_period_s = 1.0 / target_frequency_hz
        self.enable_logging = enable_logging

        self.stats = ControlLoopStats()
        self._running = False

    def run_control_loop(self,
                        reference_command: ReferenceCommand,
                        duration_s: Optional[float] = None) -> ControlLoopStats:
        """Execute MPC control loop.

        Args:
            reference_command: Reference trajectory command (balance, velocity, position)
            duration_s: Run duration in seconds (None = run until KeyboardInterrupt)

        Returns:
            ControlLoopStats: Statistics from control loop execution

        Raises:
            KeyboardInterrupt: User pressed Ctrl+C to stop
        """
        self._running = True
        start_time = time.time()
        iteration = 0

        print(f"Starting control loop at {1.0/self.target_period_s:.1f} Hz")
        print(f"Reference mode: {reference_command.mode}")
        print("Press Ctrl+C to stop")
        print("-" * 60)

        try:
            while self._running:
                loop_start_time = time.time()

                # Check duration limit
                if duration_s is not None:
                    elapsed = loop_start_time - start_time
                    if elapsed >= duration_s:
                        print(f"\nReached duration limit ({duration_s:.1f}s)")
                        break

                # Execute one control iteration
                loop_time_ms, solve_time_ms = self._control_iteration(reference_command)

                # Update statistics
                iteration += 1
                self._update_stats(iteration, loop_time_ms, solve_time_ms)

                # Print progress every 50 iterations (~1 second)
                if iteration % 50 == 0 and self.enable_logging:
                    self._print_progress(iteration, loop_time_ms, solve_time_ms)

                # Sleep to maintain target frequency
                sleep_time = self.target_period_s - (time.time() - loop_start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.stats.timing_violations += 1

        except KeyboardInterrupt:
            print("\n\nControl loop stopped by user")

        finally:
            # Emergency stop on exit
            print("Stopping motors...")
            self.i2c.emergency_stop()
            self._running = False

        # Print final statistics
        self._print_final_stats()
        return self.stats

    def _control_iteration(self, reference_command: ReferenceCommand) -> tuple[float, float]:
        """Execute one control loop iteration.

        Args:
            reference_command: Reference command for MPC

        Returns:
            Tuple of (loop_time_ms, solve_time_ms)
        """
        iter_start = time.time()

        # 1. Read sensor data from Arduino
        sensor_data = self.i2c.read_sensors()

        if sensor_data is None:
            # No data available - send zero torque and return
            self.i2c.send_motor_command(0.0, 0.0)
            return 0.0, 0.0

        # 2. Run MPC controller (from simulation!)
        control_output = self.controller.step(sensor_data, reference_command)

        # 3. Send torque commands to Arduino
        self.i2c.send_motor_command(
            control_output.torque_left_nm,
            control_output.torque_right_nm
        )

        # 4. Compute timing
        loop_time_ms = (time.time() - iter_start) * 1000.0
        solve_time_ms = control_output.timing.solve_time_s * 1000.0 if control_output.timing else 0.0

        return loop_time_ms, solve_time_ms

    def _update_stats(self, iteration: int, loop_time_ms: float, solve_time_ms: float):
        """Update running statistics.

        Args:
            iteration: Current iteration number
            loop_time_ms: Loop time for this iteration
            solve_time_ms: Solve time for this iteration
        """
        self.stats.iterations = iteration

        # Update averages using online algorithm
        self.stats.avg_loop_time_ms += (loop_time_ms - self.stats.avg_loop_time_ms) / iteration
        self.stats.avg_solve_time_ms += (solve_time_ms - self.stats.avg_solve_time_ms) / iteration

        # Update maximums
        self.stats.max_loop_time_ms = max(self.stats.max_loop_time_ms, loop_time_ms)
        self.stats.max_solve_time_ms = max(self.stats.max_solve_time_ms, solve_time_ms)

    def _print_progress(self, iteration: int, loop_time_ms: float, solve_time_ms: float):
        """Print progress update to console.

        Args:
            iteration: Current iteration number
            loop_time_ms: Loop time for this iteration
            solve_time_ms: Solve time for this iteration
        """
        target_ms = self.target_period_s * 1000.0
        violation_pct = (self.stats.timing_violations / iteration) * 100.0

        print(f"[{iteration:5d}] Loop: {loop_time_ms:5.2f}ms | "
              f"Solve: {solve_time_ms:5.2f}ms | "
              f"Target: {target_ms:.1f}ms | "
              f"Violations: {violation_pct:.1f}%")

    def _print_final_stats(self):
        """Print final statistics summary."""
        print("\n" + "=" * 60)
        print("CONTROL LOOP STATISTICS")
        print("=" * 60)
        print(f"Total iterations:        {self.stats.iterations}")
        print(f"Avg loop time:           {self.stats.avg_loop_time_ms:.2f} ms")
        print(f"Max loop time:           {self.stats.max_loop_time_ms:.2f} ms")
        print(f"Avg MPC solve time:      {self.stats.avg_solve_time_ms:.2f} ms")
        print(f"Max MPC solve time:      {self.stats.max_solve_time_ms:.2f} ms")
        print(f"Target period:           {self.target_period_s * 1000:.1f} ms")
        print(f"Timing violations:       {self.stats.timing_violations} "
              f"({self.stats.timing_violations / max(self.stats.iterations, 1) * 100:.1f}%)")
        print("=" * 60)

    def stop(self):
        """Stop the control loop."""
        self._running = False

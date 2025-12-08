"""Main balance controller orchestrating MPC and state estimation.

This module provides the top-level controller that coordinates:
1. State estimation from sensor data
2. Reference trajectory generation
3. MPC optimization
4. Control output generation
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from robot_dynamics.parameters import STATE_DIMENSION, CONTROL_DIMENSION
from mpc.linear_mpc_solver import LinearMPCSolver, MPCSolution
from mpc.reference_generator import ReferenceGenerator, ReferenceCommand
from state_estimation.complementary_filter import ComplementaryFilter, IMUReading
from control_pipeline.timing import ControlLoopTimer, IterationTiming


@dataclass
class SensorData:
    """Raw sensor measurements from hardware.

    Attributes:
        acceleration_mps2: Accelerometer readings [a_x, a_y, a_z]
        angular_velocity_radps: Gyroscope readings [omega_x, omega_y, omega_z]
        encoder_left_rad: Left wheel encoder position in radians
        encoder_right_rad: Right wheel encoder position in radians
        timestamp_s: Measurement timestamp in seconds
    """

    acceleration_mps2: np.ndarray
    angular_velocity_radps: np.ndarray
    encoder_left_rad: float
    encoder_right_rad: float
    timestamp_s: float


@dataclass
class ControlOutput:
    """Output from control pipeline.

    Attributes:
        torque_left_nm: Left wheel torque command in N*m
        torque_right_nm: Right wheel torque command in N*m
        state_estimate: Current state estimate
        predicted_trajectory: Predicted state trajectory from MPC
        timing: Timing breakdown for this iteration
        mpc_solution: Full MPC solution for diagnostics
    """

    torque_left_nm: float
    torque_right_nm: float
    state_estimate: np.ndarray
    predicted_trajectory: np.ndarray
    timing: IterationTiming
    mpc_solution: MPCSolution


class BalanceController:
    """Main control loop orchestrator for self-balancing robot.

    Coordinates state estimation, reference generation, and MPC solving
    to produce control outputs from sensor data.

    The controller maintains internal state for:
    - Complementary filter for pitch estimation
    - Previous encoder readings for velocity estimation
    - Control loop timing

    Attributes:
        mpc_solver: Linear MPC solver instance
        state_estimator: Complementary filter for pitch estimation
        reference_generator: Reference trajectory generator
        timer: Control loop timer for performance monitoring
    """

    def __init__(
        self,
        mpc_solver: LinearMPCSolver,
        state_estimator: ComplementaryFilter,
        reference_generator: ReferenceGenerator,
        sampling_period_s: float,
        wheel_radius_m: float,
        track_width_m: float,
        use_simulation_velocity: bool = False,
    ) -> None:
        """Initialize balance controller.

        Args:
            mpc_solver: Configured MPC solver
            state_estimator: Complementary filter for pitch estimation
            reference_generator: Reference trajectory generator
            sampling_period_s: Control loop period
            wheel_radius_m: Wheel radius for encoder to velocity conversion
            track_width_m: Track width for differential drive kinematics
            use_simulation_velocity: If True, use true ground velocity from simulation
                                     instead of encoder-based estimation (fixes the bug)
        """
        self._mpc_solver = mpc_solver
        self._state_estimator = state_estimator
        self._reference_generator = reference_generator
        self._sampling_period_s = sampling_period_s
        self._wheel_radius_m = wheel_radius_m
        self._track_width_m = track_width_m
        self._use_simulation_velocity = use_simulation_velocity

        # Timer for performance monitoring
        self._timer = ControlLoopTimer(deadline_s=sampling_period_s)

        # State tracking
        self._previous_encoder_left_rad: Optional[float] = None
        self._previous_encoder_right_rad: Optional[float] = None
        self._previous_timestamp_s: Optional[float] = None

        # Position and heading integration
        self._position_m: float = 0.0
        self._heading_rad: float = 0.0

        # Simulation mode: true ground velocity (set by simulation)
        self._true_ground_velocity: float = 0.0

    def step(
        self,
        sensor_data: SensorData,
        reference_command: ReferenceCommand,
    ) -> ControlOutput:
        """Execute one control step.

        Args:
            sensor_data: Raw sensor measurements
            reference_command: Desired reference mode and targets

        Returns:
            Control output with torque commands and diagnostics
        """
        self._timer.start_iteration()

        # 1. Update state estimate
        state_estimate = self._estimate_state(sensor_data)
        self._timer.mark_estimation_complete()

        # 2. Generate reference trajectory
        reference_trajectory = self._reference_generator.generate(
            reference_command, state_estimate
        )
        self._timer.mark_reference_complete()

        # 3. Solve MPC
        mpc_solution = self._mpc_solver.solve(state_estimate, reference_trajectory)
        self._timer.mark_solve_complete()

        # 4. Extract control
        optimal_control = mpc_solution.optimal_control

        # End timing
        timing = self._timer.end_iteration()

        return ControlOutput(
            torque_left_nm=optimal_control[0],
            torque_right_nm=optimal_control[1],
            state_estimate=state_estimate,
            predicted_trajectory=mpc_solution.predicted_trajectory,
            timing=timing,
            mpc_solution=mpc_solution,
        )

    def _estimate_state(self, sensor_data: SensorData) -> np.ndarray:
        """Estimate full state from sensor data.

        Combines complementary filter pitch estimate with encoder-based
        position and velocity estimates.

        Args:
            sensor_data: Raw sensor measurements

        Returns:
            State estimate [x, theta, psi, dx, dtheta, dpsi]
        """
        # Create IMU reading
        imu_reading = IMUReading(
            acceleration_mps2=sensor_data.acceleration_mps2,
            angular_velocity_radps=sensor_data.angular_velocity_radps,
        )

        # Compute timestep
        if self._previous_timestamp_s is not None:
            timestep_s = sensor_data.timestamp_s - self._previous_timestamp_s
        else:
            timestep_s = self._sampling_period_s

        # Update pitch estimate from complementary filter
        pitch_rad = self._state_estimator.update(imu_reading, timestep_s)
        pitch_rate_radps = sensor_data.angular_velocity_radps[1]  # omega_y

        # Compute wheel velocities from encoder differences
        if self._previous_encoder_left_rad is not None:
            encoder_diff_left = (
                sensor_data.encoder_left_rad - self._previous_encoder_left_rad
            )
            encoder_diff_right = (
                sensor_data.encoder_right_rad - self._previous_encoder_right_rad
            )

            wheel_velocity_left = (
                encoder_diff_left * self._wheel_radius_m / timestep_s
            )
            wheel_velocity_right = (
                encoder_diff_right * self._wheel_radius_m / timestep_s
            )

            # Differential drive kinematics
            # FIX: In simulation mode, use true ground velocity instead of encoder-based
            # The encoder measures wheel rotation relative to body, which is wrong for
            # inverted pendulum robots where body pitches while wheels rotate.
            if self._use_simulation_velocity:
                # Simulation: use ground truth velocity
                forward_velocity = self._true_ground_velocity
            else:
                # Hardware: encoder-based (BUGGY - needs proper compensation for pitch)
                # TODO: Fix for hardware by compensating for body pitch rate
                forward_velocity = (wheel_velocity_left + wheel_velocity_right) / 2

            yaw_rate = (
                (wheel_velocity_right - wheel_velocity_left) / self._track_width_m
            )

            # Integrate position and heading
            self._position_m += forward_velocity * timestep_s
            self._heading_rad += yaw_rate * timestep_s
        else:
            forward_velocity = 0.0
            yaw_rate = 0.0

        # Update tracking
        self._previous_encoder_left_rad = sensor_data.encoder_left_rad
        self._previous_encoder_right_rad = sensor_data.encoder_right_rad
        self._previous_timestamp_s = sensor_data.timestamp_s

        # Build state vector
        state = np.zeros(STATE_DIMENSION)
        state[0] = self._position_m       # x
        state[1] = pitch_rad              # theta
        state[2] = self._heading_rad      # psi
        state[3] = forward_velocity       # dx
        state[4] = pitch_rate_radps       # dtheta
        state[5] = yaw_rate               # dpsi

        return state

    def reset(self) -> None:
        """Reset controller state.

        Clears accumulated position/heading and resets state estimator.
        """
        self._state_estimator.reset()
        self._previous_encoder_left_rad = None
        self._previous_encoder_right_rad = None
        self._previous_timestamp_s = None
        self._position_m = 0.0
        self._heading_rad = 0.0

    @property
    def timer(self) -> ControlLoopTimer:
        """Control loop timer for performance monitoring."""
        return self._timer

    @property
    def position_m(self) -> float:
        """Current integrated position estimate."""
        return self._position_m

    @property
    def heading_rad(self) -> float:
        """Current integrated heading estimate."""
        return self._heading_rad

"""Main balance controller orchestrating MPC and state estimation.

This module provides the top-level controller that coordinates:
1. State estimation from sensor data
2. Reference trajectory generation
3. MPC optimization
4. Control output generation
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from robot_dynamics.parameters import (
    STATE_DIMENSION,
    CONTROL_DIMENSION,
    RobotParameters,
    VELOCITY_INDEX,
    PITCH_INDEX,
    PITCH_RATE_INDEX,
)
from robot_dynamics.linearization import linearize_at_state, build_jacobian_functions
from robot_dynamics.discretization import discretize_linear_dynamics
from mpc.linear_mpc_solver import LinearMPCSolver, MPCSolution
from mpc.reference_generator import ReferenceGenerator, ReferenceCommand, ReferenceMode
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
    state_deviation: np.ndarray
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

    BALANCE_R_SCALE: float = 0.01

    def __init__(
        self,
        mpc_solver: LinearMPCSolver,
        state_estimator: ComplementaryFilter,
        reference_generator: ReferenceGenerator,
        sampling_period_s: float,
        wheel_radius_m: float,
        track_width_m: float,
        robot_params: RobotParameters,
        equilibrium_state: Optional[np.ndarray] = None,
        equilibrium_control: Optional[np.ndarray] = None,
        online_linearization_enabled: bool = False,
        use_simulation_velocity: bool = False,
        use_simulation_heading: bool = False,
        use_simulation_state_estimate: bool = False,
    ) -> None:
        """Initialize balance controller.

        Args:
            mpc_solver: Configured MPC solver
            state_estimator: Complementary filter for pitch estimation
            reference_generator: Reference trajectory generator
            sampling_period_s: Control loop period
            wheel_radius_m: Wheel radius for encoder to velocity conversion
            track_width_m: Track width for differential drive kinematics
            robot_params: Robot parameters for online linearization
            online_linearization_enabled: If True, re-linearize at current state each step
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
        self._use_simulation_heading = use_simulation_heading
        self._use_simulation_state_estimate = use_simulation_state_estimate
        self._true_heading = 0.0
        self._true_yaw_rate = 0.0
        self._control_limit_nm = (
            mpc_solver.control_limit_nm if hasattr(mpc_solver, "control_limit_nm") else None
        )
        self._debug_q_printed = False
        self._debug_torque_sum = 0.0
        self._debug_torque_samples = 0
        self._debug_torque_reported = False
        self._debug_torque_saturation = False
        self._latest_sim_state: Optional[np.ndarray] = None

        # Store equilibrium operating point (x_eq, u_eq) so MPC works in delta coordinates.
        if equilibrium_state is None:
            equilibrium_state = np.zeros(STATE_DIMENSION)
        if equilibrium_control is None:
            equilibrium_control = np.zeros(CONTROL_DIMENSION)
        self._equilibrium_state = np.asarray(equilibrium_state, dtype=float).reshape(STATE_DIMENSION)
        self._equilibrium_control = np.asarray(equilibrium_control, dtype=float).reshape(CONTROL_DIMENSION)
        self._last_control = self._equilibrium_control.copy()

        # Online linearization
        self._robot_params = robot_params
        self._online_linearization_enabled = online_linearization_enabled

        # Cache Jacobian functions for efficient linearization
        if self._online_linearization_enabled:
            self._jacobian_functions = build_jacobian_functions(robot_params)
        else:
            self._jacobian_functions = None

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

        # 1. Update state estimate (absolute coordinates)
        if self._use_simulation_state_estimate and self._latest_sim_state is not None:
            state_estimate = self._latest_sim_state.copy()
        else:
            state_estimate = self._estimate_state(sensor_data)
        # Express deviation from equilibrium for linear MPC
        state_deviation = state_estimate - self._equilibrium_state
        self._timer.mark_estimation_complete()

        # 2. Update linearization if enabled
        if self._online_linearization_enabled:
            self._timer.start_model_update()
            self._update_linearization(state_estimate)
            self._timer.mark_model_update_complete()

        # 3. Generate reference trajectory
        reference_deviation = self._reference_generator.generate(
            reference_command, state_estimate
        )
        # The generator's "zero" corresponds to the equilibrium lean,
        # so the returned values are already deviations in MPC coordinates.
        self._timer.mark_reference_complete()

        # 4. Solve MPC (optionally override cost for balance mode)
        debug_q = os.getenv("MPC_DEBUG_Q")
        if debug_q and not self._debug_q_printed:
            q_diag = np.diag(self._mpc_solver.state_cost)
            r_diag = np.diag(self._mpc_solver.control_cost)
            print(
                "[DEBUG] MPC state indices: [x=0, theta=1, psi=2, dx=3, dtheta=4, dpsi=5], "
                f"Q diagonal: {q_diag}, R diagonal: {r_diag}"
            )
            self._debug_q_printed = True
        if (
            reference_command.mode == ReferenceMode.BALANCE
            and self._mpc_solver is not None
        ):
            q_override = self._mpc_solver.base_state_cost.copy()
            q_override[0, 0] = 40.0
            q_override[1, 1] = 40.0
            q_override[3, 3] = 400.0
            r_override = self._mpc_solver.base_control_cost.copy()
            r_override *= self.BALANCE_R_SCALE
            terminal_cost_override = self._mpc_solver.base_terminal_cost.copy()
            terminal_cost_override[VELOCITY_INDEX, VELOCITY_INDEX] *= 150.0
            terminal_cost_override[PITCH_INDEX, PITCH_INDEX] *= 20.0
            mpc_solution = self._mpc_solver.solve(
                state_deviation,
                reference_deviation,
                state_cost_override=q_override,
                control_cost_override=r_override,
                terminal_cost_override=terminal_cost_override,
                terminal_velocity_limit_override=0.05,
                terminal_pitch_limit_override=0.10,
                terminal_pitch_rate_limit_override=0.5,
            )
            if debug_q == "2":
                q_diag = np.diag(self._mpc_solver.state_cost)
                r_diag = np.diag(self._mpc_solver.control_cost)
                print(f"[DEBUG] Balance override Q diag: {q_diag}, R diag: {r_diag}")
        else:
            mpc_solution = self._mpc_solver.solve(state_deviation, reference_deviation)
        self._timer.mark_solve_complete()

        solver_status = (mpc_solution.solver_status or "").lower()

        # 4. Extract control
        optimal_control_delta = mpc_solution.optimal_control
        if solver_status != "optimal":
            optimal_control = self._last_control.copy()
        else:
            optimal_control = optimal_control_delta + self._equilibrium_control
            self._last_control = optimal_control.copy()

        if debug_q and not self._debug_torque_reported:
            self._debug_torque_sum += float(np.mean(np.abs(optimal_control)))
            self._debug_torque_samples += 1
            if (
                self._control_limit_nm is not None
                and np.any(np.abs(optimal_control) >= 0.98 * self._control_limit_nm)
            ):
                self._debug_torque_saturation = True
            elapsed = self._debug_torque_samples * self._sampling_period_s
            if elapsed >= 1.0:
                mean_torque = self._debug_torque_sum / self._debug_torque_samples
                print(
                    "[DEBUG] Balance torque mean(|u|) over "
                    f"{elapsed:.3f}s = {mean_torque:.5f} Nm, "
                    f"saturation={self._debug_torque_saturation}"
                )
                self._debug_torque_reported = True

        predicted_trajectory = mpc_solution.predicted_trajectory
        if predicted_trajectory.ndim == 1:
            predicted_trajectory = predicted_trajectory.reshape(1, -1)

        if solver_status != "optimal" or predicted_trajectory.size == 0:
            horizon_plus_one = predicted_trajectory.shape[0] or (
                self._mpc_solver.prediction_horizon_steps + 1
            )
            fallback = np.full((horizon_plus_one, STATE_DIMENSION), np.nan)
            fallback[0, :] = state_estimate
            predicted_trajectory = fallback
            print(
                f"[WARN] MPC solver returned status '{mpc_solution.solver_status}'. "
                f"Using state estimate to seed predicted trajectory."
            )
        else:
            predicted_trajectory = (
                predicted_trajectory + self._equilibrium_state
            )

        # End timing
        timing = self._timer.end_iteration()

        return ControlOutput(
            torque_left_nm=optimal_control[0],
            torque_right_nm=optimal_control[1],
            state_estimate=state_estimate,
            state_deviation=state_deviation,
            predicted_trajectory=predicted_trajectory,
            timing=timing,
            mpc_solution=mpc_solution,
        )

    def _update_linearization(self, state_estimate: np.ndarray) -> None:
        """Update MPC linearization based on current state.

        Linearizes dynamics at the current state and updates MPC solver.
        This is called every control step when online_linearization_enabled=True.

        Args:
            state_estimate: Current state estimate [x, theta, psi, dx, dtheta, dpsi]
        """
        # Linearize at current state using current equilibrium control
        control_linearization_point = self._equilibrium_control.copy()

        linearized = linearize_at_state(
            self._robot_params,
            state_estimate,
            control_linearization_point,
            jacobian_functions=self._jacobian_functions  # Use cached functions
        )

        # Discretize
        discrete = discretize_linear_dynamics(
            linearized.state_matrix,
            linearized.control_matrix,
            self._sampling_period_s
        )

        # Update MPC solver (now fast with parameter-based approach)
        self._mpc_solver.update_dynamics(discrete)

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

            if self._use_simulation_heading:
                yaw_rate = self._true_yaw_rate
                self._heading_rad = self._true_heading
            else:
                yaw_rate = (
                    (wheel_velocity_right - wheel_velocity_left) / self._track_width_m
                )
                self._heading_rad += yaw_rate * timestep_s

            # Integrate position
            self._position_m += forward_velocity * timestep_s
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
        self._last_control = self._equilibrium_control.copy()
        self._latest_sim_state = None

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

    def update_simulation_state(self, state: np.ndarray) -> None:
        """Provide ground-truth state from simulation for estimator bypass."""
        if not self._use_simulation_state_estimate:
            return
        self._latest_sim_state = np.asarray(state, dtype=float).copy()

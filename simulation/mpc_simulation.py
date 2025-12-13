"""MuJoCo-based simulation for MPC validation.

This module provides a complete simulation environment that:
1. Runs the MuJoCo physics engine
2. Extracts state from simulation
3. Feeds sensor data to the MPC controller
4. Applies control outputs to actuators
5. Logs results for analysis

Note: MuJoCo is an optional dependency. If not installed, the module
can still be imported but simulation methods will raise ImportError.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Callable, Any, Tuple
import time

import numpy as np
from scipy.optimize import least_squares, brentq

# MuJoCo is optional - only required for actual simulation
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    mujoco = None  # type: ignore

from robot_dynamics import RobotParameters
from robot_dynamics.parameters import (
    STATE_DIMENSION,
    CONTROL_DIMENSION,
    POSITION_INDEX,
    PITCH_INDEX,
    YAW_INDEX,
    VELOCITY_INDEX,
    PITCH_RATE_INDEX,
    YAW_RATE_INDEX,
)
from mpc import (
    MPCConfig,
    LinearMPCSolver,
    compute_terminal_cost_dare,
    create_constraints_from_config,
    ReferenceGenerator,
    ReferenceCommand,
    ReferenceMode,
)
from robot_dynamics import (
    linearize_at_equilibrium,
    discretize_linear_dynamics,
    compute_equilibrium_state,
    compute_state_derivative,
)
from robot_dynamics.linearization import linearize_at_state
from robot_dynamics.orientation import quat_to_yaw
from state_estimation import EstimatorConfig, ComplementaryFilter
from control_pipeline import (
    BalanceController,
    SensorData,
    ControlOutput,
)


logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for MPC simulation.

    Attributes:
        model_path: Path to MuJoCo XML model file
        robot_params_path: Path to robot parameters YAML
        mpc_params_path: Path to MPC parameters YAML
        estimator_params_path: Path to estimator parameters YAML
        duration_s: Simulation duration in seconds
        render: Whether to render the simulation
        record_video: Whether to record video (requires render=True)
        video_path: Path for video output
    """

    model_path: str = 'mujoco_sim/robot_model.xml'
    robot_params_path: str = 'config/simulation/robot_params.yaml'
    mpc_params_path: str = 'config/simulation/mpc_params.yaml'
    estimator_params_path: str = 'config/simulation/estimator_params.yaml'
    duration_s: float = 10.0
    render: bool = False
    record_video: bool = False
    video_path: str = 'simulation_output.mp4'
    use_virtual_physics: bool = False  # Toggle analytic “virtual” plant instead of MuJoCo.


@dataclass
class SimulationResult:
    """Results from a simulation run.

    Attributes:
        time_s: Array of simulation timestamps
        state_history: State trajectory (N_steps, STATE_DIMENSION)
        control_history: Control trajectory (N_steps, CONTROL_DIMENSION)
        state_estimate_history: Estimated states (N_steps, STATE_DIMENSION)
        qp_solve_time_history: MPC solve times (N_steps,)
        model_update_time_history: Linearization/discretization times (N_steps,)
        iteration_history: IPOPT iteration counts per MPC solve (N_steps,)
        reference_history: Reference trajectories (N_steps, horizon+1, STATE_DIMENSION)
        pitch_error_history: Pitch error relative to slope
        velocity_error_history: Velocity tracking error
        torque_saturation_history: Boolean flags for torque saturation
        infeasible_history: Boolean flags for solver infeasibility
        success: Whether simulation completed without falling
        total_duration_s: Total wall-clock time for simulation
        mean_qp_solve_time_ms: Mean MPC solve time in milliseconds
        max_qp_solve_time_ms: Maximum MPC solve time in milliseconds
        deadline_violations: Number of times solve exceeded deadline
    """

    time_s: np.ndarray
    state_history: np.ndarray
    control_history: np.ndarray
    state_estimate_history: np.ndarray
    qp_solve_time_history: np.ndarray
    model_update_time_history: np.ndarray
    iteration_history: np.ndarray
    reference_history: List[np.ndarray] = field(default_factory=list)
    pitch_error_history: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity_error_history: np.ndarray = field(default_factory=lambda: np.array([]))
    torque_saturation_history: np.ndarray = field(default_factory=lambda: np.array([]))
    infeasible_history: np.ndarray = field(default_factory=lambda: np.array([]))
    success: bool = True
    total_duration_s: float = 0.0
    mean_qp_solve_time_ms: float = 0.0
    max_qp_solve_time_ms: float = 0.0
    deadline_violations: int = 0

    @property
    def solve_time_history(self) -> np.ndarray:
        """Backward-compatible access to QP solve history."""
        return self.qp_solve_time_history

    @property
    def mean_solve_time_ms(self) -> float:
        """Backward-compatible access to mean QP solve time."""
        return self.mean_qp_solve_time_ms

    @property
    def max_solve_time_ms(self) -> float:
        """Backward-compatible access to max QP solve time."""
        return self.max_qp_solve_time_ms


class MPCSimulation:
    """MuJoCo-based simulation for MPC validation.

    Runs the full control pipeline with MuJoCo physics, allowing
    validation of:
    - Balance stabilization
    - Disturbance rejection
    - Trajectory tracking
    - Timing verification

    The simulation uses the robot_dynamics module for linearization,
    not the legacy LQR script in LQR/LQR.py.
    """

    # MuJoCo joint indices (with freejoint)
    # Freejoint uses: qpos[0:3] = position (x,y,z), qpos[3:7] = quaternion (w,x,y,z)
    FREE_JOINT_POS_START = 0    # Position starts at qpos[0]
    FREE_JOINT_QUAT_START = 3   # Quaternion starts at qpos[3]
    MOTOR_R_JOINT = 7           # qpos[7]
    MOTOR_L_JOINT = 8           # qpos[8]

    # Velocity indices (freejoint velocities)
    FREE_JOINT_VEL_START = 0    # Linear velocity qvel[0:3]
    FREE_JOINT_ANGVEL_START = 3 # Angular velocity qvel[3:6]

    # MuJoCo actuator indices
    LEFT_ACTUATOR = 0
    RIGHT_ACTUATOR = 1

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        """Initialize MPC simulation.

        Args:
            config: Simulation configuration (uses defaults if None)
        """
        self._config = config or SimulationConfig()

        # Load configurations
        self._robot_params = RobotParameters.from_yaml(
            self._config.robot_params_path
        )
        self._mpc_config = MPCConfig.from_yaml(self._config.mpc_params_path)
        self._estimator_config = EstimatorConfig.from_yaml(
            self._config.estimator_params_path
        )
        # When True we integrate the symbolic dynamics instead of MuJoCo so the
        # controller can run end-to-end even if the physics engine misbehaves.
        self._use_virtual_physics = self._config.use_virtual_physics

        # Use ground slope from robot parameters (must match MuJoCo model)
        self._ground_slope_rad = self._robot_params.ground_slope_rad

        # Cache equilibrium placeholders (set during controller build)
        self._equilibrium_state = np.zeros(STATE_DIMENSION)
        self._equilibrium_control = np.zeros(CONTROL_DIMENSION)

        # Build controller
        self._controller = self._build_controller()

        # MuJoCo model and data (loaded on run)
        self._model: Optional[Any] = None  # mujoco.MjModel when loaded
        self._data: Optional[Any] = None   # mujoco.MjData when loaded
        self._virtual_state = np.zeros(STATE_DIMENSION)

        # Simulation state
        self._encoder_left_rad: float = 0.0
        self._encoder_right_rad: float = 0.0

    def _build_controller(self) -> BalanceController:
        """Build the complete MPC controller.

        Returns:
            Configured BalanceController instance
        """
        # Compute equilibrium (accounting for slope) via analytic model
        eq_state, eq_control = compute_equilibrium_state(self._robot_params)
        use_linearize_at_state = False

        if not self._use_virtual_physics and abs(self._ground_slope_rad) > 1e-6:
            mujoco_equilibrium = self._estimate_equilibrium_from_mujoco(
                eq_state, eq_control,
            )
            if mujoco_equilibrium is not None:
                eq_state, eq_control = mujoco_equilibrium
                use_linearize_at_state = True

        # Cache equilibrium for later use (initial conditions, logging)
        self._equilibrium_state = eq_state
        self._equilibrium_control = eq_control
        self._virtual_state = eq_state.copy()  # seed virtual plant at operating point

        # Linearize dynamics. When MuJoCo refinement is used the analytic
        # model no longer sees an exact equilibrium, so fall back to a raw
        # state linearization if validation fails.
        if use_linearize_at_state:
            linearized = linearize_at_state(
                self._robot_params, eq_state, eq_control
            )
        else:
            try:
                linearized = linearize_at_equilibrium(
                    self._robot_params, eq_state, eq_control
                )
            except ValueError as exc:
                logger.warning(
                    "Equilibrium validation failed (%.3f slope rad): %s. "
                    "Falling back to linearize_at_state.",
                    self._ground_slope_rad,
                    exc,
                )
                linearized = linearize_at_state(
                    self._robot_params, eq_state, eq_control
                )

        # Discretize
        discrete = discretize_linear_dynamics(
            linearized.state_matrix,
            linearized.control_matrix,
            self._mpc_config.sampling_period_s,
        )

        # Build cost matrices
        Q = self._mpc_config.state_cost_matrix
        R = self._mpc_config.control_cost_matrix

        # Terminal cost: respect use_terminal_cost_dare flag
        if self._mpc_config.use_terminal_cost_dare:
            # Compute optimal terminal cost via DARE
            P = compute_terminal_cost_dare(
                discrete.state_matrix_discrete,
                discrete.control_matrix_discrete,
                Q, R,
            )
            # Apply terminal cost scaling
            P = self._mpc_config.terminal_cost_scale * P
        else:
            # Use stage cost with scaling (no DARE computation)
            # This gives a simple finite-horizon MPC without terminal stabilization
            P = self._mpc_config.terminal_cost_scale * Q

        # Create constraints
        state_constraints, input_constraints = create_constraints_from_config(
            self._mpc_config.pitch_limit_rad,
            self._mpc_config.pitch_rate_limit_radps,
            self._mpc_config.control_limit_nm,
        )

        # Create MPC solver
        mpc_solver = LinearMPCSolver(
            prediction_horizon_steps=self._mpc_config.prediction_horizon_steps,
            discrete_dynamics=discrete,
            state_cost=Q,
            control_cost=R,
            terminal_cost=P,
            state_constraints=state_constraints,
            input_constraints=input_constraints,
            terminal_pitch_limit_rad=self._mpc_config.terminal_pitch_limit_rad,
            terminal_pitch_rate_limit_radps=self._mpc_config.terminal_pitch_rate_limit_radps,
            terminal_velocity_limit_mps=self._mpc_config.terminal_velocity_limit_mps,
            preserve_warm_start_on_dynamics_update=self._mpc_config.preserve_warm_start_on_linearization,
        )

        # Create state estimator
        state_estimator = ComplementaryFilter(
            time_constant_s=self._estimator_config.complementary_filter_time_constant_s,
            sampling_period_s=self._estimator_config.sampling_period_s,
        )

        # Create reference generator
        reference_generator = ReferenceGenerator(
            sampling_period_s=self._mpc_config.sampling_period_s,
            prediction_horizon_steps=self._mpc_config.prediction_horizon_steps,
        )

        return BalanceController(
            mpc_solver=mpc_solver,
            state_estimator=state_estimator,
            reference_generator=reference_generator,
            sampling_period_s=self._mpc_config.sampling_period_s,
            wheel_radius_m=self._robot_params.wheel_radius_m,
            track_width_m=self._robot_params.track_width_m,
            robot_params=self._robot_params,
             equilibrium_state=eq_state,
             equilibrium_control=eq_control,
            online_linearization_enabled=self._mpc_config.online_linearization_enabled,
            use_simulation_velocity=True,  # FIX: Enable corrected velocity in simulation
        )

    def _resolve_model_path(self) -> Path:
        """Resolve MuJoCo model path relative to repository root."""
        model_path = Path(self._config.model_path)
        if not model_path.is_absolute():
            repo_root = Path(__file__).resolve().parent.parent
            model_path = (repo_root / model_path).resolve()
        return model_path

    def _load_model(self) -> None:
        """Load MuJoCo model and create data."""
        if self._use_virtual_physics:
            # Sensor emulation pulled directly from the virtual state so the controller
            # sees realistic IMU/encoder values even without MuJoCo.
            # Virtual mode sidesteps MuJoCo entirely; dynamics are evaluated analytically.
            self._model = None
            self._data = None
            return
        if not MUJOCO_AVAILABLE:
            raise ImportError(
                "MuJoCo is required for simulation but not installed. "
                "Install with: pip install mujoco"
            )

        model_path = self._resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

        self._model = mujoco.MjModel.from_xml_path(str(model_path))
        self._data = mujoco.MjData(self._model)

    def _estimate_equilibrium_from_mujoco(
        self,
        analytic_state: np.ndarray,
        analytic_control: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Estimate equilibrium by running MuJoCo with a PI velocity hold.

        The PI loop drives longitudinal velocity toward zero while the robot
        settles on the slope. The steady-state pitch/torque extracted from the
        tail of the run define the equilibrium used for linearization.
        """
        if not MUJOCO_AVAILABLE:
            logger.warning(
                "MuJoCo not available; cannot refine slope equilibrium."
            )
            return None

        model_path = self._resolve_model_path()
        if not model_path.exists():
            logger.warning(
                "MuJoCo model path %s does not exist; using analytic equilibrium.",
                model_path,
            )
            return None

        try:
            model = mujoco.MjModel.from_xml_path(str(model_path))
            data = mujoco.MjData(model)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Failed to load MuJoCo model for equilibrium estimation: %s",
                exc,
            )
            return None

        # Preserve MuJoCo defaults (body height, wheel offsets, etc.)
        base_qpos = data.qpos.copy()
        base_height = base_qpos[self.FREE_JOINT_POS_START + 2]

        sim_duration_s = 3.0
        sim_steps = max(1, int(sim_duration_s / model.opt.timestep))

        torque_bias = float(analytic_control[0])
        integral_v = 0.0
        torque_limit = 1.0
        velocity_kp = 2.0
        velocity_ki = 6.0
        pitch_damping = 0.3

        pitch_samples = []
        torque_samples = []

        def current_pitch() -> float:
            qw = data.qpos[self.FREE_JOINT_QUAT_START]
            qx = data.qpos[self.FREE_JOINT_QUAT_START + 1]
            qy = data.qpos[self.FREE_JOINT_QUAT_START + 2]
            qz = data.qpos[self.FREE_JOINT_QUAT_START + 3]
            return float(np.arcsin(2.0 * (qw * qy - qz * qx)))

        dt = model.opt.timestep
        for step in range(sim_steps):
            if step == 0:
                mujoco.mj_resetData(model, data)
                data.qpos[:] = base_qpos
                data.qvel[:] = 0.0
                data.qpos[self.FREE_JOINT_POS_START:self.FREE_JOINT_POS_START + 3] = [
                    0.0, 0.0, base_height
                ]

            vx = data.qvel[self.FREE_JOINT_VEL_START]
            pitch_rate = data.qvel[self.FREE_JOINT_ANGVEL_START + 1]
            integral_v += -vx * dt  # basic PI controller to drive velocity to zero
            control_torque = torque_bias + velocity_kp * (-vx) + velocity_ki * integral_v - pitch_damping * pitch_rate
            control_torque = float(np.clip(control_torque, -torque_limit, torque_limit))
            data.ctrl[self.LEFT_ACTUATOR] = control_torque
            data.ctrl[self.RIGHT_ACTUATOR] = control_torque
            mujoco.mj_step(model, data)

            if step > sim_steps * 0.6:
                pitch_samples.append(current_pitch())
                torque_samples.append(control_torque)

        if not pitch_samples:
            logger.warning("MuJoCo PI equilibrium run produced no samples.")
            return None

        eq_state = analytic_state.copy()
        eq_state[PITCH_INDEX] = float(np.mean(pitch_samples))
        eq_control = np.array([np.mean(torque_samples), np.mean(torque_samples)], dtype=float)
        logger.info(
            "MuJoCo PI equilibrium: pitch %.3f deg torque %.4f Nm",
            np.degrees(eq_state[PITCH_INDEX]),
            eq_control[0],
        )
        return eq_state, eq_control

    def _extract_state(self) -> np.ndarray:
        """Extract full state from MuJoCo.

        Maps MuJoCo generalized coordinates to our state representation.

        Returns:
            State vector [x, theta, psi, dx, dtheta, dpsi]
        """
        if self._use_virtual_physics:
            # Virtual mode keeps the state in a NumPy array; just return it.
            return self._virtual_state.copy()

        state = np.zeros(STATE_DIMENSION)

        # Position (x direction)
        state[POSITION_INDEX] = self._data.qpos[self.FREE_JOINT_POS_START]

        # Extract pitch from quaternion
        # Quaternion: [qw, qx, qy, qz] at indices [3, 4, 5, 6]
        qw = self._data.qpos[self.FREE_JOINT_QUAT_START]
        qx = self._data.qpos[self.FREE_JOINT_QUAT_START + 1]
        qy = self._data.qpos[self.FREE_JOINT_QUAT_START + 2]
        qz = self._data.qpos[self.FREE_JOINT_QUAT_START + 3]

        # Pitch (rotation about y-axis): arcsin(2*(qw*qy - qz*qx)),
        # measured in world coordinates (same convention as dynamics module).
        state[PITCH_INDEX] = np.arcsin(2.0 * (qw * qy - qz * qx))

        # Yaw (rotation about z-axis): arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
        yaw_rad = quat_to_yaw(np.array([qw, qx, qy, qz]))
        state[YAW_INDEX] = yaw_rad

        # Velocities (linear and angular from freejoint)
        state[VELOCITY_INDEX] = self._data.qvel[self.FREE_JOINT_VEL_START]      # vx
        state[PITCH_RATE_INDEX] = self._data.qvel[self.FREE_JOINT_ANGVEL_START + 1]  # wy
        state[YAW_RATE_INDEX] = self._data.qvel[self.FREE_JOINT_ANGVEL_START + 2]    # wz

        return state

    def _build_sensor_data(self, timestamp_s: float) -> SensorData:
        """Build sensor data from MuJoCo state.

        Simulates IMU and encoder readings from physics state.

        Args:
            timestamp_s: Current simulation time

        Returns:
            SensorData for controller input
        """
        if self._use_virtual_physics:
            pitch_rad = self._virtual_state[PITCH_INDEX]
            pitch_rate = self._virtual_state[PITCH_RATE_INDEX]
            yaw_rate = self._virtual_state[YAW_RATE_INDEX]
            gravity = self._robot_params.gravity_mps2
            acceleration = np.array([
                gravity * np.sin(pitch_rad),
                0.0,
                gravity * np.cos(pitch_rad),
            ])
            angular_velocity = np.array([
                0.0,
                pitch_rate,
                yaw_rate,
            ])
            self._encoder_left_rad = 0.0
            self._encoder_right_rad = 0.0
            return SensorData(
                acceleration_mps2=acceleration,
                angular_velocity_radps=angular_velocity,
                encoder_left_rad=self._encoder_left_rad,
                encoder_right_rad=self._encoder_right_rad,
                timestamp_s=timestamp_s,
            )

        # Extract pitch from quaternion
        qw = self._data.qpos[self.FREE_JOINT_QUAT_START]
        qx = self._data.qpos[self.FREE_JOINT_QUAT_START + 1]
        qy = self._data.qpos[self.FREE_JOINT_QUAT_START + 2]
        qz = self._data.qpos[self.FREE_JOINT_QUAT_START + 3]
        pitch_rad = np.arcsin(2.0 * (qw * qy - qz * qx))

        # Angular velocities from freejoint
        pitch_rate_radps = self._data.qvel[self.FREE_JOINT_ANGVEL_START + 1]
        yaw_rate_radps = self._data.qvel[self.FREE_JOINT_ANGVEL_START + 2]

        # Simulate accelerometer readings
        # At rest, accelerometer measures gravity
        gravity = self._robot_params.gravity_mps2
        acceleration = np.array([
            gravity * np.sin(pitch_rad),  # a_x (forward)
            0.0,                           # a_y (lateral)
            gravity * np.cos(pitch_rad),  # a_z (vertical)
        ])

        # Simulate gyroscope readings
        angular_velocity = np.array([
            0.0,              # omega_x (roll rate)
            pitch_rate_radps, # omega_y (pitch rate)
            yaw_rate_radps,   # omega_z (yaw rate)
        ])

        # Update encoder readings from wheel joints
        self._encoder_left_rad = self._data.qpos[self.MOTOR_L_JOINT]
        self._encoder_right_rad = self._data.qpos[self.MOTOR_R_JOINT]

        return SensorData(
            acceleration_mps2=acceleration,
            angular_velocity_radps=angular_velocity,
            encoder_left_rad=self._encoder_left_rad,
            encoder_right_rad=self._encoder_right_rad,
            timestamp_s=timestamp_s,
        )

    def _apply_control(self, torque_left_nm: float, torque_right_nm: float) -> None:
        """Apply control torques to MuJoCo actuators.

        Args:
            torque_left_nm: Left wheel torque in N*m
            torque_right_nm: Right wheel torque in N*m
        """
        if self._use_virtual_physics:
            # Store the torques so the virtual integrator can apply them.
            self._last_control = (torque_left_nm, torque_right_nm)
            return

        # Direct torque control (gear=1.0 in new model)
        self._data.ctrl[self.LEFT_ACTUATOR] = torque_left_nm
        self._data.ctrl[self.RIGHT_ACTUATOR] = torque_right_nm

    def _check_fallen(self, state: np.ndarray) -> bool:
        """Check if robot has fallen.

        Args:
            state: Current state vector

        Returns:
            True if robot has fallen beyond recovery
        """
        pitch_rad = state[PITCH_INDEX]
        return abs(pitch_rad) > self._mpc_config.pitch_limit_rad

    def run(
        self,
        duration_s: Optional[float] = None,
        initial_pitch_rad: float = 0.0,
        reference_command: Optional[ReferenceCommand] = None,
        reference_command_callback: Optional[Callable[[float], ReferenceCommand]] = None,
        disturbance_callback: Optional[Callable[[float, 'mujoco.MjData'], None]] = None,
    ) -> SimulationResult:
        """Run simulation.

        Args:
            duration_s: Simulation duration (uses config default if None)
            initial_pitch_rad: Initial pitch perturbation from equilibrium
            reference_command: Static reference command (uses BALANCE if None)
                              Ignored if reference_command_callback provided
            reference_command_callback: Optional callback for time-varying commands
                                       Called as callback(time_s) -> ReferenceCommand
                                       Takes precedence over reference_command
            disturbance_callback: Optional callback for applying disturbances
                                  Called as callback(time_s, mujoco_data)

        Returns:
            SimulationResult with logged data
        """
        duration = duration_s or self._config.duration_s

        # Determine command source (callback takes precedence)
        use_callback = reference_command_callback is not None
        if use_callback:
            # Validate callback by test-calling it
            test_command = reference_command_callback(0.0)
            if not isinstance(test_command, ReferenceCommand):
                raise TypeError(
                    f"reference_command_callback must return ReferenceCommand, "
                    f"got {type(test_command)}"
                )
            static_command = None
        else:
            static_command = reference_command or ReferenceCommand(mode=ReferenceMode.BALANCE)

        if self._use_virtual_physics:
            # Start virtual plant at equilibrium and apply the requested perturbation.
            self._virtual_state = self._equilibrium_state.copy()
            self._virtual_state[PITCH_INDEX] += initial_pitch_rad
        else:
            # Load model
            self._load_model()

            # Set initial conditions for freejoint
            # Position: start at origin
            self._data.qpos[self.FREE_JOINT_POS_START:self.FREE_JOINT_POS_START + 3] = [0, 0, 0.25]

            # Orientation: equilibrium pitch (absolute) plus optional perturbation
            pitch_target = self._equilibrium_state[PITCH_INDEX] + initial_pitch_rad
            qw = np.cos(pitch_target / 2)
            qy = np.sin(pitch_target / 2)
            self._data.qpos[self.FREE_JOINT_QUAT_START:self.FREE_JOINT_QUAT_START + 4] = [qw, 0, qy, 0]

        # Reset controller
        self._controller.reset()
        self._encoder_left_rad = 0.0
        self._encoder_right_rad = 0.0

        # Compute number of steps
        mpc_period = self._mpc_config.sampling_period_s
        if self._use_virtual_physics:
            # Break each MPC period into several smaller integration steps so the
            # virtual plant stays stable even without MuJoCo’s small timestep.
            mujoco_steps_per_mpc = 5
            virtual_dt = mpc_period / mujoco_steps_per_mpc
        else:
            mujoco_timestep = self._model.opt.timestep
            mujoco_steps_per_mpc = int(mpc_period / mujoco_timestep)
        n_mpc_steps = int(duration / mpc_period)

        # Allocate result arrays
        time_s = np.zeros(n_mpc_steps)
        state_history = np.zeros((n_mpc_steps, STATE_DIMENSION))
        control_history = np.zeros((n_mpc_steps, CONTROL_DIMENSION))
        state_estimate_history = np.zeros((n_mpc_steps, STATE_DIMENSION))
        qp_solve_time_history = np.zeros(n_mpc_steps)
        model_update_time_history = np.zeros(n_mpc_steps)
        iteration_history = np.zeros(n_mpc_steps)
        pitch_error_history = np.zeros(n_mpc_steps)
        velocity_error_history = np.zeros(n_mpc_steps)
        torque_saturation_history = np.zeros(n_mpc_steps, dtype=bool)
        infeasible_history = np.zeros(n_mpc_steps, dtype=bool)
        reference_history = []

        # Tracking
        deadline_violations = 0
        success = True
        theta_ref = -self._ground_slope_rad
        control_limit = self._mpc_config.control_limit_nm

        # Run simulation
        wall_start = time.perf_counter()

        for step_idx in range(n_mpc_steps):
            sim_time = step_idx * mpc_period

            # Get reference command for this timestep
            if use_callback:
                command = reference_command_callback(sim_time)
            else:
                command = static_command

            # Apply disturbance if callback provided
            if disturbance_callback is not None:
                disturbance_callback(sim_time, self._data)

            # Build sensor data
            sensor_data = self._build_sensor_data(sim_time)

            state_now = self._extract_state()
            if self._use_virtual_physics:
                self._controller._true_ground_velocity = state_now[VELOCITY_INDEX]
                self._controller._true_heading = state_now[YAW_INDEX]
                self._controller._true_yaw_rate = state_now[YAW_RATE_INDEX]
            else:
                # FIX: Provide true ground velocity to controller (for simulation mode)
                self._controller._true_ground_velocity = self._data.qvel[self.FREE_JOINT_VEL_START]
                # Provide true heading/yaw rate for simulation to bypass encoder yaw errors
                self._controller._true_heading = state_now[YAW_INDEX]
                self._controller._true_yaw_rate = self._data.qvel[self.FREE_JOINT_ANGVEL_START + 2]

            # Run controller
            output = self._controller.step(sensor_data, command)

            # Log data
            time_s[step_idx] = sim_time
            state_history[step_idx] = self._extract_state()
            control_history[step_idx] = np.array([
                output.torque_left_nm, output.torque_right_nm
            ])
            state_estimate_history[step_idx] = output.state_estimate
            qp_solve_time_history[step_idx] = output.timing.solve_time_s
            model_update_time_history[step_idx] = output.timing.model_update_time_s
            iteration_history[step_idx] = output.mpc_solution.solver_iterations
            reference_history.append(
                output.mpc_solution.predicted_trajectory.copy()
            )
            pitch_error_history[step_idx] = (
                state_history[step_idx, PITCH_INDEX] - theta_ref
            )
            desired_velocity = 0.0
            if command.mode == ReferenceMode.VELOCITY:
                desired_velocity = command.velocity_mps
            velocity_error_history[step_idx] = (
                state_history[step_idx, VELOCITY_INDEX] - desired_velocity
            )
            torque_saturation_history[step_idx] = (
                abs(output.torque_left_nm) >= control_limit - 1e-6
                or abs(output.torque_right_nm) >= control_limit - 1e-6
            )
            status = (output.mpc_solution.solver_status or "").lower()
            infeasible_history[step_idx] = status != 'optimal'

            # Check deadline violation
            if output.timing.total_time_s > mpc_period:
                deadline_violations += 1

            # Apply control
            self._apply_control(output.torque_left_nm, output.torque_right_nm)

            if self._use_virtual_physics:
                # Integrate the nonlinear dynamics directly using the requested torques.
                for _ in range(mujoco_steps_per_mpc):
                    control_vec = np.array([output.torque_left_nm, output.torque_right_nm])
                    deriv = compute_state_derivative(self._virtual_state, control_vec, self._robot_params)
                    self._virtual_state += deriv * virtual_dt
            else:
                # Step MuJoCo for one MPC period
                for _ in range(mujoco_steps_per_mpc):
                    mujoco.mj_step(self._model, self._data)

            # Check if fallen
            if self._check_fallen(state_history[step_idx]):
                success = False
                # Truncate arrays
                time_s = time_s[:step_idx + 1]
                state_history = state_history[:step_idx + 1]
                control_history = control_history[:step_idx + 1]
                state_estimate_history = state_estimate_history[:step_idx + 1]
                qp_solve_time_history = qp_solve_time_history[:step_idx + 1]
                model_update_time_history = model_update_time_history[:step_idx + 1]
                iteration_history = iteration_history[:step_idx + 1]
                pitch_error_history = pitch_error_history[:step_idx + 1]
                velocity_error_history = velocity_error_history[:step_idx + 1]
                torque_saturation_history = torque_saturation_history[:step_idx + 1]
                infeasible_history = infeasible_history[:step_idx + 1]
                break

        wall_duration = time.perf_counter() - wall_start

        return SimulationResult(
            time_s=time_s,
            state_history=state_history,
            control_history=control_history,
            state_estimate_history=state_estimate_history,
            qp_solve_time_history=qp_solve_time_history,
            model_update_time_history=model_update_time_history,
            iteration_history=iteration_history,
            reference_history=reference_history,
            pitch_error_history=pitch_error_history,
            velocity_error_history=velocity_error_history,
            torque_saturation_history=torque_saturation_history,
            infeasible_history=infeasible_history,
            success=success,
            total_duration_s=wall_duration,
            mean_qp_solve_time_ms=np.mean(qp_solve_time_history) * 1000,
            max_qp_solve_time_ms=np.max(qp_solve_time_history) * 1000,
            deadline_violations=deadline_violations,
        )

    def apply_disturbance(
        self,
        force_n: float,
        duration_s: float = 0.1,
    ) -> Callable[[float, Any], None]:
        """Create a disturbance callback for applying impulsive forces.

        Args:
            force_n: Force magnitude in Newtons
            duration_s: Duration of force application

        Returns:
            Callback function for run() disturbance_callback parameter
        """
        disturbance_start = None
        disturbance_active = False

        def callback(time_s: float, data: mujoco.MjData) -> None:
            nonlocal disturbance_start, disturbance_active

            if disturbance_start is None:
                # First call, set start time
                disturbance_start = time_s
                disturbance_active = True

            if disturbance_active:
                if time_s - disturbance_start < duration_s:
                    # Apply force to body (xfrc_applied is [fx, fy, fz, tx, ty, tz])
                    # Force in x direction (forward/backward)
                    data.xfrc_applied[1, 0] = force_n  # Body index 1 is robot
                else:
                    # Duration exceeded, stop applying
                    disturbance_active = False
                    data.xfrc_applied[1, :] = 0

        return callback

    def run_with_viewer(
        self,
        duration_s: Optional[float] = None,
        initial_pitch_rad: float = 0.0,
        reference_command: Optional[ReferenceCommand] = None,
        reference_command_callback: Optional[Callable[[float], ReferenceCommand]] = None,
    ) -> SimulationResult:
        """Run simulation with interactive MuJoCo viewer.

        This is a blocking call that opens the MuJoCo viewer window.
        The simulation runs indefinitely until:
        - User closes the viewer window, OR
        - Robot falls (pitch exceeds pitch_limit_rad)

        Args:
            duration_s: IGNORED in viewer mode (kept for API consistency)
            initial_pitch_rad: Initial pitch perturbation from equilibrium
            reference_command: Static reference command (uses BALANCE if None)
                              Ignored if reference_command_callback provided
            reference_command_callback: Optional callback for time-varying commands
                                       Called as callback(time_s) -> ReferenceCommand
                                       Takes precedence over reference_command

        Returns:
            SimulationResult with logged data
        """
        duration = duration_s or self._config.duration_s

        # Determine command source (callback takes precedence)
        use_callback = reference_command_callback is not None
        if use_callback:
            # Validate callback by test-calling it
            test_command = reference_command_callback(0.0)
            if not isinstance(test_command, ReferenceCommand):
                raise TypeError(
                    f"reference_command_callback must return ReferenceCommand, "
                    f"got {type(test_command)}"
                )
            static_command = None
        else:
            static_command = reference_command or ReferenceCommand(mode=ReferenceMode.BALANCE)

        if self._use_virtual_physics:
            raise NotImplementedError("Viewer mode not supported with virtual physics.")

        # Load model
        self._load_model()

        # Set initial conditions for freejoint (viewer mode)
        # Position: start at origin
        self._data.qpos[self.FREE_JOINT_POS_START:self.FREE_JOINT_POS_START + 3] = [0, 0, 0.25]

        # Orientation: equilibrium pitch (absolute) plus optional perturbation
        pitch_target = self._equilibrium_state[PITCH_INDEX] + initial_pitch_rad
        qw = np.cos(pitch_target / 2)
        qy = np.sin(pitch_target / 2)
        self._data.qpos[self.FREE_JOINT_QUAT_START:self.FREE_JOINT_QUAT_START + 4] = [qw, 0, qy, 0]

        # Reset controller
        self._controller.reset()

        # Storage for results
        time_list = []
        state_list = []
        control_list = []
        estimate_list = []
        qp_time_list = []
        model_update_time_list = []
        reference_list = []
        iteration_list = []
        pitch_error_list = []
        velocity_error_list = []
        torque_saturation_list = []
        infeasible_list = []
        theta_ref = -self._ground_slope_rad
        control_limit = self._mpc_config.control_limit_nm

        mpc_period = self._mpc_config.sampling_period_s
        mujoco_timestep = self._model.opt.timestep
        mujoco_steps_per_mpc = int(mpc_period / mujoco_timestep)

        wall_start = time.perf_counter()
        deadline_violations = 0
        success = True

        # Launch passive viewer with manual control loop
        import mujoco as mj
        import mujoco.viewer as mj_viewer

        last_mpc_time = 0.0

        with mj_viewer.launch_passive(self._model, self._data) as viewer:
            while viewer.is_running():
                current_sim_time = self._data.time

                # Run MPC when period elapses
                if current_sim_time - last_mpc_time >= mpc_period - (mujoco_timestep * 0.5):
                    # Check for duplicate step
                    if len(time_list) == 0 or abs(current_sim_time - time_list[-1]) >= mpc_period * 0.5:
                        # Get reference command for this timestep
                        if use_callback:
                            command = reference_command_callback(current_sim_time)
                        else:
                            command = static_command

                        # Build sensor data
                        sensor_data = self._build_sensor_data(current_sim_time)

                        # FIX: Provide true ground velocity to controller (for simulation mode)
                        self._controller._true_ground_velocity = self._data.qvel[self.FREE_JOINT_VEL_START]
                        # Provide true heading/yaw rate in viewer mode as well
                        state_now = self._extract_state()
                        self._controller._true_heading = state_now[YAW_INDEX]
                        self._controller._true_yaw_rate = self._data.qvel[self.FREE_JOINT_ANGVEL_START + 2]

                        # Run controller
                        output = self._controller.step(sensor_data, command)

                        # Log data
                        time_list.append(current_sim_time)
                        state_list.append(self._extract_state())
                        control_list.append([output.torque_left_nm, output.torque_right_nm])
                        estimate_list.append(output.state_estimate.copy())
                        qp_time_list.append(output.timing.solve_time_s)
                        model_update_time_list.append(output.timing.model_update_time_s)
                        reference_list.append(output.mpc_solution.predicted_trajectory.copy())
                        iteration_list.append(output.mpc_solution.solver_iterations)
                        pitch_error_list.append(state_list[-1][PITCH_INDEX] - theta_ref)
                        desired_velocity = 0.0
                        if command.mode == ReferenceMode.VELOCITY:
                            desired_velocity = command.velocity_mps
                        velocity_error_list.append(state_list[-1][VELOCITY_INDEX] - desired_velocity)
                        torque_saturation_list.append(
                            abs(output.torque_left_nm) >= control_limit - 1e-6
                            or abs(output.torque_right_nm) >= control_limit - 1e-6
                        )
                        status = (output.mpc_solution.solver_status or "").lower()
                        infeasible_list.append(status != 'optimal')

                        # Check deadline
                        if output.timing.total_time_s > mpc_period:
                            deadline_violations += 1

                        # Apply control
                        self._apply_control(output.torque_left_nm, output.torque_right_nm)

                        # Check if fallen - close viewer and exit
                        if self._check_fallen(state_list[-1]):
                            success = False
                            viewer.close()
                            break

                        last_mpc_time = current_sim_time

                # Step physics and sync viewer
                mj.mj_step(self._model, self._data)
                viewer.sync()

        wall_duration = time.perf_counter() - wall_start

        # Convert lists to arrays
        time_s = np.array(time_list)
        state_history = np.array(state_list)
        control_history = np.array(control_list)
        state_estimate_history = np.array(estimate_list)
        qp_solve_time_history = np.array(qp_time_list)
        model_update_time_history = np.array(model_update_time_list)
        iteration_history = np.array(iteration_list)
        pitch_error_history = np.array(pitch_error_list)
        velocity_error_history = np.array(velocity_error_list)
        torque_saturation_history = np.array(torque_saturation_list, dtype=bool)
        infeasible_history = np.array(infeasible_list, dtype=bool)

        return SimulationResult(
            time_s=time_s,
            state_history=state_history,
            control_history=control_history,
            state_estimate_history=state_estimate_history,
            qp_solve_time_history=qp_solve_time_history,
            model_update_time_history=model_update_time_history,
            iteration_history=iteration_history,
            reference_history=reference_list,
            pitch_error_history=pitch_error_history,
            velocity_error_history=velocity_error_history,
            torque_saturation_history=torque_saturation_history,
            infeasible_history=infeasible_history,
            success=success,
            total_duration_s=wall_duration,
            mean_qp_solve_time_ms=np.mean(qp_solve_time_history) * 1000 if len(qp_solve_time_history) > 0 else 0,
            max_qp_solve_time_ms=np.max(qp_solve_time_history) * 1000 if len(qp_solve_time_history) > 0 else 0,
            deadline_violations=deadline_violations,
        )

    @property
    def controller(self) -> BalanceController:
        """Access the balance controller."""
        return self._controller

    @property
    def config(self) -> SimulationConfig:
        """Access simulation configuration."""
        return self._config

    @property
    def equilibrium_state(self) -> np.ndarray:
        """Copy of the cached equilibrium state."""
        return self._equilibrium_state.copy()

    @property
    def equilibrium_control(self) -> np.ndarray:
        """Copy of the cached equilibrium control."""
        return self._equilibrium_control.copy()

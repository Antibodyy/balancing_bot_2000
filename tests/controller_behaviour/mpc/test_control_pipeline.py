"""Unit tests for control pipeline."""

import numpy as np
import pytest

from robot_dynamics import (
    RobotParameters,
    linearize_at_equilibrium,
    discretize_linear_dynamics,
    compute_equilibrium_state,
)
from robot_dynamics.parameters import STATE_DIMENSION, CONTROL_DIMENSION
from mpc import (
    MPCConfig,
    LinearMPCSolver,
    compute_terminal_cost_dare,
    create_constraints_from_config,
    ReferenceGenerator,
    ReferenceCommand,
    ReferenceMode,
)
from state_estimation import EstimatorConfig, ComplementaryFilter
from control_pipeline import (
    BalanceController,
    SensorData,
    ControlOutput,
    ControlLoopTimer,
    IterationTiming,
)


@pytest.fixture
def robot_params():
    """Load robot parameters."""
    return RobotParameters.from_yaml('config/simulation/robot_params.yaml')


@pytest.fixture
def mpc_config():
    """Load MPC configuration."""
    return MPCConfig.from_yaml('config/simulation/mpc_params.yaml')


@pytest.fixture
def controller(robot_params, mpc_config):
    """Create configured balance controller."""
    # Setup dynamics
    eq_state, eq_control = compute_equilibrium_state(robot_params)
    linearized = linearize_at_equilibrium(robot_params, eq_state, eq_control)
    discrete = discretize_linear_dynamics(
        linearized.state_matrix,
        linearized.control_matrix,
        mpc_config.sampling_period_s,
    )

    # Build cost matrices
    Q = mpc_config.state_cost_matrix
    R = mpc_config.control_cost_matrix
    P = compute_terminal_cost_dare(
        discrete.state_matrix_discrete,
        discrete.control_matrix_discrete,
        Q, R,
    )

    # Create constraints
    state_constraints, input_constraints = create_constraints_from_config(
        mpc_config.pitch_limit_rad,
        mpc_config.pitch_rate_limit_radps,
        mpc_config.control_limit_nm,
    )

    # Create MPC solver
    mpc_solver = LinearMPCSolver(
        prediction_horizon_steps=mpc_config.prediction_horizon_steps,
        discrete_dynamics=discrete,
        state_cost=Q,
        control_cost=R,
        terminal_cost=P,
        state_constraints=state_constraints,
        input_constraints=input_constraints,
    )

    # Create state estimator
    est_config = EstimatorConfig.from_yaml('config/simulation/estimator_params.yaml')
    state_estimator = ComplementaryFilter(
        time_constant_s=est_config.complementary_filter_time_constant_s,
        sampling_period_s=est_config.sampling_period_s,
    )

    # Create reference generator
    reference_generator = ReferenceGenerator(
        sampling_period_s=mpc_config.sampling_period_s,
        prediction_horizon_steps=mpc_config.prediction_horizon_steps,
    )

    return BalanceController(
        mpc_solver=mpc_solver,
        state_estimator=state_estimator,
        reference_generator=reference_generator,
        sampling_period_s=mpc_config.sampling_period_s,
        wheel_radius_m=robot_params.wheel_radius_m,
        track_width_m=robot_params.track_width_m,
        robot_params=robot_params,
        equilibrium_state=eq_state,
        equilibrium_control=eq_control,
    )


class TestControlLoopTimer:
    """Tests for ControlLoopTimer."""

    def test_creation(self):
        """Test basic creation."""
        timer = ControlLoopTimer(deadline_s=0.02)
        assert timer.deadline_s == 0.02
        assert timer.deadline_violations == 0

    def test_iteration_timing(self):
        """Test iteration timing tracking."""
        timer = ControlLoopTimer(deadline_s=0.02)

        timer.start_iteration()
        timer.mark_estimation_complete()
        timer.mark_reference_complete()
        timer.mark_solve_complete()
        timing = timer.end_iteration()

        assert isinstance(timing, IterationTiming)
        assert timing.total_time_s > 0

    def test_deadline_violation_counted(self):
        """Test that deadline violations are counted."""
        timer = ControlLoopTimer(deadline_s=0.0001)  # Very short deadline

        timer.start_iteration()
        import time
        time.sleep(0.001)  # Sleep longer than deadline
        timer.end_iteration()

        assert timer.deadline_violations > 0


class TestSensorData:
    """Tests for SensorData dataclass."""

    def test_creation(self):
        """Test basic creation."""
        data = SensorData(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=0.0,
            encoder_right_rad=0.0,
            timestamp_s=0.0,
        )
        assert data.timestamp_s == 0.0


class TestBalanceController:
    """Tests for BalanceController."""

    def test_creation(self, controller):
        """Test that controller creates successfully."""
        assert controller is not None

    def test_step_returns_control_output(self, controller):
        """Test that step returns ControlOutput."""
        sensor_data = SensorData(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=0.0,
            encoder_right_rad=0.0,
            timestamp_s=0.0,
        )
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)

        output = controller.step(sensor_data, command)

        assert isinstance(output, ControlOutput)

    def test_upright_produces_zero_torque(self, controller):
        """Test that upright robot produces near-zero torque."""
        # Run initial step to warm up IPOPT
        sensor_data = SensorData(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=0.0,
            encoder_right_rad=0.0,
            timestamp_s=0.0,
        )
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)

        output = controller.step(sensor_data, command)

        # At equilibrium, torques should be near zero
        assert np.abs(output.torque_left_nm) < 0.01
        assert np.abs(output.torque_right_nm) < 0.01

    def test_tilted_produces_corrective_torque(self, controller):
        """Test that tilted robot produces corrective torque."""
        gravity = 9.81
        pitch_rad = np.deg2rad(5)

        sensor_data = SensorData(
            acceleration_mps2=np.array([
                gravity * np.sin(pitch_rad),
                0.0,
                gravity * np.cos(pitch_rad),
            ]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=0.0,
            encoder_right_rad=0.0,
            timestamp_s=0.0,
        )
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)

        output = controller.step(sensor_data, command)

        # Both torques should be positive to correct forward tilt
        assert output.torque_left_nm > 0
        assert output.torque_right_nm > 0

    def test_state_estimate_shape(self, controller):
        """Test that state estimate has correct shape."""
        sensor_data = SensorData(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=0.0,
            encoder_right_rad=0.0,
            timestamp_s=0.0,
        )
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)

        output = controller.step(sensor_data, command)

        assert output.state_estimate.shape == (STATE_DIMENSION,)

    def test_predicted_trajectory_shape(self, controller, mpc_config):
        """Test that predicted trajectory has correct shape."""
        sensor_data = SensorData(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=0.0,
            encoder_right_rad=0.0,
            timestamp_s=0.0,
        )
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)

        output = controller.step(sensor_data, command)

        expected_shape = (mpc_config.prediction_horizon_steps + 1, STATE_DIMENSION)
        assert output.predicted_trajectory.shape == expected_shape

    def test_timing_recorded(self, controller):
        """Test that timing is recorded."""
        sensor_data = SensorData(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=0.0,
            encoder_right_rad=0.0,
            timestamp_s=0.0,
        )
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)

        output = controller.step(sensor_data, command)

        assert output.timing.total_time_s > 0
        assert output.timing.solve_time_s > 0

    def test_reset_clears_state(self, controller):
        """Test that reset clears accumulated state."""
        # Run a step
        sensor_data = SensorData(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=0.1,
            encoder_right_rad=0.1,
            timestamp_s=0.02,
        )
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)
        controller.step(sensor_data, command)

        # Reset
        controller.reset()

        assert controller.position_m == 0.0
        assert controller.heading_rad == 0.0

    def test_position_integrates(self, controller):
        """Test that position integrates from encoder readings."""
        # First step
        sensor_data1 = SensorData(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=0.0,
            encoder_right_rad=0.0,
            timestamp_s=0.0,
        )
        command = ReferenceCommand(mode=ReferenceMode.BALANCE)
        controller.step(sensor_data1, command)

        # Second step with encoder change
        sensor_data2 = SensorData(
            acceleration_mps2=np.array([0.0, 0.0, 9.81]),
            angular_velocity_radps=np.array([0.0, 0.0, 0.0]),
            encoder_left_rad=1.0,  # 1 radian of wheel rotation
            encoder_right_rad=1.0,
            timestamp_s=0.02,
        )
        output = controller.step(sensor_data2, command)

        # Position should have increased
        assert controller.position_m > 0

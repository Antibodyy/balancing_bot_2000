"""Reference trajectory generation for MPC.

Provides different reference generation modes:
1. Balance mode: Zero reference (upright stationary)
2. Velocity mode: Track desired forward velocity and yaw rate
3. Position mode: Track target position with smooth approach

State vector: [x, theta, psi, dx, dtheta, dpsi]
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from robot_dynamics.parameters import (
    STATE_DIMENSION,
    POSITION_INDEX,
    PITCH_INDEX,
    YAW_INDEX,
    VELOCITY_INDEX,
    PITCH_RATE_INDEX,
    YAW_RATE_INDEX,
)


class ReferenceMode(Enum):
    """Reference generation mode."""

    BALANCE = "balance"     # Upright stationary
    VELOCITY = "velocity"   # Track velocity/yaw rate
    POSITION = "position"   # Track position/heading
    CIRCULAR = "circular"   # Track circular path


@dataclass
class ReferenceCommand:
    """Command for reference generation.

    Attributes:
        mode: Reference generation mode
        velocity_mps: Desired forward velocity (for VELOCITY mode)
        yaw_rate_radps: Desired yaw rate (for VELOCITY mode)
        target_position_m: Target position (for POSITION mode)
        target_heading_rad: Target heading (for POSITION mode)
        radius_m: Circle radius (for CIRCULAR mode)
        center_x_m: Circle center X coordinate (for CIRCULAR mode)
        center_y_m: Circle center Y coordinate (for CIRCULAR mode)
        target_velocity_mps: Desired forward velocity (for CIRCULAR mode)
        clockwise: Circle direction (for CIRCULAR mode)
    """

    mode: ReferenceMode = ReferenceMode.BALANCE
    velocity_mps: float = 0.0
    yaw_rate_radps: float = 0.0
    target_position_m: float = 0.0
    target_heading_rad: float = 0.0
    radius_m: float = 1.0
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    target_velocity_mps: float = 0.2
    clockwise: bool = False


class ReferenceGenerator:
    """Generate reference trajectories for MPC.

    Produces reference state trajectories over the prediction horizon
    based on the current reference command.

    Attributes:
        sampling_period_s: Time step between prediction steps
        prediction_horizon_steps: Number of prediction steps
    """

    def __init__(
        self,
        sampling_period_s: float,
        prediction_horizon_steps: int,
    ) -> None:
        """Initialize reference generator.

        Args:
            sampling_period_s: Time step between prediction steps
            prediction_horizon_steps: Number of prediction steps N
        """
        if sampling_period_s <= 0:
            raise ValueError(
                f"sampling_period_s must be positive, got {sampling_period_s}"
            )
        if prediction_horizon_steps <= 0:
            raise ValueError(
                f"prediction_horizon_steps must be positive, "
                f"got {prediction_horizon_steps}"
            )

        self._sampling_period_s = sampling_period_s
        self._prediction_horizon_steps = prediction_horizon_steps

    def generate(
        self,
        command: ReferenceCommand,
        current_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate reference trajectory based on command.

        Args:
            command: Reference command specifying mode and targets
            current_state: Current state (required for POSITION mode)

        Returns:
            Reference trajectory array (N+1, STATE_DIMENSION)
        """
        if command.mode == ReferenceMode.BALANCE:
            return self.generate_balance_reference()
        elif command.mode == ReferenceMode.VELOCITY:
            return self.generate_velocity_reference(
                command.velocity_mps,
                command.yaw_rate_radps,
                current_state,
            )
        elif command.mode == ReferenceMode.POSITION:
            if current_state is None:
                raise ValueError(
                    "current_state is required for POSITION mode"
                )
            return self.generate_position_reference(
                current_state,
                command.target_position_m,
                command.target_heading_rad,
            )
        elif command.mode == ReferenceMode.CIRCULAR:
            return self.generate_circular_reference(
                radius_m=command.radius_m,
                target_velocity_mps=command.target_velocity_mps,
                center_x_m=command.center_x_m,
                center_y_m=command.center_y_m,
                clockwise=command.clockwise,
                current_state=current_state,
            )
        else:
            raise ValueError(f"Unknown reference mode: {command.mode}")

    def generate_balance_reference(self) -> np.ndarray:
        """Generate constant zero reference for balance mode.

        The robot should maintain upright posture at rest.
        All states are zero: position=0, pitch=0, yaw=0, velocity=0, pitch_rate=0, yaw_rate=0

        Returns:
            Reference trajectory (N+1, STATE_DIMENSION) of zeros
        """
        horizon = self._prediction_horizon_steps
        return np.zeros((horizon + 1, STATE_DIMENSION))

    def generate_velocity_reference(
        self,
        desired_velocity_mps: float,
        desired_yaw_rate_radps: float,
        current_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate velocity tracking reference.

        The robot should move at the desired velocity while maintaining
        balance (theta = 0, dtheta = 0).

        Args:
            desired_velocity_mps: Desired forward velocity
            desired_yaw_rate_radps: Desired yaw rate
            current_state: Current state to make reference relative to current position

        Returns:
            Reference trajectory (N+1, STATE_DIMENSION)
        """
        horizon = self._prediction_horizon_steps
        reference = np.zeros((horizon + 1, STATE_DIMENSION))

        # Get current position and yaw (or use 0 if not provided)
        current_position = 0.0 if current_state is None else current_state[POSITION_INDEX]
        current_yaw = 0.0 if current_state is None else current_state[YAW_INDEX]

        # Explicitly set pitch to 0 (maintain upright balance)
        reference[:, PITCH_INDEX] = 0.0
        reference[:, PITCH_RATE_INDEX] = 0.0

        # Set velocity references
        reference[:, VELOCITY_INDEX] = desired_velocity_mps
        reference[:, YAW_RATE_INDEX] = desired_yaw_rate_radps

        # Position and heading evolve based on velocity (relative to current state)
        for step_index in range(1, horizon + 1):
            time_elapsed = step_index * self._sampling_period_s
            reference[step_index, POSITION_INDEX] = (
                current_position + desired_velocity_mps * time_elapsed
            )
            reference[step_index, YAW_INDEX] = (
                current_yaw + desired_yaw_rate_radps * time_elapsed
            )

        return reference

    def generate_position_reference(
        self,
        current_state: np.ndarray,
        target_position_m: float,
        target_heading_rad: float,
        approach_velocity_mps: float = 0.1,
    ) -> np.ndarray:
        """Generate trajectory to reach target position.

        Uses a trapezoidal velocity profile for smooth approach.
        The robot decelerates as it approaches the target.

        Args:
            current_state: Current robot state (STATE_DIMENSION,)
            target_position_m: Target position
            target_heading_rad: Target heading
            approach_velocity_mps: Maximum approach velocity

        Returns:
            Reference trajectory (N+1, STATE_DIMENSION)
        """
        horizon = self._prediction_horizon_steps
        reference = np.zeros((horizon + 1, STATE_DIMENSION))

        # Explicitly set pitch to 0 (maintain upright balance)
        reference[:, PITCH_INDEX] = 0.0
        reference[:, PITCH_RATE_INDEX] = 0.0

        current_position = current_state[POSITION_INDEX]
        current_heading = current_state[YAW_INDEX]

        # Position error
        position_error = target_position_m - current_position
        heading_error = self._wrap_angle(target_heading_rad - current_heading)

        # Distance to target
        distance = abs(position_error)

        # Deceleration distance (stop smoothly)
        deceleration_distance = 0.1  # meters

        for step_index in range(horizon + 1):
            time_elapsed = step_index * self._sampling_period_s

            # Simple proportional approach with saturation
            if distance > deceleration_distance:
                # Full speed approach
                velocity = np.sign(position_error) * approach_velocity_mps
            else:
                # Decelerate near target
                velocity = (
                    np.sign(position_error)
                    * approach_velocity_mps
                    * (distance / deceleration_distance)
                )

            # Update position along trajectory
            new_position = current_position + velocity * time_elapsed

            # Clamp to target if we overshoot
            if position_error > 0:
                new_position = min(new_position, target_position_m)
            else:
                new_position = max(new_position, target_position_m)

            reference[step_index, POSITION_INDEX] = new_position

            # Heading interpolation
            heading_fraction = min(1.0, time_elapsed / (horizon * self._sampling_period_s))
            reference[step_index, YAW_INDEX] = (
                current_heading + heading_error * heading_fraction
            )

            # Velocity reference (derivative of position)
            if step_index < horizon:
                reference[step_index, VELOCITY_INDEX] = velocity

        return reference

    def generate_circular_reference(
        self,
        radius_m: float,
        target_velocity_mps: float,
        center_x_m: float = 0.0,
        center_y_m: float = 0.0,
        clockwise: bool = False,
        current_state: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate circular trajectory reference.

        Creates a reference for following a circular path at constant velocity.
        The robot maintains balance while moving along a circle defined by
        center position and radius.

        Args:
            radius_m: Circle radius (must be >= 0.3m for model validity)
            target_velocity_mps: Desired forward velocity along circle
            center_x_m: Circle center X coordinate (world frame)
            center_y_m: Circle center Y coordinate (world frame)
            clockwise: If True, rotate clockwise; if False, counter-clockwise
            current_state: Current state to determine starting angle

        Returns:
            Reference trajectory (N+1, STATE_DIMENSION)

        Raises:
            ValueError: If radius is too small or velocity is negative
        """
        # Validation
        MIN_RADIUS = 0.3  # meters - based on yaw dynamics limitations
        if radius_m < MIN_RADIUS:
            raise ValueError(
                f"Circle radius must be >= {MIN_RADIUS}m for model validity, "
                f"got {radius_m}m"
            )
        if target_velocity_mps < 0:
            raise ValueError(
                f"Velocity must be non-negative, got {target_velocity_mps}"
            )

        horizon = self._prediction_horizon_steps
        reference = np.zeros((horizon + 1, STATE_DIMENSION))

        # Compute yaw rate for circular motion: ω = v / r
        yaw_rate = target_velocity_mps / radius_m
        if clockwise:
            yaw_rate = -yaw_rate

        # Determine starting angle on circle
        if current_state is not None:
            # Use current yaw as starting angle
            # This assumes the robot is roughly facing tangent to circle
            angle_0 = current_state[YAW_INDEX]
        else:
            # Start from angle 0 (pointing along +X axis)
            angle_0 = 0.0

        # Balance references (constant - maintain upright)
        reference[:, PITCH_INDEX] = 0.0
        reference[:, PITCH_RATE_INDEX] = 0.0

        # Velocity references (constant)
        reference[:, VELOCITY_INDEX] = target_velocity_mps
        reference[:, YAW_RATE_INDEX] = yaw_rate

        # Position and heading evolve along circle
        for step_index in range(horizon + 1):
            time_elapsed = step_index * self._sampling_period_s

            # Current angle on circle
            angle = angle_0 + yaw_rate * time_elapsed

            # Heading follows angle
            reference[step_index, YAW_INDEX] = angle

            # X position on circle: x = x_c + r*sin(θ)
            # For clockwise (ω < 0), the sign is handled by angle evolution
            reference[step_index, POSITION_INDEX] = (
                center_x_m + radius_m * np.sin(angle)
            )

        return reference

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        """Wrap angle to [-pi, pi].

        Args:
            angle_rad: Angle in radians

        Returns:
            Wrapped angle in [-pi, pi]
        """
        return (angle_rad + np.pi) % (2 * np.pi) - np.pi

    @property
    def sampling_period_s(self) -> float:
        """Time step between prediction steps."""
        return self._sampling_period_s

    @property
    def prediction_horizon_steps(self) -> int:
        """Number of prediction steps."""
        return self._prediction_horizon_steps

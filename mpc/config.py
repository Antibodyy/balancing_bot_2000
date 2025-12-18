"""MPC configuration parameters.

Single source of truth for MPC controller settings.
See config/simulation/mpc_params.yaml for parameter values.
"""

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import yaml

from mpc._internal.validation import (
    validate_positive,
    validate_positive_integer,
    validate_non_negative,
    validate_state_cost_diagonal,
    validate_control_cost_diagonal,
    validate_solver_name,
)


@dataclass(frozen=True)
class MPCConfig:
    """Configuration parameters for linear MPC controller.

    All parameters immutable after construction (frozen=True).
    Units encoded in parameter names per style_guide.md section 2.1.4.

    Attributes:
        prediction_horizon_steps: Number of prediction steps (N)
        sampling_period_s: Discretization time step (T_s)
        state_cost_diagonal: Q matrix diagonal elements (6,)
        control_cost_diagonal: R matrix diagonal elements (2,)
        pitch_limit_rad: Maximum allowed pitch angle |theta|
        pitch_rate_limit_radps: Maximum allowed pitch rate |theta_dot|
        velocity_limit_mps: Optional forward velocity constraint |dx|
        control_limit_nm: Maximum allowed control torque |tau|
        use_terminal_cost_dare: If True, compute P via discrete ARE
        terminal_cost_scale: Scaling factor for terminal cost P
        solver_name: QP solver to use ('osqp' or 'qpoases')
        warm_start_enabled: Enable warm-starting from previous solution
        terminal_pitch_limit_rad: Terminal pitch constraint (None = no constraint)
        terminal_pitch_rate_limit_radps: Terminal pitch rate constraint (None = no constraint)
        terminal_velocity_limit_mps: Terminal velocity constraint (None = no constraint)
        online_linearization_enabled: If True, re-linearize dynamics at current state each step
        preserve_warm_start_on_linearization: Keep the previous MPC solution as the
            initial guess even when the dynamics matrices are refreshed online.
            Disabled by default to preserve legacy behaviour.
        velocity_limit_mps: Optional forward-velocity bound
        velocity_limit_slack_weight: Penalty weight for velocity limit slack variables
    """
    # Horizon parameters
    prediction_horizon_steps: int
    sampling_period_s: float

    # Cost function weights
    state_cost_diagonal: Tuple[float, ...]
    control_cost_diagonal: Tuple[float, ...]

    # State constraints
    pitch_limit_rad: float
    pitch_rate_limit_radps: float

    # Control constraints
    control_limit_nm: float

    # Terminal cost
    use_terminal_cost_dare: bool
    terminal_cost_scale: float

    # Solver settings
    solver_name: str
    warm_start_enabled: bool
    velocity_limit_mps: Optional[float] = None
    velocity_limit_slack_weight: float = 1e4

    # Terminal state constraints (optional - applied in addition to state constraints at step N)
    terminal_pitch_limit_rad: Optional[float] = None
    terminal_pitch_rate_limit_radps: Optional[float] = None
    terminal_velocity_limit_mps: Optional[float] = None

    # Online linearization (successive linearization at current state)
    online_linearization_enabled: bool = False
    preserve_warm_start_on_linearization: bool = False

    def __post_init__(self) -> None:
        """Validate parameters satisfy constraints."""
        # Horizon parameters
        validate_positive_integer(
            self.prediction_horizon_steps, 'prediction_horizon_steps'
        )
        validate_positive(self.sampling_period_s, 'sampling_period_s')

        # Cost function weights
        state_diagonal = np.array(self.state_cost_diagonal)
        control_diagonal = np.array(self.control_cost_diagonal)
        validate_state_cost_diagonal(state_diagonal)
        validate_control_cost_diagonal(control_diagonal)

        # Constraints
        validate_positive(self.pitch_limit_rad, 'pitch_limit_rad')
        validate_positive(self.pitch_rate_limit_radps, 'pitch_rate_limit_radps')
        validate_positive(self.control_limit_nm, 'control_limit_nm')
        if self.velocity_limit_mps is not None:
            validate_positive(self.velocity_limit_mps, 'velocity_limit_mps')
        validate_positive(self.velocity_limit_slack_weight, 'velocity_limit_slack_weight')

        # Terminal cost
        validate_positive(self.terminal_cost_scale, 'terminal_cost_scale')

        # Terminal constraints (optional)
        if self.terminal_pitch_limit_rad is not None:
            validate_non_negative(
                self.terminal_pitch_limit_rad, 'terminal_pitch_limit_rad'
            )
        if self.terminal_pitch_rate_limit_radps is not None:
            validate_non_negative(
                self.terminal_pitch_rate_limit_radps, 'terminal_pitch_rate_limit_radps'
            )
        if self.terminal_velocity_limit_mps is not None:
            validate_non_negative(
                self.terminal_velocity_limit_mps, 'terminal_velocity_limit_mps'
            )

        # Solver
        validate_solver_name(self.solver_name)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MPCConfig':
        """Load configuration from YAML file.

        Single source of truth per style_guide.md section 4.4.

        Args:
            yaml_path: Path to YAML file containing MPC parameters

        Returns:
            MPCConfig instance

        Raises:
            FileNotFoundError: If YAML file does not exist
            ValueError: If required parameters missing or invalid
        """
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        # Convert lists to tuples for immutability
        return cls(
            prediction_horizon_steps=config['prediction_horizon_steps'],
            sampling_period_s=config['sampling_period_s'],
            state_cost_diagonal=tuple(config['state_cost_diagonal']),
            control_cost_diagonal=tuple(config['control_cost_diagonal']),
            pitch_limit_rad=config['pitch_limit_rad'],
            pitch_rate_limit_radps=config['pitch_rate_limit_radps'],
            velocity_limit_mps=config.get('velocity_limit_mps', None),
            velocity_limit_slack_weight=config.get('velocity_limit_slack_weight', 1e4),
            control_limit_nm=config['control_limit_nm'],
            use_terminal_cost_dare=config['use_terminal_cost_dare'],
            terminal_cost_scale=config.get('terminal_cost_scale', 1.0),
            solver_name=config['solver_name'],
            warm_start_enabled=config['warm_start_enabled'],
            # Terminal constraints (optional, backward compatible)
            terminal_pitch_limit_rad=config.get('terminal_pitch_limit_rad', None),
            terminal_pitch_rate_limit_radps=config.get('terminal_pitch_rate_limit_radps', None),
            terminal_velocity_limit_mps=config.get('terminal_velocity_limit_mps', None),
            # Online linearization (optional, backward compatible)
            online_linearization_enabled=config.get('online_linearization_enabled', False),
            preserve_warm_start_on_linearization=config.get(
                'preserve_warm_start_on_linearization', False
            ),
        )

    @property
    def state_cost_matrix(self) -> np.ndarray:
        """Build Q matrix from diagonal elements.

        Returns:
            Diagonal state cost matrix (6, 6)
        """
        return np.diag(self.state_cost_diagonal)

    @property
    def control_cost_matrix(self) -> np.ndarray:
        """Build R matrix from diagonal elements.

        Returns:
            Diagonal control cost matrix (2, 2)
        """
        return np.diag(self.control_cost_diagonal)

    @property
    def prediction_horizon_duration_s(self) -> float:
        """Total prediction horizon duration.

        Returns:
            N * T_s in seconds
        """
        return self.prediction_horizon_steps * self.sampling_period_s

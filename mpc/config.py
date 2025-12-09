"""MPC configuration parameters.

Single source of truth for MPC controller settings.
See config/mpc_params.yaml for parameter values.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import yaml

from mpc._internal.validation import (
    validate_positive,
    validate_positive_integer,
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
        control_limit_nm: Maximum allowed control torque |tau|
        use_terminal_cost_dare: If True, compute P via discrete ARE
        solver_name: QP solver to use ('osqp' or 'qpoases')
        warm_start_enabled: Enable warm-starting from previous solution
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

    # Solver settings
    solver_name: str
    warm_start_enabled: bool

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
            control_limit_nm=config['control_limit_nm'],
            use_terminal_cost_dare=config['use_terminal_cost_dare'],
            solver_name=config['solver_name'],
            warm_start_enabled=config['warm_start_enabled'],
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

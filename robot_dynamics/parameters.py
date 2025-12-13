"""Physical parameters for self-balancing robot.

Single source of truth for all robot physical properties.
See dynamics.md for mathematical model description.
"""

from dataclasses import dataclass
import yaml
from robot_dynamics._internal.validation import (
    validate_positive,
    validate_non_negative
)


@dataclass(frozen=True)
class RobotParameters:
    """Physical parameters for two-wheeled self-balancing robot.

    All parameters immutable after construction (frozen=True).
    Units encoded in parameter names per style_guide.md section 2.1.4.

    Attributes:
        body_mass_kg: Mass of robot body (pendulum portion)
        wheel_mass_kg: Mass of one wheel (assumed identical)
        com_distance_m: Distance from wheel axle to body center of mass
        wheel_radius_m: Radius of wheels
        track_width_m: Distance between left and right wheels (d in dynamics.md)
        body_pitch_inertia_kg_m2: Moment of inertia about pitch axis (I_y)
        body_yaw_inertia_kg_m2: Moment of inertia about yaw axis (I_z)
        wheel_inertia_kg_m2: Rotational inertia of one wheel (J_w)
        gravity_mps2: Gravitational acceleration
        ground_slope_rad: Ground slope angle (0 = flat, positive = uphill)
    """

    # Masses (kg)
    body_mass_kg: float
    wheel_mass_kg: float

    # Geometry (m)
    com_distance_m: float
    wheel_radius_m: float
    track_width_m: float

    # Inertias (kg·m²)
    body_pitch_inertia_kg_m2: float
    body_yaw_inertia_kg_m2: float
    wheel_inertia_kg_m2: float

    # Environment
    gravity_mps2: float = 9.81
    ground_slope_rad: float = 0.0

    def __post_init__(self):
        """Validate parameters satisfy physical constraints."""
        # All masses must be positive
        validate_positive(self.body_mass_kg, 'body_mass_kg')
        validate_positive(self.wheel_mass_kg, 'wheel_mass_kg')

        # Dimensions must be positive
        validate_positive(self.com_distance_m, 'com_distance_m')
        validate_positive(self.wheel_radius_m, 'wheel_radius_m')
        validate_positive(self.track_width_m, 'track_width_m')

        # Inertias must be non-negative (can be zero for point masses)
        validate_non_negative(
            self.body_pitch_inertia_kg_m2, 'body_pitch_inertia_kg_m2'
        )
        validate_non_negative(
            self.body_yaw_inertia_kg_m2, 'body_yaw_inertia_kg_m2'
        )
        validate_non_negative(
            self.wheel_inertia_kg_m2, 'wheel_inertia_kg_m2'
        )

        # Gravity must be positive
        validate_positive(self.gravity_mps2, 'gravity_mps2')

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RobotParameters':
        """Load parameters from YAML configuration file.

        Single source of truth per style_guide.md section 4.4.

        Args:
            yaml_path: Path to YAML file containing robot parameters

        Returns:
            RobotParameters instance

        Raises:
            FileNotFoundError: If YAML file does not exist
            ValueError: If required parameters missing or invalid
        """
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        return cls(**config)

    @property
    def effective_mass_kg(self) -> float:
        """Effective mass M_eff = M + m.

        NOTE: In MuJoCo, wheels are modeled as separate bodies with their own DOFs.
        The wheel inertia does NOT get reflected into the effective mass.
        This differs from the quasi-static wheel assumption in dynamics.md eq 64.
        """
        return (
            self.body_mass_kg +
            2 * self.wheel_mass_kg
            # NOT including: 2 * self.wheel_inertia_kg_m2 / (self.wheel_radius_m ** 2)
        )

    @property
    def effective_pitch_inertia_kg_m2(self) -> float:
        """Effective pitch inertia I_y + m*l^2.

        See dynamics.md equation for pitch dynamics.
        """
        return (
            self.body_pitch_inertia_kg_m2 +
            self.body_mass_kg * (self.com_distance_m ** 2)
        )



# Module-level constants for state/control dimensions
STATE_DIMENSION = 6  # [x, theta, psi, dx, dtheta, dpsi]
CONTROL_DIMENSION = 2  # [tau_L, tau_R]

# State vector indices (for clarity in dynamics functions)
POSITION_INDEX = 0
PITCH_INDEX = 1
YAW_INDEX = 2
VELOCITY_INDEX = 3
PITCH_RATE_INDEX = 4
YAW_RATE_INDEX = 5

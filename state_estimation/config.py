"""State estimator configuration parameters.

Single source of truth for state estimation settings.
See config/estimator_params.yaml for parameter values.
"""

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class EstimatorConfig:
    """Configuration parameters for state estimator.

    All parameters immutable after construction (frozen=True).
    Units encoded in parameter names per style_guide.md section 2.1.4.

    Attributes:
        complementary_filter_time_constant_s: Filter time constant (tau)
            Larger values trust gyroscope more, smaller values trust
            accelerometer more. Typical range: 0.05 to 0.5 seconds.
        sampling_period_s: Expected time between updates (T_s)
    """

    complementary_filter_time_constant_s: float
    sampling_period_s: float

    def __post_init__(self) -> None:
        """Validate parameters satisfy constraints."""
        if self.complementary_filter_time_constant_s <= 0:
            raise ValueError(
                f"complementary_filter_time_constant_s must be positive, "
                f"got {self.complementary_filter_time_constant_s}"
            )
        if self.sampling_period_s <= 0:
            raise ValueError(
                f"sampling_period_s must be positive, "
                f"got {self.sampling_period_s}"
            )

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EstimatorConfig':
        """Load configuration from YAML file.

        Single source of truth per style_guide.md section 4.4.

        Args:
            yaml_path: Path to YAML file containing estimator parameters

        Returns:
            EstimatorConfig instance

        Raises:
            FileNotFoundError: If YAML file does not exist
            ValueError: If required parameters missing or invalid
        """
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)

        return cls(
            complementary_filter_time_constant_s=config[
                'complementary_filter_time_constant_s'
            ],
            sampling_period_s=config['sampling_period_s'],
        )

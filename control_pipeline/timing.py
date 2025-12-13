"""Timing utilities for real-time control loop.

Provides timing measurement and statistics for monitoring control
loop performance.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TimingStatistics:
    """Statistics for timing measurements.

    Attributes:
        mean_s: Mean timing in seconds
        max_s: Maximum timing in seconds
        min_s: Minimum timing in seconds
        std_s: Standard deviation in seconds
        count: Number of measurements
    """

    mean_s: float
    max_s: float
    min_s: float
    std_s: float
    count: int


@dataclass
class IterationTiming:
    """Timing breakdown for a single control iteration.

    Attributes:
        estimation_time_s: Time for state estimation
        model_update_time_s: Time for linearization/discretization updates
        reference_time_s: Time for reference generation
        solve_time_s: Time for MPC solve
        total_time_s: Total iteration time
    """

    estimation_time_s: float = 0.0
    model_update_time_s: float = 0.0
    reference_time_s: float = 0.0
    solve_time_s: float = 0.0
    total_time_s: float = 0.0


class ControlLoopTimer:
    """Utility for real-time control loop timing.

    Tracks timing for each phase of the control loop and provides
    statistics for performance monitoring.

    Attributes:
        deadline_s: Target iteration period (deadline)
    """

    def __init__(
        self,
        deadline_s: float,
        history_length: int = 100,
    ) -> None:
        """Initialize control loop timer.

        Args:
            deadline_s: Target iteration period (e.g., 0.02 for 50 Hz)
            history_length: Number of iterations to keep for statistics
        """
        if deadline_s <= 0:
            raise ValueError(f"deadline_s must be positive, got {deadline_s}")
        if history_length <= 0:
            raise ValueError(
                f"history_length must be positive, got {history_length}"
            )

        self._deadline_s = deadline_s
        self._history_length = history_length

        # Timing storage
        self._iteration_start_time: Optional[float] = None
        self._estimation_end_time: Optional[float] = None
        self._reference_end_time: Optional[float] = None
        self._solve_end_time: Optional[float] = None
        self._model_update_start_time: Optional[float] = None
        self._model_update_end_time: Optional[float] = None

        # History for statistics
        self._timing_history: List[IterationTiming] = []
        self._deadline_violations: int = 0

    def start_iteration(self) -> None:
        """Mark start of control iteration."""
        self._iteration_start_time = time.perf_counter()
        self._estimation_end_time = None
        self._reference_end_time = None
        self._solve_end_time = None
        self._model_update_start_time = None
        self._model_update_end_time = None

    def mark_estimation_complete(self) -> None:
        """Mark end of state estimation phase."""
        self._estimation_end_time = time.perf_counter()

    def mark_reference_complete(self) -> None:
        """Mark end of reference generation phase."""
        self._reference_end_time = time.perf_counter()

    def start_model_update(self) -> None:
        """Mark beginning of model update (linearization/discretization) phase."""
        self._model_update_start_time = time.perf_counter()

    def mark_model_update_complete(self) -> None:
        """Mark completion of model update phase."""
        self._model_update_end_time = time.perf_counter()

    def mark_solve_complete(self) -> None:
        """Mark end of MPC solve phase."""
        self._solve_end_time = time.perf_counter()

    def end_iteration(self) -> IterationTiming:
        """Mark end of control iteration and record timing.

        Returns:
            Timing breakdown for this iteration
        """
        end_time = time.perf_counter()

        if self._iteration_start_time is None:
            raise RuntimeError("start_iteration() was not called")

        # Compute phase timings
        timing = IterationTiming()

        if self._estimation_end_time is not None:
            timing.estimation_time_s = (
                self._estimation_end_time - self._iteration_start_time
            )

        if self._reference_end_time is not None and self._estimation_end_time is not None:
            timing.reference_time_s = (
                self._reference_end_time - self._estimation_end_time
            )
        if (
            self._model_update_start_time is not None
            and self._model_update_end_time is not None
        ):
            timing.model_update_time_s = (
                self._model_update_end_time - self._model_update_start_time
            )

        if self._solve_end_time is not None and self._reference_end_time is not None:
            timing.solve_time_s = (
                self._solve_end_time - self._reference_end_time
            )

        timing.total_time_s = end_time - self._iteration_start_time

        # Check deadline
        if timing.total_time_s > self._deadline_s:
            self._deadline_violations += 1

        # Update history
        self._timing_history.append(timing)
        if len(self._timing_history) > self._history_length:
            self._timing_history.pop(0)

        return timing

    def get_remaining_time_s(self) -> float:
        """Return time remaining before deadline.

        Returns:
            Remaining time in seconds (negative if overdue)
        """
        if self._iteration_start_time is None:
            return self._deadline_s

        elapsed = time.perf_counter() - self._iteration_start_time
        return self._deadline_s - elapsed

    def compute_statistics(self) -> Optional[TimingStatistics]:
        """Compute timing statistics from history.

        Returns:
            Statistics if history is not empty, None otherwise
        """
        if not self._timing_history:
            return None

        import numpy as np

        total_times = np.array([t.total_time_s for t in self._timing_history])

        return TimingStatistics(
            mean_s=float(np.mean(total_times)),
            max_s=float(np.max(total_times)),
            min_s=float(np.min(total_times)),
            std_s=float(np.std(total_times)),
            count=len(total_times),
        )

    @property
    def deadline_s(self) -> float:
        """Target iteration period."""
        return self._deadline_s

    @property
    def deadline_violations(self) -> int:
        """Number of deadline violations since creation."""
        return self._deadline_violations

    @property
    def iteration_count(self) -> int:
        """Number of completed iterations."""
        return len(self._timing_history)

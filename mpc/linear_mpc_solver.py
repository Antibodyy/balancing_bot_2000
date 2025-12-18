"""Linear MPC solver using CasADi Opti interface.

Implements a linear MPC controller for the self-balancing robot using
CasADi's Opti interface with OSQP backend for QP solving.

The MPC formulation is:
    min  sum_{k=0}^{N-1} [(x_k - x_ref)^T Q (x_k - x_ref) + u_k^T R u_k]
         + (x_N - x_ref)^T P (x_N - x_ref)

    s.t. x_{k+1} = A_d x_k + B_d u_k    (dynamics)
         x_0 = current_state             (initial condition)
         x_min <= x_k <= x_max           (state bounds)
         u_min <= u_k <= u_max           (control bounds)
"""

from dataclasses import dataclass
import time
from typing import Optional

import casadi as ca
import numpy as np

from robot_dynamics.parameters import (
    STATE_DIMENSION,
    CONTROL_DIMENSION,
    PITCH_INDEX,
    PITCH_RATE_INDEX,
    VELOCITY_INDEX,
)
from robot_dynamics.discretization import DiscreteDynamics
from mpc.constraints import StateConstraints, InputConstraints


@dataclass(frozen=True)
class MPCSolution:
    """Solution from MPC solver.

    Attributes:
        optimal_control: First control input to apply (CONTROL_DIMENSION,)
        predicted_trajectory: Predicted state trajectory (N+1, STATE_DIMENSION)
        control_sequence: Full control sequence (N, CONTROL_DIMENSION)
        solve_time_s: Wall-clock solve time in seconds
        solver_status: Solver status string ('optimal', 'infeasible', etc.)
        cost: Optimal cost value
        solver_iterations: Number of IPOPT iterations used for this solve
    """

    optimal_control: np.ndarray
    predicted_trajectory: np.ndarray
    control_sequence: np.ndarray
    solve_time_s: float
    solver_status: str
    cost: float
    solver_iterations: int = 0
    velocity_slack: Optional[np.ndarray] = None


class LinearMPCSolver:
    """Linear MPC solver using CasADi Opti with OSQP.

    This solver implements a linear time-invariant MPC for the self-balancing
    robot. It uses the discrete-time linear model obtained by linearizing
    around an equilibrium point.

    Attributes:
        prediction_horizon_steps: Number of prediction steps N
        state_cost: Q matrix (STATE_DIMENSION, STATE_DIMENSION)
        control_cost: R matrix (CONTROL_DIMENSION, CONTROL_DIMENSION)
        terminal_cost: P matrix (STATE_DIMENSION, STATE_DIMENSION)
        state_constraints: State bound constraints
        input_constraints: Control bound constraints
        warm_start_enabled: Whether to warm-start from previous solution
    """

    def __init__(
        self,
        prediction_horizon_steps: int,
        discrete_dynamics: DiscreteDynamics,
        state_cost: np.ndarray,
        control_cost: np.ndarray,
        terminal_cost: np.ndarray,
        state_constraints: StateConstraints,
        input_constraints: InputConstraints,
        solver_name: str = 'osqp',
        warm_start_enabled: bool = True,
        terminal_pitch_limit_rad: Optional[float] = None,
        terminal_pitch_rate_limit_radps: Optional[float] = None,
        terminal_velocity_limit_mps: Optional[float] = None,
        preserve_warm_start_on_dynamics_update: bool = False,
        velocity_limit_slack_weight: float = 0.0,
    ) -> None:
        """Initialize the MPC solver.

        Args:
            prediction_horizon_steps: Number of prediction steps N
            discrete_dynamics: Discrete-time dynamics (A_d, B_d)
            state_cost: State cost matrix Q
            control_cost: Control cost matrix R
            terminal_cost: Terminal cost matrix P
            state_constraints: State bound constraints (path constraints, k=0..N)
            input_constraints: Control bound constraints
            solver_name: QP solver to use ('osqp' or 'qpoases')
            warm_start_enabled: Enable warm-starting from previous solution
            terminal_pitch_limit_rad: Terminal pitch constraint (None = no constraint)
            terminal_pitch_rate_limit_radps: Terminal pitch rate constraint (None = no constraint)
            terminal_velocity_limit_mps: Terminal velocity constraint (None = no constraint)
            preserve_warm_start_on_dynamics_update: Keep previous solution as initial
                guess even when A/B matrices are refreshed (online linearization).
        """
        self._prediction_horizon_steps = prediction_horizon_steps
        self._state_cost = state_cost
        self._state_cost_diag = np.diag(self._state_cost)
        self._state_cost_base_diag = self._state_cost_diag.copy()
        self._current_state_cost_diag = self._state_cost_diag.copy()
        self._control_cost = control_cost
        self._control_cost_diag = np.diag(self._control_cost)
        self._control_cost_base_diag = self._control_cost_diag.copy()
        self._current_control_cost_diag = self._control_cost_diag.copy()
        self._terminal_cost = terminal_cost
        self._terminal_cost_diag = np.diag(self._terminal_cost)
        self._terminal_cost_base_diag = self._terminal_cost_diag.copy()
        self._current_terminal_cost_diag = self._terminal_cost_diag.copy()
        self._state_constraints = state_constraints
        self._input_constraints = input_constraints
        self._solver_name = solver_name
        self._warm_start_enabled = warm_start_enabled
        self._terminal_pitch_limit_rad = terminal_pitch_limit_rad
        self._terminal_pitch_rate_limit_radps = terminal_pitch_rate_limit_radps
        self._terminal_velocity_limit_mps = terminal_velocity_limit_mps
        self._preserve_warm_start_on_dynamics_update = (
            preserve_warm_start_on_dynamics_update
        )
        self._velocity_slack_weight = velocity_limit_slack_weight
        self._velocity_slack_var: Optional[ca.MX] = None
        self._use_velocity_slack = (
            self._state_constraints.velocity_limit_mps is not None
            and self._velocity_slack_weight > 0
        )

        # Store dynamics matrices (will be converted to parameters after Opti creation)
        self._state_matrix = discrete_dynamics.state_matrix_discrete
        self._control_matrix = discrete_dynamics.control_matrix_discrete

        # Build the optimization problem (creates Opti instance and parameters)
        self._build_optimization_problem()

        # Initialize dynamics parameters with provided matrices
        self._opti.set_value(
            self._state_matrix_param,
            discrete_dynamics.state_matrix_discrete
        )
        self._opti.set_value(
            self._control_matrix_param,
            discrete_dynamics.control_matrix_discrete
        )
        self._opti.set_value(self._state_cost_diag_param, self._state_cost_base_diag)
        self._opti.set_value(self._control_cost_diag_param, self._control_cost_base_diag)
        self._opti.set_value(self._terminal_cost_diag_param, self._terminal_cost_base_diag)

        self._terminal_pitch_limit_default = (
            self._terminal_pitch_limit_rad if self._terminal_pitch_limit_rad is not None else 1e6
        )
        self._terminal_pitch_rate_limit_default = (
            self._terminal_pitch_rate_limit_radps if self._terminal_pitch_rate_limit_radps is not None else 1e6
        )
        self._terminal_velocity_limit_default = (
            self._terminal_velocity_limit_mps if self._terminal_velocity_limit_mps is not None else 1e6
        )
        self._opti.set_value(self._terminal_pitch_limit_param, self._terminal_pitch_limit_default)
        self._opti.set_value(self._terminal_pitch_rate_limit_param, self._terminal_pitch_rate_limit_default)
        self._opti.set_value(self._terminal_velocity_limit_param, self._terminal_velocity_limit_default)

        # Warm start storage
        self._previous_state_solution: Optional[np.ndarray] = None
        self._previous_control_solution: Optional[np.ndarray] = None

    def _build_optimization_problem(self) -> None:
        """Build the CasADi Opti optimization problem."""
        horizon = self._prediction_horizon_steps
        n_states = STATE_DIMENSION
        n_controls = CONTROL_DIMENSION

        # Create Opti instance
        self._opti = ca.Opti()

        # Decision variables: state trajectory and control sequence
        # States: x_0, x_1, ..., x_N (N+1 states)
        self._state_variables = self._opti.variable(n_states, horizon + 1)
        # Controls: u_0, u_1, ..., u_{N-1} (N controls)
        self._control_variables = self._opti.variable(n_controls, horizon)

        # Parameters (set at solve time)
        self._initial_state_param = self._opti.parameter(n_states)
        self._reference_trajectory_param = self._opti.parameter(n_states, horizon + 1)

        # Dynamics matrices as parameters (for efficient online linearization)
        self._state_matrix_param = self._opti.parameter(n_states, n_states)
        self._control_matrix_param = self._opti.parameter(n_states, n_controls)

        if self._use_velocity_slack:
            self._velocity_slack_var = self._opti.variable(1, horizon + 1)
            self._opti.subject_to(self._velocity_slack_var >= 0)
        else:
            self._velocity_slack_var = None

        # Cost matrix parameters for runtime overrides
        self._state_cost_diag_param = self._opti.parameter(n_states)
        self._control_cost_diag_param = self._opti.parameter(n_controls)
        self._terminal_cost_diag_param = self._opti.parameter(n_states)

        # Build cost function
        cost = 0

        # Stage costs
        for step_index in range(horizon):
            state_error = (
                self._state_variables[:, step_index]
                - self._reference_trajectory_param[:, step_index]
            )
            control = self._control_variables[:, step_index]

            # State cost: (x - x_ref)^T Q (x - x_ref)
            q_matrix = ca.diag(self._state_cost_diag_param)
            cost += ca.mtimes([state_error.T, q_matrix, state_error])
            # Control cost: u^T R u
            r_matrix = ca.diag(self._control_cost_diag_param)
            cost += ca.mtimes([control.T, r_matrix, control])
            if self._use_velocity_slack:
                slack = self._velocity_slack_var[0, step_index]
                cost += self._velocity_slack_weight * (slack ** 2)

        # Terminal cost: (x_N - x_ref_N)^T P (x_N - x_ref_N)
        terminal_error = (
            self._state_variables[:, horizon]
            - self._reference_trajectory_param[:, horizon]
        )
        terminal_diag = ca.diag(self._terminal_cost_diag_param)
        cost += ca.mtimes([terminal_error.T, terminal_diag, terminal_error])
        if self._use_velocity_slack:
            cost += self._velocity_slack_weight * (self._velocity_slack_var[0, horizon] ** 2)

        self._opti.minimize(cost)

        # Dynamics constraints: x_{k+1} = A_d x_k + B_d u_k
        # Use parameters instead of fixed matrices for online linearization
        for step_index in range(horizon):
            next_state = (
                self._state_matrix_param @ self._state_variables[:, step_index]
                + self._control_matrix_param @ self._control_variables[:, step_index]
            )
            self._opti.subject_to(
                self._state_variables[:, step_index + 1] == next_state
            )

        # Initial condition constraint: x_0 = current_state
        self._opti.subject_to(
            self._state_variables[:, 0] == self._initial_state_param
        )

        # State constraints (box constraints with optional slack on velocity)
        state_lower, state_upper = self._state_constraints.get_bounds()
        skip_velocity_at_initial = (
            self._state_constraints.velocity_limit_mps is not None
            and not self._use_velocity_slack
        )

        for step_index in range(horizon + 1):
            if self._use_velocity_slack:
                slack = self._velocity_slack_var[0, step_index]
                limit = self._state_constraints.velocity_limit_mps or 0.0
                self._opti.subject_to(
                    self._state_variables[VELOCITY_INDEX, step_index]
                    >= -limit - slack
                )
                self._opti.subject_to(
                    self._state_variables[VELOCITY_INDEX, step_index]
                    <= limit + slack
                )
            for state_index in range(n_states):
                if self._use_velocity_slack and state_index == VELOCITY_INDEX:
                    continue
                if skip_velocity_at_initial and state_index == VELOCITY_INDEX and step_index == 0:
                    continue
                if np.isfinite(state_lower[state_index]):
                    self._opti.subject_to(
                        self._state_variables[state_index, step_index]
                        >= state_lower[state_index]
                    )
                if np.isfinite(state_upper[state_index]):
                    self._opti.subject_to(
                        self._state_variables[state_index, step_index]
                        <= state_upper[state_index]
                    )

        # ADDITIONAL terminal constraints at step N (applied on top of state constraints)
        self._terminal_pitch_limit_param = self._opti.parameter()
        self._terminal_pitch_rate_limit_param = self._opti.parameter()
        self._terminal_velocity_limit_param = self._opti.parameter()

        self._opti.subject_to(
            self._state_variables[PITCH_INDEX, horizon]
            >= -self._terminal_pitch_limit_param
        )
        self._opti.subject_to(
            self._state_variables[PITCH_INDEX, horizon]
            <= self._terminal_pitch_limit_param
        )
        self._opti.subject_to(
            self._state_variables[PITCH_RATE_INDEX, horizon]
            >= -self._terminal_pitch_rate_limit_param
        )
        self._opti.subject_to(
            self._state_variables[PITCH_RATE_INDEX, horizon]
            <= self._terminal_pitch_rate_limit_param
        )
        self._opti.subject_to(
            self._state_variables[VELOCITY_INDEX, horizon]
            >= -self._terminal_velocity_limit_param
        )
        self._opti.subject_to(
            self._state_variables[VELOCITY_INDEX, horizon]
            <= self._terminal_velocity_limit_param
        )

        # Control constraints (box constraints)
        control_lower, control_upper = self._input_constraints.get_bounds()
        for step_index in range(horizon):
            for control_index in range(n_controls):
                self._opti.subject_to(
                    self._control_variables[control_index, step_index]
                    >= control_lower[control_index]
                )
                self._opti.subject_to(
                    self._control_variables[control_index, step_index]
                    <= control_upper[control_index]
                )

        # Configure solver
        # Note: CasADi Opti uses nlpsol interface. For QP problems:
        # - IPOPT: general NLP solver, works well for QPs
        # - qpOASES: requires conic interface (not Opti)
        # - OSQP: requires conic interface (not Opti)
        # We use IPOPT through Opti for simplicity and reliability.

        ipopt_options = {
            'print_time': False,
            'ipopt': {
                'print_level': 0,
                'sb': 'yes',  # Suppress IPOPT banner
                'warm_start_init_point': 'yes' if self._warm_start_enabled else 'no',
                # QP-specific settings for faster convergence
                'max_iter': 100,
                'tol': 1e-6,
            }
        }
        self._opti.solver('ipopt', ipopt_options)

    def solve(
        self,
        current_state: np.ndarray,
        reference_trajectory: np.ndarray,
        state_cost_override: Optional[np.ndarray] = None,
        control_cost_override: Optional[np.ndarray] = None,
        terminal_cost_override: Optional[np.ndarray] = None,
        terminal_pitch_limit_override: Optional[float] = None,
        terminal_pitch_rate_limit_override: Optional[float] = None,
        terminal_velocity_limit_override: Optional[float] = None,
    ) -> MPCSolution:
        """Solve the MPC problem.

        Args:
            current_state: Current state estimate (STATE_DIMENSION,)
            reference_trajectory: Reference states over horizon.
                Either (STATE_DIMENSION,) for constant reference or
                (N+1, STATE_DIMENSION) for time-varying reference.

        Returns:
            MPCSolution with optimal control and diagnostics

        Raises:
            ValueError: If input dimensions are incorrect
        """
        horizon = self._prediction_horizon_steps

        # Validate current state
        current_state = np.asarray(current_state).flatten()
        if current_state.shape != (STATE_DIMENSION,):
            raise ValueError(
                f"current_state must have shape ({STATE_DIMENSION},), "
                f"got {current_state.shape}"
            )

        # Handle reference trajectory
        reference_trajectory = np.asarray(reference_trajectory)
        if reference_trajectory.shape == (STATE_DIMENSION,):
            # Constant reference: expand to full horizon
            reference_trajectory = np.tile(
                reference_trajectory.reshape(-1, 1), (1, horizon + 1)
            )
        elif reference_trajectory.shape == (horizon + 1, STATE_DIMENSION):
            # Time-varying reference: transpose to match CasADi format
            reference_trajectory = reference_trajectory.T
        else:
            raise ValueError(
                f"reference_trajectory must have shape ({STATE_DIMENSION},) or "
                f"({horizon + 1}, {STATE_DIMENSION}), got {reference_trajectory.shape}"
            )

        # Set state cost override if provided
        if state_cost_override is not None:
            override_diag = np.diag(state_cost_override).astype(float)
            self._opti.set_value(self._state_cost_diag_param, override_diag)
            self._current_state_cost_diag = override_diag.copy()
        else:
            self._opti.set_value(self._state_cost_diag_param, self._state_cost_base_diag)
            self._current_state_cost_diag = self._state_cost_base_diag.copy()

        # Set control cost override if provided
        if control_cost_override is not None:
            control_override_diag = np.diag(control_cost_override).astype(float)
            self._opti.set_value(self._control_cost_diag_param, control_override_diag)
            self._current_control_cost_diag = control_override_diag.copy()
        else:
            self._opti.set_value(
                self._control_cost_diag_param,
                self._control_cost_base_diag
            )
            self._current_control_cost_diag = self._control_cost_base_diag.copy()

        # Terminal cost override if provided
        if terminal_cost_override is not None:
            terminal_override_diag = np.diag(terminal_cost_override).astype(float)
            self._opti.set_value(self._terminal_cost_diag_param, terminal_override_diag)
            self._current_terminal_cost_diag = terminal_override_diag.copy()
        else:
            self._opti.set_value(
                self._terminal_cost_diag_param,
                self._terminal_cost_base_diag
            )
            self._current_terminal_cost_diag = self._terminal_cost_base_diag.copy()

        # Terminal constraint overrides
        pitch_limit = (
            float(terminal_pitch_limit_override)
            if terminal_pitch_limit_override is not None
            else self._terminal_pitch_limit_default
        )
        pitch_rate_limit = (
            float(terminal_pitch_rate_limit_override)
            if terminal_pitch_rate_limit_override is not None
            else self._terminal_pitch_rate_limit_default
        )
        velocity_limit = (
            float(terminal_velocity_limit_override)
            if terminal_velocity_limit_override is not None
            else self._terminal_velocity_limit_default
        )
        self._opti.set_value(self._terminal_pitch_limit_param, pitch_limit)
        self._opti.set_value(self._terminal_pitch_rate_limit_param, pitch_rate_limit)
        self._opti.set_value(self._terminal_velocity_limit_param, velocity_limit)

        # Set parameters
        self._opti.set_value(self._initial_state_param, current_state)
        self._opti.set_value(self._reference_trajectory_param, reference_trajectory)

        # Apply warm start if available
        if self._warm_start_enabled and self._previous_state_solution is not None:
            self._apply_warm_start()

        # Solve
        start_time = time.perf_counter()
        iter_count = 0
        velocity_slack = None
        try:
            solution = self._opti.solve()
            solve_time_s = time.perf_counter() - start_time
            solver_status = 'optimal'
            stats = solution.stats() if hasattr(solution, "stats") else {}
            iter_count = int(stats.get('iter_count', 0))

            # Extract solution
            state_trajectory = solution.value(self._state_variables).T
            control_sequence = solution.value(self._control_variables).T
            cost = solution.value(self._opti.f)
            if self._velocity_slack_var is not None:
                velocity_slack = solution.value(self._velocity_slack_var).T

            # Store for warm start
            if self._warm_start_enabled:
                self._previous_state_solution = state_trajectory
                self._previous_control_solution = control_sequence

        except RuntimeError as error:
            solve_time_s = time.perf_counter() - start_time
            solver_status = str(error)

            # Return zero control on failure
            state_trajectory = np.zeros((horizon + 1, STATE_DIMENSION))
            control_sequence = np.zeros((horizon, CONTROL_DIMENSION))
            cost = np.inf
            iter_count = 0

        return MPCSolution(
            optimal_control=control_sequence[0, :].copy(),
            predicted_trajectory=state_trajectory,
            control_sequence=control_sequence,
            solve_time_s=solve_time_s,
            solver_status=solver_status,
            cost=cost,
            solver_iterations=iter_count,
            velocity_slack=velocity_slack,
        )

    def _apply_warm_start(self) -> None:
        """Apply warm start from previous solution.

        Uses shift strategy: u_warm[k] = u_prev[k+1] for k < N-1,
        u_warm[N-1] = u_prev[N-1] (repeat last).
        """
        if self._previous_control_solution is None:
            return

        horizon = self._prediction_horizon_steps

        # Shift control sequence
        shifted_control = np.zeros((horizon, CONTROL_DIMENSION))
        shifted_control[:-1, :] = self._previous_control_solution[1:, :]
        shifted_control[-1, :] = self._previous_control_solution[-1, :]

        # Shift state trajectory (approximate)
        shifted_state = np.zeros((horizon + 1, STATE_DIMENSION))
        shifted_state[:-1, :] = self._previous_state_solution[1:, :]
        # Propagate last state with last control
        shifted_state[-1, :] = (
            self._state_matrix @ self._previous_state_solution[-1, :]
            + self._control_matrix @ shifted_control[-1, :]
        )

        # Set initial guesses
        self._opti.set_initial(self._state_variables, shifted_state.T)
        self._opti.set_initial(self._control_variables, shifted_control.T)

    def update_dynamics(self, discrete_dynamics: DiscreteDynamics) -> None:
        """Update the dynamics matrices for successive linearization.

        Efficiently updates dynamics by changing CasADi parameter values
        instead of rebuilding the entire optimization problem. This is
        critical for real-time performance in online linearization.

        Args:
            discrete_dynamics: New discrete-time dynamics (A_d, B_d)
        """
        # Update CasADi parameters (FAST - just updates values)
        self._opti.set_value(
            self._state_matrix_param,
            discrete_dynamics.state_matrix_discrete
        )
        self._opti.set_value(
            self._control_matrix_param,
            discrete_dynamics.control_matrix_discrete
        )
        self._opti.set_value(self._state_cost_diag_param, self._state_cost_base_diag)
        self._opti.set_value(self._control_cost_diag_param, self._control_cost_base_diag)

        # Update numpy copies for reference
        self._state_matrix = discrete_dynamics.state_matrix_discrete
        self._control_matrix = discrete_dynamics.control_matrix_discrete

        if not self._preserve_warm_start_on_dynamics_update:
            # Default behaviour: discard guesses so we don't seed IPOPT with states
            # computed under a different linearization model.
            self._previous_state_solution = None
            self._previous_control_solution = None

        # NO NEED to rebuild problem - parameters handle the update!

    @property
    def prediction_horizon_steps(self) -> int:
        """Number of prediction steps N."""
        return self._prediction_horizon_steps

    @property
    def state_matrix(self) -> np.ndarray:
        """Current discrete-time state transition matrix."""
        return self._state_matrix.copy()

    @property
    def control_matrix(self) -> np.ndarray:
        """Current discrete-time control matrix."""
        return self._control_matrix.copy()

    @property
    def warm_start_enabled(self) -> bool:
        return self._warm_start_enabled

    @property
    def state_cost(self) -> np.ndarray:
        """Current stage cost matrix."""
        return np.diag(self._current_state_cost_diag)

    @property
    def base_state_cost(self) -> np.ndarray:
        """Default stage cost matrix from configuration."""
        return np.diag(self._state_cost_base_diag)

    @property
    def control_cost(self) -> np.ndarray:
        """Current control cost matrix."""
        return np.diag(self._current_control_cost_diag)

    @property
    def base_control_cost(self) -> np.ndarray:
        """Default control cost matrix from configuration."""
        return np.diag(self._control_cost_base_diag)

    @property
    def control_limit_nm(self) -> float:
        """Torque limit enforced by input constraints."""
        return self._input_constraints.control_limit_nm

    @property
    def terminal_cost(self) -> np.ndarray:
        """Current terminal cost matrix."""
        return np.diag(self._current_terminal_cost_diag)

    @property
    def base_terminal_cost(self) -> np.ndarray:
        """Default terminal cost matrix from configuration."""
        return np.diag(self._terminal_cost_base_diag)

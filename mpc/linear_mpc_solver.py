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
    """

    optimal_control: np.ndarray
    predicted_trajectory: np.ndarray
    control_sequence: np.ndarray
    solve_time_s: float
    solver_status: str
    cost: float


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
        """
        self._prediction_horizon_steps = prediction_horizon_steps
        self._state_cost = state_cost
        self._control_cost = control_cost
        self._terminal_cost = terminal_cost
        self._state_constraints = state_constraints
        self._input_constraints = input_constraints
        self._solver_name = solver_name
        self._warm_start_enabled = warm_start_enabled
        self._terminal_pitch_limit_rad = terminal_pitch_limit_rad
        self._terminal_pitch_rate_limit_radps = terminal_pitch_rate_limit_radps
        self._terminal_velocity_limit_mps = terminal_velocity_limit_mps

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
            cost += ca.mtimes([state_error.T, self._state_cost, state_error])
            # Control cost: u^T R u
            cost += ca.mtimes([control.T, self._control_cost, control])

        # Terminal cost: (x_N - x_ref_N)^T P (x_N - x_ref_N)
        terminal_error = (
            self._state_variables[:, horizon]
            - self._reference_trajectory_param[:, horizon]
        )
        cost += ca.mtimes([terminal_error.T, self._terminal_cost, terminal_error])

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

        # State constraints (box constraints)
        state_lower, state_upper = self._state_constraints.get_bounds()
        for step_index in range(horizon + 1):
            for state_index in range(n_states):
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
        if (self._terminal_pitch_limit_rad is not None or
            self._terminal_pitch_rate_limit_radps is not None or
            self._terminal_velocity_limit_mps is not None):

            # Additional constraint: pitch at terminal must satisfy tighter bounds
            # This is IN ADDITION to the regular state constraint already applied above
            if self._terminal_pitch_limit_rad is not None:
                self._opti.subject_to(
                    self._state_variables[PITCH_INDEX, horizon]
                    >= -self._terminal_pitch_limit_rad
                )
                self._opti.subject_to(
                    self._state_variables[PITCH_INDEX, horizon]
                    <= self._terminal_pitch_limit_rad
                )

            # Additional constraint: pitch rate at terminal must satisfy tighter bounds
            if self._terminal_pitch_rate_limit_radps is not None:
                self._opti.subject_to(
                    self._state_variables[PITCH_RATE_INDEX, horizon]
                    >= -self._terminal_pitch_rate_limit_radps
                )
                self._opti.subject_to(
                    self._state_variables[PITCH_RATE_INDEX, horizon]
                    <= self._terminal_pitch_rate_limit_radps
                )

            # Additional constraint: velocity at terminal must satisfy tighter bounds
            if self._terminal_velocity_limit_mps is not None:
                self._opti.subject_to(
                    self._state_variables[VELOCITY_INDEX, horizon]
                    >= -self._terminal_velocity_limit_mps
                )
                self._opti.subject_to(
                    self._state_variables[VELOCITY_INDEX, horizon]
                    <= self._terminal_velocity_limit_mps
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

        # Set parameters
        self._opti.set_value(self._initial_state_param, current_state)
        self._opti.set_value(self._reference_trajectory_param, reference_trajectory)

        # Apply warm start if available
        if self._warm_start_enabled and self._previous_state_solution is not None:
            self._apply_warm_start()

        # Solve
        start_time = time.perf_counter()
        try:
            solution = self._opti.solve()
            solve_time_s = time.perf_counter() - start_time
            solver_status = 'optimal'

            # Extract solution
            state_trajectory = solution.value(self._state_variables).T
            control_sequence = solution.value(self._control_variables).T
            cost = solution.value(self._opti.f)

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

        return MPCSolution(
            optimal_control=control_sequence[0, :].copy(),
            predicted_trajectory=state_trajectory,
            control_sequence=control_sequence,
            solve_time_s=solve_time_s,
            solver_status=solver_status,
            cost=cost,
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

        Warm start is preserved across linearization updates since the
        dynamics changes are small at high sampling rates (20ms), and the
        shifted previous solution remains a good initial guess.

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

        # Update numpy copies for reference
        self._state_matrix = discrete_dynamics.state_matrix_discrete
        self._control_matrix = discrete_dynamics.control_matrix_discrete

        # NO NEED to rebuild problem - parameters handle the update!

    @property
    def prediction_horizon_steps(self) -> int:
        """Number of prediction steps N."""
        return self._prediction_horizon_steps

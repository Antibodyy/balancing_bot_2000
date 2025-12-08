"""Matplotlib plotting utilities for MPC debugging.

Provides reusable plotting functions for analyzing MPC behavior.
"""

from typing import Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from robot_dynamics.parameters import (
    POSITION_INDEX,
    PITCH_INDEX,
    YAW_INDEX,
    VELOCITY_INDEX,
    PITCH_RATE_INDEX,
    YAW_RATE_INDEX,
)

# State labels for plots
STATE_LABELS = [
    ('Position', 'x', 'm'),
    ('Pitch', 'θ', 'rad'),
    ('Yaw', 'ψ', 'rad'),
    ('Velocity', 'ẋ', 'm/s'),
    ('Pitch Rate', 'θ̇', 'rad/s'),
    ('Yaw Rate', 'ψ̇', 'rad/s'),
]


def plot_state_comparison(
    time_s: np.ndarray,
    true_state: np.ndarray,
    estimated_state: np.ndarray,
    title: str = "True vs Estimated State",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot true state vs estimated state over time.

    Args:
        time_s: Time array (N,)
        true_state: True state history (N, 6)
        estimated_state: Estimated state history (N, 6)
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    for idx, (name, symbol, unit) in enumerate(STATE_LABELS):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]

        ax.plot(time_s, true_state[:, idx], 'b-', label='True', linewidth=1.5)
        ax.plot(time_s, estimated_state[:, idx], 'r--', label='Estimated', linewidth=1.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{symbol} ({unit})')
        ax.set_title(name)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_control_analysis(
    time_s: np.ndarray,
    control_history: np.ndarray,
    control_limit: float,
    title: str = "Control Analysis",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot control commands and saturation analysis.

    Args:
        time_s: Time array (N,)
        control_history: Control history (N, 2) - [τ_L, τ_R]
        control_limit: Control saturation limit (N⋅m)
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    # Left wheel torque
    ax = axes[0, 0]
    ax.plot(time_s, control_history[:, 0], 'b-', linewidth=1.5)
    ax.axhline(y=control_limit, color='r', linestyle='--', label='Limit')
    ax.axhline(y=-control_limit, color='r', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('τ_L (N⋅m)')
    ax.set_title('Left Wheel Torque')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right wheel torque
    ax = axes[0, 1]
    ax.plot(time_s, control_history[:, 1], 'g-', linewidth=1.5)
    ax.axhline(y=control_limit, color='r', linestyle='--', label='Limit')
    ax.axhline(y=-control_limit, color='r', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('τ_R (N⋅m)')
    ax.set_title('Right Wheel Torque')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Total torque (sum)
    ax = axes[1, 0]
    total_torque = control_history[:, 0] + control_history[:, 1]
    ax.plot(time_s, total_torque, 'm-', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('τ_L + τ_R (N⋅m)')
    ax.set_title('Total Torque (pitch control)')
    ax.grid(True, alpha=0.3)

    # Saturation indicator
    ax = axes[1, 1]
    left_saturated = np.abs(control_history[:, 0]) >= control_limit * 0.99
    right_saturated = np.abs(control_history[:, 1]) >= control_limit * 0.99
    any_saturated = left_saturated | right_saturated

    ax.fill_between(time_s, 0, left_saturated.astype(float), alpha=0.5, label='Left saturated')
    ax.fill_between(time_s, 0, right_saturated.astype(float), alpha=0.5, label='Right saturated')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Saturated')
    ax.set_title(f'Saturation ({np.sum(any_saturated)}/{len(time_s)} steps)')
    ax.legend()
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_prediction_accuracy(
    time_s: np.ndarray,
    true_state: np.ndarray,
    predicted_trajectories: List[np.ndarray],
    step_indices: Optional[List[int]] = None,
    title: str = "MPC Prediction Accuracy",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot MPC predicted trajectories vs actual trajectory.

    Args:
        time_s: Time array (N,)
        true_state: True state history (N, 6)
        predicted_trajectories: List of predicted trajectories (N+1, 6) for each step
        step_indices: Which steps to show predictions for (default: every 5th)
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    if step_indices is None:
        # Show predictions at regular intervals
        step_indices = list(range(0, len(predicted_trajectories), max(1, len(predicted_trajectories) // 10)))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(title, fontsize=14)

    # Pitch trajectory with predictions
    ax = axes[0]
    ax.plot(time_s, true_state[:, PITCH_INDEX], 'b-', linewidth=2, label='Actual')

    dt = time_s[1] - time_s[0] if len(time_s) > 1 else 0.02
    for step_idx in step_indices:
        if step_idx < len(predicted_trajectories):
            pred = predicted_trajectories[step_idx]
            pred_time = time_s[step_idx] + np.arange(len(pred)) * dt
            ax.plot(pred_time, pred[:, PITCH_INDEX], 'r-', alpha=0.3, linewidth=1)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch θ (rad)')
    ax.set_title('Pitch: Actual vs Predicted')
    ax.legend(['Actual', 'Predicted'])
    ax.grid(True, alpha=0.3)

    # One-step prediction error
    ax = axes[1]
    one_step_errors = []
    for i in range(len(predicted_trajectories) - 1):
        pred_next = predicted_trajectories[i][1, PITCH_INDEX]  # Predicted next pitch
        actual_next = true_state[i + 1, PITCH_INDEX]  # Actual next pitch
        one_step_errors.append(pred_next - actual_next)

    if one_step_errors:
        ax.plot(time_s[:-1], one_step_errors, 'r-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Prediction Error (rad)')
        ax.set_title(f'One-Step Pitch Prediction Error (mean={np.mean(np.abs(one_step_errors)):.4f} rad)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_closed_loop_diagnosis(
    time_s: np.ndarray,
    true_state: np.ndarray,
    estimated_state: np.ndarray,
    control_history: np.ndarray,
    predicted_trajectories: List[np.ndarray],
    control_limit: float,
    solve_times_ms: np.ndarray,
    title: str = "Closed-Loop Diagnosis",
    save_path: Optional[str] = None,
) -> Figure:
    """Generate comprehensive 6-panel diagnostic figure.

    Args:
        time_s: Time array (N,)
        true_state: True state history (N, 6)
        estimated_state: Estimated state history (N, 6)
        control_history: Control history (N, 2)
        predicted_trajectories: Predicted trajectories for each step
        control_limit: Control saturation limit
        solve_times_ms: MPC solve times in ms
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(title, fontsize=14)

    # Panel 1: True vs Estimated Pitch
    ax = axes[0, 0]
    ax.plot(time_s, np.rad2deg(true_state[:, PITCH_INDEX]), 'b-', label='True', linewidth=1.5)
    ax.plot(time_s, np.rad2deg(estimated_state[:, PITCH_INDEX]), 'r--', label='Estimated', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (deg)')
    ax.set_title('Pitch: True vs Estimated')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: True vs Estimated Pitch Rate
    ax = axes[0, 1]
    ax.plot(time_s, np.rad2deg(true_state[:, PITCH_RATE_INDEX]), 'b-', label='True', linewidth=1.5)
    ax.plot(time_s, np.rad2deg(estimated_state[:, PITCH_RATE_INDEX]), 'r--', label='Estimated', linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch Rate (deg/s)')
    ax.set_title('Pitch Rate: True vs Estimated')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Control Commands
    ax = axes[1, 0]
    ax.plot(time_s, control_history[:, 0], 'b-', label='τ_L', linewidth=1.5)
    ax.plot(time_s, control_history[:, 1], 'g-', label='τ_R', linewidth=1.5)
    ax.axhline(y=control_limit, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=-control_limit, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (N⋅m)')
    ax.set_title('Control Commands')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Control vs Pitch Error (direction check)
    ax = axes[1, 1]
    pitch_error = estimated_state[:, PITCH_INDEX]  # Error from zero reference
    total_torque = control_history[:, 0] + control_history[:, 1]
    ax.scatter(np.rad2deg(pitch_error), total_torque, c=time_s, cmap='viridis', alpha=0.7, s=10)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Pitch Error (deg)')
    ax.set_ylabel('Total Torque (N⋅m)')
    ax.set_title('Control Direction Check\n(should have negative slope)')
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Time (s)')
    ax.grid(True, alpha=0.3)

    # Panel 5: MPC Solve Time
    ax = axes[2, 0]
    ax.plot(time_s, solve_times_ms, 'b-', linewidth=1.5)
    ax.axhline(y=20, color='r', linestyle='--', label='20ms target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Solve Time (ms)')
    ax.set_title(f'MPC Solve Time (mean={np.mean(solve_times_ms):.1f}ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 6: One-step Prediction Error
    ax = axes[2, 1]
    if len(predicted_trajectories) > 1:
        one_step_errors = []
        for i in range(len(predicted_trajectories) - 1):
            pred_next = predicted_trajectories[i][1, PITCH_INDEX]
            actual_next = true_state[i + 1, PITCH_INDEX]
            one_step_errors.append(pred_next - actual_next)
        ax.plot(time_s[:-1], np.rad2deg(one_step_errors), 'r-', linewidth=1.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_title(f'Prediction Error (mean={np.mean(np.abs(one_step_errors))*1000:.2f} mrad)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch Prediction Error (deg)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_control_direction_test(
    pitch_values_deg: np.ndarray,
    control_responses: np.ndarray,
    title: str = "Control Direction Verification",
    save_path: Optional[str] = None,
) -> Figure:
    """Plot control response for various pitch perturbations.

    For a stable controller:
    - Positive pitch (leaning forward) should get negative total torque (push back)
    - Negative pitch (leaning backward) should get positive total torque (push forward)

    Args:
        pitch_values_deg: Array of pitch values tested (in degrees)
        control_responses: Corresponding control outputs (N, 2) - [τ_L, τ_R]
        title: Figure title
        save_path: Optional path to save figure

    Returns:
        Matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    total_torque = control_responses[:, 0] + control_responses[:, 1]

    # Bar chart
    ax = axes[0]
    colors = ['red' if t * p > 0 else 'green' for t, p in zip(total_torque, pitch_values_deg)]
    bars = ax.bar(pitch_values_deg, total_torque, color=colors, alpha=0.7, width=0.8)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax.set_xlabel('Pitch Perturbation (deg)')
    ax.set_ylabel('Total Torque τ_L + τ_R (N⋅m)')
    ax.set_title('Control Response\n(green=correct, red=wrong direction)')
    ax.grid(True, alpha=0.3, axis='y')

    # Scatter with trend
    ax = axes[1]
    ax.scatter(pitch_values_deg, total_torque, c='blue', s=100, zorder=5)
    ax.plot(pitch_values_deg, total_torque, 'b--', alpha=0.5)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Add expected trend line
    if len(pitch_values_deg) > 1:
        slope = np.polyfit(pitch_values_deg, total_torque, 1)[0]
        ax.set_title(f'Pitch vs Torque (slope={slope:.3f})\n(should be NEGATIVE for stability)')
    ax.set_xlabel('Pitch (deg)')
    ax.set_ylabel('Total Torque (N⋅m)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig

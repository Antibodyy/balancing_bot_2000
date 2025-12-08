"""Debug module for MPC diagnostics and visualization.

Provides tools for debugging MPC controller behavior:
- MPCDiagnostics: Main diagnostics class with plotting methods
- plot_state_comparison: Compare true vs estimated states
- plot_control_analysis: Analyze control commands
- plot_prediction_accuracy: Compare MPC predictions vs actual
"""

from debug.mpc_diagnostics import MPCDiagnostics
from debug.plotting import (
    plot_state_comparison,
    plot_control_analysis,
    plot_prediction_accuracy,
    plot_closed_loop_diagnosis,
)

__all__ = [
    'MPCDiagnostics',
    'plot_state_comparison',
    'plot_control_analysis',
    'plot_prediction_accuracy',
    'plot_closed_loop_diagnosis',
]

"""Plotting package containing modular plotting utilities for the project.

Modules:
- types: Gemeinsame Datenstrukturen (AlgorithmResult, BudgetResult).
- utils: Hilfsfunktionen (z.B. Errorbar-Downsampling).
- heatmap: Heatmap of selected agents.
- leaves: Average leaves plots across budgets.
- regret: Cumulative regret plots across budgets.
- reward: Average reward plots across selected budgets.
- metrics: Additional metrics visualizations (e.g., leaf node boxplots).

Each module exposes small, well-documented functions that accept data and
configuration as arguments, avoiding reliance on global state.
"""

from .heatmap import plot_selected_agents
from .leaves import plot_all_average_leaves
from .metrics import plot_additional_metrics
from .regret import plot_all_cumulative_regret
from .reward import plot_all_average_reward
from .types import AlgorithmResult, BudgetResult

__all__ = [
    "AlgorithmResult",
    "BudgetResult",
    "plot_selected_agents",
    "plot_all_average_leaves",
    "plot_all_cumulative_regret",
    "plot_all_average_reward",
    "plot_additional_metrics",
]



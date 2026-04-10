import matplotlib.pyplot as plt

from .types import BudgetResult
from .utils import downsample_for_errorbars


def _plot_single_average_leaves(ax, run: BudgetResult, num_std_to_show: int) -> None:
    """Zeichnet die durchschnittliche Blattanzahl für einen BudgetResult auf eine Axes."""
    for algo in run.algorithms:
        if algo.avg_leaves is None:
            continue
        std = downsample_for_errorbars(algo.std_leaves, run.num_rounds, num_std_to_show)
        ax.errorbar(
            range(1, run.num_rounds + 1),
            algo.avg_leaves,
            yerr=std,
            label=algo.label,
            color=algo.color,
            capsize=2,
            linestyle="-",
            linewidth=2,
        )
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.set_xlabel("Arriving task $(t)$")
    ax.set_ylabel("Average leaves up to $t$")
    ax.set_title(f"Average Leaves (b = {run.budget})")
    ax.grid(True, alpha=0.3)


def plot_all_average_leaves(
    runs: list[BudgetResult],
    num_std_to_show: int,
    results_dir: str = ".",
) -> None:
    """Erstellt ein Grid mit Average-Leaves-Plots für alle BudgetResults."""
    num_plots = len(runs)
    rows = int(num_plots / 2 + (num_plots % 2))
    cols = 2
    plt.figure(figsize=(15, 12))

    for i, run in enumerate(runs):
        ax = plt.subplot(rows, cols, i + 1)
        _plot_single_average_leaves(ax, run, num_std_to_show)
        if i == 0:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{results_dir}/avg_leaves_all_budgets.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close()

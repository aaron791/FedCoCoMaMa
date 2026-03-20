import matplotlib.pyplot as plt


def _plot_single_cumulative_regret(ax, budget, omv_avg, neural_cocoma_avg, random_avg, omv_std, neural_cocoma_std, random_std, num_rounds, num_std_to_show, algorithm_colors):
    """Plot cumulative regret for a specific budget on a given axes."""
    for i in range(len(omv_std)):
        if i == 0 or i % int(num_rounds / num_std_to_show) != 0 and i != len(omv_std) - 1:
            omv_std[i] = neural_cocoma_std[i] = random_std[i] = None

    ax.errorbar(range(1, num_rounds + 1), omv_avg, yerr=omv_std,
                 label=f"CoCoMaMa (ours)", capsize=2, color=algorithm_colors['CoCoMaMa'],
                 linestyle='-', linewidth=2)
    ax.errorbar(range(1, num_rounds + 1), neural_cocoma_avg, yerr=neural_cocoma_std,
                 label=f"Neural-CoCoMaMa (ours)", capsize=2, color=algorithm_colors['Neural-CoCoMaMa (ours)'],
                 linestyle='-', linewidth=2)
    ax.errorbar(range(1, num_rounds + 1), random_avg, yerr=random_std,
                 label=f"Random", capsize=2, color=algorithm_colors['Random'],
                 linestyle='-', linewidth=2)

    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.set_xlabel("Arriving task $(t)$")
    ax.set_ylabel("Cumulative regret up to $t$")
    ax.set_title(f'Cumulative Regret (b = {budget})')


def plot_all_cumulative_regret(processed_data, budgets, num_rounds, num_std_to_show, algorithm_colors, results_dir="."):
    """Create a grid figure with cumulative regret plots for all budgets."""
    plt.figure(figsize=(15, 12))
    num_plots = len(budgets)
    rows = int(num_plots / 2 + (num_plots % 2))
    cols = 2

    for i, budget in enumerate(budgets):
        ax = plt.subplot(rows, cols, i + 1)
        budget_data = processed_data[budget]
        _plot_single_cumulative_regret(
            ax, budget,
            budget_data['cocomama_avg_regret'],
            budget_data['neural_cocoma_avg_regret'],
            budget_data['random_avg_regret'],
            budget_data['cocomama_std_regret'],
            budget_data['neural_cocoma_std_regret'],
            budget_data['random_std_regret'],
            num_rounds, num_std_to_show, algorithm_colors
        )
        if i == 0:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{results_dir}/cum_regret_2x2.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.close()

"""Runner implementation for single-router experiments."""

from __future__ import annotations

import multiprocessing
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

from algorithms.streaming_base import StreamingBenchmark, StreamingRandom
from algorithms.streaming_cocoma import StreamingCoCoMaMa
from algorithms.streaming_neural_cocomama import StreamingNeuralCoCoMaMa
from config.single_router import SingleRouterConfig
from Hyperrectangle import Hyperrectangle
from plotting import (
    AlgorithmResult,
    BudgetResult,
    plot_all_average_leaves,
    plot_all_average_reward,
    plot_all_cumulative_regret,
)
from runners.trial_result import TrialResult
from streaming_dataset import StreamingProblemModel

sns.set(style="whitegrid")

# Anzeigeeinstellungen pro Algorithmus: Label und Farbe für Plots
ALGORITHM_DISPLAY = {
    "cocomama":      {"label": "CoCoMaMa (ours)",        "color": "green"},
    "neural_cocoma": {"label": "Neural-CoCoMaMa (ours)", "color": "cyan"},
    "random":        {"label": "Random",                  "color": "purple"},
    "benchmark":     {"label": "Oracle",                  "color": "black"},
}


class SingleRouterRunner:
    """Executes the full single-router experiment pipeline."""

    def run(self, config: SingleRouterConfig) -> None:
        print(f"Initial memory usage: {get_memory_usage():.2f} MB")

        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        config.print_summary()

        if not config.only_redo_plots:
            print("Ensuring datasets are created...")
            create_dataset_if_needed(config)

            for budget in config.budgets:
                run_trials_for_budget(budget, config)

        if config.plot or config.only_redo_plots:
            print("Loading and processing streaming results for plotting...")
            runs = load_and_process_streaming_results(config)

            print("Creating streaming algorithm plots...")
            plot_all_cumulative_regret(runs, config.num_std_to_show, config.RESULTS_DIR)
            plot_all_average_reward(runs, config.num_std_to_show, config.RESULTS_DIR)
            plot_all_average_leaves(runs, config.num_std_to_show, config.RESULTS_DIR)

            plt.show()
            print("Streaming plots created successfully!")

        print("Experiment completed!")


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_trial(trial_index: int, budget: int, config: SingleRouterConfig) -> TrialResult:
    """Führt einen Trial mit allen Algorithmen aus und gibt typisierte Ergebnisse zurück."""
    root_context = _build_root_context(config.embedding_config.dimensions)
    algos = _create_algorithms(budget, config, root_context)

    print(f"Running trial {trial_index} (budget={budget})...")

    random_reward, random_regret, random_played = algos["random"].run_algorithm()
    bench_reward, bench_regret, bench_played, best_arms = algos["benchmark"].run_algorithm()
    (cocoma_reward, cocoma_regret, cocoma_played,
     cocoma_leaves, cocoma_metrics) = algos["cocomama"].run_algorithm()
    (neural_reward, neural_regret, neural_played,
     neural_leaves, neural_metrics) = algos["neural_cocoma"].run_algorithm()

    return TrialResult(
        random_reward=random_reward,
        random_regret=random_regret,
        bench_reward=bench_reward,
        cocoma_reward=cocoma_reward,
        cocoma_regret=cocoma_regret,
        cocoma_leaves=cocoma_leaves,
        cocoma_metrics=cocoma_metrics,
        neural_cocoma_reward=neural_reward,
        neural_cocoma_regret=neural_regret,
        neural_cocoma_leaves=neural_leaves,
        neural_cocoma_metrics=neural_metrics,
        bench_played_arms=bench_played,
        uniquely_best_arms=best_arms,
        cocoma_played_arms=cocoma_played,
        neural_cocoma_played_arms=neural_played,
    )


def _build_root_context(embedding_dimensions: int) -> Hyperrectangle:
    """Erstellt den initialen Kontext-Hyperrechteck für Baum-Algorithmen."""
    return Hyperrectangle(
        np.ones(embedding_dimensions * 2) * 2,
        np.zeros(embedding_dimensions * 2),
    )


def _create_algorithms(
    budget: int, config: SingleRouterConfig, root_context: Hyperrectangle
) -> dict:
    """Erstellt alle Algorithmen mit ihren jeweiligen Problem-Modellen."""
    embedding_dim = config.embedding_config.dimensions
    return {
        "random": StreamingRandom(
            get_problem_model(config), budget
        ),
        "benchmark": StreamingBenchmark(
            get_problem_model(config), budget
        ),
        "cocomama": StreamingCoCoMaMa(
            get_problem_model(config),
            config.v1, config.v2, config.N, config.rho,
            budget, root_context, config.theta,
        ),
        "neural_cocoma": StreamingNeuralCoCoMaMa(
            get_problem_model(config),
            config.v1, config.v2, config.N, config.rho,
            budget, root_context,
            embedding_dim * 2, hidden_dim=config.hidden_dim,
        ),
    }


# ---------------------------------------------------------------------------
# Result loading & aggregation
# ---------------------------------------------------------------------------

def load_and_process_streaming_results(config: SingleRouterConfig) -> list[BudgetResult]:
    """Lädt und verarbeitet Ergebnisse für alle Budgets. Gibt eine Liste von BudgetResult zurück."""
    results_dir = config.RESULTS_DIR
    budgets = config.budgets

    first_results = _load_results(results_dir, budgets[0])
    num_rounds = len(first_results[0].random_reward)
    print(f"Detected {num_rounds} rounds in the data")

    return [
        _aggregate_trials(budget, num_rounds, _load_results(results_dir, budget))
        for budget in budgets
    ]


def _load_results(results_dir: str, budget: int) -> list[TrialResult]:
    """Lädt die Pickle-Datei für ein bestimmtes Budget."""
    path = f"{results_dir}/parallel_results_budget_{budget}_streaming"
    with open(path, "rb") as f:
        return pickle.load(f)


def _aggregate_trials(
    budget: int, num_rounds: int, trial_results: list[TrialResult]
) -> BudgetResult:
    """Aggregiert eine Liste von TrialResults zu einem BudgetResult (Mittelwert + Std)."""
    num_runs = len(trial_results)

    random_reward       = np.zeros((num_runs, num_rounds))
    bench_reward        = np.zeros((num_runs, num_rounds))
    cocoma_reward       = np.zeros((num_runs, num_rounds))
    neural_cocoma_reward= np.zeros((num_runs, num_rounds))
    random_regret       = np.zeros((num_runs, num_rounds))
    cocoma_regret       = np.zeros((num_runs, num_rounds))
    neural_cocoma_regret= np.zeros((num_runs, num_rounds))
    cocoma_leaves       = np.zeros((num_runs, num_rounds))
    neural_cocoma_leaves= np.zeros((num_runs, num_rounds))

    for i, result in enumerate(trial_results):
        random_reward[i]        = _rolling_mean(result.random_reward)
        bench_reward[i]         = _rolling_mean(result.bench_reward)
        cocoma_reward[i]        = _rolling_mean(result.cocoma_reward)
        neural_cocoma_reward[i] = _rolling_mean(result.neural_cocoma_reward)

        random_regret[i]        = np.cumsum(result.random_regret)
        cocoma_regret[i]        = np.cumsum(result.cocoma_regret)
        neural_cocoma_regret[i] = np.cumsum(result.neural_cocoma_regret)

        cocoma_leaves[i]        = np.array(result.cocoma_leaves)
        neural_cocoma_leaves[i] = np.array(result.neural_cocoma_leaves)

    return BudgetResult(
        budget=budget,
        num_rounds=num_rounds,
        algorithms=[
            AlgorithmResult(
                **ALGORITHM_DISPLAY["cocomama"],
                avg_reward=np.mean(cocoma_reward, axis=0),
                std_reward=np.std(cocoma_reward, axis=0),
                avg_regret=np.mean(cocoma_regret, axis=0),
                std_regret=np.std(cocoma_regret, axis=0),
                avg_leaves=np.mean(cocoma_leaves, axis=0),
                std_leaves=np.std(cocoma_leaves, axis=0),
            ),
            AlgorithmResult(
                **ALGORITHM_DISPLAY["neural_cocoma"],
                avg_reward=np.mean(neural_cocoma_reward, axis=0),
                std_reward=np.std(neural_cocoma_reward, axis=0),
                avg_regret=np.mean(neural_cocoma_regret, axis=0),
                std_regret=np.std(neural_cocoma_regret, axis=0),
                avg_leaves=np.mean(neural_cocoma_leaves, axis=0),
                std_leaves=np.std(neural_cocoma_leaves, axis=0),
            ),
            AlgorithmResult(
                **ALGORITHM_DISPLAY["random"],
                avg_reward=np.mean(random_reward, axis=0),
                std_reward=np.std(random_reward, axis=0),
                avg_regret=np.mean(random_regret, axis=0),
                std_regret=np.std(random_regret, axis=0),
            ),
            AlgorithmResult(
                **ALGORITHM_DISPLAY["benchmark"],
                avg_reward=np.mean(bench_reward, axis=0),
                std_reward=np.std(bench_reward, axis=0),
                avg_regret=np.zeros(num_rounds),
                std_regret=np.zeros(num_rounds),
            ),
        ],
    )


def _rolling_mean(values: list[float]) -> np.ndarray:
    """Berechnet den rollenden Durchschnitt (expanding mean) einer Zeitreihe."""
    return pd.Series(values).expanding().mean().values
 

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

def run_trials_for_budget(budget: int, config: SingleRouterConfig) -> list[TrialResult]:
    """Führt alle parallelen Trials für ein Budget aus und speichert die Ergebnisse."""
    print(f"Doing budget {budget}...")
    print(f"Memory usage before parallel processing: {get_memory_usage():.2f} MB")

    num_threads = (
        int(multiprocessing.cpu_count() - 1)
        if config.num_threads_to_use == -1
        else config.num_threads_to_use
    )
    print(f"Running on {num_threads} threads")

    results = Parallel(n_jobs=num_threads)(
        delayed(run_trial)(i, budget, config) for i in tqdm(range(config.num_times_to_run))
    )

    print(f"Memory usage after parallel processing: {get_memory_usage():.2f} MB")

    results_file = f"{config.RESULTS_DIR}/parallel_results_budget_{budget}_streaming"
    with open(results_file, "wb") as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

    return results


def create_dataset_if_needed(config: SingleRouterConfig) -> None:
    """Erstellt das SPROUT-Dataset falls es nicht existiert."""
    from create_streaming_datasets import create_sprout_streaming_dataset

    os.makedirs("datasets", exist_ok=True)

    dataset_path = config.streaming_dataset_path
    if os.path.exists(dataset_path):
        print(f"SPROUT streaming dataset already exists: {dataset_path}")
        return

    if not config.create_dataset_if_missing:
        raise FileNotFoundError(
            "Streaming dataset does not exist and create_dataset_if_missing is false: "
            f"{dataset_path}"
        )

    print(f"Creating SPROUT streaming dataset: {dataset_path}")
    create_sprout_streaming_dataset(
        dataset_path,
        config.embedding_config.model_dump(),
        config.num_rounds,
        force_reload=False,
    )


def get_problem_model(config: SingleRouterConfig) -> StreamingProblemModel:
    """Erstellt und gibt das Streaming Problem Model zurück."""
    create_dataset_if_needed(config)

    embedding_dimensions = config.embedding_config.dimensions
    return StreamingProblemModel(
        config.streaming_dataset_path,
        embedding_dimensions,
        embedding_dimensions,
        config.num_rounds,
        max(config.budgets),
    )


def get_memory_usage() -> float:
    """Gibt den aktuellen Speicherverbrauch in MB zurück."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024



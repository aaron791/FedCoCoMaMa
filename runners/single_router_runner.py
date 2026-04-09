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
    plot_all_average_leaves,
    plot_all_average_reward,
    plot_all_cumulative_regret,
)
from streaming_dataset import StreamingProblemModel

sns.set(style="whitegrid")

# Algorithm colors for distinct visualization
algorithm_colors = {
    "CoCoMaMa": "green",
    "Neural-CoCoMaMa (ours)": "cyan",
    "Random": "purple",
    "Oracle": "black",
}

line_style_dict = {1: "--", 2: "-", 3: "-.", 4: ":"}


class SingleRouterRunner:
    """Executes the full single-router experiment pipeline."""

    def run(self, config: SingleRouterConfig) -> None:
        print(f"Initial memory usage: {get_memory_usage():.2f} MB")

        os.makedirs(config.RESULTS_DIR, exist_ok=True)

        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Number of rounds: {config.num_rounds}")
        print(f"Number of runs: {config.num_times_to_run}")
        print(f"Budgets: {config.budgets}")
        print(f"Results directory: {config.RESULTS_DIR}")
        print(f"Embedding model: {config.embedding_config.model_name}")
        print(f"Embedding dimensions: {config.embedding_config.dimensions}")
        print("=" * 60 + "\n")

        if not config.only_redo_plots:
            print("Ensuring datasets are created...")
            create_dataset_if_needed(config)

            for budget in config.budgets:
                run_simulation(budget, config)

        if config.plot or config.only_redo_plots:
            print("Loading and processing streaming results for plotting...")
            processed_data = load_and_process_streaming_results(config)

            num_std_to_show = config.num_std_to_show

            print("Creating streaming algorithm plots...")
            num_rounds = len(processed_data[config.budgets[0]]["cocomama_avg_reward"])
            results_dir = config.RESULTS_DIR
            plot_all_cumulative_regret(
                processed_data,
                config.budgets,
                num_rounds,
                num_std_to_show,
                algorithm_colors,
                results_dir,
            )
            plot_all_average_reward(
                processed_data,
                config.budgets,
                num_rounds,
                num_std_to_show,
                algorithm_colors,
                results_dir,
            )
            plot_all_average_leaves(
                processed_data,
                config.budgets,
                num_rounds,
                num_std_to_show,
                line_style_dict,
                algorithm_colors,
                results_dir,
            )

            plt.show()
            print("Streaming plots created successfully!")

        print("Experiment completed!")


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_one_try(num_run: int, budget: int, config: SingleRouterConfig) -> dict:
    """Run one trial with all streaming algorithms."""
    problem_model_random = get_problem_model(config)
    problem_model_benchmark = get_problem_model(config)
    problem_model_cocoma = get_problem_model(config)
    problem_model_neural_cocoma = get_problem_model(config)

    print(f"Problem model size: {problem_model_random.get_size()}")

    embedding_dimensions = config.embedding_config.dimensions
    root_context = Hyperrectangle(
        np.ones(embedding_dimensions * 2) * 2,
        np.zeros(embedding_dimensions * 2),
    )

    streaming_random = StreamingRandom(problem_model_random, budget)
    streaming_benchmark = StreamingBenchmark(problem_model_benchmark, budget)
    streaming_cocoma = StreamingCoCoMaMa(
        problem_model_cocoma,
        config.v1,
        config.v2,
        config.N,
        config.rho,
        budget,
        root_context,
        config.theta,
    )
    streaming_neural_cocoma = StreamingNeuralCoCoMaMa(
        problem_model_neural_cocoma,
        config.v1,
        config.v2,
        config.N,
        config.rho,
        budget,
        root_context,
        embedding_dimensions * 2,
        hidden_dim=config.hidden_dim,
    )

    print("Running Streaming Random...")
    (
        streaming_random_reward,
        streaming_random_regret,
        streaming_random_played_arms_arr,
    ) = streaming_random.run_algorithm()

    print("Running Streaming Benchmark...")
    (
        streaming_bench_reward,
        streaming_bench_regret,
        streaming_bench_played_arms_arr,
        streaming_uniquely_best_arms_arr,
    ) = streaming_benchmark.run_algorithm()

    print("Running Streaming CoCoMaMa...")
    (
        streaming_cocoma_reward,
        streaming_cocoma_regret,
        streaming_cocoma_played_arms_arr,
        streaming_cocoma_leaves_count_arr,
        streaming_cocoma_metrics,
    ) = streaming_cocoma.run_algorithm()

    print("Running Streaming Neural CoCoMaMa...")
    (
        streaming_neural_cocoma_reward,
        streaming_neural_cocoma_regret,
        streaming_neural_cocoma_played_arms_arr,
        streaming_neural_cocoma_leaves_count_arr,
        streaming_neural_cocoma_metrics,
    ) = streaming_neural_cocoma.run_algorithm()

    print(f"Run done: {num_run}")

    return {
        "streaming_bench_reward": streaming_bench_reward,
        "streaming_random_reward": streaming_random_reward,
        "streaming_random_regret": streaming_random_regret,
        "streaming_cocoma_reward": streaming_cocoma_reward,
        "streaming_cocoma_regret": streaming_cocoma_regret,
        "streaming_neural_cocoma_reward": streaming_neural_cocoma_reward,
        "streaming_neural_cocoma_regret": streaming_neural_cocoma_regret,
        "streaming_bench_played_arms_arr": streaming_bench_played_arms_arr,
        "streaming_uniquely_best_arms_arr": streaming_uniquely_best_arms_arr,
        "streaming_cocoma_played_arms_arr": streaming_cocoma_played_arms_arr,
        "streaming_neural_cocoma_played_arms_arr": streaming_neural_cocoma_played_arms_arr,
        "streaming_cocoma_leaves_count_arr": streaming_cocoma_leaves_count_arr,
        "streaming_neural_cocoma_leaves_count_arr": streaming_neural_cocoma_leaves_count_arr,
        "streaming_cocoma_metrics": streaming_cocoma_metrics,
        "streaming_neural_cocoma_metrics": streaming_neural_cocoma_metrics,
    }


def create_dataset_if_needed(config: SingleRouterConfig) -> None:
    """Create SPROUT dataset if it does not exist and config allows creation."""
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
    """Create and return the streaming problem model."""
    create_dataset_if_needed(config)

    embedding_dimensions = config.embedding_config.dimensions
    return StreamingProblemModel(
        config.streaming_dataset_path,
        embedding_dimensions,
        embedding_dimensions,
        config.num_rounds,
        max(config.budgets),
    )


def run_simulation(budget: int, config: SingleRouterConfig) -> list[dict]:
    """Run the simulation for a given budget."""
    print(f"Doing budget {budget}...")
    print(f"Memory usage before parallel processing: {get_memory_usage():.2f} MB")

    if config.num_threads_to_use == -1:
        num_threads = int(multiprocessing.cpu_count() - 1)
    else:
        num_threads = config.num_threads_to_use
    print(f"Running on {num_threads} threads")

    parallel_results = Parallel(n_jobs=num_threads)(
        delayed(run_one_try)(i, budget, config) for i in tqdm(range(config.num_times_to_run))
    )

    print(f"Memory usage after parallel processing: {get_memory_usage():.2f} MB")

    results_file = f"{config.RESULTS_DIR}/parallel_results_budget_{budget}_streaming"
    with open(results_file, "wb") as output:
        pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)

    return parallel_results


def load_and_process_streaming_results(config: SingleRouterConfig) -> dict:
    """Load and process streaming results for all budgets."""
    all_processed_data = {}
    results_dir = config.RESULTS_DIR
    budgets = config.budgets

    with open(f"{results_dir}/parallel_results_budget_{budgets[0]}_streaming", "rb") as input_file:
        sample_results = pickle.load(input_file)
    num_rounds = len(sample_results[0]["streaming_random_reward"])
    print(f"Detected {num_rounds} rounds in the data")

    for budget in budgets:
        with open(f"{results_dir}/parallel_results_budget_{budget}_streaming", "rb") as input_file:
            parallel_results = pickle.load(input_file)

        num_runs = len(parallel_results)

        streaming_random_reward_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_bench_reward_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_cocoma_reward_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_neural_cocoma_reward_runs_arr = np.zeros((num_runs, num_rounds))

        streaming_random_regret_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_cocoma_regret_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_neural_cocoma_regret_runs_arr = np.zeros((num_runs, num_rounds))

        streaming_cocoma_leaves_runs_arr = np.zeros((num_runs, num_rounds))
        streaming_neural_cocoma_leaves_runs_arr = np.zeros((num_runs, num_rounds))

        for i, entry in enumerate(parallel_results):
            streaming_random_reward_runs_arr[i] = (
                pd.Series(entry["streaming_random_reward"]).expanding().mean().values
            )
            streaming_bench_reward_runs_arr[i] = (
                pd.Series(entry["streaming_bench_reward"]).expanding().mean().values
            )
            streaming_cocoma_reward_runs_arr[i] = (
                pd.Series(entry["streaming_cocoma_reward"]).expanding().mean().values
            )
            streaming_neural_cocoma_reward_runs_arr[i] = (
                pd.Series(entry["streaming_neural_cocoma_reward"]).expanding().mean().values
            )

            streaming_random_regret_runs_arr[i] = np.cumsum(entry["streaming_random_regret"])
            streaming_cocoma_regret_runs_arr[i] = np.cumsum(entry["streaming_cocoma_regret"])
            streaming_neural_cocoma_regret_runs_arr[i] = np.cumsum(
                entry["streaming_neural_cocoma_regret"]
            )

            streaming_cocoma_leaves_runs_arr[i] = np.array(entry["streaming_cocoma_leaves_count_arr"])
            streaming_neural_cocoma_leaves_runs_arr[i] = np.array(
                entry["streaming_neural_cocoma_leaves_count_arr"]
            )

        all_processed_data[budget] = {
            "cocomama_avg_reward": np.mean(streaming_cocoma_reward_runs_arr, axis=0),
            "neural_cocoma_avg_reward": np.mean(streaming_neural_cocoma_reward_runs_arr, axis=0),
            "random_avg_reward": np.mean(streaming_random_reward_runs_arr, axis=0),
            "bench_avg_reward": np.mean(streaming_bench_reward_runs_arr, axis=0),
            "cocomama_std_reward": np.std(streaming_cocoma_reward_runs_arr, axis=0),
            "neural_cocoma_std_reward": np.std(streaming_neural_cocoma_reward_runs_arr, axis=0),
            "random_std_reward": np.std(streaming_random_reward_runs_arr, axis=0),
            "bench_std_reward": np.std(streaming_bench_reward_runs_arr, axis=0),
            "cocomama_avg_regret": np.mean(streaming_cocoma_regret_runs_arr, axis=0),
            "neural_cocoma_avg_regret": np.mean(streaming_neural_cocoma_regret_runs_arr, axis=0),
            "random_avg_regret": np.mean(streaming_random_regret_runs_arr, axis=0),
            "cocomama_std_regret": np.std(streaming_cocoma_regret_runs_arr, axis=0),
            "neural_cocoma_std_regret": np.std(streaming_neural_cocoma_regret_runs_arr, axis=0),
            "random_std_regret": np.std(streaming_random_regret_runs_arr, axis=0),
            "cocomama_avg_leaves": np.mean(streaming_cocoma_leaves_runs_arr, axis=0),
            "neural_cocoma_avg_leaves": np.mean(streaming_neural_cocoma_leaves_runs_arr, axis=0),
            "cocomama_std_leaves": np.std(streaming_cocoma_leaves_runs_arr, axis=0),
            "neural_cocoma_std_leaves": np.std(streaming_neural_cocoma_leaves_runs_arr, axis=0),
            "parallel_results": parallel_results,
        }
    return all_processed_data

"""
Main script for running streaming contextual bandit experiments.
This script uses streaming Arrow datasets with streaming algorithm implementations.
"""

import multiprocessing
import pickle
import psutil
import os
import argparse
import yaml

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

# Import streaming algorithms
from algorithms.streaming_base import StreamingRandom, StreamingBenchmark
from algorithms.streaming_cocoma import StreamingCoCoMaMa
from algorithms.streaming_neural_cocomama import StreamingNeuralCoCoMaMa

# Import problem models
from Hyperrectangle import Hyperrectangle
from streaming_dataset import StreamingProblemModel

# Import plotting functions
from plotting import (
    plot_selected_agents,
    plot_all_average_leaves,
    plot_all_cumulative_regret,
    plot_all_average_reward,
    plot_additional_metrics,
)

sns.set(style='whitegrid')

# Algorithm colors for distinct visualization
algorithm_colors = {
    'CoCoMaMa': 'green',
    'Neural-CoCoMaMa (ours)': 'cyan',
    'Random': 'purple',
    'Oracle': 'black',
}

line_style_dict = {1: '--', 2: '-', 3: '-.', 4: ':'}


def load_config_file(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config file: {e}")

def merge_configs(config_file, cli_args):
    """Merge config file with CLI arguments. CLI arguments take precedence."""
    merged_config = config_file.copy()
    for key, value in cli_args.items():
        if value is not None:
            merged_config[key] = value
    return merged_config

def save_config_to_file(config, filename=None):
    """Save configuration to a YAML file for reproducibility."""
    import datetime
    
    os.makedirs("configs", exist_ok=True)
    
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        problem_type = config.get("PROBLEM_MODEL_TYPE", "unknown")
        rounds = config.get("num_rounds", 100)
        filename = f"configs/config_{problem_type}_{rounds}rounds_{timestamp}.yaml"
    
    save_config = config.copy()
    
    if "create_dataset_if_missing" in save_config:
        del save_config["create_dataset_if_missing"]
    
    try:
        with open(filename, 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to: {filename}")
        return filename
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return None

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_one_try(num_run, budget, config):
    """Run one trial with all streaming algorithms."""
    problem_model_random = get_problem_model(config)
    problem_model_benchmark = get_problem_model(config)
    problem_model_cocoma = get_problem_model(config)
    problem_model_neural_cocoma = get_problem_model(config)
    
    print(f"Problem model size: {problem_model_random.get_size()}")
    
    root_context = Hyperrectangle(
        np.ones(config["embedding_config"]["dimensions"]*2)*2,
        np.zeros(config["embedding_config"]["dimensions"]*2)
    )
    
    streaming_random = StreamingRandom(problem_model_random, budget)
    streaming_benchmark = StreamingBenchmark(problem_model_benchmark, budget)
    streaming_cocoma = StreamingCoCoMaMa(
        problem_model_cocoma, config["v1"], config["v2"], config["N"], 
        config["rho"], budget, root_context, config.get("theta", 4.0)
    )
    streaming_neural_cocoma = StreamingNeuralCoCoMaMa(
        problem_model_neural_cocoma, config["v1"], config["v2"], config["N"], 
        config["rho"], budget, root_context, config["embedding_config"]["dimensions"]*2, 
        hidden_dim=config.get("hidden_dim", 64)
    )
    
    print("Running Streaming Random...")
    streaming_random_reward, streaming_random_regret, streaming_random_played_arms_arr = streaming_random.run_algorithm()
    
    print("Running Streaming Benchmark...")
    streaming_bench_reward, streaming_bench_regret, streaming_bench_played_arms_arr, streaming_uniquely_best_arms_arr = streaming_benchmark.run_algorithm()
    
    print("Running Streaming CoCoMaMa...")
    streaming_cocoma_reward, streaming_cocoma_regret, streaming_cocoma_played_arms_arr, streaming_cocoma_leaves_count_arr, streaming_cocoma_metrics = streaming_cocoma.run_algorithm()

    print("Running Streaming Neural CoCoMaMa...")
    streaming_neural_cocoma_reward, streaming_neural_cocoma_regret, streaming_neural_cocoma_played_arms_arr, streaming_neural_cocoma_leaves_count_arr, streaming_neural_cocoma_metrics = streaming_neural_cocoma.run_algorithm()
    
    print("Run done: " + str(num_run))
    
    return {
        'streaming_bench_reward': streaming_bench_reward,
        'streaming_random_reward': streaming_random_reward,
        'streaming_random_regret': streaming_random_regret,
        'streaming_cocoma_reward': streaming_cocoma_reward,
        'streaming_cocoma_regret': streaming_cocoma_regret,
        'streaming_neural_cocoma_reward': streaming_neural_cocoma_reward,
        'streaming_neural_cocoma_regret': streaming_neural_cocoma_regret,
        'streaming_bench_played_arms_arr': streaming_bench_played_arms_arr,
        'streaming_uniquely_best_arms_arr': streaming_uniquely_best_arms_arr,
        'streaming_cocoma_played_arms_arr': streaming_cocoma_played_arms_arr,
        'streaming_neural_cocoma_played_arms_arr': streaming_neural_cocoma_played_arms_arr,
        'streaming_cocoma_leaves_count_arr': streaming_cocoma_leaves_count_arr,
        'streaming_neural_cocoma_leaves_count_arr': streaming_neural_cocoma_leaves_count_arr,
        'streaming_cocoma_metrics': streaming_cocoma_metrics,
        'streaming_neural_cocoma_metrics': streaming_neural_cocoma_metrics,
    }


def create_dataset_if_needed(config):
    """Create datasets if they don't exist and are needed."""
    from create_streaming_datasets import create_sprout_streaming_dataset
    
    os.makedirs("datasets", exist_ok=True)
    
    dataset_path = config.get("streaming_dataset_path", "datasets/sprout_streaming_300.arrow")
    if not os.path.exists(dataset_path):
        print(f"Creating SPROUT streaming dataset: {dataset_path}")
        create_sprout_streaming_dataset(
            dataset_path,
            config["embedding_config"],
            config["num_rounds"],
            force_reload=False
        )
    else:
        print(f"SPROUT streaming dataset already exists: {dataset_path}")

def get_problem_model(config):
    """Create and return the streaming problem model."""
    create_dataset_if_needed(config)
    
    dataset_path = config.get("streaming_dataset_path", "datasets/sprout_streaming_300.arrow")
    return StreamingProblemModel(
        dataset_path,
        config["embedding_config"]["dimensions"],
        config["embedding_config"]["dimensions"],
        config["num_rounds"],
        max(config["budgets"])
    )


def run_simulation(budget, config):
    """Run the simulation for a given budget."""
    print(f"Doing budget {budget}...")
    print(f"Memory usage before parallel processing: {get_memory_usage():.2f} MB")
    
    if config["num_threads_to_use"] == -1:
        num_threads = int(multiprocessing.cpu_count()-1)
    else:
        num_threads = config["num_threads_to_use"]
    print(f"Running on {num_threads} threads")
    
    parallel_results = Parallel(n_jobs=num_threads)(
        delayed(run_one_try)(i, budget, config) 
        for i in tqdm(range(config["num_times_to_run"]))
    )
    
    print(f"Memory usage after parallel processing: {get_memory_usage():.2f} MB")
    
    results_file = f'{config["RESULTS_DIR"]}/parallel_results_budget_{budget}_streaming'
    with open(results_file, 'wb') as output:
        pickle.dump(parallel_results, output, pickle.HIGHEST_PROTOCOL)
    
    return parallel_results


def main():
    parser = argparse.ArgumentParser(
        description="Run streaming contextual bandit experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file
  python main_streaming.py --config_file config.yaml

  # Override config file with CLI and save result
  python main_streaming.py --config_file config.yaml --num_rounds 500 --save_config my_config.yaml
        """
    )
    
    parser.add_argument("--config_file", type=str, required=True,
                       help="Path to YAML configuration file (required).")
    
    parser.add_argument("--streaming_dataset_path", type=str, default=None,
                       help="Path to streaming Arrow dataset")
    
    parser.add_argument("--results_dir", type=str, default=None,
                       help="Results directory for output files")
    
    parser.add_argument("--num_rounds", type=int, default=None,
                       help="Number of rounds to run")
    
    parser.add_argument("--num_times_to_run", type=int, default=None,
                       help="Number of times to run each experiment")
    
    parser.add_argument("--budgets", nargs='+', type=int, default=None,
                       help="List of budgets to test")
    
    parser.add_argument("--v1", type=float, default=None,
                       help="V1 parameter for algorithms")
    
    parser.add_argument("--v2", type=float, default=None,
                       help="V2 parameter for algorithms")
    
    parser.add_argument("--rho", type=float, default=None,
                       help="Rho parameter for algorithms")
    
    parser.add_argument("--N", type=int, default=None,
                       help="N parameter for algorithms")
    
    parser.add_argument("--embedding_model", type=str, default=None,
                       help="Embedding model name")
    
    parser.add_argument("--embedding_dimensions", type=int, default=None,
                       help="Embedding dimensions")
    
    parser.add_argument("--num_threads_to_use", type=int, default=None,
                       help="Number of threads to use")
    
    parser.add_argument("--only_redo_plots", action="store_true",
                       help="Only redo plots using existing results")
    
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots after running experiments")
    
    parser.add_argument("--save_config", type=str, default=None,
                       help="Save the final configuration to a file")
    
    args = parser.parse_args()
    
    print(f"Loading configuration from: {args.config_file}")
    config_file = load_config_file(args.config_file)

    cli_args = {
        "streaming_dataset_path": args.streaming_dataset_path,
        "RESULTS_DIR": args.results_dir,
        "num_rounds": args.num_rounds,
        "num_times_to_run": args.num_times_to_run,
        "budgets": args.budgets,
        "v1": args.v1,
        "v2": args.v2,
        "rho": args.rho,
        "N": args.N,
        "num_threads_to_use": args.num_threads_to_use,
        "only_redo_plots": args.only_redo_plots if args.only_redo_plots else None,
    }

    if args.embedding_model or args.embedding_dimensions:
        embedding_config = config_file.get("embedding_config", {}).copy()
        if args.embedding_model:
            embedding_config["model_name"] = args.embedding_model
        if args.embedding_dimensions:
            embedding_config["dimensions"] = args.embedding_dimensions
            embedding_config["suffix"] = f"_{embedding_config['model_name']}_{args.embedding_dimensions}-dim"
        cli_args["embedding_config"] = embedding_config

    config = merge_configs(config_file, cli_args)

    if args.save_config:
        save_config_to_file(config, args.save_config)
    
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    
    os.makedirs(config["RESULTS_DIR"], exist_ok=True)
    
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Number of rounds: {config['num_rounds']}")
    print(f"Number of runs: {config['num_times_to_run']}")
    print(f"Budgets: {config['budgets']}")
    print(f"Results directory: {config['RESULTS_DIR']}")
    print(f"Embedding model: {config['embedding_config']['model_name']}")
    print(f"Embedding dimensions: {config['embedding_config']['dimensions']}")
    print("="*60 + "\n")
    
    if not config.get("only_redo_plots", False):
        print("Ensuring datasets are created...")
        create_dataset_if_needed(config)
        
        for budget in config["budgets"]:
            run_simulation(budget, config)
    
    if config.get("plot", False) or config.get("only_redo_plots", False):
        print("Loading and processing streaming results for plotting...")
        processed_data = load_and_process_streaming_results(config)
        
        num_std_to_show = config.get("num_std_to_show", 1)
        
        print("Creating streaming algorithm plots...")
        num_rounds = len(processed_data[config["budgets"][0]]['cocomama_avg_reward'])
        results_dir = config["RESULTS_DIR"]
        plot_all_cumulative_regret(processed_data, config["budgets"], num_rounds, num_std_to_show, algorithm_colors, results_dir)
        plot_all_average_reward(processed_data, config["budgets"], num_rounds, num_std_to_show, algorithm_colors, results_dir)
        plot_all_average_leaves(processed_data, config["budgets"], num_rounds, num_std_to_show, line_style_dict, algorithm_colors, results_dir)
        
        plt.show()
        print("Streaming plots created successfully!")
    
    print("Experiment completed!")


def load_and_process_streaming_results(config):
    """Load and process streaming results for all budgets."""
    all_processed_data = {}
    results_dir = config["RESULTS_DIR"]
    budgets = config["budgets"]
    
    with open(f'{results_dir}/parallel_results_budget_{budgets[0]}_streaming', 'rb') as input_file:
        sample_results = pickle.load(input_file)
    num_rounds = len(sample_results[0]['streaming_random_reward'])
    print(f"Detected {num_rounds} rounds in the data")
    
    for budget in budgets:
        with open(f'{results_dir}/parallel_results_budget_{budget}_streaming', 'rb') as input_file:
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
            streaming_random_reward_runs_arr[i] = pd.Series(entry['streaming_random_reward']).expanding().mean().values
            streaming_bench_reward_runs_arr[i] = pd.Series(entry['streaming_bench_reward']).expanding().mean().values
            streaming_cocoma_reward_runs_arr[i] = pd.Series(entry['streaming_cocoma_reward']).expanding().mean().values
            streaming_neural_cocoma_reward_runs_arr[i] = pd.Series(entry['streaming_neural_cocoma_reward']).expanding().mean().values
            
            streaming_random_regret_runs_arr[i] = np.cumsum(entry['streaming_random_regret'])
            streaming_cocoma_regret_runs_arr[i] = np.cumsum(entry['streaming_cocoma_regret'])
            streaming_neural_cocoma_regret_runs_arr[i] = np.cumsum(entry['streaming_neural_cocoma_regret'])
            
            streaming_cocoma_leaves_runs_arr[i] = np.array(entry['streaming_cocoma_leaves_count_arr'])
            streaming_neural_cocoma_leaves_runs_arr[i] = np.array(entry['streaming_neural_cocoma_leaves_count_arr'])
        
        all_processed_data[budget] = {
            'cocomama_avg_reward': np.mean(streaming_cocoma_reward_runs_arr, axis=0),
            'neural_cocoma_avg_reward': np.mean(streaming_neural_cocoma_reward_runs_arr, axis=0),
            'random_avg_reward': np.mean(streaming_random_reward_runs_arr, axis=0),
            'bench_avg_reward': np.mean(streaming_bench_reward_runs_arr, axis=0),
            
            'cocomama_std_reward': np.std(streaming_cocoma_reward_runs_arr, axis=0),
            'neural_cocoma_std_reward': np.std(streaming_neural_cocoma_reward_runs_arr, axis=0),
            'random_std_reward': np.std(streaming_random_reward_runs_arr, axis=0),
            'bench_std_reward': np.std(streaming_bench_reward_runs_arr, axis=0),
            
            'cocomama_avg_regret': np.mean(streaming_cocoma_regret_runs_arr, axis=0),
            'neural_cocoma_avg_regret': np.mean(streaming_neural_cocoma_regret_runs_arr, axis=0),
            'random_avg_regret': np.mean(streaming_random_regret_runs_arr, axis=0),
            
            'cocomama_std_regret': np.std(streaming_cocoma_regret_runs_arr, axis=0),
            'neural_cocoma_std_regret': np.std(streaming_neural_cocoma_regret_runs_arr, axis=0),
            'random_std_regret': np.std(streaming_random_regret_runs_arr, axis=0),
            
            'cocomama_avg_leaves': np.mean(streaming_cocoma_leaves_runs_arr, axis=0),
            'neural_cocoma_avg_leaves': np.mean(streaming_neural_cocoma_leaves_runs_arr, axis=0),
            'cocomama_std_leaves': np.std(streaming_cocoma_leaves_runs_arr, axis=0),
            'neural_cocoma_std_leaves': np.std(streaming_neural_cocoma_leaves_runs_arr, axis=0),
            'parallel_results': parallel_results
        }
    return all_processed_data


if __name__ == '__main__':
    main()

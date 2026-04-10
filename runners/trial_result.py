from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrialResult:
    """Typisierte Ergebnisse eines einzelnen Experiment-Trials.

    Ersetzt das unstrukturierte String-Key-Dict aus run_one_try().
    Wird als Pickle auf Disk gespeichert und in load_and_process_streaming_results() geladen.
    """

    random_reward: list[float]
    random_regret: list[float]
    bench_reward: list[float]
    cocoma_reward: list[float]
    cocoma_regret: list[float]
    cocoma_leaves: list[int]
    cocoma_metrics: dict
    neural_cocoma_reward: list[float]
    neural_cocoma_regret: list[float]
    neural_cocoma_leaves: list[int]
    neural_cocoma_metrics: dict
    bench_played_arms: list
    uniquely_best_arms: list
    cocoma_played_arms: list
    neural_cocoma_played_arms: list

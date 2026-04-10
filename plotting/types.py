from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AlgorithmResult:
    """Ergebnisdaten und Plot-Metadaten für einen Algorithmus."""

    label: str              # Anzeigename in der Legende, z.B. "CoCoMaMa (ours)"
    color: str              # Hex-Farbe oder Matplotlib-Farbname, z.B. "green"
    avg_reward: np.ndarray  # Rollender Mittelwert über alle Runs, Shape (num_rounds,)
    std_reward: np.ndarray  # Standardabweichung über alle Runs, Shape (num_rounds,)
    avg_regret: np.ndarray  # Kumulative Regret-Kurve, Shape (num_rounds,)
    std_regret: np.ndarray  # Standardabweichung Regret, Shape (num_rounds,)
    avg_leaves: np.ndarray | None = None  # Nur für Baum-basierte Algorithmen
    std_leaves: np.ndarray | None = None


@dataclass
class BudgetResult:
    """Aggregierte Ergebnisse (Mittelwert/Std über alle Trials) für einen Budget-Wert."""

    budget: int
    num_rounds: int
    algorithms: list[AlgorithmResult]

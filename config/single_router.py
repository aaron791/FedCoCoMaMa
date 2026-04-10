"""Strict single-router configuration schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class EmbeddingConfig(BaseModel):
    """Embedding model settings."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    dimensions: int
    suffix: str

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("embedding_config.dimensions must be > 0")
        return value


class SingleRouterConfig(BaseModel):
    """Strict single-router experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    config_type: Literal["single_router"]
    PROBLEM_MODEL_TYPE: Literal["streaming_sprout"]
    only_redo_plots: bool
    plot: bool

    RESULTS_DIR: str
    embedding_config: EmbeddingConfig

    num_times_to_run: int
    num_rounds: int
    num_std_to_show: int
    budgets: list[int]

    v1: float
    v2: float
    rho: float
    N: int
    theta: float
    hidden_dim: int

    num_threads_to_use: int

    streaming_dataset_path: str
    create_dataset_if_missing: bool

    @field_validator("num_times_to_run", "num_rounds", "num_std_to_show", "N", "hidden_dim")
    @classmethod
    def validate_positive_int_fields(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be > 0")
        return value

    @field_validator("num_threads_to_use")
    @classmethod
    def validate_num_threads_to_use(cls, value: int) -> int:
        if value == -1 or value > 0:
            return value
        raise ValueError("num_threads_to_use must be -1 or > 0")

    @field_validator("budgets")
    @classmethod
    def validate_budgets(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("budgets must not be empty")
        if any(budget <= 0 for budget in value):
            raise ValueError("all budgets must be > 0")
        return value

    def print_summary(self) -> None:
        """Gibt eine übersichtliche Zusammenfassung der Konfiguration aus."""
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Number of rounds:     {self.num_rounds}")
        print(f"Number of runs:       {self.num_times_to_run}")
        print(f"Budgets:              {self.budgets}")
        print(f"Results directory:    {self.RESULTS_DIR}")
        print(f"Embedding model:      {self.embedding_config.model_name}")
        print(f"Embedding dimensions: {self.embedding_config.dimensions}")
        print("=" * 60 + "\n")

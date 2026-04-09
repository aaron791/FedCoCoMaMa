"""Strict single-router configuration schema and loader."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator


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


def load_single_router_config(config_file: str | Path) -> SingleRouterConfig:
    """Load and strictly validate a single-router YAML configuration."""

    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:

        raise ValueError(f"Error parsing YAML config file: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a YAML mapping/object at the root.")

    try:
        return SingleRouterConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid config file '{config_path}':\n{exc}") from exc

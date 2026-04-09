"""Strict multi-router configuration schema (MVP placeholder)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


class MultiRouterOutputConfig(BaseModel):
    """Output configuration for multi-router runs."""

    model_config = ConfigDict(extra="forbid")

    results_dir: str
    plot: bool
    num_std_to_show: int

    @field_validator("num_std_to_show")
    @classmethod
    def validate_num_std_to_show(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("output.num_std_to_show must be > 0")
        return value


class MultiRouterNodeConfig(BaseModel):
    """Minimal router definition for the multi-router MVP."""

    model_config = ConfigDict(extra="forbid")

    id: str
    dataset_path: str
    enabled_algorithms: list[str]

    @field_validator("enabled_algorithms")
    @classmethod
    def validate_enabled_algorithms(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("routers[].enabled_algorithms must not be empty")
        return value


class MultiRouterConfig(BaseModel):
    """Strict multi-router config used for dispatch and placeholder execution."""

    model_config = ConfigDict(extra="forbid")

    config_type: Literal["multi_router"]
    experiment_name: str
    num_rounds: int
    num_runs: int
    routers: list[MultiRouterNodeConfig]
    output: MultiRouterOutputConfig

    @field_validator("num_rounds", "num_runs")
    @classmethod
    def validate_positive_int_fields(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be > 0")
        return value

    @field_validator("routers")
    @classmethod
    def validate_routers(cls, value: list[MultiRouterNodeConfig]) -> list[MultiRouterNodeConfig]:
        if not value:
            raise ValueError("routers must not be empty")
        return value

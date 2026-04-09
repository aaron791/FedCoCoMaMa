"""App-level config loader with strict discriminated-union validation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import Field, TypeAdapter, ValidationError

from .multi_router import MultiRouterConfig
from .single_router import SingleRouterConfig

AppConfig = Annotated[
    SingleRouterConfig | MultiRouterConfig,
    Field(discriminator="config_type"),
]

_app_config_adapter = TypeAdapter(AppConfig)


def load_app_config(config_file: str | Path) -> AppConfig:
    """Load YAML config and validate it via discriminated union on `config_type`."""

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
        return _app_config_adapter.validate_python(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid config file '{config_path}':\n{exc}") from exc

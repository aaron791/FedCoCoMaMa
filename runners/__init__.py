"""Runner dispatch and public runner interfaces."""

from __future__ import annotations

from typing import Protocol

from config.loader import AppConfig
from config.multi_router import MultiRouterConfig
from config.single_router import SingleRouterConfig


class Runner(Protocol):
    """Shared protocol for all runners."""

    def run(self, config: AppConfig) -> None:
        """Execute the configured experiment."""


def selectRunner(config: AppConfig) -> Runner:
    """Return the runner implementation for the given config type."""

    if isinstance(config, SingleRouterConfig):
        from .single_router_runner import SingleRouterRunner
        return SingleRouterRunner()

    if isinstance(config, MultiRouterConfig):
        from .multi_router_runner import MultiRouterRunner
        return MultiRouterRunner()
        
    raise ValueError(f"No runner available for config type: {type(config).__name__}")


__all__ = ["Runner", "selectRunner"]

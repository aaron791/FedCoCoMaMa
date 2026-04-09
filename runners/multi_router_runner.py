"""Placeholder runner for multi-router configuration."""

from __future__ import annotations

from config.multi_router import MultiRouterConfig


class MultiRouterRunner:
    """Placeholder implementation for multi-router execution."""

    def run(self, config: MultiRouterConfig) -> None:
        print("\n" + "=" * 60)
        print("MULTI ROUTER CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Experiment: {config.experiment_name}")
        print(f"Rounds: {config.num_rounds}")
        print(f"Runs: {config.num_runs}")
        print(f"Routers: {len(config.routers)}")
        print(f"Output directory: {config.output.results_dir}")
        print("=" * 60 + "\n")

        print(
            "Multi-router runner is recognized and validated, "
            "but execution is not implemented yet (placeholder only)."
        )

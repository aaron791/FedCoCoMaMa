"""CLI entry point for streaming experiments."""

from __future__ import annotations

import argparse

from config.loader import load_app_config
from runners import selectRunner


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run streaming contextual bandit experiments")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to YAML configuration file (required).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_app_config(args.config_file)
    runner = selectRunner(config)
    runner.run(config)


if __name__ == "__main__":
    main()

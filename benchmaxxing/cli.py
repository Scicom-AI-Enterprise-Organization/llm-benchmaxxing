#!/usr/bin/env python3
"""
CLI entry point for benchmaxxing.

Usage:
    benchmaxxing <config.yaml>
    benchmaxxing examples/run_single.yaml
"""

import sys

from .runner import run


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: benchmaxxing <config.yaml>")
        print("Example: benchmaxxing examples/run_single.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    run(config_path)


if __name__ == "__main__":
    main()

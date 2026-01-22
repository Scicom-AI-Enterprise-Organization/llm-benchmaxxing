#!/usr/bin/env python3
"""
LLM Benchmark Runner

Dispatches benchmark runs to the appropriate engine (vllm, tensorrt-llm, sglang, etc.)
"""

import importlib
import os
import sys

import yaml


SUPPORTED_ENGINES = ["vllm"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <config.yaml>")
        print("Example: python run.py config/my-benchmark.yaml")
        sys.exit(1)

    config_path = sys.argv[1]

    # Resolve config path
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    runs = config.get("runs", [])
    if not runs:
        print("Error: No runs defined in config")
        sys.exit(1)

    # Get engine from first run (default: vllm)
    engine = runs[0].get("engine", "vllm")

    if engine not in SUPPORTED_ENGINES:
        print(f"Error: Unsupported engine '{engine}'. Supported: {SUPPORTED_ENGINES}")
        sys.exit(1)

    print()
    print("=" * 64)
    print(f"ENGINE: {engine}")
    print("=" * 64)

    # Import and run the engine module
    engine_module = importlib.import_module(engine)
    engine_module.run(config)


if __name__ == "__main__":
    main()

"""Benchmaq - LLM performance benchmarking toolkit.

Usage:
    import benchmaq.vllm.bench as bench
    result = bench.from_yaml("config.yaml")

    import benchmaq.runpod.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq vllm bench config.yaml
    benchmaq runpod bench config.yaml
"""

__version__ = "0.5.0"

from . import vllm
from . import runpod

__all__ = ["__version__", "vllm", "runpod"]

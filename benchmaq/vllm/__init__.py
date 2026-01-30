"""vLLM benchmarking module.

Usage:
    import benchmaq.vllm.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq vllm bench config.yaml
"""

from . import bench
from .core import VLLMServer, run_benchmark

__all__ = ["bench", "VLLMServer", "run_benchmark"]

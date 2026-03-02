"""vLLM benchmarking module.

Usage:
    import benchmaq.vllm.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq vllm bench config.yaml
"""

from . import bench
from . import stt
from .core import VLLMServer, run_benchmark

__all__ = ["bench", "stt", "VLLMServer", "run_benchmark"]

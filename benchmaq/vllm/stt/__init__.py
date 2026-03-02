"""STT (Speech-to-Text) benchmarking module.

Uses vLLM HTTP server to serve STT models and a custom benchmark client
to measure RTF (Real-Time Factor), throughput, and per-request timing.

Usage:
    import benchmaq.vllm.stt.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq vllm stt bench config.yaml
"""

from . import bench
from .core import run_benchmark

__all__ = ["bench", "run_benchmark"]

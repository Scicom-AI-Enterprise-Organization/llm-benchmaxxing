"""SGLang benchmarking module.

Usage:
    import benchmaq.sglang.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq sglang bench config.yaml
"""

from . import bench
from .core import SGLangServer, run_benchmark

__all__ = ["bench", "SGLangServer", "run_benchmark"]

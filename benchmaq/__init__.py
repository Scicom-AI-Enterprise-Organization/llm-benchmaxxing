"""Benchmaq - LLM performance benchmarking toolkit.

Usage:
    import benchmaq.vllm.bench as bench
    result = bench.from_yaml("config.yaml")

    import benchmaq.sglang.bench as bench
    result = bench.from_yaml("config.yaml")

    import benchmaq.runpod.bench as bench
    result = bench.from_yaml("config.yaml")

    import benchmaq.skypilot.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq bench config.yaml
    benchmaq vllm bench config.yaml
    benchmaq sglang bench config.yaml
    benchmaq runpod bench config.yaml
    benchmaq sky bench -c config.yaml
"""

__version__ = "0.6.0"

from . import vllm
from . import sglang
from . import runpod
from . import skypilot

__all__ = ["__version__", "vllm", "sglang", "runpod", "skypilot"]

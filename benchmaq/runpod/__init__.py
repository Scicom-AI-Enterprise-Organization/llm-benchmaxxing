"""RunPod benchmarking module.

Usage:
    import benchmaq.runpod.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq runpod bench config.yaml
"""

from . import bench

__all__ = ["bench"]

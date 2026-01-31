"""SkyPilot integration for benchmaq.

Provides end-to-end benchmark execution on SkyPilot-managed cloud infrastructure.

Usage:
    CLI:
        benchmaq sky bench --config config.yaml
    
    Python API:
        import benchmaq.skypilot.bench as bench
        result = bench.from_yaml("config.yaml")

Prerequisites:
    Users must authenticate with SkyPilot before using this module:
    - Run `sky auth` or
    - Set SKYPILOT_API_SERVER_URL and SKYPILOT_API_KEY environment variables
"""

from . import bench

__all__ = ["bench"]

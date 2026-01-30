"""Configuration utilities for benchmaq."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import yaml


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, 
            "status": self.status, 
            "metrics": self.metrics, 
            "duration": self.duration, 
            "error": self.error
        }


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        return yaml.safe_load(f)

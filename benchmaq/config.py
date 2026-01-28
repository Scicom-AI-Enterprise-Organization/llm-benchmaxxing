"""Configuration utilities for benchmaq."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import os
import yaml


@dataclass
class BenchConfig:
    """Main configuration for benchmaq. Picklable for multiprocessing."""
    name: str = "benchmark"
    engine: str = "vllm"
    model_path: str = ""
    hf_token: Optional[str] = None
    port: int = 8000
    tensor_parallel: int = 1
    data_parallel: int = 1
    pipeline_parallel: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    max_num_seqs: Optional[int] = None
    dtype: Optional[str] = None
    disable_log_requests: bool = False
    enable_expert_parallel: bool = False
    output_dir: str = "./benchmark_results"
    context_sizes: List[int] = field(default_factory=lambda: [1024])
    concurrency: List[int] = field(default_factory=lambda: [50])
    num_prompts: List[int] = field(default_factory=lambda: [100])
    output_len: List[int] = field(default_factory=lambda: [128])
    save_results: bool = False

    def to_run_config(self) -> dict:
        return {
            "name": self.name,
            "engine": self.engine,
            "model": {"hf_token": self.hf_token},
            "vllm_serve": {
                "model_path": self.model_path,
                "port": self.port,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "max_num_seqs": self.max_num_seqs,
                "dtype": self.dtype,
                "disable_log_requests": self.disable_log_requests,
                "enable_expert_parallel": self.enable_expert_parallel,
                "parallelism_pairs": [{
                    "tensor_parallel": self.tensor_parallel,
                    "data_parallel": self.data_parallel,
                    "pipeline_parallel": self.pipeline_parallel,
                }],
            },
            "benchmark": {
                "output_dir": self.output_dir,
                "context_size": self.context_sizes,
                "concurrency": self.concurrency,
                "num_prompts": self.num_prompts,
                "output_len": self.output_len,
                "save_results": self.save_results,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    name: str
    status: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "metrics": self.metrics, "duration": self.duration, "error": self.error}


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_config(base: dict, overrides: dict) -> dict:
    """Deep merge two config dicts, with overrides taking precedence."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def kwargs_to_run_config(**kwargs) -> dict:
    """Convert flat kwargs to a run config dict."""
    return {
        "runs": [{
            "name": kwargs.get("name", "benchmark"),
            "engine": kwargs.get("engine", "vllm"),
            "model": {"hf_token": kwargs.get("hf_token")},
            "vllm_serve": {
                "model_path": kwargs.get("model_path", ""),
                "port": kwargs.get("port", 8000),
                "gpu_memory_utilization": kwargs.get("gpu_memory_utilization", 0.9),
                "max_model_len": kwargs.get("max_model_len"),
                "max_num_seqs": kwargs.get("max_num_seqs"),
                "dtype": kwargs.get("dtype"),
                "disable_log_requests": kwargs.get("disable_log_requests", False),
                "enable_expert_parallel": kwargs.get("enable_expert_parallel", False),
                "parallelism_pairs": kwargs.get("parallelism_pairs") or [{
                    "tensor_parallel": kwargs.get("tensor_parallel", 1),
                    "data_parallel": kwargs.get("data_parallel", 1),
                    "pipeline_parallel": kwargs.get("pipeline_parallel", 1),
                }],
            },
            "benchmark": {
                "output_dir": kwargs.get("output_dir", "./benchmark_results"),
                "context_size": kwargs.get("context_sizes", [1024]),
                "concurrency": kwargs.get("concurrency", [50]),
                "num_prompts": kwargs.get("num_prompts", [100]),
                "output_len": kwargs.get("output_len", [128]),
                "save_results": kwargs.get("save_results", False),
            },
        }]
    }


def kwargs_to_remote_config(**kwargs) -> dict:
    """Convert flat kwargs to a remote config dict."""
    return {
        "host": kwargs.get("host", ""),
        "port": kwargs.get("port", 22),
        "username": kwargs.get("username", "root"),
        "password": kwargs.get("password"),
        "key_filename": kwargs.get("key_filename"),
        "uv": {
            "path": kwargs.get("uv_path", "~/.benchmark-venv"),
            "python_version": kwargs.get("python_version", "3.11"),
        },
        "dependencies": kwargs.get("dependencies", ["pyyaml", "requests", "vllm==0.11.0", "huggingface_hub"]),
    }


def kwargs_to_runpod_config(**kwargs) -> dict:
    """Convert flat kwargs to a runpod config dict."""
    return {
        "runpod": {
            "runpod_api_key": kwargs.get("api_key"),
            "ssh_private_key": kwargs.get("ssh_private_key"),
            "pod": {
                "name": kwargs.get("name"),
                "gpu_type": kwargs.get("gpu_type", ""),
                "gpu_count": kwargs.get("gpu_count", 1),
                "instance_type": kwargs.get("instance_type", "spot"),
                "bid_per_gpu": kwargs.get("bid_per_gpu"),
                "secure_cloud": kwargs.get("secure_cloud", True),
            },
            "container": {
                "image": kwargs.get("image", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"),
                "disk_size": kwargs.get("container_disk_size", 20),
            },
            "storage": {
                "volume_size": kwargs.get("disk_size", 100),
                "mount_path": kwargs.get("mount_path", "/workspace"),
            },
            "ports": {
                "http": kwargs.get("ports_http", [8888, 8000]),
                "tcp": kwargs.get("ports_tcp", [22]),
            },
            "env": kwargs.get("env", {}),
        },
        "remote": kwargs_to_remote_config(**kwargs),
        "runs": kwargs_to_run_config(**kwargs).get("runs", []),
    }

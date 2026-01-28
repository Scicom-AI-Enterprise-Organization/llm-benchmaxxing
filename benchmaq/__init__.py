"""Benchmaq - LLM performance benchmarking toolkit."""

__version__ = "0.4.0"

from typing import Optional, List, Dict, Any

from . import vllm
from . import runpod
from .config import (
    BenchConfig,
    BenchmarkResult,
    load_config,
    merge_config,
    kwargs_to_run_config,
    kwargs_to_remote_config,
)


def bench(
    config_path: Optional[str] = None,
    *,
    host: Optional[str] = None,
    port: int = 22,
    username: str = "root",
    password: Optional[str] = None,
    key_filename: Optional[str] = None,
    uv_path: str = "~/.benchmark-venv",
    python_version: str = "3.11",
    dependencies: Optional[List[str]] = None,
    name: str = "benchmark",
    model_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    server_port: int = 8000,
    tensor_parallel: int = 1,
    data_parallel: int = 1,
    pipeline_parallel: int = 1,
    parallelism_pairs: Optional[List[Dict[str, int]]] = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    dtype: Optional[str] = None,
    disable_log_requests: bool = False,
    enable_expert_parallel: bool = False,
    context_sizes: Optional[List[int]] = None,
    concurrency: Optional[List[int]] = None,
    num_prompts: Optional[List[int]] = None,
    output_len: Optional[List[int]] = None,
    output_dir: str = "./benchmark_results",
    save_results: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run benchmarks locally or remotely.

    If `host` is provided, runs on remote GPU server via SSH. Otherwise runs locally.
    """
    from .runner import run as _run, run_remote as _run_remote

    config = {}
    if config_path:
        config = load_config(config_path)

    is_remote = host is not None or config.get("remote", {}).get("host")
    kwargs_config = {}

    if host:
        kwargs_config["remote"] = kwargs_to_remote_config(
            host=host,
            port=port,
            username=username,
            password=password,
            key_filename=key_filename,
            uv_path=uv_path,
            python_version=python_version,
            dependencies=dependencies or ["pyyaml", "requests", "vllm==0.11.0", "huggingface_hub"],
        )

    if model_path:
        kwargs_config.update(kwargs_to_run_config(
            name=name,
            model_path=model_path,
            hf_token=hf_token,
            port=server_port,
            tensor_parallel=tensor_parallel,
            data_parallel=data_parallel,
            pipeline_parallel=pipeline_parallel,
            parallelism_pairs=parallelism_pairs,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            dtype=dtype,
            disable_log_requests=disable_log_requests,
            enable_expert_parallel=enable_expert_parallel,
            context_sizes=context_sizes or [1024],
            concurrency=concurrency or [50],
            num_prompts=num_prompts or [100],
            output_len=output_len or [128],
            output_dir=output_dir,
            save_results=save_results,
        ))

    config = merge_config(config, kwargs_config)

    if not config.get("runs"):
        raise ValueError("No benchmark runs defined. Provide model_path or config file with 'runs' section.")

    try:
        if is_remote:
            remote_cfg = config.get("remote", {})
            if not remote_cfg.get("host"):
                raise ValueError("Remote host is required for remote benchmark.")
            _run_remote(config, remote_cfg)
            return {"status": "success", "mode": "remote", "host": remote_cfg.get("host")}
        else:
            return vllm.run(config)
    except Exception as e:
        return {"status": "error", "error": str(e), "mode": "remote" if is_remote else "local"}


__all__ = ["__version__", "bench", "vllm", "runpod", "BenchConfig", "BenchmarkResult", "load_config"]

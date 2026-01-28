"""vLLM benchmarking module."""

import os
import time
from typing import Optional, List, Dict, Any

from .core import VLLMServer, run_benchmark


def run(config: dict) -> Dict[str, Any]:
    """Run vLLM benchmarks based on config dict."""
    results = []

    for run_cfg in config.get("runs", []):
        name = run_cfg.get("name", "")
        model_cfg = run_cfg.get("model", {})
        vllm_serve_cfg = run_cfg.get("vllm_serve", run_cfg)
        benchmark_cfg = run_cfg.get("benchmark", run_cfg)

        hf_token = model_cfg.get("hf_token") or config.get("hf_token")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        model_path = vllm_serve_cfg.get("model_path", "")
        port = vllm_serve_cfg.get("port", 8000)
        gpu_memory_utilization = vllm_serve_cfg.get("gpu_memory_utilization", 0.9)
        max_model_len = vllm_serve_cfg.get("max_model_len")
        max_num_seqs = vllm_serve_cfg.get("max_num_seqs")
        dtype = vllm_serve_cfg.get("dtype")
        disable_log_requests = vllm_serve_cfg.get("disable_log_requests", False)
        enable_expert_parallel = vllm_serve_cfg.get("enable_expert_parallel", False)
        parallelism_pairs = vllm_serve_cfg.get("parallelism_pairs", [])

        output_dir = benchmark_cfg.get("output_dir", "./benchmark_results")
        context_sizes = benchmark_cfg.get("context_size", [])
        concurrencies = benchmark_cfg.get("concurrency", [])
        num_prompts_list = benchmark_cfg.get("num_prompts", [])
        output_lens = benchmark_cfg.get("output_len", [])
        save_results = benchmark_cfg.get("save_results", False)

        if not name or not model_path:
            continue

        if save_results:
            os.makedirs(output_dir, exist_ok=True)

        for pair in parallelism_pairs:
            tp = pair.get("tensor_parallel", 1)
            dp = pair.get("data_parallel", 1)
            pp = pair.get("pipeline_parallel", 1)

            print()
            print("=" * 64)
            print(f"RUN: {name} | TP={tp} DP={dp} PP={pp}")
            print("=" * 64)

            with VLLMServer(
                model_path, port, tp, dp, pp,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                dtype=dtype,
                disable_log_requests=disable_log_requests,
                enable_expert_parallel=enable_expert_parallel
            ) as server:
                for ctx in context_sizes:
                    for concurrency in concurrencies:
                        for num_prompts in num_prompts_list:
                            for output_len in output_lens:
                                result_name = f"{name}_TP{tp}_DP{dp}_CTX{ctx}_C{concurrency}_P{num_prompts}_O{output_len}"
                                run_benchmark(
                                    model_path, port, output_dir, result_name,
                                    ctx, output_len, num_prompts, concurrency,
                                    save_results=save_results
                                )
                                results.append({
                                    "name": result_name, "tp": tp, "dp": dp, "pp": pp,
                                    "ctx": ctx, "concurrency": concurrency,
                                    "num_prompts": num_prompts, "output_len": output_len,
                                })

            time.sleep(5)

    return {"status": "success", "results": results}


def bench(
    config_path: Optional[str] = None,
    *,
    name: str = "benchmark",
    model_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    port: int = 8000,
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
    """Run vLLM benchmarks with kwargs or config file."""
    from ..config import load_config, merge_config, kwargs_to_run_config

    config = {}
    if config_path:
        config = load_config(config_path)

    if model_path:
        kwargs_config = kwargs_to_run_config(
            name=name, model_path=model_path, hf_token=hf_token, port=port,
            tensor_parallel=tensor_parallel, data_parallel=data_parallel, pipeline_parallel=pipeline_parallel,
            parallelism_pairs=parallelism_pairs, gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, max_num_seqs=max_num_seqs, dtype=dtype,
            disable_log_requests=disable_log_requests, enable_expert_parallel=enable_expert_parallel,
            context_sizes=context_sizes or [1024], concurrency=concurrency or [50],
            num_prompts=num_prompts or [100], output_len=output_len or [128],
            output_dir=output_dir, save_results=save_results,
        )
        config = merge_config(config, kwargs_config)

    if not config.get("runs"):
        raise ValueError("No benchmark runs defined. Provide model_path or config file with 'runs' section.")

    try:
        return run(config)
    except Exception as e:
        return {"status": "error", "error": str(e)}


__all__ = ["VLLMServer", "run_benchmark", "run", "bench"]

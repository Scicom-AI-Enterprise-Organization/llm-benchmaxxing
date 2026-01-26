import os
import time

from .core import VLLMServer, run_benchmark


def run(config: dict):
    """Run vLLM benchmarks based on config."""
    for run_cfg in config.get("runs", []):
        name = run_cfg.get("name", "")
        model_cfg = run_cfg.get("model", {})

        vllm_serve_cfg = run_cfg.get("vllm_serve", run_cfg)
        benchmark_cfg = run_cfg.get("benchmark", run_cfg)

        # Set HF_TOKEN for gated models (config takes priority over env)
        hf_token = model_cfg.get("hf_token") or config.get("hf_token")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Server options
        model_path = vllm_serve_cfg.get("model_path", "")
        port = vllm_serve_cfg.get("port", 8000)
        gpu_memory_utilization = vllm_serve_cfg.get("gpu_memory_utilization", 0.9)
        max_model_len = vllm_serve_cfg.get("max_model_len")
        max_num_seqs = vllm_serve_cfg.get("max_num_seqs")
        dtype = vllm_serve_cfg.get("dtype")
        disable_log_requests = vllm_serve_cfg.get("disable_log_requests", False)
        enable_expert_parallel = vllm_serve_cfg.get("enable_expert_parallel", False)
        parallelism_pairs = vllm_serve_cfg.get("parallelism_pairs", [])

        # Benchmark options
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

            time.sleep(5)

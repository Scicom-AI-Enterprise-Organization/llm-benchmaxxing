import os
import time

from .core import VLLMServer, run_benchmark


def run(config: dict):
    """Run vLLM benchmarks based on config."""
    for run_cfg in config.get("runs", []):
        name = run_cfg.get("name", "")

        serve_cfg = run_cfg.get("serve", run_cfg)
        bench_cfg = run_cfg.get("bench", run_cfg)

        # Server options
        model_path = serve_cfg.get("model_path", "")
        port = serve_cfg.get("port", 8000)
        gpu_memory_utilization = serve_cfg.get("gpu_memory_utilization", 0.9)
        max_model_len = serve_cfg.get("max_model_len")
        max_num_seqs = serve_cfg.get("max_num_seqs")
        dtype = serve_cfg.get("dtype")
        disable_log_requests = serve_cfg.get("disable_log_requests", False)
        enable_expert_parallel = serve_cfg.get("enable_expert_parallel", False)
        tp_dp_pairs = serve_cfg.get("tp_dp_pairs", [])

        # Benchmark options
        output_dir = bench_cfg.get("output_dir", "./benchmark_results")
        context_sizes = bench_cfg.get("context_size", [])
        concurrencies = bench_cfg.get("concurrency", [])
        num_prompts_list = bench_cfg.get("num_prompts", [])
        output_lens = bench_cfg.get("output_len", [])
        save_results = bench_cfg.get("save_results", False)

        if not name or not model_path:
            continue

        if save_results:
            os.makedirs(output_dir, exist_ok=True)

        for pair in tp_dp_pairs:
            tp = pair.get("tp", 1)
            dp = pair.get("dp", 1)
            pp = pair.get("pp", 1)

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

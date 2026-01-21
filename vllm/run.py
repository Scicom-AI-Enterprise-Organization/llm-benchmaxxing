#!/usr/bin/env python3

import argparse
import os
import signal
import socket
import subprocess
import time

import requests
import yaml


class VLLMServer:
    def __init__(self, model_path, port, tp, dp, pp, 
                 gpu_memory_utilization=0.9,
                 max_model_len=None,
                 max_num_seqs=None,
                 dtype=None,
                 disable_log_requests=False,
                 enable_expert_parallel=False):
        self.model_path = model_path
        self.port = port
        self.tp = tp
        self.dp = dp
        self.pp = pp
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.dtype = dtype
        self.disable_log_requests = disable_log_requests
        self.enable_expert_parallel = enable_expert_parallel
        self.process = None
        self.base_url = f"http://localhost:{port}"

    def start(self):
        cmd = [
            "vllm", "serve", self.model_path,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp),
            "--pipeline-parallel-size", str(self.pp),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]
        
        # data parallelism via num replicas if dp > 1
        if self.dp > 1:
            cmd.extend(["--data-parallel-size", str(self.dp)])
        
        # optional arguments
        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.max_num_seqs:
            cmd.extend(["--max-num-seqs", str(self.max_num_seqs)])
        if self.dtype:
            cmd.extend(["--dtype", self.dtype])
        if self.disable_log_requests:
            cmd.append("--disable-log-requests")
        if self.enable_expert_parallel:
            cmd.append("--enable-expert-parallel")

        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, text=True)
        return self._wait_for_health()

    def _wait_for_health(self, max_attempts=200, interval=5.0):
        health_url = f"{self.base_url}/health"
        print(f"Waiting for server at {health_url}...")

        for attempt in range(max_attempts):
            try:
                resp = requests.get(health_url, timeout=5.0)
                if resp.status_code == 200:
                    print(f"Server healthy after {attempt + 1} attempts")
                    return True
            except requests.RequestException:
                pass

            if attempt % 10 == 0:
                print(f"Health check attempt {attempt + 1}...")
            time.sleep(interval)

        print(f"Server failed to become healthy after {max_attempts} attempts")
        return False

    def stop(self):
        if self.process is None or self.process.poll() is not None:
            return

        print(f"Stopping vLLM server on port {self.port}...")
        self.process.send_signal(signal.SIGINT)

        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"Force killing server...")
            self.process.kill()
            self.process.wait()

        # wait for port to be released
        for _ in range(30):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", self.port))
                print(f"Port {self.port} released")
                break
            except OSError:
                time.sleep(1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def run_benchmark(model_path, port, output_dir, result_name, ctx, output_len, num_prompts, concurrency):
    print()
    print("=" * 64)
    print(f"BENCHMARK: {result_name}")
    print("=" * 64)

    subprocess.run([
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--base-url", f"http://localhost:{port}",
        "--model", model_path,
        "--endpoint", "/v1/completions",
        "--dataset-name", "random",
        "--random-input-len", str(ctx),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--ignore-eos",
        "--save-result",
        "--result-dir", output_dir,
        "--result-filename", f"{result_name}.json",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Config file name (looked up in runs/ folder) or full path")
    args = parser.parse_args()

    # Resolve config path - check runs/ folder first, then treat as direct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, "runs")
    
    config_path = args.config
    # Try multiple resolution strategies for relative paths
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        # 1. Try relative to script directory (e.g., runs/config.yaml)
        script_relative = os.path.join(script_dir, config_path)
        # 2. Try in runs/ folder (e.g., just config.yaml or config)
        runs_path = os.path.join(runs_dir, config_path)
        runs_path_yaml = runs_path + '.yaml'
        
        if os.path.exists(script_relative):
            config_path = script_relative
        elif os.path.exists(runs_path):
            config_path = runs_path
        elif not config_path.endswith(('.yaml', '.yml')) and os.path.exists(runs_path_yaml):
            config_path = runs_path_yaml

    print(f"Using config: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for run in config.get("runs", []):
        name = run.get("name", "")
        
        # Support nested structure (serve/bench) or flat structure
        serve_cfg = run.get("serve", run)  # fallback to run itself for flat config
        bench_cfg = run.get("bench", run)  # fallback to run itself for flat config
        
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

        if not name or not model_path:
            continue

        os.makedirs(output_dir, exist_ok=True)

        for pair in tp_dp_pairs:
            tp = pair.get("tp", 1)
            dp = pair.get("dp", 1)
            pp = pair.get("pp", 1)

            print()
            print("=" * 64)
            print(f"STARTING SERVER: {name} | TP={tp} DP={dp} PP={pp}")
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
                                    ctx, output_len, num_prompts, concurrency
                                )

            # small delay between server restarts
            time.sleep(5)


if __name__ == "__main__":
    main()

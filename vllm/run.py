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
    def __init__(self, model_path, port, tp, dp, pp, gpu_memory_utilization=0.9):
        self.model_path = model_path
        self.port = port
        self.tp = tp
        self.dp = dp
        self.pp = pp
        self.gpu_memory_utilization = gpu_memory_utilization
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
        model_path = run.get("model_path", "")
        port = run.get("port", 8000)
        output_dir = run.get("output_dir", "./benchmark_results")
        gpu_memory_utilization = run.get("gpu_memory_utilization", 0.9)

        if not name or not model_path:
            continue

        os.makedirs(output_dir, exist_ok=True)

        for pair in run.get("tp_dp_pairs", []):
            tp = pair.get("tp", 1)
            dp = pair.get("dp", 1)
            pp = pair.get("pp", 1)

            print()
            print("=" * 64)
            print(f"STARTING SERVER: {name} | TP={tp} DP={dp} PP={pp}")
            print("=" * 64)

            with VLLMServer(model_path, port, tp, dp, pp, gpu_memory_utilization) as server:
                for ctx in run.get("context_size", []):
                    for concurrency in run.get("concurrency", []):
                        for num_prompts in run.get("num_prompts", []):
                            for output_len in run.get("output_len", []):
                                result_name = f"{name}_TP{tp}_DP{dp}_CTX{ctx}_C{concurrency}_P{num_prompts}_O{output_len}"
                                run_benchmark(
                                    model_path, port, output_dir, result_name,
                                    ctx, output_len, num_prompts, concurrency
                                )

            # small delay between server restarts
            time.sleep(5)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LLM Benchmark Runner

Dispatches benchmark runs to the appropriate engine (vllm, tensorrt-llm, sglang, etc.)
Supports remote execution on GPU servers via pyremote with live streaming.
"""

import importlib
import os
import sys

import yaml


SUPPORTED_ENGINES = ["vllm"]


def _write_local_logs_from_dict(config: dict, logs: dict):
    """Write benchmark logs to local .txt files."""
    if not logs:
        print("No logs to save")
        return
    
    # Get output_dir from config
    output_dir = "./benchmark_results"
    save_results = False
    for run_cfg in config.get("runs", []):
        benchmark_cfg = run_cfg.get("benchmark", {})
        if benchmark_cfg.get("save_results"):
            save_results = True
            output_dir = benchmark_cfg.get("output_dir", "./benchmark_results")
            break
    
    if not save_results:
        print("save_results is False, skipping log files")
        return
    
    output_dir = output_dir.lstrip("./")
    os.makedirs(output_dir, exist_ok=True)
    
    for result_name, log_lines in logs.items():
        log_path = os.path.join(output_dir, f"{result_name}.txt")
        with open(log_path, "w") as f:
            f.write(f"BENCHMARK: {result_name}\n")
            f.write("=" * 64 + "\n")
            for line in log_lines:
                f.write(line + "\n")
        print(f"  Saved: {result_name}.txt")


def _download_results(config: dict, remote_cfg: dict):
    """Download benchmark results (.json and .txt) from remote pod to local machine."""
    import paramiko
    from scp import SCPClient
    
    # Get output_dir from config
    output_dir = "./benchmark_results"
    for run_cfg in config.get("runs", []):
        benchmark_cfg = run_cfg.get("benchmark", {})
        if benchmark_cfg.get("save_results"):
            output_dir = benchmark_cfg.get("output_dir", "./benchmark_results")
            break
    
    host = remote_cfg["host"]
    port = remote_cfg.get("port", 22)
    username = remote_cfg.get("username", "root")
    key_filename = remote_cfg.get("key_filename")
    
    # Expand key path
    if key_filename:
        key_filename = os.path.expanduser(key_filename)
    
    print(f"Connecting to {host}:{port}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, key_filename=key_filename)
        
        # Check if results directory exists on remote
        stdin, stdout, stderr = ssh.exec_command(f"ls -la {output_dir} 2>/dev/null || echo 'NOT_FOUND'")
        output = stdout.read().decode()
        
        if "NOT_FOUND" in output:
            print(f"No results found at {output_dir} on remote")
            return
        
        # Create local results directory
        local_output_dir = output_dir.lstrip("./")
        os.makedirs(local_output_dir, exist_ok=True)
        
        # Download both .json and .txt results using SCP
        print(f"Downloading results from {output_dir} to {local_output_dir}/...")
        
        with SCPClient(ssh.get_transport()) as scp:
            # List .json files and download each
            stdin, stdout, stderr = ssh.exec_command(f"ls {output_dir}/*.json 2>/dev/null")
            json_files = stdout.read().decode().strip().split("\n")
            
            for remote_file in json_files:
                if remote_file:
                    filename = os.path.basename(remote_file)
                    local_file = os.path.join(local_output_dir, filename)
                    try:
                        scp.get(remote_file, local_file)
                        print(f"  Downloaded: {filename}")
                    except Exception as e:
                        print(f"  Failed to download {filename}: {e}")
            
            # List .txt files and download each
            stdin, stdout, stderr = ssh.exec_command(f"ls {output_dir}/*.txt 2>/dev/null")
            txt_files = stdout.read().decode().strip().split("\n")
            
            for remote_file in txt_files:
                if remote_file:
                    filename = os.path.basename(remote_file)
                    local_file = os.path.join(local_output_dir, filename)
                    try:
                        scp.get(remote_file, local_file)
                        print(f"  Downloaded: {filename}")
                    except Exception as e:
                        print(f"  Failed to download {filename}: {e}")
        
        print(f"Results saved to {local_output_dir}/")
        
    finally:
        ssh.close()


def run_e2e(config: dict):
    """
    End-to-end benchmark execution with RunPod:
    1. Deploy RunPod pod
    2. Run benchmarks on the pod
    3. Delete pod when done
    """
    from .runpod.core.client import deploy, delete, set_api_key
    
    runpod_cfg = config.get("runpod", {})
    remote_cfg = config.get("remote", {})
    
    if not runpod_cfg:
        raise ValueError("No 'runpod' section found in config")
    
    # Set API key
    api_key = runpod_cfg.get("runpod_api_key") or os.environ.get("RUNPOD_API_KEY")
    if api_key:
        set_api_key(api_key)
    else:
        raise ValueError("RunPod API key not found. Set 'runpod.runpod_api_key' in config or RUNPOD_API_KEY env var")
    
    # Prepare deploy config from runpod section
    pod_cfg = runpod_cfg.get("pod", {})
    container_cfg = runpod_cfg.get("container", {})
    storage_cfg = runpod_cfg.get("storage", {})
    ports_cfg = runpod_cfg.get("ports", {})
    env_cfg = runpod_cfg.get("env", {})
    
    # Format ports
    ports = []
    for p in ports_cfg.get("http", []):
        ports.append(f"{p}/http")
    for p in ports_cfg.get("tcp", []):
        ports.append(f"{p}/tcp")
    
    instance_type = pod_cfg.get("instance_type", "spot")
    spot = instance_type == "spot"
    ssh_key_path = runpod_cfg.get("ssh_private_key")
    
    deploy_kwargs = {
        "name": pod_cfg.get("name"),
        "gpu_type": pod_cfg.get("gpu_type"),
        "gpu_count": pod_cfg.get("gpu_count"),
        "spot": spot,
        "bid_per_gpu": pod_cfg.get("bid_per_gpu"),
        "secure_cloud": pod_cfg.get("secure_cloud", True),
        "image": container_cfg.get("image"),
        "container_disk_size": container_cfg.get("disk_size", 20),
        "disk_size": storage_cfg.get("volume_size"),
        "volume_mount_path": storage_cfg.get("mount_path", "/workspace"),
        "ports": ports if ports else None,
        "env": env_cfg if env_cfg else None,
        "ssh_key_path": ssh_key_path,
        "wait_for_ready": True,
        "deploy_retries": pod_cfg.get("deploy_retries", 10),
        "deploy_retry_interval": pod_cfg.get("deploy_retry_interval", 30.0),
    }
    
    pod_id = None
    
    try:
        print()
        print("=" * 64)
        print("STEP 1: DEPLOYING RUNPOD POD")
        print("=" * 64)
        
        instance = deploy(**deploy_kwargs)
        pod_id = instance["id"]
        
        print()
        print(f"Pod deployed: {pod_id}")
        print(f"Pod name: {instance.get('name')}")
        
        if "ssh" not in instance:
            raise Exception("SSH info not available from pod deployment")
        
        ssh_info = instance["ssh"]
        print(f"SSH: {ssh_info['command']}")
        
        # Build remote config from pod SSH info
        auto_remote_cfg = {
            "host": ssh_info["ip"],
            "port": ssh_info["port"],
            "username": "root",
            "key_filename": ssh_key_path or remote_cfg.get("key_filename"),
            "uv": remote_cfg.get("uv", {}),
            "dependencies": remote_cfg.get("dependencies", []),
        }
        
        print()
        print("=" * 64)
        print("STEP 2: RUNNING BENCHMARKS")
        print("=" * 64)
        
        # Inject pod name into run names for result identification
        pod_name = instance.get('name', '')
        if pod_name:
            for run_cfg in config.get("runs", []):
                original_name = run_cfg.get("name", "benchmark")
                run_cfg["name"] = f"{pod_name}_{original_name}"
        
        # Run benchmarks
        run_remote(config, auto_remote_cfg)
        
        # Download results (.json and .txt) from remote pod
        print()
        print("=" * 64)
        print("STEP 3: DOWNLOADING RESULTS")
        print("=" * 64)
        
        try:
            _download_results(config, auto_remote_cfg)
        except Exception as e:
            print(f"Warning: Failed to download results: {e}")
        
        print()
        print("=" * 64)
        print("STEP 4: CLEANING UP POD")
        print("=" * 64)
        
    except Exception as e:
        print()
        print("=" * 64)
        print(f"ERROR: {e}")
        print("=" * 64)
        raise
    
    finally:
        if pod_id:
            print()
            print(f"Deleting pod: {pod_id}")
            try:
                result = delete(pod_id=pod_id)
                print(f"Pod deleted: {result}")
            except Exception as e:
                print(f"Warning: Failed to delete pod {pod_id}: {e}")
                print("Please delete the pod manually via RunPod dashboard")
    
    print()
    print("=" * 64)
    print("END-TO-END BENCHMARK COMPLETED!")
    print("=" * 64)


def run_remote(config: dict, remote_cfg: dict):
    """Execute benchmark on a remote GPU server via pyremote with live streaming."""
    from pyremote import remote, UvConfig
    
    host = remote_cfg["host"]
    port = remote_cfg.get("port", 22)
    username = remote_cfg.get("username", "root")
    password = remote_cfg.get("password")
    key_filename = remote_cfg.get("key_filename")
    
    uv_cfg = remote_cfg.get("uv", {})
    uv_path = uv_cfg.get("path", "~/.benchmark-venv")
    python_version = uv_cfg.get("python_version", "3.11")
    
    deps = remote_cfg.get("dependencies", [
        "pyyaml",
        "requests",
        "vllm==0.11.0",
        "huggingface_hub",
    ])
    
    if key_filename:
        key_filename = os.path.expanduser(key_filename)
    
    print(f"Connecting to remote server: {username}@{host}:{port}")
    if key_filename:
        print(f"Using SSH key: {key_filename}")
    elif password:
        print("Using password authentication")
    print(f"UV environment: {uv_path} (Python {python_version})")
    print(f"Dependencies: {deps}")
    print()

    remote_kwargs = {
        "uv": UvConfig(path=uv_path, python_version=python_version),
        "dependencies": deps,
    }
    if password:
        remote_kwargs["password"] = password
    if key_filename:
        remote_kwargs["key_filename"] = key_filename
    if port != 22:
        remote_kwargs["port"] = port

    @remote(host, username, **remote_kwargs)
    def execute_benchmark():
        import os
        import signal
        import socket
        import subprocess
        import sys
        import time
        
        import requests
        
        class VLLMServer:
            def __init__(self, model_path, port, tp, dp, pp,
                         gpu_memory_utilization=0.9, max_model_len=None,
                         max_num_seqs=None, dtype=None,
                         disable_log_requests=False, enable_expert_parallel=False):
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
                if self.dp > 1:
                    cmd.extend(["--data-parallel-size", str(self.dp)])
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
                sys.stdout.flush()
                self.process = subprocess.Popen(cmd, text=True)
                return self._wait_for_health()

            def _wait_for_health(self):
                health_url = f"{self.base_url}/health"
                print(f"Waiting for server at {health_url} (unlimited retries)...")
                sys.stdout.flush()
                attempt = 0
                while True:
                    try:
                        resp = requests.get(health_url, timeout=5.0)
                        if resp.status_code == 200:
                            print(f"Server healthy after {attempt + 1} attempts")
                            sys.stdout.flush()
                            return True
                    except requests.RequestException:
                        pass
                    if attempt % 10 == 0:
                        print(f"Health check attempt {attempt + 1}...")
                        sys.stdout.flush()
                    attempt += 1
                    time.sleep(10.0)

            def stop(self):
                if self.process is None or self.process.poll() is not None:
                    return
                print(f"Stopping vLLM server on port {self.port}...")
                sys.stdout.flush()
                self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("Force killing server...")
                    self.process.kill()
                    self.process.wait()
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

        def run_benchmark(model_path, port, output_dir, result_name, ctx, output_len, num_prompts, concurrency, save_results=False):
            print()
            print("=" * 64)
            print(f"BENCHMARK: {result_name}")
            print("=" * 64)
            sys.stdout.flush()

            cmd = [
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
                "--percentile-metrics", "ttft,tpot,itl,e2el",
            ]

            if save_results:
                cmd.extend([
                    "--save-result",
                    "--result-dir", output_dir,
                    "--result-filename", f"{result_name}.json",
                ])

            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            log_lines = []
            for line in process.stdout:
                print(line, end='', flush=True)
                # Filter out APIServer logs for storage
                if "(APIServer)" not in line:
                    log_lines.append(line.rstrip('\n'))
            process.wait()
            
            # Save .txt log file on remote if save_results is enabled
            if save_results:
                log_path = os.path.join(output_dir, f"{result_name}.txt")
                with open(log_path, "w") as f:
                    f.write(f"BENCHMARK: {result_name}\n")
                    f.write("=" * 64 + "\n")
                    for line in log_lines:
                        f.write(line + "\n")
                print(f"Saved log: {log_path}")
                sys.stdout.flush()

        def download_model(repo_id, local_dir, hf_token=None):
            print()
            print("=" * 64)
            print(f"DOWNLOADING MODEL: {repo_id}")
            print("=" * 64)
            sys.stdout.flush()
            
            os.makedirs(local_dir, exist_ok=True)
            
            # Set HF_TOKEN if provided (config takes priority over env)
            env = os.environ.copy()
            token = hf_token or os.environ.get("HF_TOKEN")
            if token:
                env["HF_TOKEN"] = token
            
            cmd = ["huggingface-cli", "download", repo_id, "--local-dir", local_dir]
            print(f"Running: {' '.join(cmd)}")
            sys.stdout.flush()
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            for line in process.stdout:
                print(line, end='', flush=True)
            process.wait()
            
            if process.returncode != 0:
                raise Exception(f"Model download failed with exit code {process.returncode}")
            
            print()
            print("✓ Model download completed!")
            sys.stdout.flush()

        # Run benchmarks
        for run_cfg in config.get("runs", []):
            name = run_cfg.get("name", "")
            model_cfg = run_cfg.get("model", {})
            vllm_serve_cfg = run_cfg.get("vllm_serve", run_cfg)
            benchmark_cfg = run_cfg.get("benchmark", run_cfg)

            # Set HF_TOKEN for gated models (config takes priority over env)
            hf_token = model_cfg.get("hf_token") or config.get("hf_token")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token

            # Download model if specified
            if model_cfg.get("repo_id") and model_cfg.get("local_dir"):
                download_model(model_cfg["repo_id"], model_cfg["local_dir"], hf_token)

            model_path = vllm_serve_cfg.get("model_path", model_cfg.get("local_dir", ""))
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
                sys.stdout.flush()

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

        print()
        print("=" * 64)
        print("BENCHMARK COMPLETED!")
        print("=" * 64)
        return {"status": "completed"}

    result = execute_benchmark()
    
    print()
    print("=" * 64)
    print("Remote execution completed!")
    print("=" * 64)
    
    return result


def run(config_or_path):
    """
    Main entry point for running benchmarks.
    
    Args:
        config_or_path: Either a config dict or a path to a YAML config file
        
    Returns:
        Dict with status and results
    """
    from typing import Union
    
    # Handle both dict and string (path) inputs
    if isinstance(config_or_path, dict):
        config = config_or_path
    else:
        config_path = config_or_path
        if not os.path.isabs(config_path):
            config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

        print(f"Loading config: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)

    runs = config.get("runs", [])
    if not runs:
        print("Error: No runs defined in config")
        sys.exit(1)

    engine = runs[0].get("engine", "vllm")

    if engine not in SUPPORTED_ENGINES:
        print(f"Error: Unsupported engine '{engine}'. Supported: {SUPPORTED_ENGINES}")
        sys.exit(1)

    print()
    print("=" * 64)
    print(f"ENGINE: {engine}")
    print("=" * 64)

    remote_cfg = config.get("remote")
    if remote_cfg:
        print("REMOTE EXECUTION ENABLED")
        print("=" * 64)
        run_remote(config, remote_cfg)
        
        # Download results from remote
        print()
        print("=" * 64)
        print("DOWNLOADING RESULTS")
        print("=" * 64)
        
        try:
            _download_results(config, remote_cfg)
        except Exception as e:
            print(f"Warning: Failed to download results: {e}")
        
        return {"status": "success", "mode": "remote", "host": remote_cfg.get("host")}
    else:
        # Import from benchmaq.vllm instead of vllm
        engine_module = importlib.import_module(f"benchmaq.{engine}")
        return engine_module.run(config)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m benchmaq.runner <config.yaml>")
        print("Example: python -m benchmaq.runner examples/run_single.yaml")
        sys.exit(1)
    
    run(sys.argv[1])

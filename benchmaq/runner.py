#!/usr/bin/env python3
"""
LLM Benchmark Runner

Handles remote execution on GPU servers via pyremote with live streaming.
Used by both vllm.bench and runpod.bench modules.
"""

import os
import sys
import hashlib


def _get_results_config(config: dict) -> dict:
    """Extract results configuration from config."""
    for run_cfg in config.get("benchmark", []):
        results_cfg = run_cfg.get("results", {})
        if results_cfg.get("save_result"):
            return {
                "save_result": True,
                "result_dir": results_cfg.get("result_dir", "./benchmark_results"),
            }
    return {"save_result": False, "result_dir": "./benchmark_results"}


def _download_results(config: dict, remote_cfg: dict):
    """Download benchmark results (.json and .txt) from remote to local."""
    import paramiko
    from scp import SCPClient
    
    results_cfg = _get_results_config(config)
    output_dir = results_cfg.get("result_dir", "./benchmark_results")
    
    host = remote_cfg["host"]
    port = remote_cfg.get("port", 22)
    username = remote_cfg.get("username", "root")
    key_filename = remote_cfg.get("key_filename")
    
    if key_filename:
        key_filename = os.path.expanduser(key_filename)
    
    print(f"Connecting to {host}:{port}...")
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        ssh.connect(host, port=port, username=username, key_filename=key_filename)
        
        stdin, stdout, stderr = ssh.exec_command(f"ls -la {output_dir} 2>/dev/null || echo 'NOT_FOUND'")
        output = stdout.read().decode()
        
        if "NOT_FOUND" in output:
            print(f"No results found at {output_dir} on remote")
            return
        
        local_output_dir = output_dir.lstrip("./")
        os.makedirs(local_output_dir, exist_ok=True)
        
        print(f"Downloading results from {output_dir} to {local_output_dir}/...")
        
        with SCPClient(ssh.get_transport()) as scp:
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
    3. Download results
    4. Delete pod
    """
    from .runpod.core.client import deploy, delete, set_api_key
    
    runpod_cfg = config.get("runpod", {})
    remote_cfg = config.get("remote", {})
    
    if not runpod_cfg:
        raise ValueError("No 'runpod' section found in config")
    
    api_key = runpod_cfg.get("runpod_api_key") or os.environ.get("RUNPOD_API_KEY")
    if api_key:
        set_api_key(api_key)
    else:
        raise ValueError("RunPod API key not found. Set 'runpod.runpod_api_key' in config or RUNPOD_API_KEY env var")
    
    pod_cfg = runpod_cfg.get("pod", {})
    container_cfg = runpod_cfg.get("container", {})
    storage_cfg = runpod_cfg.get("storage", {})
    ports_cfg = runpod_cfg.get("ports", {})
    env_cfg = runpod_cfg.get("env", {})
    
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
        "gpu_count": pod_cfg.get("gpu_count", 1),
        "spot": spot,
        "bid_per_gpu": pod_cfg.get("bid_per_gpu"),
        "secure_cloud": pod_cfg.get("secure_cloud", True),
        "image": container_cfg.get("image", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"),
        "container_disk_size": container_cfg.get("disk_size", 20),
        "disk_size": storage_cfg.get("volume_size", 100),
        "volume_mount_path": storage_cfg.get("mount_path", "/workspace"),
        "ports": ports if ports else None,
        "env": env_cfg if env_cfg else None,
        "ssh_key_path": ssh_key_path,
        "wait_for_ready": True,
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
        
        pod_name = instance.get('name', '')
        if pod_name:
            for bench_cfg in config.get("benchmark", []):
                original_name = bench_cfg.get("name", "benchmark")
                bench_cfg["name"] = f"{pod_name}_{original_name}"
        
        run_remote(config, auto_remote_cfg)
        
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
        
    except KeyboardInterrupt:
        print()
        print("=" * 64)
        print("INTERRUPTED BY USER")
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
            print("=" * 64)
            print("CLEANING UP POD")
            print("=" * 64)
            print(f"Deleting pod: {pod_id}")
            try:
                result = delete(pod_id=pod_id)
                print(f"Pod deleted: {result}")
            except Exception as e:
                print(f"Warning: Failed to delete pod {pod_id}: {e}")
                print("Please delete the pod manually via RunPod dashboard")
    
    print()
    print("=" * 64)
    print("BENCHMARK COMPLETED!")
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
        "vllm",
        "huggingface_hub[hf_transfer]",
        "hf_transfer",
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
        import hashlib
        
        import requests
        
        def kwargs_to_cli_args(kwargs):
            """Convert kwargs dict to CLI arguments list."""
            args = []
            for key, value in kwargs.items():
                arg_name = f"--{key.replace('_', '-')}"
                if isinstance(value, bool):
                    if value:
                        args.append(arg_name)
                elif value is not None:
                    args.extend([arg_name, str(value)])
            return args
        
        def generate_result_name(config_name, index, bench_cfg):
            """Generate a unique result name."""
            cfg_str = str(sorted(bench_cfg.items()))
            cfg_hash = hashlib.md5(cfg_str.encode()).hexdigest()[:6]
            parts = [config_name]
            if "random_input_len" in bench_cfg:
                parts.append(f"in{bench_cfg['random_input_len']}")
            if "random_output_len" in bench_cfg:
                parts.append(f"out{bench_cfg['random_output_len']}")
            if "num_prompts" in bench_cfg:
                parts.append(f"p{bench_cfg['num_prompts']}")
            if "max_concurrency" in bench_cfg:
                parts.append(f"c{bench_cfg['max_concurrency']}")
            parts.append(cfg_hash)
            return "_".join(parts)
        
        class VLLMServer:
            """vLLM Server manager."""
            
            def __init__(self, model, port=8000, **kwargs):
                self.model = model
                self.port = port
                self.serve_kwargs = kwargs
                self.process = None
                self.base_url = f"http://localhost:{port}"

            def _build_cmd(self):
                cmd = ["vllm", "serve", self.model, "--port", str(self.port)]
                cmd.extend(kwargs_to_cli_args(self.serve_kwargs))
                return cmd

            def start(self):
                cmd = self._build_cmd()
                print(f"Starting vLLM server: {' '.join(cmd)}")
                sys.stdout.flush()
                self.process = subprocess.Popen(cmd, text=True)
                return self._wait_for_health()

            def _wait_for_health(self, max_attempts=200, interval=5.0):
                health_url = f"{self.base_url}/health"
                print(f"Waiting for server at {health_url}...")
                sys.stdout.flush()
                for attempt in range(max_attempts):
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
                    time.sleep(interval)
                print(f"Server failed to become healthy after {max_attempts} attempts")
                sys.stdout.flush()
                return False

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

        def run_benchmark(model, port, result_name, results_config=None, **kwargs):
            """Run vLLM bench serve."""
            print()
            print("=" * 64)
            print(f"BENCHMARK: {result_name}")
            print("=" * 64)
            sys.stdout.flush()

            cmd = ["vllm", "bench", "serve",
                   "--base-url", f"http://localhost:{port}",
                   "--model", model]
            cmd.extend(kwargs_to_cli_args(kwargs))
            
            results_config = results_config or {}
            save_result = results_config.get("save_result", False)
            result_dir = results_config.get("result_dir", "./benchmark_results")
            result_filename = results_config.get("result_filename")
            save_detailed = results_config.get("save_detailed", False)
            
            if save_result:
                os.makedirs(result_dir, exist_ok=True)
                cmd.append("--save-result")
                cmd.extend(["--result-dir", result_dir])
                if result_filename:
                    cmd.extend(["--result-filename", result_filename])
                else:
                    cmd.extend(["--result-filename", f"{result_name}.json"])
                if save_detailed:
                    cmd.append("--save-detailed")

            print(f"Running: {' '.join(cmd)}")
            sys.stdout.flush()
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            log_lines = []
            for line in process.stdout:
                print(line, end='', flush=True)
                if "(APIServer)" not in line:
                    log_lines.append(line.rstrip('\n'))
            process.wait()
            
            if save_result:
                log_path = os.path.join(result_dir, f"{result_name}.txt")
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
            
            env = os.environ.copy()
            token = hf_token or os.environ.get("HF_TOKEN")
            if token:
                env["HF_TOKEN"] = token
            
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            
            cmd = ["huggingface-cli", "download", repo_id, "--local-dir", local_dir]
            print(f"Running: {' '.join(cmd)} (with hf_transfer enabled)")
            sys.stdout.flush()
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            for line in process.stdout:
                print(line, end='', flush=True)
            process.wait()
            
            if process.returncode != 0:
                raise Exception(f"Model download failed with exit code {process.returncode}")
            
            print()
            print("Model download completed!")
            sys.stdout.flush()

        # Run benchmarks
        if "benchmark" not in config:
            raise ValueError("No 'benchmark:' section found in config")
        
        for run_cfg in config.get("benchmark", []):
            name = run_cfg.get("name", "benchmark")
            engine = run_cfg.get("engine", "vllm")
            
            if engine != "vllm":
                print(f"Skipping {name}: engine '{engine}' not supported")
                continue
            
            model_cfg = run_cfg.get("model", {})
            serve_cfg = run_cfg.get("serve", {}).copy()
            bench_configs = run_cfg.get("bench", [])
            results_cfg = run_cfg.get("results", {})
            
            hf_token = model_cfg.get("hf_token") or os.environ.get("HF_TOKEN")
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
            
            if model_cfg.get("repo_id") and model_cfg.get("local_dir"):
                download_model(model_cfg["repo_id"], model_cfg["local_dir"], hf_token)
            
            model = (
                serve_cfg.pop("model", None) or 
                serve_cfg.pop("model_path", None) or 
                model_cfg.get("local_dir") or 
                model_cfg.get("repo_id", "")
            )
            port = serve_cfg.pop("port", 8000)
            
            if not model:
                print(f"Skipping {name}: no model specified")
                continue
            
            if not bench_configs:
                print(f"Skipping {name}: no 'bench:' configurations found")
                continue
            
            print()
            print("=" * 64)
            print(f"CONFIGURATION: {name}")
            print(f"Model: {model}")
            print("=" * 64)
            sys.stdout.flush()
            
            with VLLMServer(model=model, port=port, **serve_cfg) as server:
                for i, bench_cfg in enumerate(bench_configs):
                    result_name = generate_result_name(name, i, bench_cfg)
                    run_benchmark(
                        model=model,
                        port=port,
                        result_name=result_name,
                        results_config=results_cfg,
                        **bench_cfg
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

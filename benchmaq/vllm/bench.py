"""vLLM benchmark runner.

Usage:
    import benchmaq.vllm.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq vllm bench config.yaml
"""

import os
import time
import hashlib
from typing import Optional, List, Dict, Any

from .core import VLLMServer, run_benchmark


def from_yaml(config_path: str) -> Dict[str, Any]:
    """Run vLLM benchmarks from YAML config.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict with status and results
        
    Example YAML structure:
        benchmark:
          - name: my_benchmark
            engine: vllm
            model:
              repo_id: "model/repo"
              local_dir: "/path/to/model"
            serve:
              tensor_parallel_size: 8
              max_num_seqs: 256
            bench:
              - backend: vllm
                endpoint: /v1/completions
                dataset_name: random
                random_input_len: 1024
                num_prompts: 100
            results:
              save_result: true
              result_dir: "./results"
        
        # Optional: for remote execution
        remote:
          host: "gpu-server.example.com"
          port: 22
          username: "root"
          key_filename: "~/.ssh/id_rsa"
    """
    from benchmaq.config import load_config
    
    config = load_config(config_path)
    return _run(config)


def _run(config: dict) -> Dict[str, Any]:
    """Run vLLM benchmarks based on config dict."""
    # Check for remote execution
    remote_cfg = config.get("remote")
    if remote_cfg:
        from benchmaq.runner import run_remote, _download_results
        
        print("Remote execution enabled")
        print(f"Host: {remote_cfg.get('host')}:{remote_cfg.get('port', 22)}")
        print()
        
        run_remote(config, remote_cfg)
        
        # Download results from remote
        try:
            _download_results(config, remote_cfg)
        except Exception as e:
            print(f"Warning: Failed to download results: {e}")
        
        return {"status": "success", "mode": "remote", "host": remote_cfg.get("host")}
    
    # Local execution
    if "benchmark" not in config:
        raise ValueError("No 'benchmark:' section found in config.")
    
    results = _run_benchmarks(config)
    return {"status": "success", "results": results}


def _download_model(repo_id: str, local_dir: str, hf_token: Optional[str] = None):
    """Download model from HuggingFace Hub."""
    import subprocess
    
    print()
    print("=" * 64)
    print(f"DOWNLOADING MODEL: {repo_id}")
    print(f"Destination: {local_dir}")
    print("=" * 64)
    
    os.makedirs(local_dir, exist_ok=True)
    
    env = os.environ.copy()
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        env["HF_TOKEN"] = token
    
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    
    cmd = ["huggingface-cli", "download", repo_id, "--local-dir", local_dir]
    print(f"Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    for line in process.stdout:
        print(line, end='', flush=True)
    process.wait()
    
    if process.returncode != 0:
        raise Exception(f"Model download failed with exit code {process.returncode}")
    
    print()
    print("Model download completed!")


def _run_benchmarks(config: dict) -> List[Dict[str, Any]]:
    """Run benchmarks from config."""
    results = []
    
    for run_cfg in config.get("benchmark", []):
        name = run_cfg.get("name", "benchmark")
        engine = run_cfg.get("engine", "vllm")
        
        if engine != "vllm":
            print(f"Skipping {name}: engine '{engine}' not supported (only 'vllm' supported)")
            continue
        
        model_cfg = run_cfg.get("model", {})
        serve_cfg = run_cfg.get("serve", {}).copy()
        bench_configs = run_cfg.get("bench", [])
        results_cfg = run_cfg.get("results", {})
        
        # Handle HF token
        hf_token = model_cfg.get("hf_token") or os.environ.get("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        
        # Download model if needed
        if model_cfg.get("repo_id") and model_cfg.get("local_dir"):
            _download_model(model_cfg["repo_id"], model_cfg["local_dir"], hf_token)
        
        # Determine model path
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
        print(f"Serve kwargs: {serve_cfg}")
        print("=" * 64)
        
        with VLLMServer(model=model, port=port, **serve_cfg) as server:
            for i, bench_cfg in enumerate(bench_configs):
                result_name = _generate_result_name(name, i, bench_cfg)
                
                print()
                print(f"--- Benchmark {i + 1}/{len(bench_configs)}: {result_name} ---")
                
                run_benchmark(
                    model=model,
                    port=port,
                    result_name=result_name,
                    results_config=results_cfg,
                    **bench_cfg
                )
                
                results.append({
                    "name": result_name,
                    "config": name,
                    "bench_index": i,
                    **bench_cfg
                })
        
        time.sleep(5)
    
    return results


def _generate_result_name(config_name: str, index: int, bench_cfg: dict) -> str:
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

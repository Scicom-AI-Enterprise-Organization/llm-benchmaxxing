"""RunPod end-to-end benchmark runner.

Usage:
    import benchmaq.runpod.bench as bench
    result = bench.from_yaml("config.yaml")

CLI:
    benchmaq runpod bench config.yaml
"""

import os
from typing import Dict, Any

from .core.client import set_api_key


def from_yaml(config_path: str) -> Dict[str, Any]:
    """Run end-to-end RunPod benchmark from YAML config.
    
    This deploys a pod, runs benchmarks, downloads results, and deletes the pod.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict with status and results
        
    Example YAML structure:
        runpod:
          runpod_api_key: "..."  # or set RUNPOD_API_KEY env var
          ssh_private_key: "~/.ssh/id_ed25519"
          pod:
            name: "benchmark-pod"
            gpu_type: "NVIDIA A100 80GB PCIe"
            gpu_count: 2
            instance_type: "spot"  # or "on-demand"
          container:
            image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
          storage:
            volume_size: 100
            mount_path: "/workspace"
        
        remote:
          uv:
            path: "~/.benchmark-venv"
            python_version: "3.11"
          dependencies:
            - vllm
            - huggingface_hub
        
        benchmark:
          - name: my_benchmark
            engine: vllm
            model:
              repo_id: "Qwen/Qwen2.5-7B-Instruct"
              local_dir: "/workspace/model"
            serve:
              tensor_parallel_size: 2
              max_model_len: 8192
            bench:
              - backend: vllm
                dataset_name: random
                random_input_len: 1024
                num_prompts: 100
            results:
              save_result: true
              result_dir: "./benchmark_results"
    """
    from benchmaq.config import load_config
    from benchmaq.runner import run_e2e
    
    config = load_config(config_path)
    
    # Ensure API key is set
    runpod_cfg = config.get("runpod", {})
    api_key = runpod_cfg.get("runpod_api_key") or os.environ.get("RUNPOD_API_KEY")
    if api_key:
        set_api_key(api_key)
    
    try:
        run_e2e(config)
        return {
            "status": "success",
            "gpu_type": runpod_cfg.get("pod", {}).get("gpu_type"),
            "gpu_count": runpod_cfg.get("pod", {}).get("gpu_count"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

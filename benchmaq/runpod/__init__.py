"""RunPod cloud GPU module."""

from typing import Optional, List, Dict, Any
import os

from .core.client import (
    deploy as _deploy,
    delete as _delete,
    find as _find,
    find_by_name as _find_by_name,
    start as _start,
    stop as _stop,
    list_pods as _list_pods,
    set_api_key,
    get_api_key,
    wait_for_pod,
    get_ssh_info,
    check_ssh,
)


def _ensure_api_key(api_key: Optional[str] = None) -> str:
    if api_key:
        set_api_key(api_key)
        return api_key
    existing = get_api_key()
    if existing:
        return existing
    env_key = os.environ.get("RUNPOD_API_KEY")
    if env_key:
        set_api_key(env_key)
        return env_key
    raise ValueError("RunPod API key required. Provide api_key or set RUNPOD_API_KEY env var.")


def deploy(
    config_path: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    gpu_type: Optional[str] = None,
    gpu_count: int = 1,
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    disk_size: int = 100,
    container_disk_size: int = 20,
    volume_mount_path: str = "/workspace",
    secure_cloud: bool = True,
    spot: bool = True,
    bid_per_gpu: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
    ports: Optional[str] = None,
    ssh_key_path: Optional[str] = None,
    wait_for_ready: bool = True,
    health_check_retries: int = 60,
    health_check_interval: float = 10.0,
    **kwargs,
) -> Dict[str, Any]:
    """Deploy a RunPod GPU pod.
    
    Can be called with a YAML config file or with individual parameters.
    
    Examples:
        # Using YAML config
        deploy("runpod-config.yaml")
        
        # Using parameters
        deploy(gpu_type="NVIDIA B200", gpu_count=8)
        
        # Mix: YAML config with overrides
        deploy("config.yaml", gpu_count=4)
    """
    import yaml
    
    # Load config from YAML if provided
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        runpod_cfg = config.get("runpod", {})
        pod_cfg = runpod_cfg.get("pod", {})
        container_cfg = runpod_cfg.get("container", {})
        storage_cfg = runpod_cfg.get("storage", {})
        ports_cfg = runpod_cfg.get("ports", {})
        env_cfg = runpod_cfg.get("env", {})
        
        # Extract values from config (parameters override config)
        final_api_key = api_key or runpod_cfg.get("runpod_api_key")
        final_gpu_type = gpu_type or pod_cfg.get("gpu_type")
        final_gpu_count = gpu_count if gpu_type else pod_cfg.get("gpu_count", 1)
        final_name = name or pod_cfg.get("name")
        final_image = image if gpu_type else container_cfg.get("image", image)
        final_container_disk_size = container_disk_size if gpu_type else container_cfg.get("disk_size", container_disk_size)
        final_disk_size = disk_size if gpu_type else storage_cfg.get("volume_size", disk_size)
        final_volume_mount_path = volume_mount_path if gpu_type else storage_cfg.get("mount_path", volume_mount_path)
        final_secure_cloud = secure_cloud if gpu_type else pod_cfg.get("secure_cloud", secure_cloud)
        final_ssh_key_path = ssh_key_path or runpod_cfg.get("ssh_private_key")
        
        # Instance type: spot vs on_demand
        instance_type = pod_cfg.get("instance_type", "spot")
        final_spot = spot if gpu_type else (instance_type == "spot")
        final_bid_per_gpu = bid_per_gpu or pod_cfg.get("bid_per_gpu")
        
        # Ports
        if ports:
            final_ports = ports
        elif ports_cfg:
            port_list = []
            for p in ports_cfg.get("http", []):
                port_list.append(f"{p}/http")
            for p in ports_cfg.get("tcp", []):
                port_list.append(f"{p}/tcp")
            final_ports = ",".join(port_list) if port_list else None
        else:
            final_ports = None
        
        # Env
        final_env = env or env_cfg or None
        
        # Deploy retries from config
        deploy_retries = pod_cfg.get("deploy_retries", kwargs.get("deploy_retries", 10))
        deploy_retry_interval = pod_cfg.get("deploy_retry_interval", kwargs.get("deploy_retry_interval", 30.0))
        kwargs["deploy_retries"] = deploy_retries
        kwargs["deploy_retry_interval"] = deploy_retry_interval
    else:
        # No config file, use parameters directly
        if not gpu_type:
            raise ValueError("gpu_type is required when not using a config file")
        final_api_key = api_key
        final_gpu_type = gpu_type
        final_gpu_count = gpu_count
        final_name = name
        final_image = image
        final_container_disk_size = container_disk_size
        final_disk_size = disk_size
        final_volume_mount_path = volume_mount_path
        final_secure_cloud = secure_cloud
        final_spot = spot
        final_bid_per_gpu = bid_per_gpu
        final_ports = ports
        final_env = env
        final_ssh_key_path = ssh_key_path
    
    _ensure_api_key(final_api_key)
    return _deploy(
        gpu_type=final_gpu_type, gpu_count=final_gpu_count, image=final_image, disk_size=final_disk_size,
        container_disk_size=final_container_disk_size, volume_mount_path=final_volume_mount_path,
        secure_cloud=final_secure_cloud, spot=final_spot, bid_per_gpu=final_bid_per_gpu, env=final_env,
        name=final_name, ports=final_ports, ssh_key_path=final_ssh_key_path, wait_for_ready=wait_for_ready,
        health_check_retries=health_check_retries, health_check_interval=health_check_interval,
        **kwargs,
    )


def delete(api_key: Optional[str] = None, *, pod_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    """Delete a RunPod pod."""
    _ensure_api_key(api_key)
    return _delete(pod_id=pod_id, name=name)


def find(api_key: Optional[str] = None, *, pod_id: str) -> Dict[str, Any]:
    """Find a pod by ID."""
    _ensure_api_key(api_key)
    return _find(pod_id)


def find_by_name(api_key: Optional[str] = None, *, name: str) -> Optional[Dict[str, Any]]:
    """Find a pod by name."""
    _ensure_api_key(api_key)
    return _find_by_name(name)


def start(api_key: Optional[str] = None, *, pod_id: str, gpu_count: int = 1) -> Dict[str, Any]:
    """Start a stopped pod."""
    _ensure_api_key(api_key)
    return _start(pod_id, gpu_count=gpu_count)


def stop(api_key: Optional[str] = None, *, pod_id: str) -> Dict[str, Any]:
    """Stop a running pod."""
    _ensure_api_key(api_key)
    return _stop(pod_id)


def list_pods(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all pods."""
    _ensure_api_key(api_key)
    return _list_pods()


def bench(
    config_path: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    gpu_type: Optional[str] = None,
    gpu_count: Optional[int] = None,
    image: str = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    disk_size: int = 100,
    container_disk_size: int = 20,
    mount_path: str = "/workspace",
    secure_cloud: bool = True,
    spot: bool = True,
    instance_type: Optional[str] = None,
    bid_per_gpu: Optional[float] = None,
    pod_name: Optional[str] = None,
    ports_http: Optional[List[int]] = None,
    ports_tcp: Optional[List[int]] = None,
    env: Optional[Dict[str, str]] = None,
    ssh_private_key: Optional[str] = None,
    uv_path: str = "~/.benchmark-venv",
    python_version: str = "3.11",
    dependencies: Optional[List[str]] = None,
    name: str = "benchmark",
    model_path: Optional[str] = None,
    hf_token: Optional[str] = None,
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
    """End-to-end RunPod benchmarking: deploy -> bench -> delete."""
    from ..config import load_config, merge_config, kwargs_to_runpod_config
    from ..runner import run_e2e as _run_e2e

    config = {}
    if config_path:
        config = load_config(config_path)

    if "runpod" not in config:
        config["runpod"] = {}
    if "pod" not in config["runpod"]:
        config["runpod"]["pod"] = {}

    if gpu_count is not None:
        config["runpod"]["pod"]["gpu_count"] = gpu_count
    if gpu_type:
        config["runpod"]["pod"]["gpu_type"] = gpu_type
    if api_key:
        config["runpod"]["runpod_api_key"] = api_key
    if ssh_private_key:
        config["runpod"]["ssh_private_key"] = ssh_private_key
    if pod_name:
        config["runpod"]["pod"]["name"] = pod_name
    if instance_type:
        config["runpod"]["pod"]["instance_type"] = instance_type

    if config.get("runs"):
        for run_cfg in config["runs"]:
            if "benchmark" not in run_cfg:
                run_cfg["benchmark"] = {}
            # Override run name with pod_name if provided (for multiprocessing identification)
            if pod_name and name == "benchmark":
                run_cfg["name"] = f"{run_cfg.get('name', 'benchmark')}_{pod_name}"
            elif name != "benchmark":
                run_cfg["name"] = name
            if context_sizes is not None:
                run_cfg["benchmark"]["context_size"] = context_sizes
            if concurrency is not None:
                run_cfg["benchmark"]["concurrency"] = concurrency
            if num_prompts is not None:
                run_cfg["benchmark"]["num_prompts"] = num_prompts
            if output_len is not None:
                run_cfg["benchmark"]["output_len"] = output_len
            if save_results:
                run_cfg["benchmark"]["save_results"] = save_results
            if output_dir != "./benchmark_results":
                run_cfg["benchmark"]["output_dir"] = output_dir
            if parallelism_pairs is not None:
                if "vllm_serve" not in run_cfg:
                    run_cfg["vllm_serve"] = {}
                run_cfg["vllm_serve"]["parallelism_pairs"] = parallelism_pairs
            if model_path:
                if "vllm_serve" not in run_cfg:
                    run_cfg["vllm_serve"] = {}
                run_cfg["vllm_serve"]["model_path"] = model_path

    elif gpu_type or model_path:
        use_spot = spot if instance_type is None else instance_type == "spot"
        kwargs_config = kwargs_to_runpod_config(
            api_key=api_key, gpu_type=gpu_type or "", gpu_count=gpu_count if gpu_count is not None else 1,
            image=image, disk_size=disk_size, container_disk_size=container_disk_size, mount_path=mount_path,
            secure_cloud=secure_cloud, instance_type="spot" if use_spot else "on-demand", bid_per_gpu=bid_per_gpu,
            name=pod_name, ports_http=ports_http or [8888, 8000], ports_tcp=ports_tcp or [22], env=env or {},
            ssh_private_key=ssh_private_key, uv_path=uv_path, python_version=python_version,
            dependencies=dependencies or ["pyyaml", "requests", "vllm==0.11.0", "huggingface_hub"],
            model_path=model_path or "", hf_token=hf_token, tensor_parallel=tensor_parallel,
            data_parallel=data_parallel, pipeline_parallel=pipeline_parallel, parallelism_pairs=parallelism_pairs,
            gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
            dtype=dtype, disable_log_requests=disable_log_requests, enable_expert_parallel=enable_expert_parallel,
            context_sizes=context_sizes or [1024], concurrency=concurrency or [50],
            num_prompts=num_prompts or [100], output_len=output_len or [128],
            output_dir=output_dir, save_results=save_results,
        )
        config = merge_config(config, kwargs_config)

    runpod_cfg = config.get("runpod", {})
    final_api_key = api_key or runpod_cfg.get("runpod_api_key") or os.environ.get("RUNPOD_API_KEY")
    if final_api_key:
        set_api_key(final_api_key)
        config["runpod"]["runpod_api_key"] = final_api_key

    if not config.get("runpod", {}).get("pod", {}).get("gpu_type") and not gpu_type:
        raise ValueError("gpu_type is required.")
    if not config.get("runs"):
        raise ValueError("No benchmark runs defined.")

    try:
        _run_e2e(config)
        return {
            "status": "success",
            "gpu_type": config.get("runpod", {}).get("pod", {}).get("gpu_type"),
            "gpu_count": config.get("runpod", {}).get("pod", {}).get("gpu_count"),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def run_benchmark(config_path: str) -> Dict[str, Any]:
    """
    Run benchmark from config file. Designed for multiprocessing.
    
    This function is importable from the module level, making it compatible
    with Python's multiprocessing on macOS (spawn method).
    
    Usage:
        from multiprocessing import Pool
        from benchmaq.runpod import run_benchmark
        
        configs = ["config1.yaml", "config2.yaml"]
        with Pool(processes=len(configs)) as pool:
            results = pool.map(run_benchmark, configs)
    
    Args:
        config_path: Path to the benchmark config YAML file.
        
    Returns:
        Dict with 'config', 'result' on success, or 'config', 'error' on failure.
    """
    try:
        result = bench(config_path)
        return {"config": config_path, "result": result}
    except Exception as e:
        return {"config": config_path, "error": str(e)}


__all__ = ["deploy", "delete", "find", "find_by_name", "start", "stop", "list_pods", "bench", "run_benchmark"]

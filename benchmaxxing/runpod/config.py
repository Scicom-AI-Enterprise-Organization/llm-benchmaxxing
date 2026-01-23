import os

import yaml


def load_config(config_path: str) -> dict:
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    
    runpod_cfg = raw.get("runpod", raw)
    
    pod = runpod_cfg.get("pod", {})
    container = runpod_cfg.get("container", {})
    storage = runpod_cfg.get("storage", {})
    ports_cfg = runpod_cfg.get("ports", {})
    env = runpod_cfg.get("env", {})
    
    ports = []
    for p in ports_cfg.get("http", []):
        ports.append(f"{p}/http")
    for p in ports_cfg.get("tcp", []):
        ports.append(f"{p}/tcp")
    
    instance_type = pod.get("instance_type", "spot")
    spot = instance_type == "spot"
    
    ssh_key_path = runpod_cfg.get("ssh_key")
    
    return {
        "api_key": runpod_cfg.get("api_key"),
        "ssh_key_path": ssh_key_path,
        "name": pod.get("name"),
        "gpu_type": pod.get("gpu_type"),
        "gpu_count": pod.get("gpu_count"),
        "spot": spot,
        "secure_cloud": pod.get("secure_cloud", True),
        "image": container.get("image"),
        "container_disk_size": container.get("disk_size", 20),
        "disk_size": storage.get("volume_size"),
        "volume_mount_path": storage.get("mount_path", "/workspace"),
        "ports": ports if ports else None,
        "env": env if env else None,
    }

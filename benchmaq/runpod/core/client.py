"""RunPod API client for pod management."""

import os
import time
import subprocess
from typing import Optional

import runpod
from runpod.api.graphql import run_graphql_query


def set_api_key(key: str):
    runpod.api_key = key


def get_api_key() -> Optional[str]:
    return runpod.api_key or os.environ.get("RUNPOD_API_KEY")


def _format_env_for_graphql(env: dict) -> str:
    """Format env dict as GraphQL input array."""
    if not env:
        return "[]"
    items = [f'{{key: "{k}", value: "{v}"}}' for k, v in env.items()]
    return f"[{', '.join(items)}]"


def get_minimum_bid_price(gpu_type: str, gpu_count: int = 1, secure_cloud: bool = True) -> Optional[float]:
    """Query the current minimum bid price for a GPU type."""
    query = f"""
    query {{
        gpuTypes(input: {{id: "{gpu_type}"}}) {{
            id
            lowestPrice(input: {{gpuCount: 1, secureCloud: {str(secure_cloud).lower()}}}) {{
                minimumBidPrice
            }}
        }}
    }}
    """
    try:
        result = run_graphql_query(query)
        gpu_types = result.get("data", {}).get("gpuTypes", [])
        if gpu_types and gpu_types[0].get("lowestPrice"):
            return gpu_types[0]["lowestPrice"].get("minimumBidPrice")
    except Exception as e:
        print(f"Warning: Could not fetch minimum bid price: {e}")
    return None


def deploy(
    gpu_type: str,
    gpu_count: int,
    image: str,
    disk_size: int,
    container_disk_size: int = 20,
    volume_mount_path: str = "/workspace",
    secure_cloud: bool = True,
    spot: bool = True,
    bid_per_gpu: Optional[float] = None,
    env: Optional[dict] = None,
    name: Optional[str] = None,
    ports: Optional[str] = None,
    ssh_key_path: Optional[str] = None,
    wait_for_ready: bool = True,
    **kwargs,
) -> dict:
    """Deploy a RunPod GPU pod."""
    if env is None:
        env = {}
    
    if ports is None:
        ports = "8888/http,8000/http,22/tcp"
    elif isinstance(ports, list):
        ports = ",".join(ports)
    
    if name is None:
        name = f"{gpu_type}_{gpu_count}".replace(" ", "_")
    
    cloud_type = "SECURE" if secure_cloud else "ALL"
    env_graphql = _format_env_for_graphql(env)
    
    if spot:
        if bid_per_gpu is not None:
            bid = bid_per_gpu
        else:
            bid = get_minimum_bid_price(gpu_type, gpu_count, secure_cloud)
            if bid:
                print(f"Using spot instance with minimum bid: ${bid}/GPU/hour")
            else:
                bid = 0.0
        
        query = f"""
        mutation {{
            podRentInterruptable(input: {{
                bidPerGpu: {bid},
                cloudType: {cloud_type},
                gpuCount: {gpu_count},
                volumeInGb: {disk_size},
                containerDiskInGb: {container_disk_size},
                gpuTypeId: "{gpu_type}",
                name: "{name}",
                imageName: "{image}",
                ports: "{ports}",
                volumeMountPath: "{volume_mount_path}",
                startSsh: true,
                env: {env_graphql}
            }}) {{
                id
                imageName
                machineId
            }}
        }}
        """
        
        result = run_graphql_query(query)
        if "errors" in result:
            error_msg = result["errors"][0].get("message", "Unknown error")
            raise Exception(error_msg)
        pod = result["data"]["podRentInterruptable"]
    else:
        print("Using on-demand instance")
        query = f"""
        mutation {{
            podFindAndDeployOnDemand(input: {{
                cloudType: {cloud_type},
                gpuCount: {gpu_count},
                volumeInGb: {disk_size},
                containerDiskInGb: {container_disk_size},
                gpuTypeId: "{gpu_type}",
                name: "{name}",
                imageName: "{image}",
                ports: "{ports}",
                volumeMountPath: "{volume_mount_path}",
                startSsh: true,
                env: {env_graphql}
            }}) {{
                id
                imageName
                machineId
            }}
        }}
        """
        result = run_graphql_query(query)
        if "errors" in result:
            error_msg = result["errors"][0].get("message", "Unknown error")
            raise Exception(error_msg)
        pod = result["data"]["podFindAndDeployOnDemand"]
    
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    
    instance = {
        "id": pod_id,
        "name": name,
        "url": f"https://{pod_id}-8000.proxy.runpod.net",
    }
    
    if wait_for_ready:
        ssh_info = _wait_for_ssh(pod_id, ssh_key_path)
        if ssh_info:
            instance["ssh"] = ssh_info
            print(f"SSH ready: {ssh_info['command']}")
        else:
            raise Exception(f"Pod {name} failed to become SSH accessible")
    
    return instance


def _wait_for_ssh(pod_id: str, ssh_key_path: Optional[str] = None, timeout: int = 600) -> Optional[dict]:
    """Wait for pod SSH to be ready. Returns SSH info or None."""
    print("Waiting for pod to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            pod = runpod.get_pod(pod_id)
            if not pod:
                time.sleep(10)
                continue
            
            status = pod.get("desiredStatus")
            if status != "RUNNING":
                print(f"  Pod status: {status}")
                time.sleep(10)
                continue
            
            runtime = pod.get("runtime")
            if not runtime:
                time.sleep(10)
                continue
            
            # Get SSH info
            ports = runtime.get("ports", [])
            for port in ports:
                if port.get("privatePort") == 22:
                    ip = port.get("ip")
                    public_port = port.get("publicPort")
                    if ip and public_port:
                        key_path = ssh_key_path or "~/.ssh/id_ed25519"
                        ssh_info = {
                            "ip": ip,
                            "port": public_port,
                            "command": f"ssh root@{ip} -p {public_port} -i {key_path}"
                        }
                        # Test SSH connection
                        if _check_ssh(ip, public_port, ssh_key_path):
                            return ssh_info
                        print(f"  SSH not ready yet, retrying...")
            
            time.sleep(10)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(10)
    
    return None


def _check_ssh(ip: str, port: int, ssh_key_path: Optional[str] = None) -> bool:
    """Test SSH connection."""
    key_path = os.path.expanduser(ssh_key_path or "~/.ssh/id_ed25519")
    try:
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-i", key_path,
            "-p", str(port),
            f"root@{ip}",
            "echo ok"
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=15)
        return result.returncode == 0
    except Exception:
        return False


def delete(pod_id: Optional[str] = None, name: Optional[str] = None) -> dict:
    """Delete a RunPod pod."""
    if name and not pod_id:
        pods = runpod.get_pods()
        for pod in pods:
            if pod.get("name") == name:
                pod_id = pod["id"]
                break
        if not pod_id:
            raise Exception(f"Pod with name '{name}' not found")
    
    if not pod_id:
        raise Exception("Either pod_id or name is required")
    
    runpod.terminate_pod(pod_id)
    return {"status": "deleted", "id": pod_id, "name": name}

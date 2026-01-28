import os
import time
import subprocess
from typing import Optional

import runpod
from runpod.api.graphql import run_graphql_query
from tqdm import tqdm


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
    """Query the current minimum bid price for a GPU type (per GPU)."""
    # Query for 1 GPU to get the per-GPU price
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
    health_check_retries: int = 60,
    health_check_interval: float = 10.0,
    deploy_retries: int = 10,
    deploy_retry_interval: float = 30.0,
    **kwargs,
) -> dict:
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
        # Use podRentInterruptable for spot instances
        # Fetch minimum bid price if not specified
        if bid_per_gpu is not None:
            bid = bid_per_gpu
        else:
            bid = get_minimum_bid_price(gpu_type, gpu_count, secure_cloud)
            if bid:
                print(f"Using spot instance with minimum bid: ${bid}/GPU/hour")
            else:
                bid = 0.0  # Fallback to 0 if query fails
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
        
        # Retry loop for spot instance availability
        pod = None
        last_error = None
        for attempt in range(deploy_retries):
            try:
                result = run_graphql_query(query)
                if "errors" in result:
                    error_msg = result["errors"][0].get("message", "Unknown error")
                    if "no longer available" in error_msg.lower() or "could not be created" in error_msg.lower():
                        last_error = error_msg
                        print(f"Spot instance unavailable (attempt {attempt + 1}/{deploy_retries}), retrying in {deploy_retry_interval}s...")
                        time.sleep(deploy_retry_interval)
                        continue
                    else:
                        raise Exception(error_msg)
                pod = result["data"]["podRentInterruptable"]
                break
            except Exception as e:
                error_str = str(e)
                if "no longer available" in error_str.lower() or "could not be created" in error_str.lower():
                    last_error = error_str
                    print(f"Spot instance unavailable (attempt {attempt + 1}/{deploy_retries}), retrying in {deploy_retry_interval}s...")
                    time.sleep(deploy_retry_interval)
                    continue
                raise
        
        if pod is None:
            raise Exception(f"Failed to create spot instance after {deploy_retries} attempts. Last error: {last_error}")
    else:
        # Use podFindAndDeployOnDemand for on-demand instances
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
        pod = result["data"]["podFindAndDeployOnDemand"]
    
    pod_id = pod["id"]
    print(f"Pod created: {pod_id}")
    
    instance = {
        "id": pod_id,
        "name": name,
        "url": f"https://{pod_id}-8000.proxy.runpod.net",
    }
    
    if wait_for_ready:
        result = wait_for_pod(
            pod_id=pod_id,
            retry=health_check_retries,
            sleep_time=health_check_interval,
            ssh_key_path=ssh_key_path,
        )
        if not result["ready"]:
            raise Exception(f"Pod {name} failed to start after {health_check_retries} attempts.")
        
        print("\n✓ Done!")
        if result["ssh"]:
            instance["ssh"] = result["ssh"]
            print(f"  SSH: {result['ssh']['command']}")
    
    return instance


def list_pods() -> list:
    return runpod.get_pods()


def find(pod_id: str) -> dict:
    return runpod.get_pod(pod_id)


def find_by_name(name: str) -> Optional[dict]:
    pods = runpod.get_pods()
    for pod in pods:
        if pod.get("name") == name:
            return pod
    return None


def start(pod_id: str, gpu_count: int = 1) -> dict:
    return runpod.resume_pod(pod_id, gpu_count=gpu_count)


def stop(pod_id: str) -> dict:
    return runpod.stop_pod(pod_id)


def delete(pod_id: Optional[str] = None, name: Optional[str] = None) -> dict:
    if name and not pod_id:
        pod = find_by_name(name)
        if not pod:
            raise Exception(f"Pod with name '{name}' not found")
        pod_id = pod["id"]
    
    if not pod_id:
        raise Exception("Either pod_id or name is required")
    
    runpod.terminate_pod(pod_id)
    return {"status": "deleted", "id": pod_id, "name": name}


def get_ssh_info(pod_id: str, ssh_key_path: Optional[str] = None) -> Optional[dict]:
    pod = runpod.get_pod(pod_id)
    
    key_path = ssh_key_path or "~/.ssh/id_ed25519"
    
    if not pod:
        return None
    
    runtime = pod.get("runtime")
    
    if not runtime:
        return None
    
    ports = runtime.get("ports", [])
    
    for port in ports:
        if port.get("privatePort") == 22:
            ip = port.get("ip")
            public_port = port.get("publicPort")
            if ip and public_port:
                return {
                    "ip": ip,
                    "port": public_port,
                    "command": f"ssh root@{ip} -p {public_port} -i {key_path}"
                }
    
    return None


def check_ssh(ip: str, port: int, ssh_key_path: Optional[str] = None, timeout: float = 10.0) -> bool:
    key_path = os.path.expanduser(ssh_key_path or "~/.ssh/id_ed25519")
    
    try:
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={int(timeout)}",
            "-o", "BatchMode=yes",
            "-i", key_path,
            "-p", str(port),
            f"root@{ip}",
            "echo ok"
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=timeout + 5)
        return result.returncode == 0
    except Exception:
        return False


def wait_for_pod(
    pod_id: str,
    retry: int = 60,
    sleep_time: float = 10.0,
    ssh_key_path: Optional[str] = None,
) -> dict:
    pbar = tqdm(range(retry), desc="Waiting for pod")
    
    for i in pbar:
        try:
            pod = runpod.get_pod(pod_id)
            
            if not pod:
                pbar.set_description("Pod not found...")
                time.sleep(sleep_time)
                continue
            
            status = pod.get("desiredStatus")
            runtime = pod.get("runtime")
            
            if status != "RUNNING":
                pbar.set_description(f"Status: {status}")
                time.sleep(sleep_time)
                continue
            
            if not runtime:
                pbar.set_description("Waiting for runtime...")
                time.sleep(sleep_time)
                continue
            
            ssh_info = get_ssh_info(pod_id, ssh_key_path)
            
            if ssh_info:
                pbar.set_description(f"Trying SSH {ssh_info['ip']}:{ssh_info['port']}...")
                if check_ssh(ssh_info["ip"], ssh_info["port"], ssh_key_path):
                    pbar.set_description("Pod ready!")
                    return {"ready": True, "ssh": ssh_info}
            else:
                pbar.set_description("Waiting for SSH port...")
            
        except Exception as e:
            pbar.set_description(f"Error: {str(e)[:30]}")
        
        time.sleep(sleep_time)
    
    return {"ready": False, "ssh": None}

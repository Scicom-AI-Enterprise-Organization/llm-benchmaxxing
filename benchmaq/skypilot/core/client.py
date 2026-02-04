"""SkyPilot API client wrapper.

Provides functions to interact with SkyPilot for cluster management.

Prerequisites:
    Users must authenticate with SkyPilot before using these functions:
    - Run `sky auth` or
    - Set SKYPILOT_API_SERVER_URL and SKYPILOT_API_KEY environment variables
"""

import os
from typing import Dict, Any, Optional, List

import sky


def launch_cluster(
    task_yaml: str,
    cluster_name: str,
    down: bool = True,
    idle_minutes_to_autostop: Optional[int] = None,
) -> Dict[str, Any]:
    """Launch a SkyPilot cluster from task YAML string.
    
    Args:
        task_yaml: YAML string defining the SkyPilot task configuration.
        cluster_name: Name for the cluster.
        down: If True, tear down cluster after job completes.
        idle_minutes_to_autostop: Auto-stop after this many idle minutes.
    
    Returns:
        Dict with job_id and handle information.
    
    Raises:
        Various SkyPilot exceptions if launch fails.
    """
    task = sky.Task.from_yaml_str(task_yaml)
    
    request_id = sky.launch(
        task,
        cluster_name=cluster_name,
        down=down,
        idle_minutes_to_autostop=idle_minutes_to_autostop,
    )
    
    # Stream launch/provisioning logs and wait for completion
    job_id, handle = sky.stream_and_get(request_id)
    
    # Tail the job logs to show setup and run output
    if job_id is not None:
        print()
        print("=" * 64)
        print("STREAMING JOB LOGS")
        print("=" * 64)
        print()
        sky.tail_logs(cluster_name, job_id, follow=True)
    
    return {
        "job_id": job_id,
        "handle": handle,
    }


def teardown_cluster(cluster_name: str, purge: bool = False) -> None:
    """Tear down a SkyPilot cluster.
    
    Args:
        cluster_name: Name of the cluster to tear down.
        purge: If True, forcefully remove from SkyPilot's cluster table
               even if actual termination fails.
    """
    request_id = sky.down(cluster_name, purge=purge)
    sky.get(request_id)


def get_cluster_status(cluster_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Get status of SkyPilot clusters.
    
    Args:
        cluster_names: List of cluster names to query. If None, returns all clusters.
    
    Returns:
        List of cluster status dictionaries.
    """
    request_id = sky.status(cluster_names)
    return sky.get(request_id)


def stop_cluster(cluster_name: str) -> None:
    """Stop a SkyPilot cluster (preserves disk).
    
    Args:
        cluster_name: Name of the cluster to stop.
    """
    request_id = sky.stop(cluster_name)
    sky.get(request_id)


def start_cluster(cluster_name: str) -> None:
    """Start a stopped SkyPilot cluster.
    
    Args:
        cluster_name: Name of the cluster to start.
    """
    request_id = sky.start(cluster_name)
    sky.get(request_id)


def _find_ssh_key_for_cluster(cluster_name: Optional[str] = None) -> Optional[str]:
    """Find SSH private key for a SkyPilot cluster.
    
    SkyPilot stores SSH keys in ~/.sky/generated/ssh-keys/ with format:
    <cluster-name>.key
    
    Args:
        cluster_name: Optional cluster name to find specific key for.
        
    Returns:
        Path to SSH private key, or None if not found.
    """
    import glob
    
    sky_keys_dir = os.path.expanduser("~/.sky/generated/ssh-keys")
    
    # First, try to find key for specific cluster
    if cluster_name and os.path.isdir(sky_keys_dir):
        cluster_key = os.path.join(sky_keys_dir, f"{cluster_name}.key")
        if os.path.exists(cluster_key):
            return cluster_key
    
    # Try to find most recently modified key in SkyPilot directory
    if os.path.isdir(sky_keys_dir):
        key_files = glob.glob(os.path.join(sky_keys_dir, "*.key"))
        if key_files:
            # Sort by modification time, most recent first
            key_files.sort(key=os.path.getmtime, reverse=True)
            return key_files[0]
    
    # Fallback to standard SSH key locations
    key_locations = [
        "~/.ssh/sky-key",
        "~/.ssh/id_rsa",
        "~/.ssh/id_ed25519",
        "~/.ssh/id_ecdsa",
    ]
    
    for key_path in key_locations:
        expanded = os.path.expanduser(key_path)
        if os.path.exists(expanded):
            return expanded
    
    return None


def _find_ssh_key() -> Optional[str]:
    """Find a valid SSH private key (without cluster context)."""
    return _find_ssh_key_for_cluster(None)


def _extract_ssh_info_from_handle(handle: Any, debug: bool = False) -> Dict[str, Any]:
    """Extract SSH connection info from a SkyPilot handle.
    
    Args:
        handle: SkyPilot ResourceHandle object.
        debug: If True, print debug info about handle attributes.
        
    Returns:
        Dict with head_ip, ssh_user, ssh_key_path, ssh_port (some may be None).
    """
    result = {
        "head_ip": None,
        "ssh_user": None,
        "ssh_key_path": None,
        "ssh_port": 22,
    }
    
    if handle is None:
        return result
    
    if debug:
        print(f"[DEBUG] Handle type: {type(handle)}")
        print(f"[DEBUG] Handle attributes: {[a for a in dir(handle) if not a.startswith('_')]}")
    
    # Try to get head_ip
    result["head_ip"] = getattr(handle, "head_ip", None)
    
    # Try to get ssh_user from various attributes
    ssh_user = getattr(handle, "ssh_user", None)
    if ssh_user is None:
        # Try to get from launched_resources
        launched_resources = getattr(handle, "launched_resources", None)
        if launched_resources:
            ssh_user = getattr(launched_resources, "ssh_user", None)
            if debug:
                print(f"[DEBUG] launched_resources: {launched_resources}")
    
    # Try from cluster_info
    if ssh_user is None:
        cluster_info = getattr(handle, "cluster_info", None)
        if cluster_info:
            ssh_user = getattr(cluster_info, "ssh_user", None)
            if debug:
                print(f"[DEBUG] cluster_info: {cluster_info}")
                print(f"[DEBUG] cluster_info attrs: {[a for a in dir(cluster_info) if not a.startswith('_')]}")
    
    result["ssh_user"] = ssh_user
    
    # Try to get SSH key from various attributes
    ssh_key = None
    
    # Direct attribute
    ssh_key = getattr(handle, "ssh_private_key", None)
    
    # From credentials dict
    if ssh_key is None:
        credentials = getattr(handle, "credentials", None)
        if credentials:
            if isinstance(credentials, dict):
                ssh_key = credentials.get("ssh_private_key")
            else:
                ssh_key = getattr(credentials, "ssh_private_key", None)
            if debug:
                print(f"[DEBUG] credentials type: {type(credentials)}")
    
    # From cluster_info
    if ssh_key is None:
        cluster_info = getattr(handle, "cluster_info", None)
        if cluster_info:
            ssh_key = getattr(cluster_info, "ssh_private_key", None)
            # Also try ssh_key_path
            if ssh_key is None:
                ssh_key = getattr(cluster_info, "ssh_key_path", None)
    
    # Try stable_ssh_ports for the port
    ssh_ports = getattr(handle, "stable_ssh_ports", None)
    if ssh_ports and isinstance(ssh_ports, (list, tuple)) and len(ssh_ports) > 0:
        result["ssh_port"] = ssh_ports[0]
    
    result["ssh_key_path"] = ssh_key
    
    if debug:
        print(f"[DEBUG] Extracted SSH info: {result}")
    
    return result


def _get_cluster_ssh_info(cluster_name: str) -> Optional[Dict[str, Any]]:
    """Get SSH connection info for a SkyPilot cluster.
    
    Args:
        cluster_name: Name of the SkyPilot cluster.
        
    Returns:
        Dict with head_ip, ssh_user, ssh_key_path, or None if not available.
    """
    try:
        # Get cluster status with credentials
        request_id = sky.status(
            cluster_names=[cluster_name],
            _include_credentials=True,
        )
        clusters = sky.get(request_id)
        
        if not clusters:
            print(f"Cluster {cluster_name} not found")
            return None
        
        cluster = clusters[0]
        handle = cluster.get("handle")
        
        if handle is None:
            print(f"No handle found for cluster {cluster_name}")
            return None
        
        # Extract SSH info from handle
        ssh_info = _extract_ssh_info_from_handle(handle)
        
        if not ssh_info["head_ip"]:
            print(f"Could not get head IP for cluster {cluster_name}")
            return None
        
        # Apply defaults for missing values
        if not ssh_info["ssh_user"]:
            ssh_info["ssh_user"] = "ubuntu"  # Common default for cloud VMs
        
        if not ssh_info["ssh_key_path"]:
            ssh_info["ssh_key_path"] = _find_ssh_key()
        
        return ssh_info
        
    except Exception as e:
        print(f"Error getting SSH info for cluster {cluster_name}: {e}")
        return None


def download_results(
    cluster_name: str,
    remote_dir: str,
    local_dir: str,
    handle: Optional[Any] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Download benchmark results from SkyPilot cluster.
    
    Uses SSH/SCP to download files after refreshing the cluster status
    to ensure SSH config is up-to-date.
    
    Args:
        cluster_name: Name of the SkyPilot cluster.
        remote_dir: Path to results directory on the cluster.
        local_dir: Local directory to download results to.
        handle: Optional cluster handle (not used, kept for compatibility).
        debug: If True, print debug info.
        
    Returns:
        Dict with status and list of downloaded files.
    """
    import subprocess
    import time
    import glob
    
    # Normalize paths
    if remote_dir.startswith("./"):
        remote_path = remote_dir[2:]
    elif remote_dir.startswith("/"):
        remote_path = remote_dir
    else:
        remote_path = remote_dir
    
    local_dir_normalized = local_dir.lstrip("./")
    os.makedirs(local_dir_normalized, exist_ok=True)
    
    # Step 1: Refresh cluster status via sky status CLI
    # IMPORTANT: Must use CLI (not Python API) to update ~/.ssh/config
    print(f"Refreshing cluster status for {cluster_name}...")
    max_status_retries = 3
    cluster_ready = False
    
    for attempt in range(max_status_retries):
        try:
            # Use CLI command to ensure SSH config is updated
            # The Python API sky.status() does NOT update ~/.ssh/config
            status_cmd = ["sky", "status", cluster_name]
            if debug:
                print(f"[DEBUG] Running: {' '.join(status_cmd)}")
            
            result = subprocess.run(
                status_cmd, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if debug:
                print(f"[DEBUG] sky status stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
                if result.stderr:
                    print(f"[DEBUG] sky status stderr: {result.stderr[:200]}")
            
            # Check if cluster is UP from output
            if "UP" in result.stdout:
                cluster_ready = True
                if debug:
                    print(f"[DEBUG] Cluster status: UP")
                break
            elif cluster_name not in result.stdout and "No cluster" in result.stdout:
                error_msg = f"Cluster {cluster_name} not found"
                print(f"Error: {error_msg}")
                return {"status": "error", "error": error_msg}
            else:
                print(f"  Cluster not ready yet, waiting...")
                time.sleep(10)
                
        except subprocess.TimeoutExpired:
            print(f"  sky status timed out (attempt {attempt + 1})")
            if attempt < max_status_retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"  Failed to get cluster status (attempt {attempt + 1}): {e}")
            if attempt < max_status_retries - 1:
                time.sleep(5)
    
    if not cluster_ready:
        print("Warning: Could not confirm cluster is UP, attempting download anyway...")
    
    # Step 2: Download using SSH/SCP
    print(f"Downloading results via SSH/SCP...")
    return _download_results_via_ssh(cluster_name, remote_path, local_dir_normalized, debug)


def _download_results_via_ssh(
    cluster_name: str,
    remote_path: str,
    local_dir: str,
    debug: bool = False,
) -> Dict[str, Any]:
    """Download results using raw SSH/SCP commands with fallback patterns."""
    import subprocess
    import time
    
    max_retries = 5
    retry_delay = 10
    ssh_connected = False
    
    # File patterns to try in order (fallback logic)
    file_patterns = [
        f"{remote_path}/*.json {remote_path}/*.txt",      # Try json + txt first
        f"{remote_path}/*.jsonl {remote_path}/*.txt",     # Then jsonl + txt
        f"{remote_path}/*.txt",                            # Then just txt
    ]
    
    remote_files = []
    
    for attempt in range(max_retries):
        current_delay = retry_delay * (1.5 ** attempt)
        
        try:
            # Try each file pattern until we find files
            for pattern in file_patterns:
                check_cmd = [
                    "ssh", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=no",
                    cluster_name, 
                    f"ls {pattern} 2>/dev/null"
                ]
                if debug:
                    print(f"[DEBUG] Attempt {attempt + 1}: Running: {' '.join(check_cmd)}")
                
                result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=45)
                
                if debug:
                    print(f"[DEBUG] stdout: {result.stdout[:200] if result.stdout else '(empty)'}")
                    print(f"[DEBUG] stderr: {result.stderr[:200] if result.stderr else '(empty)'}")
                
                # Check for connection errors
                if "Connection refused" in result.stderr or "Connection timed out" in result.stderr or "No route to host" in result.stderr:
                    print(f"  SSH connection failed (attempt {attempt + 1}/{max_retries})")
                    if debug:
                        print(f"  [DEBUG] {result.stderr.strip()}")
                    break  # Break pattern loop to retry connection
                
                # Check for hostname resolution errors
                if "Could not resolve hostname" in result.stderr:
                    print(f"  SSH hostname not found (attempt {attempt + 1}/{max_retries})")
                    print(f"  Hint: Run 'sky status' to refresh SSH config")
                    break  # Break pattern loop to retry connection
                
                ssh_connected = True
                
                # Parse found files (filter out errors and empty lines)
                stdout_lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                found_files = [f for f in stdout_lines if not f.startswith("ls:") and f]
                
                if found_files:
                    remote_files = found_files
                    if debug:
                        print(f"[DEBUG] Found {len(remote_files)} files with pattern: {pattern}")
                    break  # Found files, stop trying patterns
            
            if remote_files or ssh_connected:
                break  # Either found files or connected successfully
                
            if attempt < max_retries - 1:
                print(f"  Retrying in {int(current_delay)} seconds...")
                time.sleep(current_delay)
            
        except subprocess.TimeoutExpired:
            print(f"  SSH timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(current_delay)
        except Exception as e:
            print(f"  Error: {e} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(current_delay)
    
    if not ssh_connected:
        error_msg = f"Could not establish SSH connection to cluster {cluster_name}"
        print(f"Error: {error_msg}")
        return {"status": "error", "error": error_msg}
    
    if not remote_files:
        return {
            "status": "success",
            "local_dir": local_dir,
            "files": [],
            "message": "No results files found on remote",
        }
    
    # Download files using scp
    print(f"Downloading {len(remote_files)} files from {cluster_name}:{remote_path}/ to {local_dir}/...")
    
    downloaded_files = []
    
    try:
        for remote_file in remote_files:
            filename = os.path.basename(remote_file)
            local_file = os.path.join(local_dir, filename)
            
            scp_cmd = ["scp", "-o", "StrictHostKeyChecking=no", f"{cluster_name}:{remote_file}", local_file]
            try:
                result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print(f"  Downloaded: {filename}")
                    downloaded_files.append(filename)
                else:
                    print(f"  Failed to download {filename}: {result.stderr}")
            except Exception as e:
                print(f"  Failed to download {filename}: {e}")
        
        return {
            "status": "success",
            "local_dir": local_dir,
            "files": downloaded_files,
        }
        
    except Exception as e:
        print(f"Error downloading results: {e}")
        return {"status": "error", "error": str(e)}

# Runpod Module

Deploy and manage RunPod GPU pods.

## CLI Usage

### Deploy a Pod

Create and start a new GPU pod on RunPod.

```bash
benchmaq runpod deploy config.yaml
```

Output:
```
Pod created: abc123xyz
✓ Done!
  SSH: ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519
```

### Get Pod Info

Retrieve pod status, IP address, and SSH connection details.

```bash
benchmaq runpod find config.yaml
```

Output:
```
Pod: my-pod (abc123xyz)
  Status: RUNNING
  SSH: ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519
```

### Start a Stopped Pod

Resume a previously stopped pod without losing data.

```bash
benchmaq runpod start config.yaml
```

### Delete a Pod

Terminate and remove a pod permanently.

```bash
benchmaq runpod delete config.yaml
```

## Python API

```python
import benchmaq

# RunPod end-to-end: deploy -> benchmark -> cleanup
benchmaq.runpod.bench("examples/5_config_runpod.yaml")

# RunPod deploy / delete
benchmaq.runpod.deploy("examples/4_remote_gpu_runpod.yaml")
benchmaq.runpod.delete("examples/4_remote_gpu_runpod.yaml")

# Pod utilities (requires RUNPOD_API_KEY env var)
pods = benchmaq.runpod.list_pods()
pod = benchmaq.runpod.find(pod_id="abc123")
pod = benchmaq.runpod.find_by_name(name="my-pod")
benchmaq.runpod.stop(pod_id="abc123")
benchmaq.runpod.start(pod_id="abc123")
```

### Multiprocessing (Parallel Benchmarks)

```python
from multiprocessing import Pool
# run_benchmark is a module-level function that supports multiprocessing (macOS spawn method)
from benchmaq.runpod import run_benchmark

configs = [
    "examples/5_config_runpod_multiprocess_1.yaml",
    "examples/5_config_runpod_multiprocess_2.yaml",
]

with Pool(processes=len(configs)) as pool:
    results = pool.map(run_benchmark, configs)
```

## Configuration

```yaml
runpod:
  runpod_api_key: "" # or export RUNPOD_API_KEY
  ssh_private_key: "/path/to/your/private/key"
  pod:
    name: "my-pod"
    gpu_type: "NVIDIA H100 80GB HBM3" # https://docs.runpod.io/references/gpu-types#gpu-types
    gpu_count: 2
    instance_type: on_demand # (spot|on_demand)
    secure_cloud: true
  container:
    image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" # docker image
    disk_size: 200 # temporary storage (GB)
  storage:
    volume_size: 200 # persistent storage (GB)
    mount_path: "/workspace"
  ports:
    http: [8888, 8000]
    tcp: [22]
  env:
    HF_HOME: "/workspace/hf_home" # HuggingFace cache directory
```

See [examples/](../../examples/) for more.

# Runpod CLI

Deploy and manage RunPod GPU pods.

## Usage

### Deploy a Pod

Create and start a new GPU pod on RunPod.

```bash
benchmaxxing runpod deploy config.yaml
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
benchmaxxing runpod find config.yaml
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
benchmaxxing runpod start config.yaml
```

### Delete a Pod

Terminate and remove a pod permanently.

```bash
benchmaxxing runpod delete config.yaml
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

# RunPod CLI

Deploy and manage RunPod GPU pods.

## Commands

| Command | Description |
|---------|-------------|
| `deploy` | Create and start a new pod |
| `delete` | Terminate and remove a pod |
| `find` | Get pod info (status, IP, SSH) |
| `start` | Resume a stopped pod |

## Usage

```bash
# Deploy a new pod
benchmaxxing runpod deploy config.yaml

# Get pod info
benchmaxxing runpod find config.yaml

# Delete pod
benchmaxxing runpod delete config.yaml

# Start stopped pod
benchmaxxing runpod start config.yaml
```

## Config Format

```yaml
api_key: "your-runpod-api-key"
ssh_key: "~/.ssh/id_ed25519"

pod:
  name: "my-pod"
  gpu_type: "NVIDIA H100 80GB HBM3"
  gpu_count: 2
  instance_type: spot

container:
  image: "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
  disk_size: 20

storage:
  volume_size: 200
  mount_path: "/workspace"

ports:
  http: [8888, 8000]
  tcp: [22]

env:
  HF_HOME: "/workspace/hf_home"
```

## Example Configs

| File | GPUs |
|------|------|
| `examples/2x_h100_sxm.yaml` | 2x H100 |
| `examples/4x_h100_sxm.yaml` | 4x H100 |
| `examples/8x_h100_sxm.yaml` | 8x H100 |

See [examples/](./examples/) for more configs.

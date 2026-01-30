# Runpod Module

End-to-end RunPod GPU benchmarking: deploy pod, run benchmarks, download results, delete pod.

## CLI Usage

```bash
benchmaq runpod bench config.yaml
```

This will:
1. Deploy a RunPod GPU pod
2. Run benchmarks on the pod
3. Download results to local machine
4. Delete the pod automatically

If you press `Ctrl+C`, the pod will still be cleaned up.

## Python API

```python
import benchmaq

# RunPod end-to-end: deploy -> benchmark -> cleanup
benchmaq.runpod.bench.from_yaml("examples/6_example_runpod_config.yaml")
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

# Benchmaxxing

Seamless scripts for LLM performance benchmaxxing.

## Supported Engines

- [x] [vLLM](./benchmaxxing/vllm/) - vLLM inference server
- [ ] TensorRT-LLM - *(coming soon)*
- [ ] SGLang - *(coming soon)*

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.11
source .venv/bin/activate

uv pip install -e .
```

## Quick Start: 2-Step Workflow

### Step 1: Deploy RunPod Instance

```bash
benchmaxxing runpod deploy config.yaml
```

Output:
```
Pod created: abc123xyz
✓ Done!
  SSH: ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519
```

### Step 2: Run Benchmarks

Copy the SSH info to your config's `remote` section, then:

```bash
benchmaxxing bench config.yaml
```

### Step 3: Cleanup

```bash
benchmaxxing runpod delete config.yaml
```

## Config Format

Single config file for both RunPod deployment and benchmarking:

```yaml
runpod:
  api_key: ""
  ssh_key: "~/.ssh/id_ed25519"
  
  pod:
    name: "my-benchmark-pod"
    gpu_type: "NVIDIA H100 80GB HBM3"
    gpu_count: 8
    instance_type: on_demand
    secure_cloud: true

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

remote:
  host: ""
  port: 22
  username: "root"
  key_filename: "~/.ssh/id_ed25519"
  uv:
    path: "~/.benchmark-venv"
    python_version: "3.11"
  dependencies:
    - vllm==0.11.0
    - pyyaml
    - requests
    - huggingface_hub

runs:
  - name: "llama-70b-benchmark"
    engine: "vllm"
    model:
      repo_id: "meta-llama/Llama-2-70b-hf"
      local_dir: "/workspace/models/llama-70b"
    serve:
      model_path: "/workspace/models/llama-70b"
      port: 8000
      gpu_memory_utilization: 0.9
      max_model_len: 4096
      dtype: "bfloat16"
      disable_log_requests: true
      tp_dp_pairs:
        - tp: 8
          dp: 1
          pp: 1
    bench:
      output_dir: "/workspace/benchmark_results"
      context_size: [1024, 2048]
      concurrency: [50, 100]
      num_prompts: [100]
      output_len: [128]
      save_results: true
```

## CLI Commands

### RunPod Management

```bash
benchmaxxing runpod deploy config.yaml      # Deploy pod
benchmaxxing runpod find config.yaml        # Get pod info
benchmaxxing runpod delete config.yaml      # Delete pod
benchmaxxing runpod start config.yaml       # Start stopped pod
```

### Benchmarking

```bash
benchmaxxing bench config.yaml              # Run benchmarks
```

## RunPod Templates

Pre-configured templates in `benchmaxxing/runpod/examples/`:

| File | GPUs | Description |
|------|------|-------------|
| `2x_h100_sxm.yaml` | 2x H100 | Development/testing |
| `4x_h100_sxm.yaml` | 4x H100 | Medium models |
| `8x_h100_sxm.yaml` | 8x H100 | Large models (70B+) |

## Examples

See [examples/](./examples/) for more config samples.

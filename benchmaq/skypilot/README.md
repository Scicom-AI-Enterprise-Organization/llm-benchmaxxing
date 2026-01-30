# SkyPilot Module

End-to-end SkyPilot GPU benchmarking: launch cluster, run benchmarks, download results, tear down cluster.

## Prerequisites

Users must authenticate with SkyPilot before using this module:

```bash
# Option 1: Use sky auth (interactive)
sky auth

# Option 2: Set environment variables for remote API server
export SKYPILOT_API_SERVER_URL="https://your-skypilot-api.example.com"
export SKYPILOT_API_KEY="your-api-key"
```

## CLI Usage

```bash
benchmaq sky bench --config config.yaml
# or short form:
benchmaq sky bench -c config.yaml
```

This will:
1. Launch a SkyPilot cluster on your configured cloud provider
2. Run benchmarks on the cluster
3. Download results automatically via SSH/SCP (if `save_result: true`)
4. Tear down the cluster automatically

If you press `Ctrl+C`, the cluster will still be cleaned up.

## Python API

```python
import benchmaq

# SkyPilot end-to-end: launch -> benchmark -> download -> cleanup
result = benchmaq.skypilot.bench.from_yaml("examples/7_example_skypilot_config.yaml")

print(result)
# {'status': 'success', 'cluster_name': '...', 'job_id': 1, 'results_dirs': ['./benchmark_results']}
```

## Configuration

The config file has two main sections:
- `skypilot`: SkyPilot task configuration (passed directly to sky.Task)
- `benchmark`: Benchmark configuration (read by benchmaq on the remote cluster)

```yaml
# SkyPilot task configuration
skypilot:
  name: my-benchmark-cluster
  workdir: .
  resources:
    accelerators: A100-80GB:2
    disk_size: 500
    any_of:
      - cloud: runpod
      - cloud: aws
      - cloud: gcp
  envs:
    HF_TOKEN: ""  # Set via env var or here for gated models
  setup: |
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    uv venv --python 3.11
    source .venv/bin/activate
    uv pip install "benchmaq[vllm] @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git"
  run: |
    export PATH="$HOME/.local/bin:$PATH"
    source .venv/bin/activate
    # $config is replaced with the --config path automatically
    benchmaq vllm bench $config

# Benchmark configuration (read by benchmaq on the cluster)
benchmark:
  - name: tp2_dp1
    engine: vllm
    model:
      repo_id: "Qwen/Qwen2.5-7B-Instruct"
      local_dir: "/workspace/model"
    serve:
      tensor_parallel_size: 2
      max_model_len: 8192
    bench:
      - backend: vllm
        dataset_name: random
        random_input_len: 1024
        num_prompts: 100
    results:
      save_result: true
      result_dir: "./benchmark_results"
```

## Multiple Benchmarks

You can define multiple benchmark configurations with different result directories:

```yaml
benchmark:
  - name: tp1_dp1
    results:
      save_result: true
      result_dir: "./benchmark_results_tp1"
    # ...

  - name: tp2_dp1
    results:
      save_result: true
      result_dir: "./benchmark_results_tp2"
    # ...
```

All result directories will be downloaded automatically.

## How It Works

1. **Local**: `benchmaq sky bench --config config.yaml` reads the `skypilot:` section
2. **Substitution**: `$config` in the YAML is replaced with `config.yaml`
3. **SkyPilot**: Launches a cluster with the specified resources and runs the task
4. **Remote**: The `run:` command executes `benchmaq vllm bench config.yaml`
5. **Remote**: `benchmaq vllm bench` reads the `benchmark:` section and runs benchmarks
6. **Download**: Results are downloaded from `~/sky_workdir/<result_dir>` via SSH/SCP
7. **Cleanup**: Cluster is torn down after results are downloaded

## Supported Clouds

SkyPilot supports many cloud providers:
- AWS
- GCP
- Azure
- RunPod
- Lambda Labs
- Kubernetes
- And more...

See [SkyPilot documentation](https://docs.skypilot.co/) for the full list and configuration.

## Example Configs

See [examples/7_example_skypilot_config.yaml](../../examples/7_example_skypilot_config.yaml) for a complete example.

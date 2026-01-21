# vLLM Benchmark

Benchmarking tool for vLLM models.

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Make script executable
chmod u+x run.sh

# Download model
huggingface-cli download Scicom-intl/gpt-oss-120b-Malaysian-Reasoning-SFT-v0.1
```

## Config

Create a YAML config in `runs/`:

```yaml
runs:
  - name: "my-benchmark"
    serve:
      model_path: "Scicom-intl/gpt-oss-120b-Malaysian-Reasoning-SFT-v0.1"
      port: 8000
      gpu_memory_utilization: 0.9
      max_model_len: 12000
      max_num_seqs: 256
      dtype: "bfloat16"
      disable_log_requests: true
      enable_expert_parallel: true
      tp_dp_pairs:
        - tp: 4
          dp: 1
          pp: 1
    bench:
      output_dir: "./benchmark_results"
      context_size: [1024, 2048, 4096]
      concurrency: [100]
      num_prompts: [100]
      output_len: [128]
```

### Multiple Runs

Add multiple entries under `runs:` for different output naming or benchmarking different models (make sure vLLM supports both models).

```yaml
runs:
  - name: "run-1"
    serve:
      model_path: "/path/to/model-a"
      port: 8000
      gpu_memory_utilization: 0.9
      max_model_len: 12000
      max_num_seqs: 256
      dtype: "bfloat16"
      disable_log_requests: true
      enable_expert_parallel: true
      tp_dp_pairs:
        - tp: 8
          dp: 1
          pp: 1
    bench:
      output_dir: "./benchmark_results"
      context_size: [1024]
      concurrency: [100]
      num_prompts: [100]
      output_len: [128]
  
  - name: "run-2"
    serve:
      model_path: "/path/to/model-b"
      port: 8000
      gpu_memory_utilization: 0.9
      max_model_len: 12000
      max_num_seqs: 256
      dtype: "bfloat16"
      disable_log_requests: true
      enable_expert_parallel: true
      tp_dp_pairs:
        - tp: 2
          dp: 4
          pp: 1
        - tp: 4
          dp: 2
          pp: 1
    bench:
      output_dir: "./benchmark_results_segment_B"
      context_size: [1024, 2048]
      concurrency: [100]
      num_prompts: [100]
      output_len: [128]
```

## Run Benchmark

```bash
# Using default config.yaml
./run.sh

# Using custom config (auto-resolves from runs/ folder)
./run.sh my-config.yaml
```

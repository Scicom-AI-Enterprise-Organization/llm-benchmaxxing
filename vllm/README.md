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

# Run benchmark
./run.sh gpt-oss-120b-session-1.yaml
```

## Config

Create a YAML config in `runs/`:

```yaml
runs:
  - name: "my-benchmark"
    model_path: "Scicom-intl/gpt-oss-120b-Malaysian-Reasoning-SFT-v0.1"
    port: 8000
    output_dir: "./benchmark_results"
    tp_dp_pairs:
      - tp: 8
        dp: 1
        pp: 1
      - tp: 4
        dp: 2
        pp: 1
      - tp: 2
        dp: 4
        pp: 1
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
    model_path: "/path/to/model-a"
    output_dir: "./benchmark_results"
    tp_dp_pairs:
      - tp: 8
        dp: 1
        pp: 1
    context_size: [1024]
    concurrency: [100]
    num_prompts: [100]
    output_len: [128]
  
  - name: "run-2"
    model_path: "/path/to/model-b"
    output_dir: "./benchmark_results"
    tp_dp_pairs:
      - tp: 2
        dp: 4
        pp: 1
      - tp: 4
        dp: 2
        pp: 1
    context_size: [1024, 2048]
    concurrency: [100]
    num_prompts: [100]
    output_len: [128]
```

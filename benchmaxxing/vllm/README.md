# vLLM Engine

Benchmarking module for [vLLM](https://github.com/vllm-project/vllm).

## How It Works

1. Starts vLLM server
2. Runs benchmarks across all parameter combinations
3. Saves results to JSON
4. Repeats for each TP/DP/PP configuration

## Usage

```bash
# Install with vllm
uv pip install "benchmaxxing[vllm] @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaxxing.git"

# Download model
huggingface-cli download <huggingface_model_path> \
  --local-dir /download/dir

# single run
benchmaxxing bench examples/1_run_single.yaml

# multiple run
benchmaxxing bench examples/2_run_multiple.yaml
```
## Config Format

```yaml
runs:
  - name: "my-benchmark"
    engine: "vllm"

    model:
      repo_id: "meta-llama/Llama-2-7b-hf"      # HuggingFace model repo
      local_dir: "/path/to/model"              # optional, custom local path

    vllm_serve:
      model_path: "meta-llama/Llama-2-7b-hf"
      port: 8000
      gpu_memory_utilization: 0.9
      max_model_len: 4096
      max_num_seqs: 256
      dtype: "bfloat16"
      disable_log_requests: true
      enable_expert_parallel: false
      parallelism_pairs:
        - tensor_parallel: 1
          data_parallel: 1
          pipeline_parallel: 1

    benchmark:
      save_results: true
      output_dir: "./results"
      context_size: [512, 1024]
      concurrency: [50, 100]
      num_prompts: [100]
      output_len: [128]
```

## Model Configuration

The `model` section controls where models are downloaded/loaded from:

| Field | Required | Description |
|-------|----------|-------------|
| `repo_id` | Yes | HuggingFace model repository ID |
| `local_dir` | No | Custom local path for model storage |

**When to use `local_dir`:**

- **Not needed** if you set `HF_HOME` environment variable (e.g., in RunPod config). Models will be cached automatically in `$HF_HOME/hub/`.
- **Use it** when you need a specific path (e.g., shared storage, pre-downloaded models, or custom mount points).

## Output

Results saved as JSON in `output_dir`:

```
results/
├── my-benchmark_TP1_DP1_CTX512_C50_P100_O128.json
└── ...
```

Metrics: TTFT, TPOT, ITL, E2EL, throughput.

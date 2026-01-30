# vLLM Engine

Benchmarking module for [vLLM](https://github.com/vllm-project/vllm).

## How It Works

1. Starts vLLM server
2. Runs benchmarks across all parameter combinations
3. Saves results to JSON
4. Repeats for each TP/DP/PP configuration

## Usage

### CLI

```bash
# Install with vllm
uv pip install "benchmaq[vllm] @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git"

# Run benchmark (local)
benchmaq vllm bench examples/5_example_local_config.yaml

# Run benchmark (remote via SSH)
benchmaq vllm bench examples/5_example_remote_config.yaml
```

### Python API

```python
import benchmaq

# Run benchmark (local or remote SSH)
benchmaq.vllm.bench.from_yaml("examples/5_example_local_config.yaml")
benchmaq.vllm.bench.from_yaml("examples/5_example_remote_config.yaml")
```
## Config Format

```yaml
benchmark:
  - name: tp2_dp1
    engine: vllm

    model:
      repo_id: "meta-llama/Llama-3.1-8B-Instruct"
      local_dir: "/workspace/models/llama-3.1-8b"
      hf_token: ""  # Use HF_TOKEN env var instead
    
    serve:
      tensor_parallel_size: 2
      max_model_len: 8192
      max_num_seqs: 128
      gpu_memory_utilization: 0.9
      disable_log_requests: true
    
    bench:
      - backend: vllm
        endpoint: /v1/completions
        dataset_name: random
        random_input_len: 512
        random_output_len: 128
        num_prompts: 50
        max_concurrency: 50
        request_rate: inf
        ignore_eos: true
      
      - backend: vllm
        endpoint: /v1/completions
        dataset_name: random
        random_input_len: 1024
        random_output_len: 256
        num_prompts: 50
        max_concurrency: 50
        request_rate: inf
        ignore_eos: true
    
    results:
      save_result: true
      result_dir: "./benchmark_results"
      save_detailed: true
```

## Output

Results saved as JSON in `result_dir`:

```
benchmark_results/
├── tp2_dp1_TP2_IN512_OUT128_C50.json
├── tp2_dp1_TP2_IN1024_OUT256_C50.json
└── ...
```

Metrics: TTFT, TPOT, ITL, E2EL, throughput.

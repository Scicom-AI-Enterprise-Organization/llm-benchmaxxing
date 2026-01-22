# vLLM Engine

Benchmarking module for [vLLM](https://github.com/vllm-project/vllm) inference server.

## How It Works

1. Starts a vLLM server with your specified configuration (model, TP/DP/PP, etc.)
2. Waits for server to be healthy
3. Runs benchmarks across all combinations of context sizes, concurrency levels, etc.
4. Saves results to JSON files
5. Stops the server and moves to the next TP/DP configuration

## Module Structure

```
vllm/
├── __init__.py         # Entry point - run(config) function
└── core/
    ├── server.py       # VLLMServer class (start/stop/health check)
    └── benchmark.py    # run_benchmark() function
```

## Config Options

### serve (Server Configuration)

| Option | Type | Description |
|--------|------|-------------|
| `model_path` | string | HuggingFace model path or local path |
| `port` | int | Server port (default: 8000) |
| `gpu_memory_utilization` | float | GPU memory usage ratio (0.0-1.0) |
| `max_model_len` | int | Maximum sequence length |
| `max_num_seqs` | int | Maximum concurrent sequences |
| `dtype` | string | Data type (`bfloat16`, `float16`, `auto`) |
| `disable_log_requests` | bool | Disable request logging |
| `enable_expert_parallel` | bool | Enable expert parallelism (for MoE models) |
| `tp_dp_pairs` | list | List of TP/DP/PP configurations to test |

### bench (Benchmark Configuration)

| Option | Type | Description |
|--------|------|-------------|
| `output_dir` | string | Directory to save benchmark results |
| `context_size` | list[int] | Input context sizes to test |
| `concurrency` | list[int] | Concurrency levels to test |
| `num_prompts` | list[int] | Number of prompts per benchmark |
| `output_len` | list[int] | Output token lengths to test |

## Example Config

```yaml
runs:
  - name: "my-benchmark"
    engine: "vllm"
    serve:
      model_path: "meta-llama/Llama-2-7b-hf"
      port: 8000
      gpu_memory_utilization: 0.9
      max_model_len: 4096
      max_num_seqs: 256
      dtype: "bfloat16"
      disable_log_requests: true
      enable_expert_parallel: false
      tp_dp_pairs:
        - tp: 1
          dp: 1
          pp: 1
    bench:
      output_dir: "./benchmark_results"
      context_size: [512, 1024, 2048]
      concurrency: [50, 100]
      num_prompts: [100]
      output_len: [128]
```

## Output

Benchmark results are saved as JSON files in `output_dir`:

```
benchmark_results/
├── my-benchmark_TP1_DP1_CTX512_C50_P100_O128.json
├── my-benchmark_TP1_DP1_CTX512_C100_P100_O128.json
├── my-benchmark_TP1_DP1_CTX1024_C50_P100_O128.json
└── ...
```

Each JSON contains metrics like:
- TTFT (Time to First Token)
- TPOT (Time per Output Token)
- ITL (Inter-Token Latency)
- E2EL (End-to-End Latency)
- Throughput (tokens/sec)

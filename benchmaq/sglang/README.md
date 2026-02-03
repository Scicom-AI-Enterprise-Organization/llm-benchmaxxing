# SGLang Engine

Benchmarking module for [SGLang](https://github.com/sgl-project/sglang).

## How It Works

1. Starts SGLang server via `python -m sglang.launch_server`
2. Runs benchmarks via `python -m sglang.bench_serving`
3. Saves results to JSONL
4. Repeats for each TP/DP configuration

## Usage

### CLI

```bash
# Install with sglang
uv pip install "benchmaq[sglang] @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git"

# Run benchmark (remote via SSH)
benchmaq sglang bench examples/8_example_sglang_remote_config.yaml
```

### Python API

```python
import benchmaq

# Run benchmark (local or remote SSH)
benchmaq.sglang.bench.from_yaml("examples/8_example_sglang_remote_config.yaml")
```

## Config Format

```yaml
benchmark:
  - name: tp2_dp1
    engine: sglang

    model:
      repo_id: "meta-llama/Llama-3.1-8B-Instruct"
      local_dir: "/workspace/models/llama-3.1-8b"
      # hf_token: ""  # Use HF_TOKEN env var instead
    
    serve:
      # Parallelism
      tensor_parallel_size: 2        # --tensor-parallel-size
      # data_parallel_size: 1        # --data-parallel-size
      
      # Memory
      mem_fraction_static: 0.9       # --mem-fraction-static
      context_length: 32000          # --context-length
      
      # Server
      host: "0.0.0.0"                # --host
      port: 30000                    # --port
      trust_remote_code: true        # --trust-remote-code
      log_level: info                # --log-level
    
    bench:
      # Random dataset benchmark
      - backend: sglang              # or sglang-oai, sglang-oai-chat
        dataset_name: random
        random_input_len: 512
        random_output_len: 128
        num_prompts: 50
        max_concurrency: 50
        request_rate: inf            # burst mode
      
      # ShareGPT dataset benchmark
      - backend: sglang
        dataset_name: sharegpt
        num_prompts: 100
        max_concurrency: 50
        sharegpt_output_len: 256
    
    results:
      save_result: true
      result_dir: "./benchmark_results"
      output_details: true
```

## Server Arguments (serve:)

Arguments use exact SGLang CLI names (snake_case in YAML → kebab-case in CLI).

### Common Server Arguments

| YAML Key | CLI Argument | Description |
|----------|--------------|-------------|
| `tensor_parallel_size` | `--tensor-parallel-size` | Tensor parallelism size |
| `data_parallel_size` | `--data-parallel-size` | Data parallelism size |
| `mem_fraction_static` | `--mem-fraction-static` | GPU memory fraction (default: 0.9) |
| `context_length` | `--context-length` | Max context length |
| `host` | `--host` | Server host (default: 127.0.0.1) |
| `port` | `--port` | Server port (default: 30000) |
| `dtype` | `--dtype` | Data type: auto, half, bfloat16, float |
| `quantization` | `--quantization` | Quantization: awq, fp8, gptq, etc. |
| `trust_remote_code` | `--trust-remote-code` | Allow custom model code |
| `log_level` | `--log-level` | Logging level: info, debug, warning |

See [SGLang Server Arguments](https://docs.sglang.io/advanced_features/server_arguments.html) for full list.

## Benchmark Arguments (bench:)

Arguments use exact SGLang CLI names (snake_case in YAML → kebab-case in CLI).

### Backend Options

| Backend | Endpoint |
|---------|----------|
| `sglang` | POST /generate |
| `sglang-oai` | POST /v1/completions |
| `sglang-oai-chat` | POST /v1/chat/completions |

### Common Benchmark Arguments

| YAML Key | CLI Argument | Description |
|----------|--------------|-------------|
| `backend` | `--backend` | Backend type (see above) |
| `dataset_name` | `--dataset-name` | Dataset: random, sharegpt, random-ids, image |
| `num_prompts` | `--num-prompts` | Number of requests |
| `random_input_len` | `--random-input-len` | Input token length (random dataset) |
| `random_output_len` | `--random-output-len` | Output token length (random dataset) |
| `max_concurrency` | `--max-concurrency` | Max concurrent requests |
| `request_rate` | `--request-rate` | Requests/sec (use `inf` for burst) |
| `sharegpt_output_len` | `--sharegpt-output-len` | Override output length (sharegpt) |

See [SGLang Bench Serving Guide](https://docs.sglang.io/developer_guide/bench_serving.html) for full list.

## Results Configuration

| Argument | Description |
|----------|-------------|
| `save_result` | Save results to file |
| `result_dir` | Directory for result files |
| `output_details` | Include per-request details (ttfts, itls, etc.) |

## Output

Results saved as JSONL in `result_dir`:

```
benchmark_results/
├── tp2_dp1_in512_out128_p50_c50_abc123.jsonl   # Structured metrics
├── tp2_dp1_in512_out128_p50_c50_abc123.txt     # Console log
└── ...
```

### Metrics

- Request throughput (req/s)
- Input/Output token throughput (tok/s)
- Time to First Token (TTFT) - mean/median/p99
- Inter-Token Latency (ITL) - mean/median/p95/p99
- End-to-End Latency (E2EL) - mean/median/p99
- TPOT (time per output token after first)

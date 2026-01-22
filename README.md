# Benchmaxxing

Seamless scripts for LLM performance benchmaxxing.

## Supported Engines

- [x] [vLLM](./benchmaxxing/vllm/) - vLLM inference server
- [ ] TensorRT-LLM - *(coming soon)*
- [ ] SGLang - *(coming soon)*

## Usage

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.11
source .venv/bin/activate

```

### Running Remotely

```bash
# Install base package
uv pip install git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmark.git

# Run benchmark on remote GPU server
benchmaxxing examples/remote_gpu.yaml
```

### Running on GPU Server

```bash
# Install with vllm
uv pip install "benchmaxxing[vllm] @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmark.git"

# Download model
huggingface-cli download <huggingface_model_path> \
  --local-dir /download/dir

# Run benchmark locally
benchmaxxing examples/run_single.yaml
```

## Config Format Examples

### vLLM Benchmarking config format

```yaml
runs:
  - name: "run name"
    engine: "vllm"
    serve:
      model_path: "/model/path"
      port: 8000
      gpu_memory_utilization: 0.9
      max_model_len: 12000
      max_num_seqs: 256
      dtype: "bfloat16"
      disable_log_requests: true
      enable_expert_parallel: false
      tp_dp_pairs:
        - tp: 4
          dp: 1
          pp: 1
    bench:
      save_results: false
      output_dir: "./benchmark_results"
      context_size: [1024, 2048, 4096, 8192]
      concurrency: [100]
      num_prompts: [100]
      output_len: [128]
```

### SGLang Benchmarking config format

```bash
# coming soon
```

### TensorRT-LLM Benchmarking config format

```bash
# coming soon
```

### Remote Execution

Run benchmarks on remote GPU servers locally. Add a `remote` section to your config:

```yaml
remote:
  host: "gpu-server.example.com"
  username: "ubuntu"
  password: "your-password"
  uv:
    path: "~/.benchmark-venv"
    python_version: "3.11"
  dependencies:
    - vllm==0.11.0
    - pyyaml
    - requests
    - huggingface_hub

runs:
  - name: "remote-benchmark"
    engine: "vllm"
    # ... rest of config
```
See [examples/](./examples/) for more config samples.

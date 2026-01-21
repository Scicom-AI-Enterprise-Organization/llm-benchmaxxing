# vLLM Benchmark Script

Simple benchmarking tool for vLLM models.

## Pre-requisites

```bash
# Install yq
wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq
chmod +x /usr/bin/yq

# Install jq
apt-get install jq -y
```

## Quick Start

### 1. Start vLLM Server

```bash
vllm serve /path/to/your/model \
    --tensor-parallel-size 4 \
    --data-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 12000 \
    --max-num-seqs 256 \
    --dtype bfloat16 \
    --port 8000 \
    --disable-log-requests
```

### 2. Configure Benchmark

Edit `config.yaml` or create a new config:

```yaml
runs:
  - name: "my-model-run1"
    model_path: "/path/to/model"
    port: 8000
    output_dir: "./benchmark_results"
    tp_dp_pairs:
      - tp: 4
        dp: 2
        pp: 1
    context_size: [1024, 2048, 4096]
    concurrency: [100]
    num_prompts: [100]
    output_len: [128]
```

### 3. Run Benchmark

```bash
# Using default config.yaml
./benchmark.sh

# Using custom config
./benchmark.sh my-config.yaml
```

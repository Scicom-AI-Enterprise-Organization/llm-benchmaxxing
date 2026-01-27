# LLM-Benchmaq

Seamless scripts for LLM performance benchmarking, written in Northern Malaysia slang, with end 'k' sounds replaced by a 'q' sound.

## Supported Engines

- [x] [vLLM](./benchmaxxing/vllm/) - vLLM inference server
- [ ] TensorRT-LLM - *(coming soon)*
- [ ] SGLang - *(coming soon)*

## Installation

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv --python 3.11
source .venv/bin/activate

uv pip install "benchmaxxing @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaq.git"
```

## Usage

### 1. Benchmark locally (GPU Server)

```bash
# Install with vllm
uv pip install "benchmaxxing[vllm] @ git+https://github.com/Scicom-AI-Enterprise-Organization/llm-benchmaxxing.git"

# single run
benchmaxxing bench examples/1_run_single.yaml

# multiple run
benchmaxxing bench examples/2_run_multiple.yaml
```

### 2. Benchmark Remotely via SSH

```bash
benchmaxxing bench examples/3_remote_gpu_ssh_password.yaml
```

### 3. Benchmark Remotely on Runpod

#### Deploy RunPod Instance

```bash
benchmaxxing runpod deploy examples/4_remote_gpu_runpod.yaml
```

Output:
```
Pod created: abc123xyz
✓ Done!
  SSH: ssh root@1.2.3.4 -p 12345 -i ~/.ssh/id_ed25519
```

#### Run Benchmarks

Copy the SSH info to your config's `remote` section, then:

```bash
benchmaxxing bench examples/4_remote_gpu_runpod.yaml
```

#### Delete RunPod Instance

```bash
benchmaxxing runpod delete examples/4_remote_gpu_runpod.yaml
```

## Config Format

See [examples/](./examples/) for more config samples.

# LLM Benchmarking

Seamless scripts for LLM performance benchmarking. This repository provides ready-to-use benchmarking tools for various inference frameworks.

## Table of Contents

- [x] [vLLM](./vllm/) - Benchmark scripts for vLLM inference server
- [ ] TensorRT-LLM - *(coming soon)*
- [ ] SGLang - *(coming soon)*


## Structure

```
llm-benchmarking/
├── vllm/                    # vLLM benchmarking
│   ├── benchmark.sh         
│   ├── config.yaml          
│   └── <model>/             
│       ├── config.yaml
│       └── benchmark_results/
├── tensorrt-llm/            # (coming soon)
└── sglang/                  # (coming soon)
```

## Metrics

Each benchmark captures:
- **TTFT** - Time to First Token
- **TPOT** - Time per Output Token
- **ITL** - Inter-token Latency
- **E2EL** - End-to-end Latency
- **Throughput** - Requests/s, Tokens/s

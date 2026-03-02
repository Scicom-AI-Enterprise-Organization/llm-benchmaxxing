# STT Benchmark Results

Whisper STT benchmarks using vLLM across 4 GPU types on RunPod via SkyPilot.

**Test config:** 100 requests, 100 max concurrency, 100 req/s, 12.27s audio file, single GPU per run.

**Attention backends:**
- A100, H100, H200: default (FlashAttention2 / FlashInfer)
- B200 (Blackwell sm_100): encoder uses `TORCH_SDPA`, decoder uses `TRITON_ATTN` (FA2/FlashInfer PTX not compiled for sm_100)

---

## Whisper Large V3

### FP16 (`dtype: float16`)

| GPU | Backend | Total Time (s) | Throughput (req/s) | Mean Proc. Time (s) | Median Proc. Time (s) | Mean RTF | Min RTF | Max RTF |
|-----|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A100-80GB | default | 7.59 | 13.18 | 4.68 | 4.79 | 0.382 | 0.189 | 0.534 |
| H100-SXM | default | 2.57 | 38.97 | 1.23 | 1.27 | 0.100 | 0.057 | 0.136 |
| H200-SXM | default | 2.22 | 45.14 | 0.94 | 0.96 | 0.077 | 0.042 | 0.107 |
| B200 | TORCH_SDPA + TRITON_ATTN | 3.05 | 32.80 | 1.39 | 1.35 | 0.113 | 0.057 | 0.165 |

### FP32 (`dtype: float32`)

| GPU | Backend | Total Time (s) | Throughput (req/s) | Mean Proc. Time (s) | Median Proc. Time (s) | Mean RTF | Min RTF | Max RTF |
|-----|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A100-80GB | default | 6.69 | 14.95 | 4.07 | 4.19 | 0.332 | 0.163 | 0.461 |
| H100-SXM | default | 4.12 | 24.30 | 2.31 | 2.34 | 0.188 | 0.099 | 0.274 |
| H200-SXM | default | 2.90 | 34.46 | 1.41 | 1.45 | 0.115 | 0.063 | 0.160 |
| B200 | TORCH_SDPA + TRITON_ATTN | 2.64 | 37.87 | 1.26 | 1.30 | 0.102 | 0.055 | 0.140 |

---

## Whisper Large V3 Turbo

### FP16 (`dtype: float16`)

| GPU | Backend | Total Time (s) | Throughput (req/s) | Mean Proc. Time (s) | Median Proc. Time (s) | Mean RTF | Min RTF | Max RTF |
|-----|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A100-80GB | default | 3.88 | 25.81 | 2.26 | 2.37 | 0.184 | 0.102 | 0.238 |
| H100-SXM | default | 1.64 | 61.04 | 0.57 | 0.58 | 0.047 | 0.027 | 0.067 |
| H200-SXM | default | 1.45 | 69.14 | 0.41 | 0.41 | 0.034 | 0.014 | 0.051 |
| B200 | TORCH_SDPA + TRITON_ATTN | 2.59 | 38.61 | 1.15 | 1.21 | 0.093 | 0.054 | 0.136 |

### FP32 (`dtype: float32`)

| GPU | Backend | Total Time (s) | Throughput (req/s) | Mean Proc. Time (s) | Median Proc. Time (s) | Mean RTF | Min RTF | Max RTF |
|-----|---------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A100-80GB | default | 3.46 | 28.91 | 1.98 | 2.08 | 0.161 | 0.093 | 0.207 |
| H100-SXM | default | 2.84 | 35.23 | 1.42 | 1.47 | 0.116 | 0.069 | 0.167 |
| H200-SXM | default | 2.36 | 42.34 | 1.06 | 1.09 | 0.086 | 0.052 | 0.125 |
| B200 | TORCH_SDPA + TRITON_ATTN | 2.22 | 45.16 | 0.98 | 1.03 | 0.080 | 0.051 | 0.109 |

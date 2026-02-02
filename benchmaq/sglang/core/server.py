import signal
import socket
import subprocess
import time
from typing import Any

import requests


class SGLangServer:
    """SGLang Server wrapper that accepts dynamic kwargs for CLI args.
    
    All kwargs are converted to CLI arguments using the pattern:
    - key_name -> --key-name (snake_case to kebab-case)
    - Boolean True -> flag added (--key-name)
    - Boolean False -> flag omitted
    - Other values -> --key-name value
    
    Example:
        SGLangServer(
            model_path="meta-llama/Llama-3-8B-Instruct",
            port=30000,
            tensor_parallel_size=4,
            mem_fraction_static=0.9,
            trust_remote_code=True
        )
        
        Produces: python -m sglang.launch_server --model-path meta-llama/Llama-3-8B-Instruct 
                  --port 30000 --tensor-parallel-size 4 --mem-fraction-static 0.9 --trust-remote-code
    
    Common SGLang Arguments (use exact names from SGLang docs):
        Model & Tokenizer:
            model_path: Model path or HuggingFace repo ID (required)
            tokenizer_path: Custom tokenizer path
            context_length: Max context length (--context-length)
            trust_remote_code: Allow custom model code (--trust-remote-code)
            revision: Specific model version (--revision)
        
        Parallelism:
            tensor_parallel_size: Tensor parallelism size (--tensor-parallel-size)
            data_parallel_size: Data parallelism size (--data-parallel-size)
            pipeline_parallel_size: Pipeline parallelism size (--pipeline-parallel-size)
        
        Memory:
            mem_fraction_static: Fraction of memory for static allocation (--mem-fraction-static)
            max_running_requests: Max concurrent requests (--max-running-requests)
            max_total_tokens: Max tokens in memory pool (--max-total-tokens)
            chunked_prefill_size: Max tokens per chunk (--chunked-prefill-size)
        
        Quantization:
            dtype: Data type (--dtype): auto, half, bfloat16, float
            quantization: Quantization method (--quantization): awq, fp8, gptq, etc.
            kv_cache_dtype: KV cache data type (--kv-cache-dtype)
        
        Serving:
            host: Server host (--host, default: 127.0.0.1)
            port: Server port (--port, default: 30000)
            api_key: API authentication key (--api-key)
            served_model_name: Override model name in API (--served-model-name)
            chat_template: Custom chat template path (--chat-template)
        
        Optimization:
            enable_mixed_chunk: Mix prefill and decode in batches (--enable-mixed-chunk)
            enable_dp_attention: Data parallelism for attention (--enable-dp-attention)
            disable_radix_cache: Disable prefix caching (--disable-radix-cache)
            disable_cuda_graph: Disable CUDA graph optimization (--disable-cuda-graph)
        
        Logging:
            log_level: Logging level (--log-level): info, debug, warning, error
            log_requests: Log all request metadata (--log-requests)
            enable_metrics: Enable prometheus metrics (--enable-metrics)
    
    See https://docs.sglang.io/advanced_features/server_arguments.html for full list.
    """
    
    def __init__(self, model_path: str, port: int = 30000, host: str = "0.0.0.0", **kwargs):
        """Initialize SGLangServer with dynamic kwargs.
        
        Args:
            model_path: Model path or HuggingFace repo ID (required)
            port: Server port (default: 30000)
            host: Server host (default: 0.0.0.0)
            **kwargs: Any SGLang launch_server arguments, converted to --key-name format
        """
        self.model_path = model_path
        self.port = port
        self.host = host
        self.serve_kwargs = kwargs
        self.process = None
        self.base_url = f"http://localhost:{port}"

    def _build_cmd(self) -> list:
        """Build the sglang launch_server command from kwargs."""
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
        ]
        
        for key, value in self.serve_kwargs.items():
            # Convert Python snake_case to CLI kebab-case
            arg_name = f"--{key.replace('_', '-')}"
            
            if isinstance(value, bool):
                # Boolean flags: only add if True
                if value:
                    cmd.append(arg_name)
            elif value is not None:
                # Other values: add as --key value
                cmd.extend([arg_name, str(value)])
        
        return cmd

    def start(self):
        """Start the SGLang server and wait for it to be healthy."""
        cmd = self._build_cmd()
        print(f"Starting SGLang server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, text=True)
        return self._wait_for_health()

    def _wait_for_health(self, max_attempts=200, interval=5.0):
        """Wait for the server to become healthy."""
        health_url = f"{self.base_url}/health"
        print(f"Waiting for server at {health_url}...")

        for attempt in range(max_attempts):
            try:
                resp = requests.get(health_url, timeout=5.0)
                if resp.status_code == 200:
                    print(f"Server healthy after {attempt + 1} attempts")
                    return True
            except requests.RequestException:
                pass

            if attempt % 10 == 0:
                print(f"Health check attempt {attempt + 1}...")
            time.sleep(interval)

        print(f"Server failed to become healthy after {max_attempts} attempts")
        return False

    def stop(self):
        """Stop the SGLang server."""
        if self.process is None or self.process.poll() is not None:
            return

        print(f"Stopping SGLang server on port {self.port}...")
        self.process.send_signal(signal.SIGINT)

        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Force killing server...")
            self.process.kill()
            self.process.wait()

        # Wait for port to be released
        for _ in range(30):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", self.port))
                print(f"Port {self.port} released")
                break
            except OSError:
                time.sleep(1.0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

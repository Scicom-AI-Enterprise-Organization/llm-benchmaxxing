import signal
import socket
import subprocess
import time
from typing import Any

import requests


class VLLMServer:
    """vLLM Server wrapper that accepts dynamic kwargs for CLI args.
    
    All kwargs are converted to CLI arguments using the pattern:
    - key_name -> --key-name
    - Boolean True -> flag added (--key-name)
    - Boolean False -> flag omitted
    - Other values -> --key-name value
    
    Example:
        VLLMServer(
            model="meta-llama/Llama-2-7b",
            port=8000,
            tensor_parallel_size=4,
            enable_expert_parallel=True,
            max_model_len=32000
        )
        
        Produces: vllm serve meta-llama/Llama-2-7b --port 8000 
                  --tensor-parallel-size 4 --enable-expert-parallel --max-model-len 32000
    """
    
    def __init__(self, model: str, port: int = 8000, **kwargs):
        """Initialize VLLMServer with dynamic kwargs.
        
        Args:
            model: Model path or HuggingFace repo ID (required)
            port: Server port (default: 8000)
            **kwargs: Any vLLM serve arguments, converted to --key-name format
        """
        self.model = model
        self.port = port
        self.serve_kwargs = kwargs
        self.process = None
        self.base_url = f"http://localhost:{port}"

    def _build_cmd(self) -> list:
        """Build the vllm serve command from kwargs."""
        cmd = ["vllm", "serve", self.model, "--port", str(self.port)]
        
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
        """Start the vLLM server and wait for it to be healthy."""
        cmd = self._build_cmd()
        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, text=True)
        return self._wait_for_health()

    def _wait_for_health(self, max_attempts=200, interval=5.0):
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
        if self.process is None or self.process.poll() is not None:
            return

        print(f"Stopping vLLM server on port {self.port}...")
        self.process.send_signal(signal.SIGINT)

        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print("Force killing server...")
            self.process.kill()
            self.process.wait()

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

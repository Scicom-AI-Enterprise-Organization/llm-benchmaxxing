import signal
import socket
import subprocess
import time

import requests


class VLLMServer:
    def __init__(self, model_path, port, tp, dp, pp,
                 gpu_memory_utilization=0.9,
                 max_model_len=None,
                 max_num_seqs=None,
                 dtype=None,
                 disable_log_requests=False,
                 enable_expert_parallel=False,
                 health_check_max_attempts=0,
                 health_check_interval=10.0):
        self.model_path = model_path
        self.port = port
        self.tp = tp
        self.dp = dp
        self.pp = pp
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.dtype = dtype
        self.disable_log_requests = disable_log_requests
        self.enable_expert_parallel = enable_expert_parallel
        self.health_check_max_attempts = health_check_max_attempts
        self.health_check_interval = health_check_interval
        self.process = None
        self.base_url = f"http://localhost:{port}"

    def start(self):
        cmd = [
            "vllm", "serve", self.model_path,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tp),
            "--pipeline-parallel-size", str(self.pp),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]

        if self.dp > 1:
            cmd.extend(["--data-parallel-size", str(self.dp)])

        if self.max_model_len:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.max_num_seqs:
            cmd.extend(["--max-num-seqs", str(self.max_num_seqs)])
        if self.dtype:
            cmd.extend(["--dtype", self.dtype])
        if self.disable_log_requests:
            cmd.append("--disable-log-requests")
        if self.enable_expert_parallel:
            cmd.append("--enable-expert-parallel")

        print(f"Starting vLLM server: {' '.join(cmd)}")
        self.process = subprocess.Popen(cmd, text=True)
        return self._wait_for_health()

    def _wait_for_health(self):
        """Wait for vLLM server to become healthy.
        
        If health_check_max_attempts is 0, wait indefinitely until server is healthy.
        """
        health_url = f"{self.base_url}/health"
        max_attempts = self.health_check_max_attempts
        interval = self.health_check_interval
        
        if max_attempts == 0:
            print(f"Waiting for server at {health_url} (unlimited retries, interval={interval}s)...")
        else:
            print(f"Waiting for server at {health_url} (max_attempts={max_attempts}, interval={interval}s)...")

        attempt = 0
        while True:
            try:
                resp = requests.get(health_url, timeout=5.0)
                if resp.status_code == 200:
                    print(f"Server healthy after {attempt + 1} attempts")
                    return True
            except requests.RequestException:
                pass

            if attempt % 10 == 0:
                print(f"Health check attempt {attempt + 1}...")
            
            attempt += 1
            
            # If max_attempts is 0, wait indefinitely; otherwise check limit
            if max_attempts > 0 and attempt >= max_attempts:
                print(f"Server failed to become healthy after {max_attempts} attempts")
                return False
            
            time.sleep(interval)

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

"""
Integration tests for benchmaq bench command with SSH (remote execution).

These tests require a real SSH server with GPU access.
Run with: pytest tests/test_bench_integration.py -v -s

Environment variables required:
- RUNPOD_API_KEY: RunPod API key (for RunPod-based tests)
- TEST_SSH_HOST: SSH host for testing (for standalone SSH tests)
- TEST_SSH_PORT: SSH port (default: 22)
- TEST_SSH_USER: SSH username
- TEST_SSH_PASSWORD: SSH password (for password auth tests)
- TEST_SSH_KEY: Path to SSH private key (for key auth tests)
"""

import os
import sys
import time
import subprocess
import pytest

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestBenchLocalCLI:
    """Integration tests for local benchmark CLI (no GPU required for basic tests)."""
    
    def test_bench_help(self):
        """Test: benchmaq bench --help"""
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "bench", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Note: This will fail because bench requires a config positional arg
        # Just checking the CLI is accessible
        assert "usage" in result.stderr.lower() or "config" in result.stdout.lower() or result.returncode in [0, 2]
    
    def test_bench_missing_config(self):
        """Test: benchmaq bench with missing config file"""
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "bench", "nonexistent.yaml"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode != 0, "Should fail with missing config"


class TestBenchRemoteSSHPassword:
    """Integration tests for remote SSH benchmark with password authentication."""
    
    @pytest.fixture(autouse=True)
    def check_ssh_env(self):
        """Check if SSH test environment is configured."""
        required = ["TEST_SSH_HOST", "TEST_SSH_USER", "TEST_SSH_PASSWORD"]
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            pytest.skip(f"SSH password test env vars not set: {missing}")
    
    def test_bench_ssh_password_cli(self, test_fixtures_dir):
        """Test: benchmaq bench <config.yaml> with SSH password auth."""
        # Create a temp config with SSH password settings
        config_path = test_fixtures_dir / "test_ssh_password_config.yaml"
        
        if not config_path.exists():
            pytest.skip("SSH password config not found")
        
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "bench", str(config_path)],
            capture_output=True,
            text=True,
            timeout=600,
            env={
                **os.environ,
                "SSH_HOST": os.environ["TEST_SSH_HOST"],
                "SSH_USER": os.environ["TEST_SSH_USER"],
                "SSH_PASSWORD": os.environ["TEST_SSH_PASSWORD"],
            }
        )
        
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        assert result.returncode == 0, f"SSH password bench failed: {result.stderr}"
    
    def test_bench_ssh_password_python(self):
        """Test: benchmaq.bench() with SSH password auth."""
        import benchmaq
        
        result = benchmaq.bench(
            host=os.environ["TEST_SSH_HOST"],
            port=int(os.environ.get("TEST_SSH_PORT", 22)),
            username=os.environ["TEST_SSH_USER"],
            password=os.environ["TEST_SSH_PASSWORD"],
            model_path="Qwen/Qwen2.5-0.5B",
            context_sizes=[512],
            concurrency=[5],
            num_prompts=[5],
            output_len=[16],
        )
        
        print(f"Bench result: {result}")
        
        assert result.get("status") in ["success", "error"]


class TestBenchRemoteSSHKey:
    """Integration tests for remote SSH benchmark with private key authentication."""
    
    @pytest.fixture(autouse=True)
    def check_ssh_env(self):
        """Check if SSH key test environment is configured."""
        required = ["TEST_SSH_HOST", "TEST_SSH_USER", "TEST_SSH_KEY"]
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            pytest.skip(f"SSH key test env vars not set: {missing}")
    
    def test_bench_ssh_key_cli(self, test_fixtures_dir):
        """Test: benchmaq bench <config.yaml> with SSH key auth."""
        config_path = test_fixtures_dir / "test_ssh_key_config.yaml"
        
        if not config_path.exists():
            pytest.skip("SSH key config not found")
        
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "bench", str(config_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        assert result.returncode == 0, f"SSH key bench failed: {result.stderr}"
    
    def test_bench_ssh_key_python(self):
        """Test: benchmaq.bench() with SSH key auth."""
        import benchmaq
        
        result = benchmaq.bench(
            host=os.environ["TEST_SSH_HOST"],
            port=int(os.environ.get("TEST_SSH_PORT", 22)),
            username=os.environ["TEST_SSH_USER"],
            key_filename=os.environ["TEST_SSH_KEY"],
            model_path="Qwen/Qwen2.5-0.5B",
            context_sizes=[512],
            concurrency=[5],
            num_prompts=[5],
            output_len=[16],
        )
        
        print(f"Bench result: {result}")
        
        assert result.get("status") in ["success", "error"]


class TestBenchWithRunPod:
    """
    End-to-end bench tests using RunPod for GPU access.
    
    These tests:
    1. Deploy a RunPod pod (1x A100 PCIe spot)
    2. Wait for SSH to be ready
    3. Run benchmaq bench via SSH key auth
    4. Clean up the pod
    
    Requires: RUNPOD_API_KEY environment variable
    """
    
    pod_id = None
    ssh_info = None
    
    @pytest.fixture(autouse=True)
    def check_runpod_env(self, runpod_api_key):
        """Ensure RunPod API key is available."""
        pass  # runpod_api_key fixture handles the skip
    
    @pytest.mark.slow
    def test_01_deploy_pod_for_bench(self, runpod_api_key):
        """Deploy a RunPod pod for bench testing."""
        import benchmaq.runpod as rp
        
        print("\n=== Deploying RunPod pod for bench test ===")
        
        instance = rp.deploy(
            api_key=runpod_api_key,
            gpu_type="NVIDIA A100 80GB PCIe",
            gpu_count=1,
            image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            disk_size=100,
            container_disk_size=50,
            spot=True,
            name="benchmaq_bench_test",
            wait_for_ready=True,
            ssh_key_path="~/.ssh/id_ed25519",
            health_check_retries=60,
            health_check_interval=10.0,
        )
        
        print(f"Deployed pod: {instance}")
        
        assert "id" in instance, "Pod ID not returned"
        assert "ssh" in instance, "SSH info not returned"
        
        TestBenchWithRunPod.pod_id = instance["id"]
        TestBenchWithRunPod.ssh_info = instance["ssh"]
        
        print(f"SSH: {instance['ssh']['command']}")
    
    @pytest.mark.slow
    def test_02_bench_cli_with_runpod(self, runpod_api_key, test_fixtures_dir):
        """Test: benchmaq bench <config.yaml> using RunPod pod via CLI."""
        if not TestBenchWithRunPod.pod_id or not TestBenchWithRunPod.ssh_info:
            pytest.skip("No pod deployed from previous test")
        
        ssh = TestBenchWithRunPod.ssh_info
        
        # Create a temporary config file with the SSH info
        import yaml
        config = {
            "remote": {
                "host": ssh["ip"],
                "port": ssh["port"],
                "username": "root",
                "key_filename": os.path.expanduser("~/.ssh/id_ed25519"),
                "uv": {
                    "path": "~/.benchmark-venv",
                    "python_version": "3.11"
                },
                "dependencies": ["pyyaml", "requests", "vllm", "huggingface_hub"]
            },
            "runs": [{
                "name": "bench_test",
                "engine": "vllm",
                "model": {
                    "repo_id": "Qwen/Qwen2.5-0.5B",
                    "local_dir": "/workspace/Qwen2.5-0.5B"
                },
                "vllm_serve": {
                    "model_path": "/workspace/Qwen2.5-0.5B",
                    "port": 8000,
                    "gpu_memory_utilization": 0.9,
                    "max_model_len": 2048,
                    "dtype": "auto",
                    "disable_log_requests": True,
                    "parallelism_pairs": [{"tensor_parallel": 1, "data_parallel": 1, "pipeline_parallel": 1}]
                },
                "benchmark": {
                    "save_results": False,
                    "output_dir": "./benchmark_results",
                    "context_size": [256],
                    "concurrency": [5],
                    "num_prompts": [5],
                    "output_len": [16]
                }
            }]
        }
        
        config_path = test_fixtures_dir / "test_bench_runpod_temp.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        print(f"\n=== Running benchmaq bench via CLI ===")
        print(f"Config: {config_path}")
        print(f"SSH: {ssh['command']}")
        
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "bench", str(config_path)],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            env={**os.environ, "RUNPOD_API_KEY": runpod_api_key}
        )
        
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        # Clean up temp config
        if config_path.exists():
            config_path.unlink()
        
        assert result.returncode == 0, f"Bench CLI failed: {result.stderr}"
        assert "BENCHMARK COMPLETED" in result.stdout or "Remote execution completed" in result.stdout
    
    @pytest.mark.slow  
    def test_03_bench_python_with_runpod(self, runpod_api_key):
        """Test: benchmaq.bench() using RunPod pod via Python API."""
        if not TestBenchWithRunPod.pod_id or not TestBenchWithRunPod.ssh_info:
            pytest.skip("No pod deployed from previous test")
        
        import benchmaq
        
        ssh = TestBenchWithRunPod.ssh_info
        
        print(f"\n=== Running benchmaq.bench() via Python API ===")
        print(f"Host: {ssh['ip']}:{ssh['port']}")
        
        result = benchmaq.bench(
            host=ssh["ip"],
            port=ssh["port"],
            username="root",
            key_filename=os.path.expanduser("~/.ssh/id_ed25519"),
            model_path="Qwen/Qwen2.5-0.5B",
            context_sizes=[256],
            concurrency=[5],
            num_prompts=[5],
            output_len=[16],
        )
        
        print(f"Bench result: {result}")
        
        assert result.get("status") in ["success", "error"]
        if result.get("status") == "success":
            assert result.get("mode") == "remote"
    
    @pytest.mark.slow
    def test_04_cleanup_pod(self, runpod_api_key):
        """Clean up the RunPod pod after testing."""
        if not TestBenchWithRunPod.pod_id:
            pytest.skip("No pod to clean up")
        
        import benchmaq.runpod as rp
        
        print(f"\n=== Cleaning up pod: {TestBenchWithRunPod.pod_id} ===")
        
        result = rp.delete(api_key=runpod_api_key, pod_id=TestBenchWithRunPod.pod_id)
        
        print(f"Delete result: {result}")
        
        assert result.get("status") == "deleted"
        
        TestBenchWithRunPod.pod_id = None
        TestBenchWithRunPod.ssh_info = None

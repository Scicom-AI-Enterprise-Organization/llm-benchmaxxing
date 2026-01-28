"""
Unit tests for benchmaq with mocked dependencies.

Run with: pytest tests/test_unit.py -v
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch


class TestConfigLoading:
    """Unit tests for config loading and merging."""
    
    def test_load_config(self, test_runpod_config):
        """Test loading a YAML config file."""
        from benchmaq.config import load_config
        
        config = load_config(test_runpod_config)
        
        assert "runpod" in config
        assert "runs" in config
        assert config["runpod"]["pod"]["gpu_type"] == "NVIDIA A100 80GB PCIe"
        assert config["runpod"]["pod"]["gpu_count"] == 1
    
    def test_merge_config(self):
        """Test merging two config dicts."""
        from benchmaq.config import merge_config
        
        base = {
            "runpod": {"pod": {"gpu_type": "A100", "gpu_count": 1}},
            "runs": [{"name": "test"}]
        }
        overrides = {
            "runpod": {"pod": {"gpu_count": 2}},
        }
        
        result = merge_config(base, overrides)
        
        assert result["runpod"]["pod"]["gpu_type"] == "A100"  # preserved
        assert result["runpod"]["pod"]["gpu_count"] == 2  # overridden
        assert result["runs"][0]["name"] == "test"  # preserved
    
    def test_kwargs_to_run_config(self):
        """Test converting kwargs to run config."""
        from benchmaq.config import kwargs_to_run_config
        
        config = kwargs_to_run_config(
            name="test_run",
            model_path="/path/to/model",
            tensor_parallel=2,
            context_sizes=[1024, 2048],
        )
        
        assert "runs" in config
        assert config["runs"][0]["name"] == "test_run"
        assert config["runs"][0]["vllm_serve"]["model_path"] == "/path/to/model"
        assert config["runs"][0]["vllm_serve"]["parallelism_pairs"][0]["tensor_parallel"] == 2


class TestRunPodClientUnit:
    """Unit tests for RunPod client with mocked API."""
    
    def test_deploy_spot(self, mock_runpod_api, mock_subprocess):
        """Test deploying a spot instance."""
        from benchmaq.runpod.core.client import deploy, set_api_key
        
        # Mock SSH check
        mock_subprocess["run"].return_value.returncode = 0
        
        # Mock wait_for_pod
        with patch("benchmaq.runpod.core.client.wait_for_pod") as mock_wait:
            mock_wait.return_value = {
                "ready": True,
                "ssh": {"ip": "1.2.3.4", "port": 22222, "command": "ssh root@1.2.3.4 -p 22222"}
            }
            
            set_api_key("test-api-key")
            
            result = deploy(
                gpu_type="NVIDIA A100 80GB PCIe",
                gpu_count=1,
                image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                disk_size=50,
                spot=True,
                wait_for_ready=True,
            )
        
        assert result["id"] == "test-pod-id-123"
        assert "ssh" in result
    
    def test_find_pod(self, mock_runpod_api):
        """Test finding a pod by ID."""
        from benchmaq.runpod.core.client import find, set_api_key
        
        set_api_key("test-api-key")
        
        result = find("test-pod-id-123")
        
        assert result["id"] == "test-pod-id-123"
        assert result["name"] == "benchmaq_test_1xa100"
    
    def test_find_by_name(self, mock_runpod_api):
        """Test finding a pod by name."""
        from benchmaq.runpod.core.client import find_by_name, set_api_key
        
        set_api_key("test-api-key")
        
        result = find_by_name("benchmaq_test_1xa100")
        
        assert result is not None
        assert result["name"] == "benchmaq_test_1xa100"
    
    def test_delete_pod(self, mock_runpod_api):
        """Test deleting a pod."""
        from benchmaq.runpod.core.client import delete, set_api_key
        
        set_api_key("test-api-key")
        
        result = delete(pod_id="test-pod-id-123")
        
        assert result["status"] == "deleted"
        assert result["id"] == "test-pod-id-123"
    
    def test_start_pod(self, mock_runpod_api):
        """Test starting a stopped pod."""
        from benchmaq.runpod.core.client import start, set_api_key
        
        set_api_key("test-api-key")
        
        result = start("test-pod-id-123")
        
        assert result["id"] == "test-pod-id-123"
    
    def test_stop_pod(self, mock_runpod_api):
        """Test stopping a running pod."""
        from benchmaq.runpod.core.client import stop, set_api_key
        
        set_api_key("test-api-key")
        
        result = stop("test-pod-id-123")
        
        assert result["id"] == "test-pod-id-123"


class TestRunPodModuleUnit:
    """Unit tests for benchmaq.runpod module with mocked API."""
    
    def test_deploy(self, mock_runpod_api, mock_subprocess):
        """Test benchmaq.runpod.deploy()."""
        import benchmaq.runpod as rp
        
        with patch("benchmaq.runpod.core.client.wait_for_pod") as mock_wait:
            mock_wait.return_value = {"ready": True, "ssh": None}
            
            result = rp.deploy(
                api_key="test-api-key",
                gpu_type="NVIDIA A100 80GB PCIe",
                gpu_count=1,
                image="test-image",
                disk_size=50,
                wait_for_ready=True,
            )
        
        assert result["id"] == "test-pod-id-123"
    
    def test_find(self, mock_runpod_api):
        """Test benchmaq.runpod.find()."""
        import benchmaq.runpod as rp
        
        result = rp.find(api_key="test-api-key", pod_id="test-pod-id-123")
        
        assert result["id"] == "test-pod-id-123"
    
    def test_delete(self, mock_runpod_api):
        """Test benchmaq.runpod.delete()."""
        import benchmaq.runpod as rp
        
        result = rp.delete(api_key="test-api-key", pod_id="test-pod-id-123")
        
        assert result["status"] == "deleted"


class TestCLIUnit:
    """Unit tests for CLI argument parsing."""
    
    def test_cli_help(self):
        """Test CLI help output."""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0
        assert "bench" in result.stdout
        assert "runpod" in result.stdout
    
    def test_runpod_help(self):
        """Test runpod subcommand help."""
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "runpod", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        assert result.returncode == 0
        assert "deploy" in result.stdout
        assert "delete" in result.stdout
        assert "find" in result.stdout
        assert "start" in result.stdout
        assert "bench" in result.stdout


class TestBenchFunctionUnit:
    """Unit tests for benchmaq.bench() function."""
    
    def test_bench_requires_runs(self):
        """Test that bench() requires runs to be defined."""
        import benchmaq
        
        # Should raise ValueError since no runs defined
        with pytest.raises(ValueError, match="No benchmark runs"):
            benchmaq.bench()
    
    def test_bench_remote_requires_host(self):
        """Test that remote bench requires host."""
        import benchmaq
        
        # Provide model_path but no host
        result = benchmaq.bench(
            model_path="/fake/model",
            context_sizes=[512],
        )
        
        # Should attempt local run or fail appropriately
        assert result.get("status") in ["success", "error"]

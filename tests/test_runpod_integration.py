"""
Integration tests for RunPod operations.

These tests make REAL API calls to RunPod and will incur costs.
Run with: pytest tests/test_runpod_integration.py -v -s

Environment variables required:
- RUNPOD_API_KEY: Your RunPod API key
- HF_TOKEN: (optional) HuggingFace token for gated models
"""

import os
import sys
import time
import pytest
import subprocess

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


class TestRunPodCLI:
    """Integration tests for benchmaq runpod CLI commands."""
    
    pod_id = None
    
    def test_01_runpod_deploy(self, test_runpod_config, runpod_api_key):
        """Test: benchmaq runpod deploy <config.yaml>"""
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "runpod", "deploy", test_runpod_config, "--no-wait"],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "RUNPOD_API_KEY": runpod_api_key}
        )
        
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        assert result.returncode == 0, f"Deploy failed: {result.stderr}"
        assert '"id"' in result.stdout, "Pod ID not found in output"
        
        # Extract JSON from output (may have log lines before JSON)
        import json
        stdout = result.stdout
        json_start = stdout.find('{')
        if json_start >= 0:
            json_str = stdout[json_start:]
            output = json.loads(json_str)
            TestRunPodCLI.pod_id = output["id"]
            print(f"Deployed pod: {TestRunPodCLI.pod_id}")
    
    def test_02_runpod_find_by_id(self, runpod_api_key):
        """Test: benchmaq runpod find <pod_id>"""
        if not TestRunPodCLI.pod_id:
            pytest.skip("No pod ID from previous deploy test")
        
        # Wait a bit for pod to be queryable
        time.sleep(5)
        
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "runpod", "find", TestRunPodCLI.pod_id],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "RUNPOD_API_KEY": runpod_api_key}
        )
        
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        assert result.returncode == 0, f"Find failed: {result.stderr}"
        assert TestRunPodCLI.pod_id in result.stdout, "Pod ID not found in output"
    
    def test_03_runpod_find_by_config(self, test_runpod_config, runpod_api_key):
        """Test: benchmaq runpod find <config.yaml>"""
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "runpod", "find", test_runpod_config],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "RUNPOD_API_KEY": runpod_api_key}
        )
        
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        assert result.returncode == 0, f"Find by config failed: {result.stderr}"
    
    def test_04_runpod_stop(self, test_runpod_config):
        """Test: benchmaq runpod stop <config.yaml> (if stop command exists)"""
        # Note: Current CLI doesn't have stop, but let's test via Python API
        pytest.skip("Stop command not implemented in CLI - test via Python API")
    
    def test_05_runpod_start(self, test_runpod_config, runpod_api_key):
        """Test: benchmaq runpod start <config.yaml>
        
        Note: Spot pods cannot be resumed - they must be re-deployed.
        This test will skip for spot instances.
        """
        if not TestRunPodCLI.pod_id:
            pytest.skip("No pod ID from previous deploy test")
        
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "runpod", "start", test_runpod_config],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "RUNPOD_API_KEY": runpod_api_key}
        )
        
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        # Spot pods cannot be resumed - they fail with specific error
        # This is expected behavior for spot instances
        if "cannot resume a spot pod" in result.stderr.lower():
            pytest.skip("Spot pods cannot be resumed - expected behavior")
        
        # May fail if pod is already running, which is fine
        assert result.returncode == 0 or "already running" in result.stderr.lower()
    
    def test_06_runpod_delete(self, runpod_api_key):
        """Test: benchmaq runpod delete <pod_id>"""
        if not TestRunPodCLI.pod_id:
            pytest.skip("No pod ID from previous deploy test")
        
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "runpod", "delete", TestRunPodCLI.pod_id],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "RUNPOD_API_KEY": runpod_api_key}
        )
        
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        assert result.returncode == 0, f"Delete failed: {result.stderr}"
        assert '"status": "deleted"' in result.stdout, "Delete confirmation not found"
        
        TestRunPodCLI.pod_id = None


class TestRunPodPython:
    """Integration tests for benchmaq.runpod Python API."""
    
    pod_id = None
    
    def test_01_deploy(self, runpod_api_key):
        """Test: benchmaq.runpod.deploy()"""
        import benchmaq.runpod as rp
        
        instance = rp.deploy(
            api_key=runpod_api_key,
            gpu_type="NVIDIA A100 80GB PCIe",
            gpu_count=1,
            image="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            disk_size=50,
            container_disk_size=50,
            spot=True,
            name="benchmaq_pytest_1xa100",
            wait_for_ready=False,  # Don't wait to speed up test
            ssh_key_path="~/.ssh/id_ed25519",
        )
        
        print(f"Deployed instance: {instance}")
        
        assert "id" in instance, "Pod ID not returned"
        TestRunPodPython.pod_id = instance["id"]
    
    def test_02_find(self, runpod_api_key):
        """Test: benchmaq.runpod.find()"""
        if not TestRunPodPython.pod_id:
            pytest.skip("No pod ID from previous deploy test")
        
        import benchmaq.runpod as rp
        
        # Wait a bit
        time.sleep(5)
        
        pod = rp.find(api_key=runpod_api_key, pod_id=TestRunPodPython.pod_id)
        
        print(f"Found pod: {pod}")
        
        assert pod is not None, "Pod not found"
        assert pod.get("id") == TestRunPodPython.pod_id
    
    def test_03_find_by_name(self, runpod_api_key):
        """Test: benchmaq.runpod.find_by_name()"""
        import benchmaq.runpod as rp
        
        pod = rp.find_by_name(api_key=runpod_api_key, name="benchmaq_pytest_1xa100")
        
        print(f"Found pod by name: {pod}")
        
        # May or may not find it depending on timing
        if pod:
            assert pod.get("name") == "benchmaq_pytest_1xa100"
    
    def test_04_list_pods(self, runpod_api_key):
        """Test: benchmaq.runpod.list_pods()"""
        import benchmaq.runpod as rp
        
        pods = rp.list_pods(api_key=runpod_api_key)
        
        print(f"Listed {len(pods)} pods")
        
        assert isinstance(pods, list)
    
    def test_05_stop(self, runpod_api_key):
        """Test: benchmaq.runpod.stop()"""
        if not TestRunPodPython.pod_id:
            pytest.skip("No pod ID from previous deploy test")
        
        import benchmaq.runpod as rp
        
        try:
            result = rp.stop(api_key=runpod_api_key, pod_id=TestRunPodPython.pod_id)
            print(f"Stop result: {result}")
        except Exception as e:
            # May fail if pod is not yet running
            print(f"Stop failed (may be expected): {e}")
    
    def test_06_start(self, runpod_api_key):
        """Test: benchmaq.runpod.start()"""
        if not TestRunPodPython.pod_id:
            pytest.skip("No pod ID from previous deploy test")
        
        import benchmaq.runpod as rp
        
        # Wait for stop to complete
        time.sleep(10)
        
        try:
            result = rp.start(api_key=runpod_api_key, pod_id=TestRunPodPython.pod_id)
            print(f"Start result: {result}")
        except Exception as e:
            # May fail if pod is already running
            print(f"Start failed (may be expected): {e}")
    
    def test_07_delete(self, runpod_api_key):
        """Test: benchmaq.runpod.delete()"""
        if not TestRunPodPython.pod_id:
            pytest.skip("No pod ID from previous deploy test")
        
        import benchmaq.runpod as rp
        
        result = rp.delete(api_key=runpod_api_key, pod_id=TestRunPodPython.pod_id)
        
        print(f"Delete result: {result}")
        
        assert result.get("status") == "deleted"
        TestRunPodPython.pod_id = None


class TestRunPodE2EBench:
    """End-to-end benchmark test (deploy -> bench -> delete)."""
    
    @pytest.mark.slow
    def test_runpod_bench_cli(self, test_runpod_config, runpod_api_key):
        """Test: benchmaq runpod bench <config.yaml>
        
        WARNING: This test runs a full benchmark and may take 10-30 minutes!
        """
        os.environ["RUNPOD_API_KEY"] = runpod_api_key
        
        result = subprocess.run(
            [sys.executable, "-m", "benchmaq.cli", "runpod", "bench", test_runpod_config],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        assert result.returncode == 0, f"E2E bench failed: {result.stderr}"
        assert "BENCHMARK COMPLETED" in result.stdout or "END-TO-END BENCHMARK COMPLETED" in result.stdout
    
    @pytest.mark.slow
    def test_runpod_bench_python(self, test_runpod_config, runpod_api_key):
        """Test: benchmaq.runpod.bench()
        
        WARNING: This test runs a full benchmark and may take 10-30 minutes!
        """
        import benchmaq.runpod as rp
        
        result = rp.bench(
            config_path=test_runpod_config,
            api_key=runpod_api_key,
        )
        
        print(f"Bench result: {result}")
        
        assert result.get("status") == "success", f"Bench failed: {result.get('error')}"

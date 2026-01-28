"""Pytest configuration and fixtures for benchmaq tests."""

import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env
load_dotenv(PROJECT_ROOT / ".env")


@pytest.fixture(scope="session")
def runpod_api_key():
    """Get RunPod API key from environment."""
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        pytest.skip("RUNPOD_API_KEY not set in environment")
    return key


@pytest.fixture(scope="session")
def hf_token():
    """Get HuggingFace token from environment."""
    return os.environ.get("HF_TOKEN", "")


@pytest.fixture
def test_fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_runpod_config(test_fixtures_dir):
    """Path to test RunPod config file."""
    return str(test_fixtures_dir / "test_runpod_config.yaml")


@pytest.fixture
def test_ssh_password_config(test_fixtures_dir):
    """Path to test SSH password config file."""
    return str(test_fixtures_dir / "test_ssh_password_config.yaml")


@pytest.fixture
def test_ssh_key_config(test_fixtures_dir):
    """Path to test SSH key config file."""
    return str(test_fixtures_dir / "test_ssh_key_config.yaml")


@pytest.fixture
def test_local_config(test_fixtures_dir):
    """Path to test local benchmark config file."""
    return str(test_fixtures_dir / "test_local_config.yaml")


# Mocking fixtures for unit tests
@pytest.fixture
def mock_runpod_api(mocker):
    """Mock RunPod API calls."""
    mock_deploy = mocker.patch("benchmaq.runpod.core.client.run_graphql_query")
    mock_deploy.return_value = {
        "data": {
            "podRentInterruptable": {
                "id": "test-pod-id-123",
                "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
                "machineId": "test-machine-id"
            }
        }
    }
    
    mock_get_pod = mocker.patch("runpod.get_pod")
    mock_get_pod.return_value = {
        "id": "test-pod-id-123",
        "name": "benchmaq_test_1xa100",
        "desiredStatus": "RUNNING",
        "runtime": {
            "ports": [
                {"privatePort": 22, "publicPort": 22222, "ip": "123.45.67.89"}
            ]
        }
    }
    
    mock_get_pods = mocker.patch("runpod.get_pods")
    mock_get_pods.return_value = [
        {
            "id": "test-pod-id-123",
            "name": "benchmaq_test_1xa100",
            "desiredStatus": "RUNNING"
        }
    ]
    
    mock_terminate = mocker.patch("runpod.terminate_pod")
    mock_terminate.return_value = {"status": "terminated"}
    
    mock_resume = mocker.patch("runpod.resume_pod")
    mock_resume.return_value = {"id": "test-pod-id-123", "desiredStatus": "RUNNING"}
    
    mock_stop = mocker.patch("runpod.stop_pod")
    mock_stop.return_value = {"id": "test-pod-id-123", "desiredStatus": "EXITED"}
    
    return {
        "deploy": mock_deploy,
        "get_pod": mock_get_pod,
        "get_pods": mock_get_pods,
        "terminate": mock_terminate,
        "resume": mock_resume,
        "stop": mock_stop,
    }


@pytest.fixture
def mock_ssh(mocker):
    """Mock SSH/Paramiko calls."""
    mock_client = mocker.MagicMock()
    mock_paramiko = mocker.patch("paramiko.SSHClient")
    mock_paramiko.return_value = mock_client
    
    # Mock exec_command
    mock_stdout = mocker.MagicMock()
    mock_stdout.read.return_value = b"ok\n"
    mock_client.exec_command.return_value = (mocker.MagicMock(), mock_stdout, mocker.MagicMock())
    
    return mock_client


@pytest.fixture
def mock_subprocess(mocker):
    """Mock subprocess calls."""
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value = mocker.MagicMock(returncode=0)
    
    mock_popen = mocker.patch("subprocess.Popen")
    mock_process = mocker.MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = iter(["line1\n", "line2\n"])
    mock_process.wait.return_value = 0
    mock_popen.return_value = mock_process
    
    return {"run": mock_run, "popen": mock_popen}

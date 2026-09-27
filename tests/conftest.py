import logging
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from dotenv import load_dotenv
from hypothesis import HealthCheck, settings

from app.helpers.logger import get_logger


load_dotenv(dotenv_path=".env_testing")


# Hypothesis global profiles: fast by default for local runs
settings.register_profile(
    "dev",
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "ci",
    max_examples=100,
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def mock_datetime():
    """Mock datetime to return a fixed value"""
    fixed_datetime = datetime(2023, 12, 20, 10, 30, 0)
    with patch("app.helpers.logger.file.datetime") as mock_dt:
        mock_dt.utcnow.return_value = fixed_datetime
        yield mock_dt


@pytest.fixture
def file_logger(temp_log_dir):
    """Create a logger instance with temporary directory"""
    logger = get_logger("test-service")
    return logger


@pytest.fixture
def mock_logger():
    """Mock the underlying logger object"""
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = Mock(spec=logging.Logger)
        mock_get_logger.return_value = mock_logger
        yield mock_logger


@pytest.fixture(autouse=True)
def _setup_testing_environment(monkeypatch):
    """Setup test environment variables to override .env"""
    monkeypatch.setenv("STORAGE_TYPE", "local")
    monkeypatch.setenv("APP_ENVIRONMENT", "test")
    from app.helpers.environment import env

    env.cache_clear()
    yield
    env.cache_clear()


@pytest.fixture(autouse=True)
def _prevent_external_calls(monkeypatch, request):
    """Prevent any accidental external calls during testing"""

    # Skip socket blocking for E2E tests
    if hasattr(request, "fspath") and "e2e" in str(request.fspath):
        return  # Don't block sockets for E2E tests

    # Skip socket blocking for stock API tests
    if request.node.name and (
        "stock" in request.node.name.lower() or "test_stocks" in request.module.__name__
        if hasattr(request, "module")
        else False
    ):
        return  # Don't block sockets for stock tests

    def mock_external_call(*args, **kwargs):
        raise RuntimeError("External calls are not allowed during testing")

    # Add any external calls you want to prevent
    monkeypatch.setattr("socket.socket", mock_external_call)


@pytest.fixture
def mock_tracer_config():
    """Fixture to provide standard tracer configuration"""
    return {"service_name": "test_service", "environment": "test"}


@pytest.fixture
def mock_env_cloud():
    """Fixture to mock environment for cloud configuration"""

    def mock_getenv(key, default=None):  # Add default parameter
        if key == "TRACER_TYPE":
            return "cloud"
        if key == "SERVICE_NAME":
            return "test_service"
        return default  # Return default instead of None

    with patch("app.helpers.tracer.factory.os.getenv", side_effect=mock_getenv) as mock:
        yield mock


# Main conftest now delegates to specific test type conftest files
# Unit tests: tests/unit/conftest.py handles mocking
# E2E tests: tests/e2e/conftest.py handles real services

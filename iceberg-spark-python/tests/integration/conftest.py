"""Shared fixtures for integration tests — Docker lifecycle management."""

from __future__ import annotations

import os
import socket
import subprocess
import time

import pytest

DOCKER_DIR = os.path.join(os.path.dirname(__file__), "..", "docker")


def _is_docker_available() -> bool:
    """Check if docker and docker compose are available."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _wait_for_port(host: str, port: int, timeout: int = 90) -> bool:
    """Poll until a TCP port is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    return False


def _compose_up(compose_dir: str, project_name: str) -> None:
    """Start docker compose services."""
    subprocess.run(
        [
            "docker", "compose", "-p", project_name,
            "up", "-d", "--wait", "--timeout", "120",
        ],
        cwd=compose_dir,
        check=True,
        capture_output=True,
    )


def _compose_down(compose_dir: str, project_name: str) -> None:
    """Stop docker compose services and remove volumes."""
    subprocess.run(
        [
            "docker", "compose", "-p", project_name,
            "down", "-v", "--remove-orphans",
        ],
        cwd=compose_dir,
        capture_output=True,
    )


# ---------------------------------------------------------------------------
# REST catalog Docker fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rest_catalog_docker():
    """Start REST catalog Docker stack. Yields connection info dict."""
    if not _is_docker_available():
        pytest.skip("Docker not available")
    compose_dir = os.path.join(DOCKER_DIR, "rest-catalog")
    project = "iceberg-spark-rest-test"
    _compose_up(compose_dir, project)
    try:
        assert _wait_for_port("localhost", 8181, timeout=90), "REST catalog not ready"
        assert _wait_for_port("localhost", 9000, timeout=90), "MinIO not ready"
        yield {"uri": "http://localhost:8181", "s3_endpoint": "http://localhost:9000"}
    finally:
        _compose_down(compose_dir, project)


@pytest.fixture(scope="module")
def rest_session(rest_catalog_docker):
    """IcebergSession connected to Dockerized REST catalog."""
    from iceberg_spark import IcebergSession

    sess = (
        IcebergSession.builder()
        .catalog(
            "rest",
            name="rest_test",
            uri=rest_catalog_docker["uri"],
            warehouse="s3://icebergdata/demo",
            **{
                "s3.endpoint": rest_catalog_docker["s3_endpoint"],
                "s3.access-key-id": "admin",
                "s3.secret-access-key": "password",
                "s3.region": "us-east-1",
            },
        )
        .build()
    )
    yield sess
    sess.stop()


# ---------------------------------------------------------------------------
# Glue catalog Docker fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def glue_catalog_docker():
    """Start Glue (moto) + MinIO Docker stack. Yields connection info dict."""
    if not _is_docker_available():
        pytest.skip("Docker not available")
    try:
        import boto3  # noqa: F401
    except ImportError:
        pytest.skip("boto3 not installed — required for Glue catalog tests")
    compose_dir = os.path.join(DOCKER_DIR, "glue-catalog")
    project = "iceberg-spark-glue-test"
    _compose_up(compose_dir, project)
    try:
        assert _wait_for_port("localhost", 5000, timeout=90), "Moto not ready"
        assert _wait_for_port("localhost", 9002, timeout=90), "MinIO not ready"
        yield {
            "glue_endpoint": "http://localhost:5000",
            "s3_endpoint": "http://localhost:9002",
        }
    finally:
        _compose_down(compose_dir, project)


@pytest.fixture(scope="module")
def glue_session(glue_catalog_docker):
    """IcebergSession connected to Dockerized Glue (moto) catalog."""
    from iceberg_spark import IcebergSession

    sess = (
        IcebergSession.builder()
        .catalog(
            "glue",
            name="glue_test",
            warehouse="s3://warehouse/hive",
            **{
                "glue.endpoint": glue_catalog_docker["glue_endpoint"],
                "glue.access-key-id": "testing",
                "glue.secret-access-key": "testing",
                "glue.region": "us-east-1",
                "s3.endpoint": glue_catalog_docker["s3_endpoint"],
                "s3.access-key-id": "admin",
                "s3.secret-access-key": "password",
                "s3.region": "us-east-1",
            },
        )
        .build()
    )
    yield sess
    sess.stop()

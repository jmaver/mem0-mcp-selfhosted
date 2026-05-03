"""Integration test fixtures — skip gracefully when infrastructure is unavailable.

Fixture hierarchy:
    qdrant_url (session)  +  ollama_url (session)
                          ↓
                  memory_instance (session)  ←  uses mem0_integration_test collection
                          ↓                      drops collection in teardown
                  test_user_id (function)    ←  unique per test: "inttest-<test_name>"
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest


def _load_dotenv_once() -> None:
    """Load .env for integration tests only.

    Unit tests use monkeypatch/patch.dict and must not be affected by a
    developer's local .env file.  Shell env vars take precedence
    (python-dotenv's default ``override=False``).
    """
    from dotenv import load_dotenv

    load_dotenv()


_load_dotenv_once()

# Mark all integration tests
pytestmark = pytest.mark.integration

TEST_COLLECTION = "mem0_integration_test"


@pytest.fixture(scope="session")
def qdrant_url():
    """Health-check Qdrant REST API; skip entire suite if unreachable."""
    url = os.environ.get("MEM0_QDRANT_URL", "http://localhost:6333")
    try:
        urllib.request.urlopen(f"{url}/healthz", timeout=3)
    except Exception:
        pytest.skip(f"Qdrant not reachable at {url}")
    return url


@pytest.fixture(scope="session")
def ollama_url():
    """Health-check Ollama API; skip entire suite if unreachable."""
    url = os.environ.get("MEM0_EMBED_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{url}/api/tags", timeout=3)
    except Exception:
        pytest.skip(f"Ollama not reachable at {url}")
    return url


@pytest.fixture(scope="session")
def memory_instance(qdrant_url, ollama_url):
    """Create a real Memory instance against live infrastructure.

    Uses a dedicated test collection, dropped entirely in teardown.
    Anthropic token is only required when MEM0_LLM_PROVIDER=anthropic.
    """
    llm_provider = os.environ.get("MEM0_LLM_PROVIDER", "anthropic")
    if llm_provider == "anthropic":
        from mem0_mcp_selfhosted.auth import resolve_token

        token = resolve_token()
        if not token:
            pytest.skip("No Anthropic token available (required for MEM0_LLM_PROVIDER=anthropic)")
    # Override collection name for test isolation
    original_collection = os.environ.get("MEM0_COLLECTION")
    os.environ["MEM0_COLLECTION"] = TEST_COLLECTION

    from mem0_mcp_selfhosted.config import build_config
    from mem0_mcp_selfhosted.server import register_providers

    config_dict, providers_info = build_config()
    register_providers(providers_info)

    from mem0 import Memory

    memory = Memory.from_config(config_dict)

    yield memory

    # Teardown: drop the entire test collection
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url)
        client.delete_collection(TEST_COLLECTION)
    except Exception:
        pass  # Collection may not exist

    # Restore original env
    if original_collection is not None:
        os.environ["MEM0_COLLECTION"] = original_collection
    else:
        os.environ.pop("MEM0_COLLECTION", None)


@pytest.fixture
def test_user_id(request):
    """Generate a unique user_id per test function."""
    return f"inttest-{request.node.name}"

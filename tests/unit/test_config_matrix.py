"""Config matrix tests — parametrized provider combination tests.

Verifies that build_config() produces valid config dicts for all meaningful
combinations of llm_provider (v3: no graph permutations).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from mem0_mcp_selfhosted.config import build_config

# Base env vars required for all tests
_BASE_ENV = {
    "MEM0_QDRANT_URL": "http://localhost:6333",
    "MEM0_EMBED_URL": "http://localhost:11434",
    "MEM0_EMBED_MODEL": "bge-m3",
    "MEM0_EMBED_DIMS": "1024",
}


def _make_env(llm_provider: str) -> dict:
    """Build env dict for a provider combination."""
    env = dict(_BASE_ENV)
    env["MEM0_LLM_PROVIDER"] = llm_provider
    return env


# (llm_provider, test_id)
MATRIX = [
    ("anthropic", "anthropic"),
    ("ollama", "ollama"),
]


class TestProviderMatrix:
    @pytest.mark.parametrize(
        "llm_provider,test_id",
        MATRIX,
        ids=[m[1] for m in MATRIX],
    )
    @patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-ant-api-fake")
    def test_provider_combination(self, mock_token, llm_provider, test_id, monkeypatch):
        env = _make_env(llm_provider)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        # Clean up env vars not in this combination
        for key in ("MEM0_LLM_MODEL", "MEM0_LLM_URL", "MEM0_EMBED_PROVIDER", "MEM0_OPENAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        config_dict, providers_info = build_config()

        # Basic shape validation
        assert config_dict["llm"]["provider"] == llm_provider
        assert config_dict["embedder"]["provider"] == "ollama"
        assert config_dict["vector_store"]["provider"] == "qdrant"

        # v3: no graph_store in config
        assert "graph_store" not in config_dict

        # Provider info — always includes Ollama, plus Anthropic when applicable
        provider_names = [pi["name"] for pi in providers_info]
        assert "ollama" in provider_names  # Always registered
        if llm_provider == "anthropic":
            assert "anthropic" in provider_names

        # Provider-specific config keys
        if llm_provider == "ollama":
            assert "ollama_base_url" in config_dict["llm"]["config"]
            assert "api_key" not in config_dict["llm"]["config"]
        elif llm_provider == "anthropic":
            assert "api_key" in config_dict["llm"]["config"]
            assert "max_tokens" in config_dict["llm"]["config"]


class TestDefaultModelAcrossMatrix:
    @pytest.mark.parametrize(
        "llm_provider,expected_model",
        [
            ("anthropic", "claude-opus-4-6"),
            ("ollama", "qwen3.5:4b"),
        ],
    )
    @patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-ant-api-fake")
    def test_default_model(self, mock_token, llm_provider, expected_model, monkeypatch):
        env = _make_env(llm_provider)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("MEM0_LLM_MODEL", raising=False)

        config_dict, _ = build_config()
        assert config_dict["llm"]["config"]["model"] == expected_model

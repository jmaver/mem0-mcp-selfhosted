"""Tests for config.py — build_config() with various env var combinations."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestBuildConfig:
    def _build_with_env(self, env: dict):
        """Build config with the given env vars, mocking resolve_token."""
        # Clear env vars that could leak from integration tests or CLI
        leak_keys = [k for k in os.environ if k.startswith("MEM0_")]
        with patch.dict("os.environ", env, clear=False) as patched_env:
            for k in leak_keys:
                if k not in env:
                    patched_env.pop(k, None)
            with patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-test-token"):
                from mem0_mcp_selfhosted.config import build_config

                config_dict, providers_info = build_config()
                return config_dict, providers_info

    def test_defaults(self):
        """All defaults applied when no env vars set."""
        config_dict, provider_info = self._build_with_env({})

        assert config_dict["llm"]["provider"] == "anthropic"
        assert config_dict["llm"]["config"]["model"] == "claude-opus-4-6"
        assert config_dict["embedder"]["provider"] == "ollama"
        assert config_dict["embedder"]["config"]["model"] == "bge-m3"
        assert config_dict["vector_store"]["provider"] == "qdrant"
        assert config_dict["vector_store"]["config"]["collection_name"] == "mem0_mcp_selfhosted"
        assert "graph_store" not in config_dict
        assert config_dict["version"] == "v1.1"

    def test_env_overrides(self):
        """Environment variables override defaults."""
        env = {
            "MEM0_LLM_MODEL": "claude-sonnet-4-5-20250929",
            "MEM0_EMBED_MODEL": "nomic-embed-text",
            "MEM0_COLLECTION": "custom_collection",
        }
        config_dict, *_ = self._build_with_env(env)

        assert config_dict["llm"]["config"]["model"] == "claude-sonnet-4-5-20250929"
        assert config_dict["embedder"]["config"]["model"] == "nomic-embed-text"
        assert config_dict["vector_store"]["config"]["collection_name"] == "custom_collection"

    def test_no_graph_store_in_config(self):
        """v3: graph_store never appears in config dict."""
        config_dict, *_ = self._build_with_env({})
        assert "graph_store" not in config_dict

    def test_build_config_returns_two_tuple(self):
        """v3: build_config returns (config_dict, providers_info) — 2-tuple only."""
        result = self._build_with_env({})
        assert len(result) == 2
        config_dict, providers_info = result
        assert isinstance(config_dict, dict)
        assert isinstance(providers_info, list)

    def test_explicit_embedder_provider(self):
        """Embedder provider is always explicit (never default to openai)."""
        config_dict, *_ = self._build_with_env({})
        assert config_dict["embedder"]["provider"] == "ollama"

    def test_provider_info_structure(self):
        """Provider info list includes Anthropic and Ollama entries."""
        _, providers_info = self._build_with_env({})

        provider_names = [pi["name"] for pi in providers_info]
        assert "ollama" in provider_names  # Always registered
        assert "anthropic" in provider_names

        anthropic_pi = next(pi for pi in providers_info if pi["name"] == "anthropic")
        assert "AnthropicOATLLM" in anthropic_pi["class_path"]

        ollama_pi = next(pi for pi in providers_info if pi["name"] == "ollama")
        assert "OllamaToolLLM" in ollama_pi["class_path"]

    def test_qdrant_optional_fields(self):
        """Optional Qdrant fields only included when env vars set."""
        config_dict, *_ = self._build_with_env({})
        assert "api_key" not in config_dict["vector_store"]["config"]

        env = {"MEM0_QDRANT_API_KEY": "test-key"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["vector_store"]["config"]["api_key"] == "test-key"

    # --- Provider selection and config branching (7.x) ---

    def test_default_llm_provider_is_anthropic(self):
        """Default provider is anthropic when MEM0_LLM_PROVIDER not set."""
        config_dict, *_ = self._build_with_env({})
        assert config_dict["llm"]["provider"] == "anthropic"

    def test_ollama_llm_provider(self):
        """MEM0_LLM_PROVIDER=ollama sets the LLM provider to ollama."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["provider"] == "ollama"

    def test_unsupported_llm_provider_raises(self):
        """Unsupported MEM0_LLM_PROVIDER raises ValueError."""
        leak_keys = [k for k in os.environ if k.startswith("MEM0_")]
        env = {"MEM0_LLM_PROVIDER": "gemini"}
        with patch.dict("os.environ", env, clear=False) as patched_env:
            for k in leak_keys:
                if k not in env:
                    patched_env.pop(k, None)
            with patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-test"):
                from mem0_mcp_selfhosted.config import build_config

                with pytest.raises(ValueError, match="Unsupported MEM0_LLM_PROVIDER='gemini'"):
                    build_config()

    def test_anthropic_config_has_api_key_and_max_tokens(self):
        """Anthropic LLM config includes api_key and max_tokens."""
        config_dict, *_ = self._build_with_env({})
        llm_cfg = config_dict["llm"]["config"]
        assert llm_cfg["api_key"] == "sk-test-token"
        assert llm_cfg["max_tokens"] == 16384

    def test_ollama_config_has_base_url_no_api_key(self):
        """Ollama LLM config includes ollama_base_url, no api_key or max_tokens."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        config_dict, *_ = self._build_with_env(env)
        llm_cfg = config_dict["llm"]["config"]
        assert "ollama_base_url" in llm_cfg
        assert "api_key" not in llm_cfg
        assert "max_tokens" not in llm_cfg

    def test_ollama_default_model(self):
        """Ollama provider defaults to qwen3:14b when MEM0_LLM_MODEL not set."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["config"]["model"] == "qwen3:14b"

    def test_ollama_llm_url_custom(self):
        """MEM0_LLM_URL sets ollama_base_url when provider is ollama."""
        env = {"MEM0_LLM_PROVIDER": "ollama", "MEM0_LLM_URL": "http://gpu:11434"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["config"]["ollama_base_url"] == "http://gpu:11434"

    def test_ollama_llm_url_default(self):
        """MEM0_LLM_URL defaults to localhost:11434 when not set."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"

    def test_llm_url_not_read_for_anthropic(self):
        """MEM0_LLM_URL is not included in anthropic config."""
        env = {"MEM0_LLM_URL": "http://gpu:11434"}
        config_dict, *_ = self._build_with_env(env)
        assert "ollama_base_url" not in config_dict["llm"]["config"]

    # --- Conditional provider registration (8.x) ---

    def test_providers_info_includes_anthropic(self):
        """providers_info includes Anthropic when LLM provider is anthropic."""
        _, providers_info = self._build_with_env({})
        provider_names = [pi["name"] for pi in providers_info]
        assert "anthropic" in provider_names
        assert "ollama" in provider_names  # Always included

    def test_providers_info_ollama_only(self):
        """providers_info includes only Ollama when LLM provider is ollama."""
        env = {"MEM0_LLM_PROVIDER": "ollama"}
        _, providers_info = self._build_with_env(env)
        provider_names = [pi["name"] for pi in providers_info]
        assert "ollama" in provider_names
        assert "anthropic" not in provider_names

    # --- Qdrant timeout (11.x) ---

    def test_qdrant_timeout_creates_preconfigured_client(self):
        """MEM0_QDRANT_TIMEOUT creates a pre-configured QdrantClient via 'client' field."""
        env = {"MEM0_QDRANT_TIMEOUT": "30"}
        config_dict, *_ = self._build_with_env(env)
        vc = config_dict["vector_store"]["config"]
        # "timeout" must NOT be a direct key (QdrantConfig rejects it)
        assert "timeout" not in vc
        # A pre-configured QdrantClient should be in the "client" field
        from qdrant_client import QdrantClient

        assert isinstance(vc["client"], QdrantClient)

    def test_qdrant_timeout_absent_when_not_set(self):
        """No client or timeout key in vector_config when MEM0_QDRANT_TIMEOUT is not set."""
        config_dict, *_ = self._build_with_env({})
        vc = config_dict["vector_store"]["config"]
        assert "timeout" not in vc
        assert "client" not in vc

    # --- MEM0_PROVIDER cascade (12.x) ---

    def test_mem0_provider_cascades_to_llm(self):
        """MEM0_PROVIDER=ollama alone sets LLM provider to ollama."""
        env = {"MEM0_PROVIDER": "ollama"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["provider"] == "ollama"

    def test_llm_provider_overrides_mem0_provider(self):
        """MEM0_LLM_PROVIDER takes precedence over MEM0_PROVIDER."""
        env = {"MEM0_PROVIDER": "ollama", "MEM0_LLM_PROVIDER": "anthropic"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["llm"]["provider"] == "anthropic"

    def test_neither_provider_set_defaults_to_anthropic(self):
        """Neither MEM0_PROVIDER nor MEM0_LLM_PROVIDER → defaults to anthropic."""
        config_dict, *_ = self._build_with_env({})
        assert config_dict["llm"]["provider"] == "anthropic"

    def test_mem0_provider_does_not_cascade_to_embed(self):
        """MEM0_PROVIDER does NOT cascade to embed provider (stays ollama)."""
        env = {"MEM0_PROVIDER": "anthropic"}
        config_dict, *_ = self._build_with_env(env)
        assert config_dict["embedder"]["provider"] == "ollama"

    def test_invalid_mem0_provider_raises_valueerror(self):
        """Invalid MEM0_PROVIDER raises ValueError."""
        leak_keys = [k for k in os.environ if k.startswith("MEM0_")]
        env = {"MEM0_PROVIDER": "unsupported"}
        with patch.dict("os.environ", env, clear=False) as patched_env:
            for k in leak_keys:
                if k not in env:
                    patched_env.pop(k, None)
            with patch("mem0_mcp_selfhosted.config.resolve_token", return_value="sk-test"):
                from mem0_mcp_selfhosted.config import build_config

                with pytest.raises(ValueError, match="Unsupported MEM0_PROVIDER"):
                    build_config()

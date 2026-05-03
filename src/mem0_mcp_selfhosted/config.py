"""Environment-driven configuration for mem0-mcp-selfhosted.

Reads all config from env vars with sensible defaults, constructs a
mem0ai MemoryConfig dict, and returns provider registration info.
"""

from __future__ import annotations

from typing import Any, TypedDict

from mem0_mcp_selfhosted.auth import resolve_token
from mem0_mcp_selfhosted.env import bool_env, env, opt_env


class ProviderInfo(TypedDict):
    """Custom LLM provider registration info for LlmFactory."""

    name: str
    class_path: str


def _resolve_ollama_url(*env_keys: str) -> str:
    """Resolve the Ollama base URL from a priority chain of env vars.

    Checks each key in *env_keys* first, then falls back to
    ``MEM0_OLLAMA_URL``, then ``"http://localhost:11434"``.
    """
    for key in env_keys:
        val = env(key)
        if val:
            return val
    return env("MEM0_OLLAMA_URL") or "http://localhost:11434"


def build_config() -> tuple[dict[str, Any], list[ProviderInfo]]:
    """Build mem0ai MemoryConfig dict and provider registration info.

    Returns:
        (config_dict, providers_info) where:
        - providers_info: list of ProviderInfo dicts (name + class_path)
    """
    token = resolve_token()

    # --- Top-level provider default (cascades to LLM provider when unset) ---
    _provider_default = env("MEM0_PROVIDER", "anthropic")
    _supported_llm_providers = ("anthropic", "ollama", "openai")
    if _provider_default not in _supported_llm_providers:
        raise ValueError(f"Unsupported MEM0_PROVIDER={_provider_default!r}. Supported: {list(_supported_llm_providers)}")

    # --- LLM ---
    llm_provider = env("MEM0_LLM_PROVIDER", _provider_default)
    if llm_provider not in _supported_llm_providers:
        raise ValueError(f"Unsupported MEM0_LLM_PROVIDER={llm_provider!r}. Supported: {list(_supported_llm_providers)}")

    _llm_model_defaults = {"anthropic": "claude-opus-4-6", "ollama": "qwen3:14b", "openai": ""}
    llm_model = env("MEM0_LLM_MODEL", _llm_model_defaults[llm_provider])
    if llm_provider == "openai" and not llm_model:
        raise ValueError("MEM0_LLM_MODEL is required for the 'openai' provider (e.g. 'qwen3-14b')")
    llm_max_tokens = int(env("MEM0_LLM_MAX_TOKENS", "16384"))

    llm_config: dict[str, Any] = {"model": llm_model}
    if llm_provider == "anthropic":
        llm_config["max_tokens"] = llm_max_tokens
        if token:
            llm_config["api_key"] = token
    elif llm_provider == "ollama":
        llm_config["ollama_base_url"] = _resolve_ollama_url("MEM0_LLM_URL")
    elif llm_provider == "openai":
        llm_config["max_tokens"] = llm_max_tokens
        openai_base_url = opt_env("MEM0_LLM_URL")
        if openai_base_url:
            llm_config["openai_base_url"] = openai_base_url
        llm_config["api_key"] = opt_env("MEM0_OPENAI_API_KEY") or "not-needed"

    # --- Embedder ---
    _supported_embed_providers = ("ollama", "openai")
    embed_provider = env("MEM0_EMBED_PROVIDER", "ollama")
    if embed_provider not in _supported_embed_providers:
        raise ValueError(f"Unsupported MEM0_EMBED_PROVIDER={embed_provider!r}. Supported: {list(_supported_embed_providers)}")
    _embed_model_defaults = {"ollama": "bge-m3", "openai": "text-embedding-3-small"}
    _embed_dims_defaults = {"ollama": 1024, "openai": 1536}
    embed_model = env("MEM0_EMBED_MODEL", _embed_model_defaults[embed_provider])
    embed_dims = int(env("MEM0_EMBED_DIMS", str(_embed_dims_defaults[embed_provider])))

    embedder_config: dict[str, Any] = {
        "model": embed_model,
    }
    if embed_provider == "ollama":
        embedder_config["ollama_base_url"] = _resolve_ollama_url("MEM0_EMBED_URL")
    elif embed_provider == "openai":
        openai_embed_url = opt_env("MEM0_EMBED_URL") or opt_env("MEM0_LLM_URL")
        if openai_embed_url:
            embedder_config["openai_base_url"] = openai_embed_url
        embedder_config["api_key"] = opt_env("MEM0_OPENAI_API_KEY") or "not-needed"

    # --- Vector Store ---
    qdrant_url = env("MEM0_QDRANT_URL", "http://localhost:6333")
    collection = env("MEM0_COLLECTION", "mem0_mcp_selfhosted")
    qdrant_api_key = opt_env("MEM0_QDRANT_API_KEY")
    qdrant_on_disk = bool_env("MEM0_QDRANT_ON_DISK")

    vector_config: dict[str, Any] = {
        "collection_name": collection,
        "url": qdrant_url,
        "embedding_model_dims": embed_dims,
    }
    if qdrant_api_key:
        vector_config["api_key"] = qdrant_api_key
    if qdrant_on_disk:
        vector_config["on_disk"] = True
    qdrant_timeout = opt_env("MEM0_QDRANT_TIMEOUT")
    if qdrant_timeout:
        # QdrantConfig's Pydantic model does not accept "timeout" directly.
        # Create a pre-configured QdrantClient with the timeout and pass it
        # via the "client" field, which mem0ai uses as-is.
        from qdrant_client import QdrantClient

        client_kwargs: dict[str, Any] = {
            "url": qdrant_url,
            "timeout": int(qdrant_timeout),
        }
        if qdrant_api_key:
            client_kwargs["api_key"] = qdrant_api_key
        vector_config["client"] = QdrantClient(**client_kwargs)

    # --- History ---
    history_db_path = opt_env("MEM0_HISTORY_DB_PATH")

    # --- Build config dict ---
    config_dict: dict[str, Any] = {
        "llm": {
            "provider": llm_provider,
            "config": llm_config,
        },
        "embedder": {
            "provider": embed_provider,  # Explicit — never rely on mem0ai's openai default
            "config": embedder_config,
        },
        "vector_store": {
            "provider": "qdrant",
            "config": vector_config,
        },
        "version": "v1.1",
    }

    if history_db_path:
        config_dict["history_db_path"] = history_db_path

    # --- Provider registration info ---
    # Always register custom Ollama provider — strict superset of upstream
    # OllamaLLM (restores tool-calling removed in mem0ai PR #3241).
    # Registering even when not used has no side effects.
    providers_info: list[ProviderInfo] = [
        {
            "name": "ollama",
            "class_path": "mem0_mcp_selfhosted.llm_ollama.OllamaToolLLM",
        },
    ]
    # Register OpenAI-compat provider when openai is used — overrides
    # mem0ai's built-in OpenAILLM to strip unsupported json_object response_format
    # (LM Studio / vLLM / llama.cpp only accept json_schema or text).
    if llm_provider == "openai":
        providers_info.append(
            {
                "name": "openai",
                "class_path": "mem0_mcp_selfhosted.llm_openai_compat.OpenAICompatLLM",
            }
        )

    if llm_provider == "anthropic":
        providers_info.append(
            {
                "name": "anthropic",
                "class_path": "mem0_mcp_selfhosted.llm_anthropic.AnthropicOATLLM",
            }
        )

    return config_dict, providers_info

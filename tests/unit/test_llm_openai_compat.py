"""Tests for llm_openai_compat.py — OpenAICompatLLM response_format stripping."""

from __future__ import annotations

from unittest.mock import patch


class TestOpenAICompatLLM:
    """Test that OpenAICompatLLM strips json_object response_format."""

    def _make_llm(self):
        """Instantiate OpenAICompatLLM with a minimal mocked config."""
        with patch("mem0.llms.openai.OpenAI"):
            from mem0_mcp_selfhosted.llm_openai_compat import OpenAICompatLLM
            from mem0.configs.llms.openai import OpenAIConfig

            config = OpenAIConfig(model="test-model", api_key="not-needed")
            return OpenAICompatLLM(config)

    def test_json_object_is_replaced_with_text(self):
        """response_format={'type': 'json_object'} is replaced with {'type': 'text'} before super()."""
        llm = self._make_llm()

        with patch("mem0.llms.openai.OpenAILLM.generate_response") as mock_super:
            mock_super.return_value = {"content": "ok", "tool_calls": None}
            llm.generate_response(
                messages=[{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )

        _, kwargs = mock_super.call_args
        assert kwargs["response_format"] == {"type": "text"}

    def test_json_schema_is_passed_through_unchanged(self):
        """response_format={'type': 'json_schema', ...} is forwarded as-is."""
        llm = self._make_llm()
        schema_format = {"type": "json_schema", "json_schema": {"name": "Foo", "schema": {}}}

        with patch("mem0.llms.openai.OpenAILLM.generate_response") as mock_super:
            mock_super.return_value = {"content": "ok", "tool_calls": None}
            llm.generate_response(
                messages=[{"role": "user", "content": "hi"}],
                response_format=schema_format,
            )

        _, kwargs = mock_super.call_args
        assert kwargs["response_format"] is schema_format

    def test_none_response_format_is_passed_through(self):
        """response_format=None is forwarded as-is."""
        llm = self._make_llm()

        with patch("mem0.llms.openai.OpenAILLM.generate_response") as mock_super:
            mock_super.return_value = {"content": "ok", "tool_calls": None}
            llm.generate_response(
                messages=[{"role": "user", "content": "hi"}],
                response_format=None,
            )

        _, kwargs = mock_super.call_args
        assert kwargs["response_format"] is None


class TestGpt5MaxTokensTranslation:
    """gpt-5.x rejects max_tokens — translate to max_completion_tokens, don't drop."""

    def _make_llm(self, model: str):
        with patch("mem0.llms.openai.OpenAI"):
            from mem0_mcp_selfhosted.llm_openai_compat import OpenAICompatLLM
            from mem0.configs.llms.openai import OpenAIConfig

            config = OpenAIConfig(model=model, api_key="not-needed")
            return OpenAICompatLLM(config)

    def test_gpt5_translates_max_tokens(self):
        """For gpt-5 models, max_tokens is renamed to max_completion_tokens (not dropped).

        Regression for codex review: prior code dropped max_tokens entirely, making
        MEM0_LLM_MAX_TOKENS silently ineffective on gpt-5 deployments.
        """
        llm = self._make_llm("gpt-5.4-nano")
        params = llm._get_common_params(max_tokens=4096)
        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 4096

    def test_chatgpt5_also_translates(self):
        llm = self._make_llm("chatgpt-5-mini")
        params = llm._get_common_params(max_tokens=8192)
        assert "max_tokens" not in params
        assert params["max_completion_tokens"] == 8192

    def test_non_gpt5_keeps_max_tokens(self):
        """Non-gpt-5 models keep max_tokens as-is."""
        llm = self._make_llm("qwen3-14b")
        params = llm._get_common_params(max_tokens=4096)
        assert params["max_tokens"] == 4096
        assert "max_completion_tokens" not in params

    def test_gpt5_default_max_tokens_also_translated(self):
        """The mem0 OpenAILLM default (2000) also gets translated for gpt-5 models."""
        llm = self._make_llm("gpt-5-turbo")
        params = llm._get_common_params()
        assert "max_tokens" not in params
        assert "max_completion_tokens" in params
        assert params["max_completion_tokens"] == 2000

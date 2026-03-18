"""OpenAI-compatible LLM provider for local servers (LM Studio, vLLM, llama.cpp).

Subclasses mem0ai's OpenAILLM and transforms unsupported response_format values.
LM Studio (and many local servers) reject {"type": "json_object"} — they only
accept {"type": "json_schema"} or {"type": "text"}.  mem0ai's prompts already
request JSON explicitly, so falling back to "text" is safe: the model still
returns JSON and mem0ai's JSON extraction logic handles it.

Also strips artifacts that some local models leak into response content:
- <|im_end|> and similar EOS tokens (Qwen family via LM Studio)
- <think>...</think> blocks (Qwen3 reasoning mode)
- Markdown code fences (```json ... ```)

Sanitizes EOS token strings from outgoing message content so that stored
memories containing literal EOS token text (e.g. "<|im_end|>") don't cause
the model to stop generation prematurely mid-response.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from mem0.llms.openai import OpenAILLM

logger = logging.getLogger(__name__)

# Matches <think>...</think> blocks including any leading/trailing whitespace
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Matches ```[lang]\n...\n``` code fences
_FENCE_RE = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)
# EOS tokens leaked by some models
_EOS_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|end|>")


def _clean_response(text: str) -> str:
    """Strip thinking blocks, EOS tokens, and markdown fences from LLM output."""
    text = _THINK_RE.sub("", text)
    for tok in _EOS_TOKENS:
        text = text.replace(tok, "")
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1)
    return text.strip()


def _sanitize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Replace EOS token strings in message content with safe placeholders.

    Stored memories may contain literal EOS token text (e.g. "<|im_end|>").
    When those memories are fed back into a prompt, the model generates the
    EOS token as output and LM Studio stops generation mid-JSON.  Replacing
    them with their HTML-entity equivalent is invisible to the model's
    reasoning but prevents premature generation termination.
    """
    if not any(tok in str(m.get("content", "")) for m in messages for tok in _EOS_TOKENS):
        return messages
    sanitized = []
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str) and any(tok in content for tok in _EOS_TOKENS):
            m = {**m, "content": content}
            for tok in _EOS_TOKENS:
                # Replace angle brackets so the token is no longer syntactically
                # recognised by the tokenizer (e.g. <|im_end|> → &lt;|im_end|&gt;)
                safe = tok.replace("<", "&lt;").replace(">", "&gt;")
                m["content"] = m["content"].replace(tok, safe)
            logger.debug("OpenAICompatLLM: sanitized EOS tokens in message content")
        sanitized.append(m)
    return sanitized


class OpenAICompatLLM(OpenAILLM):
    """Drop-in replacement for OpenAILLM that strips json_object response_format
    and cleans up response artifacts from local model servers.

    Registered as the "openai" provider so it transparently replaces the
    built-in when MEM0_LLM_PROVIDER=openai is used with a local server.
    """

    def generate_response(
        self,
        messages: list[dict[str, str]],
        response_format: Any = None,
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
        **kwargs: Any,
    ):
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            logger.debug(
                "OpenAICompatLLM: replacing unsupported response_format={'type': 'json_object'} "
                "with {'type': 'text'} for local server compatibility"
            )
            response_format = {"type": "text"}

        messages = _sanitize_messages(messages)

        result = super().generate_response(
            messages=messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        if isinstance(result, str):
            cleaned = _clean_response(result)
            if cleaned != result:
                logger.debug("OpenAICompatLLM: cleaned response artifacts")
            return cleaned

        return result

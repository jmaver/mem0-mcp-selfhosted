"""One-shot probe: dump the qwen36 response shape so the parser can be fixed
without guessing at where the reply text lives in ``response.content``.
"""

from __future__ import annotations

import json
import os
import sys

import anthropic

from mem0_mcp_selfhosted.llm_anthropic import _extract_text_block


def main() -> int:
    base_url = os.environ.get("MEM0_QWEN_ANTHROPIC_BASE_URL", "")
    api_key = os.environ.get("MEM0_QWEN_ANTHROPIC_AUTH_TOKEN", "")
    model = os.environ.get("MEM0_QWEN_ANTHROPIC_MODEL", "")
    missing = [n for n, v in [
        ("MEM0_QWEN_ANTHROPIC_BASE_URL", base_url),
        ("MEM0_QWEN_ANTHROPIC_AUTH_TOKEN", api_key),
        ("MEM0_QWEN_ANTHROPIC_MODEL", model),
    ] if not v]
    if missing:
        print(f"Missing env: {missing}", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=api_key, base_url=base_url)

    print(f"endpoint: {base_url}")
    print(f"model:    {model}")
    print()

    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[
            {"role": "user", "content": "Reply with exactly: pong"},
        ],
    )

    print("=== top-level attrs ===")
    for attr in ("id", "model", "stop_reason", "stop_sequence", "type", "role", "usage"):
        print(f"  {attr}: {getattr(response, attr, '<missing>')!r}")

    print()
    print("=== content (raw) ===")
    print(f"  type(response.content): {type(response.content).__name__}")
    print(f"  len(response.content):  {len(response.content) if hasattr(response.content, '__len__') else '<not sized>'}")

    if response.content:
        for i, block in enumerate(response.content):
            print(f"  --- block[{i}] ---")
            print(f"    type(block):     {type(block).__name__}")
            print(f"    block.type:      {getattr(block, 'type', '<missing>')!r}")
            print(f"    block.text:      {getattr(block, 'text', '<missing>')!r}")
            print(f"    repr(block)[:240]: {repr(block)[:240]}")
            try:
                print(f"    model_dump:      {json.dumps(block.model_dump(), indent=2)}")
            except Exception as exc:
                print(f"    model_dump failed: {exc}")

    print()
    print("=== full response.model_dump (truncated to 800 chars) ===")
    try:
        full = json.dumps(response.model_dump(), indent=2, default=str)
        print(full[:800] + ("…" if len(full) > 800 else ""))
    except Exception as exc:
        print(f"model_dump failed: {exc}")

    print()
    print("=== _extract_text_block result ===")
    print(f"  -> {_extract_text_block(response)!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

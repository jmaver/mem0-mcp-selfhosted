"""Shared utilities for mem0-mcp-selfhosted.

- patch_gemini_parse_response(): null-content guard for mem0ai's GeminiLLM
- _mem0_call(): error wrapper for all mem0ai calls
- safe_bulk_delete(): iterate + individual delete (never memory.delete_all())
- get_default_user_id(): default user_id injection
- make_project_user_id() / search_with_project(): project-scoped memory isolation
- list_entities_facet(): Qdrant Facet API entity listing with scroll fallback
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from mem0_mcp_selfhosted.env import env

logger = logging.getLogger(__name__)


def patch_gemini_parse_response() -> None:
    """Monkey-patch mem0ai's GeminiLLM to guard against null content responses.

    The upstream ``GeminiLLM._parse_response`` accesses
    ``response.candidates[0].content.parts`` without checking that ``.content``
    is not ``None``.  When the Gemini API returns a candidate with null content
    (safety block, empty response, transient error), this raises
    ``AttributeError: 'NoneType' object has no attribute 'parts'``.

    Must be called AFTER mem0 modules are imported but BEFORE Memory.from_config().
    """
    try:
        from mem0.llms.gemini import GeminiLLM
    except ImportError:
        logger.debug("mem0.llms.gemini not available — skipping Gemini null guard patch")
        return

    original = getattr(GeminiLLM, "_parse_response", None)
    if original is None:
        logger.debug("GeminiLLM._parse_response not found — skipping patch")
        return

    def _safe_parse_response(self, response, *args, **kwargs):  # noqa: ANN001
        if response.candidates and response.candidates[0].content is not None and response.candidates[0].content.parts:
            return original(self, response, *args, **kwargs)
        logger.warning("[mem0] Gemini returned null content — returning empty string")
        return ""

    GeminiLLM._parse_response = _safe_parse_response
    logger.info("Patched GeminiLLM._parse_response for null content guard")


PROJECT_GLOBAL = "global"
"""Sentinel value for the ``project`` parameter meaning 'no project scope'."""


def get_default_user_id() -> str:
    """Get the default user_id from MEM0_USER_ID env var."""
    return env("MEM0_USER_ID", "user")


def make_project_user_id(user_id: str, project: str | None) -> str:
    """Build a project-scoped user_id.

    - ``project=None`` or ``project=PROJECT_GLOBAL`` → bare ``user_id`` (global)
    - otherwise → ``user_id:project``
    """
    if not project or project == PROJECT_GLOBAL:
        return user_id
    return f"{user_id}:{project}"


def search_with_project(
    mem: Any,
    query: str,
    user_id: str,
    project: str | None,
    **kwargs: Any,
) -> list[dict]:
    """Search memories across project scope + global, deduplicated.

    When *project* is set (and not ``"global"``), runs two searches:
    1. project-scoped (``user_id:project``)
    2. global (bare ``user_id``)

    Results are deduplicated by memory ID, project results first.
    When *project* is None or ``"global"``, searches global only.

    v3 API: ALL entity IDs (``user_id``, ``agent_id``, ``run_id``) go inside
    the ``filters`` dict — mem0 2.x ``Memory.search`` rejects them as top-level
    kwargs. ``agent_id`` and ``run_id`` accepted here as kwargs are auto-folded
    into ``filters`` so callers don't have to re-encode the v3 contract.
    Caller-supplied ``filters`` is merged with these — but a ``user_id`` value
    inside caller filters is dropped: it would otherwise override the project /
    global scope this function computed and break project isolation.
    """
    # v3 renamed `limit` to `top_k`; accept either at the helper boundary so
    # callers that still pass `limit=` don't silently fall through to the
    # mem0 default of 20 (Memory.search swallows unknown kwargs).
    top_k = kwargs.pop("top_k", kwargs.pop("limit", 15))
    extra_filters = dict(kwargs.pop("filters", None) or {})
    if "user_id" in extra_filters:
        # Project scope is authoritative; ignoring caller-supplied user_id
        # rather than silently letting it cross-scope the search.
        logger.warning("search_with_project: ignoring caller-supplied filters['user_id'] (project scope is authoritative)")
        extra_filters.pop("user_id")
    for entity_kw in ("agent_id", "run_id"):
        val = kwargs.pop(entity_kw, None)
        if val:
            extra_filters[entity_kw] = val
    seen: set[str] = set()
    merged: list[dict] = []

    def _collect(results: list[dict], scope: str) -> None:
        for r in results:
            mid = r.get("id")
            if mid and mid not in seen:
                seen.add(mid)
                r["scope"] = scope
                merged.append(r)

    def _extract(raw: Any) -> list[dict]:
        if isinstance(raw, dict):
            return raw.get("results", [])
        if isinstance(raw, list):
            return raw
        return []

    if project and project != PROJECT_GLOBAL:
        project_uid = make_project_user_id(user_id, project)
        filters = {"user_id": project_uid, **extra_filters}
        raw = mem.search(query=query, filters=filters, top_k=top_k, **kwargs)
        _collect(_extract(raw), "project")

    filters = {"user_id": user_id, **extra_filters}
    raw = mem.search(query=query, filters=filters, top_k=top_k, **kwargs)
    _collect(_extract(raw), "global")

    return merged


def _mem0_call(func: Callable, *args: Any, **kwargs: Any) -> str:
    """Wrap a mem0ai call with structured error handling.

    Returns a JSON string in all cases (success or error).
    """
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        exc_type = type(exc).__name__
        is_memory_error = any(cls.__name__ == "MemoryError" for cls in type(exc).__mro__)
        if is_memory_error:
            logger.error("Mem0 call failed: %s", exc)
            return json.dumps(
                {
                    "error": str(exc),
                    "error_code": getattr(exc, "error_code", None),
                    "details": getattr(exc, "details", None),
                    "suggestion": getattr(exc, "suggestion", None),
                },
                ensure_ascii=False,
            )
        logger.error("Unexpected error: %s", exc)
        return json.dumps(
            {"error": exc_type, "detail": str(exc)},
            ensure_ascii=False,
        )
    return json.dumps(result, ensure_ascii=False)


_BULK_PAGE_SIZE = 1000


def _iter_vector_store_list(memory: Any, filters: dict[str, Any]) -> Any:
    """Yield every record matching ``filters`` from the vector store.

    Qdrant has the only paged API among the stores mem0 supports, but its
    ``Qdrant.list()`` wrapper drops the cursor: it accepts only ``filters``
    and ``top_k`` (≤ 100 by default) and returns ``(records, next_offset)``
    from a single ``client.scroll`` call without continuing. For
    ``safe_bulk_delete`` and ``list_entities_facet``'s scroll fallback to
    actually visit every record, we have to drive ``client.scroll`` directly
    when the wrapper exposes the underlying Qdrant client. Everything else
    falls back to the wrapper's single page.
    """
    store = memory.vector_store
    qdrant_client = getattr(store, "client", None)
    create_filter = getattr(store, "_create_filter", None)
    collection_name = getattr(store, "collection_name", None)

    if qdrant_client is not None and callable(create_filter) and collection_name is not None:
        scroll_filter = create_filter(filters) if filters else None
        offset: Any = None
        while True:
            page, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=_BULK_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            yield from page
            if not next_offset or len(page) < _BULK_PAGE_SIZE:
                return
            offset = next_offset

    # Non-Qdrant vector stores: best-effort single page from the wrapper.
    # If a caller hits this with > _BULK_PAGE_SIZE records they'll silently
    # see only the first page; warn so it's at least visible in logs.
    result = store.list(filters=filters, top_k=_BULK_PAGE_SIZE)
    page = result[0] if isinstance(result, tuple) else result
    yield from page
    if len(page) >= _BULK_PAGE_SIZE:
        logger.warning(
            "vector_store has no Qdrant client; pagination unsupported. "
            "Returned %d records — additional records may exist.",
            len(page),
        )


def _extract_id(item: Any) -> str:
    if hasattr(item, "id"):
        return item.id
    if isinstance(item, dict):
        return item.get("id", "")
    return str(item)


def safe_bulk_delete(memory: Any, filters: dict[str, Any]) -> int:
    """Delete all memories matching filters; iterate+delete individually.

    Never calls memory.delete_all() (which triggers vector_store.reset()).
    Materializes the full ID list before issuing deletes so the scroll
    cursor isn't mutated mid-iteration.
    """
    ids = [_extract_id(item) for item in _iter_vector_store_list(memory, filters)]
    count = 0
    for memory_id in ids:
        try:
            memory.delete(memory_id)
            count += 1
        except Exception as exc:
            logger.warning("Failed to delete memory %s: %s", memory_id, exc)
    return count


def list_entities_facet(memory: Any) -> dict[str, list[dict]]:
    """List entities using Qdrant Facet API with scroll fallback.

    Primary: Facet API (Qdrant v1.12+) — server-side distinct value aggregation.
    Fallback: scroll+dedupe for older Qdrant versions.

    Returns: {"users": [{"value": ..., "count": ...}], "agents": [...], "runs": [...]}
    """
    client = memory.vector_store.client
    collection = memory.vector_store.collection_name

    result: dict[str, list[dict]] = {"users": [], "agents": [], "runs": []}
    entity_keys = {"users": "user_id", "agents": "agent_id", "runs": "run_id"}

    try:
        for result_key, payload_key in entity_keys.items():
            facet_response = client.facet(
                collection_name=collection,
                key=payload_key,
            )
            result[result_key] = [{"value": hit.value, "count": hit.count} for hit in facet_response.hits]
        return result
    except Exception as exc:
        logger.warning(
            "Qdrant Facet API unavailable (%s). Falling back to scroll+dedupe. Upgrade to Qdrant v1.12+ for better performance.",
            exc,
        )
        return _list_entities_scroll_fallback(memory)


def _list_entities_scroll_fallback(memory: Any) -> dict[str, list[dict]]:
    """Fallback entity listing via scroll+dedupe."""
    entities: dict[str, dict[str, int]] = {
        "user_id": {},
        "agent_id": {},
        "run_id": {},
    }

    # mem0 v3: Qdrant.list() defaults top_k=100; page through to avoid
    # silently capping entity discovery on large collections.
    for item in _iter_vector_store_list(memory, {}):
        payload = item.payload if hasattr(item, "payload") else item
        if isinstance(payload, dict):
            for key in entities:
                val = payload.get(key)
                if val:
                    entities[key][val] = entities[key].get(val, 0) + 1

    return {
        "users": [{"value": v, "count": c} for v, c in entities["user_id"].items()],
        "agents": [{"value": v, "count": c} for v, c in entities["agent_id"].items()],
        "runs": [{"value": v, "count": c} for v, c in entities["run_id"].items()],
    }

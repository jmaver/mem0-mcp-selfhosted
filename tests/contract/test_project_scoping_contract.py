"""Contract test: ``user_id`` encoded as ``user:project`` is the
source-of-truth for cross-project memory isolation.

The hook_and_scoping bench validates 3-way isolation end-to-end
(alice/projA, alice/projB, bob/projA, plus global). This file pins the
*wire-format* invariant in tests/contract/ so a regression where someone
adds a top-level ``user_id=`` kwarg (or otherwise bypasses
``make_project_user_id``) gets caught fast — before isolation breaks in
production.

Two sides of the contract:

1. mem0 v3 only honors ``user_id`` inside ``filters={}`` (the
   ``test_memory_search_no_top_level_user_id`` check in
   ``test_mem0ai_internals.py`` locks this on mem0's side).
2. ``search_with_project`` always encodes the project scope as
   ``filters={"user_id": user:project | user}`` and never reaches mem0
   with a top-level ``user_id``. That's what this file pins.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.contract


def _all_search_kwargs(mock_mem: MagicMock) -> list[dict]:
    return [c.kwargs for c in mock_mem.search.call_args_list]


def test_project_scope_routes_through_filters_user_id():
    """``project=p`` produces filters['user_id'] == 'user:p' on the project leg
    and filters['user_id'] == 'user' on the global leg, with *zero* top-level
    ``user_id=`` kwargs to mem.search."""
    from mem0_mcp_selfhosted.helpers import search_with_project

    mock_mem = MagicMock()
    mock_mem.search.return_value = {"results": []}

    search_with_project(mock_mem, "q", "alice", "projA")

    calls = _all_search_kwargs(mock_mem)
    assert len(calls) == 2, "project scope must run project + global searches"

    project_filters, global_filters = calls[0]["filters"], calls[1]["filters"]
    assert project_filters["user_id"] == "alice:projA", "INVARIANT: project leg must filter on user:project"
    assert global_filters["user_id"] == "alice", "INVARIANT: global leg must filter on bare user_id"

    for kwargs in calls:
        assert "user_id" not in kwargs, (
            "INVARIANT BROKEN: user_id leaked to top-level kwargs on mem.search. "
            "v3 only honors user_id inside filters={}; a top-level kwarg silently "
            "skips project isolation."
        )


def test_caller_filters_cannot_override_project_scope():
    """A caller-supplied ``filters={'user_id': ...}`` must be dropped, not
    forwarded — otherwise project scope becomes a suggestion."""
    from mem0_mcp_selfhosted.helpers import search_with_project

    mock_mem = MagicMock()
    mock_mem.search.return_value = {"results": []}

    search_with_project(
        mock_mem,
        "q",
        "alice",
        "projA",
        filters={"user_id": "bob", "category": {"eq": "note"}},
    )

    for kwargs in _all_search_kwargs(mock_mem):
        assert kwargs["filters"]["user_id"] in ("alice:projA", "alice"), (
            "INVARIANT BROKEN: caller-supplied filters['user_id'] overrode "
            "project scope. Project scope must be authoritative."
        )
        assert kwargs["filters"]["category"] == {"eq": "note"}, "non-entity filters must pass through"


def test_make_project_user_id_encoding_is_user_colon_project():
    """The encoding contract: ``user:project`` (colon-delimited). If this ever
    changes shape, every persisted memory's user_id payload becomes
    unsearchable under the new encoding — so this is a hard invariant."""
    from mem0_mcp_selfhosted.helpers import PROJECT_GLOBAL, make_project_user_id

    assert make_project_user_id("alice", "projA") == "alice:projA"
    assert make_project_user_id("alice", PROJECT_GLOBAL) == "alice"
    assert make_project_user_id("alice", None) == "alice"
    assert make_project_user_id("alice", "") == "alice"

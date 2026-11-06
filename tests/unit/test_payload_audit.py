"""主管载荷瘦身与审计辅助（prune_empty / estimate_tokens）。"""

from __future__ import annotations

from so_agent.workflow_support import estimate_tokens, prune_empty


def test_prune_empty_removes_none_and_empty_containers():
    payload = {
        "phase": "planning",
        "objective": "目标",
        "empty_list": [],
        "empty_dict": {},
        "empty_str": "",
        "none": None,
    }
    assert prune_empty(payload) == {"phase": "planning", "objective": "目标"}


def test_prune_empty_keeps_semantic_falsy_values():
    payload = {
        "success": False,
        "count": 0,
        "score": 0.0,
        "nested": {"allowed": False, "items": [0, False]},
    }
    assert prune_empty(payload) == payload


def test_prune_empty_recurses_into_lists_and_keeps_order():
    payload = {
        "items": [
            {"id": "a", "note": ""},
            {"id": "b", "note": None},
            "x",
        ]
    }
    assert prune_empty(payload) == {"items": [{"id": "a"}, {"id": "b"}, "x"]}


def test_estimate_tokens_counts_cjk_and_latin_separately():
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好世界") == 4
    assert estimate_tokens("abcd") == 1

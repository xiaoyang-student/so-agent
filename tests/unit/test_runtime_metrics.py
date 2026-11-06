"""运行指标事件聚合与重启恢复合并。"""

from __future__ import annotations

from so_agent.runtime.metrics import aggregate_events, merge_event_summaries


def test_aggregate_events_counts_known_types():
    summary = aggregate_events(
        [
            [{"event_type": "tool_called"}, {"event_type": "tool_called"}],
            [{"event_type": "tool_denied"}],
        ]
    )
    assert summary["total"] == 3
    assert summary["tool_calls"] == 2
    assert summary["tool_denials"] == 1


def test_merge_event_summaries_keeps_counts_after_restart():
    live = aggregate_events([[{"event_type": "tool_called"}]])
    stored = aggregate_events(
        [
            [
                {"event_type": "tool_called"},
                {"event_type": "replan"},
                {"event_type": "human_escalation"},
            ]
        ]
    )
    merged = merge_event_summaries(live, stored)
    assert merged["total"] == 4
    assert merged["tool_calls"] == 2
    assert merged["replans"] == 1
    assert merged["human_escalations"] == 1
    assert merged["by_type"]["tool_called"] == 2

"""评审输出预算：只裁剪叙述性字段，不改变评审事实与必填语义。"""

from __future__ import annotations

from so_agent.config import Settings
from so_agent.models import ReviewDecision
from so_agent.workflow_supervisor import _clip_items, _trim_review_decision


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


class TestClipHelpers:
    def test_short_values_untouched(self):
        assert _clip_items(["a", "b"], 5, 10) == ["a", "b"]

    def test_items_and_chars_clipped_with_markers(self):
        clipped = _clip_items(["abcdef", "ghijkl", "mnopqr"], 2, 3)
        assert clipped == ["abc…（已截断）", "ghi…（已截断）", "…（另有 1 条已省略）"]

    def test_zero_limits_disable_clipping(self):
        assert _clip_items(["abcdef"], 0, 0) == ["abcdef"]


class TestTrimReviewDecision:
    def test_pass_decision_is_length_trimmed_only(self):
        decision = ReviewDecision(
            stage="plan_review",
            passed=True,
            issues=["x" * 30],
            suggestions=[f"s{index}" for index in range(5)],
            summary="y" * 50,
        )
        trimmed = _trim_review_decision(
            decision,
            _settings(
                review_issues_max_items=1,
                review_suggestions_max_items=2,
                review_item_max_chars=10,
                review_summary_max_chars=12,
            ),
        )
        assert trimmed.passed is True
        assert trimmed.required_fixes == []
        assert len(trimmed.issues) == 1
        assert len(trimmed.suggestions) == 3  # 2 条 + 省略标记
        assert trimmed.suggestions[-1] == "…（另有 3 条已省略）"
        assert trimmed.summary.startswith("y" * 12)
        assert trimmed.summary.endswith("…（已截断）")

    def test_fail_decision_keeps_required_fixes_non_empty(self):
        decision = ReviewDecision(
            stage="result_review",
            passed=False,
            required_fixes=[f"fix{index}" for index in range(10)],
        )
        trimmed = _trim_review_decision(
            decision,
            _settings(
                review_required_fixes_max_items=1,
                review_item_max_chars=10,
            ),
        )
        assert trimmed.passed is False
        assert trimmed.required_fixes
        assert trimmed.required_fixes[0] == "fix0"

    def test_defaults_are_applied_from_settings(self):
        decision = ReviewDecision(
            stage="plan_review",
            passed=True,
            summary="z" * 1000,
        )
        trimmed = _trim_review_decision(decision, _settings())
        assert len(trimmed.summary) <= 400 + len("…（已截断）")

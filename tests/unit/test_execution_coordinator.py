"""单元测试：ExecutionCoordinator._parse_execution_result 降级路径防御。

背景：主管 Agent 偶发返回内容为空壳或语义为"待执行/未完成"的 JSON，
降级抢救路径（快速契约校验失败后从内嵌 JSON 提取字段）可能把无效
结果误判为成功并混入最终交付物。两道防御：

- 防御一（output 有效性）：降级判定为成功但 output 为空/长度 < 20
  字符时，翻转为 ``success=False``（error=输出内容过短，疑似无效结果）；
- 防御二（关键词）：output/error 携带"待执行/未完成/pending/
  not started"（大小写不敏感）时强制 ``success=False``。

同时回归验证：完整合法的 ExecutionResult 仍走快速契约路径，不受
防御影响；显式失败结果原样保留错误信息。
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

from so_agent.context import ProjectContext
from so_agent.config import Settings
from so_agent.execution_coordinator import ExecutionCoordinator, missing_write_scope
from so_agent.models import (
    ExecutionResult,
    SubtaskSpec,
    TaskRequest,
    VerificationRule,
)
from so_agent.runtime.events import (
    EVENT_TOOL_CALLED,
    EVENT_TOOL_EVIDENCE_MISSING,
    EVENT_VERIFICATION_FAILED,
    EVENT_WRITE_SCOPE_UNMET,
    EventLogger,
)
from so_agent.state_machine import WorkflowState


def make_spec(subtask_id: str = "s1") -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"模拟子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        allowed_tools=[],
        max_attempts=3,
    )


def parse(text: str, subtask_id: str = "s1") -> ExecutionResult:
    """直接调用被测静态方法（static，无需构造协调器）。"""
    return ExecutionCoordinator._parse_execution_result(text, make_spec(subtask_id))


# 长度 ≥ 20 的正常产出（不触发防御一，只用于隔离防御二）
_LONG_OUTPUT = "已完成数据分析并生成完整报告，包含结论与建议明细"


class TestMergedTurnBudget:
    """合并步骤的回合上限只作用于合并产生的那一个子任务。"""

    def _coordinator(self, **settings) -> ExecutionCoordinator:
        coordinator = object.__new__(ExecutionCoordinator)
        coordinator._deps = SimpleNamespace(
            settings=Settings(_env_file=None, **settings)
        )
        coordinator._state = WorkflowState()
        return coordinator

    def test_merged_subtask_gets_configured_bonus(self):
        coordinator = self._coordinator(
            max_model_turns=8, plan_merge_turn_bonus=6
        )
        coordinator._state.merged_subtask_ids = {"s1"}

        assert coordinator._merged_turn_budget(make_spec("s1")) == 14

    def test_unmerged_subtask_keeps_default_budget(self):
        coordinator = self._coordinator(
            max_model_turns=8, plan_merge_turn_bonus=6
        )

        assert coordinator._merged_turn_budget(make_spec("s1")) is None

    def test_zero_bonus_disables_override(self):
        coordinator = self._coordinator(
            max_model_turns=8, plan_merge_turn_bonus=0
        )
        coordinator._state.merged_subtask_ids = {"s1"}

        assert coordinator._merged_turn_budget(make_spec("s1")) is None


class TestPendingKeywordDefense:
    """防御二：output/error 携带未完成标记时强制判失败。"""

    def test_chinese_keyword_in_output_forces_failure(self):
        # 长产出 + "待执行"：只触发关键词防御
        text = json.dumps(
            {"success": True, "output": f"该子任务标记为待执行状态，{_LONG_OUTPUT}"},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False
        assert "待执行" in result.error

    def test_pending_in_output_forces_failure(self):
        text = json.dumps(
            {
                "success": True,
                "output": f"The task is still pending, {_LONG_OUTPUT}",
            },
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False
        assert "pending" in result.error

    def test_keyword_in_error_forces_failure(self):
        # output 正常但 error 声明未完成：整体强制失败（保留原 error）
        text = json.dumps(
            {
                "success": True,
                "output": _LONG_OUTPUT,
                "error": "子任务未完成：外部依赖尚未就绪",
            },
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False
        assert result.error == "子任务未完成：外部依赖尚未就绪"

    def test_not_started_case_insensitive_forces_failure(self):
        # "Not Started" 大小写不敏感命中 "not started"
        text = json.dumps(
            {
                "success": True,
                "output": _LONG_OUTPUT,
                "error": "Execution Not Started yet",
            },
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False

    def test_inferred_success_with_keyword_forces_failure(self):
        # 无 success 字段（推断成功）+ 关键词：同样强制失败
        text = json.dumps(
            {"output": f"待执行清单如下：{_LONG_OUTPUT}"},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False


class TestShortOutputDefense:
    """防御一：降级判定成功但产出为空/过短时翻转为失败。"""

    def test_short_output_flips_to_failure(self):
        text = json.dumps({"success": True, "output": "OK"}, ensure_ascii=False)
        result = parse(text)
        assert result.success is False
        assert result.error == "输出内容过短，疑似无效结果"

    def test_missing_output_flips_to_failure(self):
        # 无 output 字段：降级路径用整个候选 JSON 充当 output（空壳仍过短）
        text = json.dumps({"success": True}, ensure_ascii=False)
        result = parse(text)
        assert result.success is False
        assert "过短" in result.error

    def test_boundary_length_output_passes(self):
        # 恰好 20 字符（strip 后）：不触发防御一
        text = json.dumps(
            {"success": True, "output": "一二三四五六七八九十一二三四五六七八九十"},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is True


class TestValidResultsUnaffected:
    """回归：正常结果与显式失败不受防御影响。"""

    def test_valid_long_output_stays_success(self):
        text = json.dumps(
            {"success": True, "output": _LONG_OUTPUT},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is True
        assert result.output == _LONG_OUTPUT

    def test_inferred_success_long_output_stays_success(self):
        # 无 success 无 error：推断成功，产出充足时不误伤
        text = json.dumps({"output": _LONG_OUTPUT}, ensure_ascii=False)
        result = parse(text)
        assert result.success is True

    def test_explicit_failure_keeps_original_error(self):
        text = json.dumps(
            {"success": False, "error": "沙箱执行超时", "output": _LONG_OUTPUT},
            ensure_ascii=False,
        )
        result = parse(text)
        assert result.success is False
        assert result.error == "沙箱执行超时"

    def test_fast_path_contract_ignores_defenses(self):
        # 完整合法契约（含 subtask_id）：走快速路径，短 output 也不误伤
        payload = ExecutionResult(subtask_id="s1", success=True, output="42")
        result = parse(payload.model_dump_json())
        assert result.success is True
        assert result.subtask_id == "s1"
        assert result.output == "42"

    def test_unparseable_text_still_fails_gracefully(self):
        result = parse("这不是 JSON，也不是任何可抢救的结构")
        assert result.success is False
        assert "未能从主管 Agent 输出解析" in result.error


class TestWriteScopeVerification:
    """write_scope 确定性验收：文件系统事实优先于子 Agent 自述成功。"""

    def test_missing_file_detected(self, tmp_path):
        (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
        assert missing_write_scope(tmp_path, ["a.txt", "b.txt"]) == ["b.txt"]

    def test_backslash_and_dot_prefix_normalized(self, tmp_path):
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "report.md").write_text("ok", encoding="utf-8")
        assert missing_write_scope(tmp_path, [".\\out\\report.md"]) == []

    def test_wildcard_matches_any_file(self, tmp_path):
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "a.csv").write_text("x", encoding="utf-8")
        assert missing_write_scope(tmp_path, ["out/*.csv"]) == []
        assert missing_write_scope(tmp_path, ["out/*.json"]) == ["out/*.json"]

    def test_directory_requires_at_least_one_file(self, tmp_path):
        (tmp_path / "out").mkdir()
        assert missing_write_scope(tmp_path, ["out"]) == ["out"]
        (tmp_path / "out" / "nested").mkdir()
        (tmp_path / "out" / "nested" / "a.txt").write_text("x", encoding="utf-8")
        assert missing_write_scope(tmp_path, ["out"]) == []

    def test_traversal_and_absolute_paths_rejected(self, tmp_path):
        outside = tmp_path.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        assert missing_write_scope(tmp_path, ["../outside.txt"]) == ["../outside.txt"]
        assert missing_write_scope(tmp_path, [str(outside)]) == [str(outside)]

    def test_duplicate_entries_reported_once(self, tmp_path):
        assert missing_write_scope(tmp_path, ["x.txt", "x.txt"]) == ["x.txt"]

    def test_empty_file_does_not_count(self, tmp_path):
        (tmp_path / "empty.txt").write_text("", encoding="utf-8")
        assert missing_write_scope(tmp_path, ["empty.txt"]) == ["empty.txt"]

    def test_since_rejects_artifact_from_previous_run(self, tmp_path):
        target = tmp_path / "old.txt"
        target.write_text("old", encoding="utf-8")
        os.utime(target, (1_000_000, 1_000_000))
        assert missing_write_scope(
            tmp_path, ["old.txt"], since=2_000_000
        ) == ["old.txt"]


def make_coordinator(tmp_path) -> tuple[ExecutionCoordinator, EventLogger]:
    events = EventLogger()
    coordinator = ExecutionCoordinator.__new__(ExecutionCoordinator)
    state = WorkflowState(events=events)
    state.task_request = TaskRequest(task_id="t1", objective="demo")
    coordinator._state = state
    coordinator._deps = SimpleNamespace(
        context=ProjectContext(sandbox_dir=tmp_path, project_name="demo"),
        events=events,
    )
    return coordinator, events


def scope_spec(scope: list[str]) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id="s1",
        title="写文件",
        instructions="写文件",
        role="code",
        write_scope=scope,
    )


class TestWriteScopeEnforcement:
    def test_self_reported_success_flipped_when_file_missing(self, tmp_path):
        coordinator, events = make_coordinator(tmp_path)
        result = coordinator._enforce_write_scope(
            scope_spec(["report.md"]),
            ExecutionResult(subtask_id="s1", success=True, output="已生成完整报告内容"),
        )
        assert result.success is False
        assert "写入范围未产出：report.md" in result.error
        logged = [e for e in events.events if e["event_type"] == EVENT_WRITE_SCOPE_UNMET]
        assert logged[0]["payload"]["missing"] == ["report.md"]

    def test_success_kept_when_declared_file_exists(self, tmp_path):
        (tmp_path / "report.md").write_text("ok", encoding="utf-8")
        coordinator, events = make_coordinator(tmp_path)
        result = coordinator._enforce_write_scope(
            scope_spec(["report.md"]),
            ExecutionResult(subtask_id="s1", success=True, output="已生成完整报告内容"),
        )
        assert result.success is True
        assert not [e for e in events.events if e["event_type"] == EVENT_WRITE_SCOPE_UNMET]

    def test_stale_file_flips_success(self, tmp_path):
        target = tmp_path / "report.md"
        target.write_text("old", encoding="utf-8")
        os.utime(target, (1_000_000, 1_000_000))
        coordinator, events = make_coordinator(tmp_path)
        result = coordinator._enforce_write_scope(
            scope_spec(["report.md"]),
            ExecutionResult(subtask_id="s1", success=True, output="已生成完整报告内容"),
            since=2_000_000,
        )
        assert result.success is False
        assert "写入范围未产出：report.md" in result.error
        logged = [e for e in events.events if e["event_type"] == EVENT_WRITE_SCOPE_UNMET]
        assert logged[0]["payload"]["missing"] == ["report.md"]

    def test_empty_scope_not_checked(self, tmp_path):
        coordinator, _ = make_coordinator(tmp_path)
        result = coordinator._enforce_write_scope(
            scope_spec([]),
            ExecutionResult(subtask_id="s1", success=True, output="只读核对完成，无需落盘"),
        )
        assert result.success is True

    def test_existing_failure_not_rewritten(self, tmp_path):
        coordinator, _ = make_coordinator(tmp_path)
        result = coordinator._enforce_write_scope(
            scope_spec(["report.md"]),
            ExecutionResult(subtask_id="s1", success=False, error="沙箱执行超时"),
        )
        assert result.error == "沙箱执行超时"


def make_evidence_coordinator(
    tmp_path, *, required: bool = True
) -> tuple[ExecutionCoordinator, EventLogger]:
    coordinator, events = make_coordinator(tmp_path)
    coordinator._deps.settings = Settings(
        _env_file=None, subtask_tool_evidence_required=required
    )
    return coordinator, events


def _tool_event(
    events: EventLogger,
    *,
    subtask_id: str = "s1",
    attempt: int = 1,
    action: str = "invoke_completed",
    result_ok: bool | None = True,
) -> None:
    events.log_event(
        EVENT_TOOL_CALLED,
        {
            "task_id": "t1",
            "subtask_id": subtask_id,
            "attempt": attempt,
            "action": action,
            "result_ok": result_ok,
            "tool_name": "code_agent",
        },
    )


class TestToolEvidenceEnforcement:
    """工具证据验收：声明产出必须对应本次尝试完成的工具调用。"""

    def _success(self) -> ExecutionResult:
        return ExecutionResult(
            subtask_id="s1",
            success=True,
            output="已生成报告并写入 report.md 文件，内容完整可读",
            artifacts=["report.md"],
        )

    def test_completed_tool_call_keeps_success(self, tmp_path):
        coordinator, events = make_evidence_coordinator(tmp_path)
        _tool_event(events)
        result = coordinator._enforce_tool_evidence(
            scope_spec(["report.md"]), self._success(), 1
        )
        assert result.success is True

    def test_missing_tool_evidence_flips_to_failure(self, tmp_path):
        coordinator, events = make_evidence_coordinator(tmp_path)
        result = coordinator._enforce_tool_evidence(
            scope_spec(["report.md"]), self._success(), 1
        )
        assert result.success is False
        assert "缺少工具调用证据" in result.error
        logged = [
            e
            for e in events.events
            if e["event_type"] == EVENT_TOOL_EVIDENCE_MISSING
        ]
        assert logged[0]["payload"]["attempt"] == 1
        assert logged[0]["payload"]["declared"] == ["report.md"]

    def test_other_attempt_evidence_is_not_reused(self, tmp_path):
        coordinator, events = make_evidence_coordinator(tmp_path)
        _tool_event(events, attempt=1)
        result = coordinator._enforce_tool_evidence(
            scope_spec(["report.md"]), self._success(), 2
        )
        assert result.success is False

    def test_other_subtask_evidence_is_not_reused(self, tmp_path):
        coordinator, events = make_evidence_coordinator(tmp_path)
        _tool_event(events, subtask_id="s2")
        result = coordinator._enforce_tool_evidence(
            scope_spec(["report.md"]), self._success(), 1
        )
        assert result.success is False

    def test_failed_tool_call_is_not_evidence(self, tmp_path):
        coordinator, events = make_evidence_coordinator(tmp_path)
        _tool_event(events, action="invoke_failed", result_ok=False)
        result = coordinator._enforce_tool_evidence(
            scope_spec(["report.md"]), self._success(), 1
        )
        assert result.success is False

    def test_started_but_unsettled_call_is_not_evidence(self, tmp_path):
        coordinator, events = make_evidence_coordinator(tmp_path)
        _tool_event(events, action="invoke_started", result_ok=None)
        result = coordinator._enforce_tool_evidence(
            scope_spec(["report.md"]), self._success(), 1
        )
        assert result.success is False

    def test_no_declared_artifacts_not_checked(self, tmp_path):
        coordinator, _ = make_evidence_coordinator(tmp_path)
        result = coordinator._enforce_tool_evidence(
            scope_spec([]),
            ExecutionResult(
                subtask_id="s1",
                success=True,
                output="只读核对完成，无需落盘，结论见上",
            ),
            1,
        )
        assert result.success is True

    def test_disabled_setting_skips_check(self, tmp_path):
        coordinator, _ = make_evidence_coordinator(tmp_path, required=False)
        result = coordinator._enforce_tool_evidence(
            scope_spec(["report.md"]), self._success(), 1
        )
        assert result.success is True

    def test_existing_failure_not_rewritten(self, tmp_path):
        coordinator, _ = make_evidence_coordinator(tmp_path)
        result = coordinator._enforce_tool_evidence(
            scope_spec(["report.md"]),
            ExecutionResult(subtask_id="s1", success=False, error="沙箱超时"),
            1,
        )
        assert result.error == "沙箱超时"


class TestVerificationEnforcement:
    """可机检验收：声明的 verification 规则全部通过才保留成功。"""

    def _spec(self, *rules: VerificationRule) -> SubtaskSpec:
        return SubtaskSpec(
            subtask_id="s1",
            title="产出报告",
            instructions="产出报告",
            role="code",
            verification=list(rules),
        )

    def test_rules_pass_keep_success(self, tmp_path):
        (tmp_path / "report.md").write_text("销售总额：100", encoding="utf-8")
        coordinator, _ = make_coordinator(tmp_path)
        result = coordinator._enforce_verification(
            self._spec(
                VerificationRule(kind="file_non_empty", path="report.md"),
                VerificationRule(
                    kind="text_contains", path="report.md", text="销售总额"
                ),
            ),
            ExecutionResult(subtask_id="s1", success=True, output="完成"),
        )
        assert result.success is True

    def test_rule_failure_flips_and_logs(self, tmp_path):
        coordinator, events = make_coordinator(tmp_path)
        result = coordinator._enforce_verification(
            self._spec(VerificationRule(kind="file_exists", path="report.md")),
            ExecutionResult(subtask_id="s1", success=True, output="完成"),
        )
        assert result.success is False
        assert "可机检验收未通过" in result.error
        assert "report.md" in result.error
        logged = [
            e
            for e in events.events
            if e["event_type"] == EVENT_VERIFICATION_FAILED
        ]
        assert logged[0]["payload"]["subtask_id"] == "s1"
        assert logged[0]["payload"]["failures"]

    def test_no_rules_not_checked(self, tmp_path):
        coordinator, _ = make_coordinator(tmp_path)
        result = coordinator._enforce_verification(
            self._spec(), ExecutionResult(subtask_id="s1", success=True, output="完成")
        )
        assert result.success is True

    def test_existing_failure_not_rewritten(self, tmp_path):
        coordinator, _ = make_coordinator(tmp_path)
        result = coordinator._enforce_verification(
            self._spec(VerificationRule(kind="file_exists", path="x")),
            ExecutionResult(subtask_id="s1", success=False, error="沙箱超时"),
        )
        assert result.error == "沙箱超时"

    def test_escape_path_fails_closed(self, tmp_path):
        coordinator, _ = make_coordinator(tmp_path)
        result = coordinator._enforce_verification(
            self._spec(
                VerificationRule(
                    kind="file_exists", path=str(tmp_path.parent / "outside.txt")
                )
            ),
            ExecutionResult(subtask_id="s1", success=True, output="完成"),
        )
        assert result.success is False

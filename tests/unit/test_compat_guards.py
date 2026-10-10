"""兼容护栏：冻结对外契约与安全不变量（任务 44 基线冻结）。

与既有测试的分工：各功能模块的深度行为测试（权限、授权生命周期、
评审契约等）已存在于对应测试文件；本文件只做**契约冻结**——把
外部可见的接口形态固化为显式断言，任何破坏向后兼容的改动都会在此
先红灯：

- CLI 参数形态与默认值（main.build_parser）；
- 退出码契约（0 completed / 1 其余终态 / 2 配置错误 / 130 中断）；
- 标准输出 JSON 顶层字段集合（OrchestratorOutput 序列化形态）；
- OrchestratorOutput 字段冻结；
- WorkflowEngine.run 签名冻结；
- 事件类型名向后兼容（26 个冻结类型必须继续登记 + 常量值一一对应；
  允许新增事件，不做精确长度/全集相等）；
- 安全治理不变量：默认拒绝、任意出口撤销幂等、passed 唯一语义信号、
  星型通信、replacement 深度一、reuse_from 不参与语义指纹。
"""

from __future__ import annotations

import inspect
import json
import typing
from collections.abc import Iterable, Mapping

import pytest

from so_agent import main as main_mod
from so_agent.config import Settings
from so_agent.fingerprint import spec_fingerprint
from so_agent.main import build_parser, main
from so_agent.models import (
    ReviewDecision,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.events import ALL_EVENT_TYPES
from so_agent.runtime import events as events_mod
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.supervisor_router import SupervisorRouter
from so_agent.state_machine import _review_contract_contradiction
from so_agent.workflow import WorkflowEngine


# ---------------------------------------------------------------------------
# CLI 参数契约
# ---------------------------------------------------------------------------


class TestCliArgumentContract:
    def test_default_invocation(self):
        """位置目标 + 各可选参数的默认值冻结。"""
        args = build_parser().parse_args(["实现一个工具"])
        assert args.objective == "实现一个工具"
        assert args.constraints == ""
        assert args.acceptance_criteria == ""
        assert args.config is None
        assert args.sandbox is None
        assert args.project_name == "so-agent"
        assert args.daemon is False

    def test_full_option_invocation(self):
        """全参数形态与短选项别名冻结。"""
        args = build_parser().parse_args(
            [
                "目标",
                "-c", "约束A",
                "--acceptance-criteria", "验收B",
                "--config", "runtime_config.yaml",
                "--sandbox", "custom_sandbox",
                "--project-name", "proj",
                "--daemon",
            ]
        )
        assert args.objective == "目标"
        assert args.constraints == "约束A"
        assert args.acceptance_criteria == "验收B"
        assert args.config == "runtime_config.yaml"
        assert args.sandbox == "custom_sandbox"
        assert args.project_name == "proj"
        assert args.daemon is True

    def test_objective_optional_for_daemon(self):
        """daemon 模式允许省略任务目标。"""
        args = build_parser().parse_args(["--daemon"])
        assert args.daemon is True
        assert args.objective is None


# ---------------------------------------------------------------------------
# 退出码与 stdout 契约（FakeRouter 注入，不触网）
# ---------------------------------------------------------------------------


def _install_fake_router(monkeypatch, *, output, route_exc=None):
    """把 main 命名空间中的 TaskRouter 替换为返回固定输出的假路由器。"""

    class _FakeRouter:
        def __init__(self, context, settings, *, log_file=None):
            self.context = context
            self.settings = settings
            self.log_file = log_file

        async def route(self, objective, constraints, acceptance_criteria):
            if route_exc is not None:
                raise route_exc
            return output

    monkeypatch.setattr(main_mod, "TaskRouter", _FakeRouter)


class TestExitCodeContract:
    def test_completed_exits_zero(self, monkeypatch, capsys, tmp_path):
        output = OrchestratorOutput(
            task_id="t-1", status=TaskStatus.COMPLETED, final_result="完成"
        )
        _install_fake_router(monkeypatch, output=output)
        code = main(["--sandbox", str(tmp_path), "目标"])
        assert code == 0

    @pytest.mark.parametrize(
        "status",
        [TaskStatus.FAILED, TaskStatus.HUMAN_REQUIRED, TaskStatus.CANCELLED],
    )
    def test_non_completed_statuses_exit_one(
        self, monkeypatch, capsys, tmp_path, status
    ):
        output = OrchestratorOutput(task_id="t-1", status=status)
        _install_fake_router(monkeypatch, output=output)
        code = main(["--sandbox", str(tmp_path), "目标"])
        assert code == 1

    def test_missing_config_file_exits_two(self, capsys, tmp_path):
        code = main([
            "--config", str(tmp_path / "nope.yaml"),
            "--sandbox", str(tmp_path),
            "目标",
        ])
        assert code == 2

    def test_invalid_config_field_exits_two(self, capsys, tmp_path):
        config = tmp_path / "bad.yaml"
        config.write_text("max_concurrency: not-an-int\n", encoding="utf-8")
        code = main(["--config", str(config), "--sandbox", str(tmp_path), "目标"])
        assert code == 2

    def test_keyboard_interrupt_exits_130(self, monkeypatch, capsys, tmp_path):
        _install_fake_router(monkeypatch, output=None, route_exc=KeyboardInterrupt())
        code = main(["--sandbox", str(tmp_path), "目标"])
        assert code == 130


class TestStdoutJsonContract:
    def test_stdout_is_orchestrator_output_json(self, monkeypatch, capsys, tmp_path):
        """标准输出为 OrchestratorOutput 的 JSON；顶层字段集合冻结。"""
        output = OrchestratorOutput(
            task_id="t-json",
            status=TaskStatus.COMPLETED,
            final_result="结果文本",
        )
        _install_fake_router(monkeypatch, output=output)
        main(["--sandbox", str(tmp_path), "目标"])

        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert set(payload) == {
            "task_id",
            "status",
            "final_result",
            "plan_versions",
            "agent_history",
            "review_summary",
            "human_escalation",
        }
        assert payload["task_id"] == "t-json"
        assert payload["status"] == "completed"
        assert payload["final_result"] == "结果文本"


# ---------------------------------------------------------------------------
# OrchestratorOutput 与 WorkflowEngine.run 契约冻结
# ---------------------------------------------------------------------------


class TestOutputContractFreeze:
    def test_orchestrator_output_fields_frozen(self):
        assert list(OrchestratorOutput.model_fields) == [
            "task_id",
            "status",
            "final_result",
            "plan_versions",
            "agent_history",
            "review_summary",
            "human_escalation",
        ]

    def test_workflow_run_signature_frozen(self):
        signature = inspect.signature(WorkflowEngine.run)
        assert list(signature.parameters) == [
            "self",
            "objective",
            "constraints",
            "acceptance_criteria",
        ]
        assert signature.parameters["constraints"].default == ""
        assert signature.parameters["acceptance_criteria"].default == ""
        hints = typing.get_type_hints(WorkflowEngine.run)
        assert hints["return"] is OrchestratorOutput
        assert hints["objective"] is str


# ---------------------------------------------------------------------------
# 事件类型名向后兼容（冻结集合是 runtime/events.py 登记表的子集）
# ---------------------------------------------------------------------------

_FROZEN_EVENT_CONSTANTS: dict[str, str] = {
    "EVENT_TASK_CREATED": "task_created",
    "EVENT_STATUS_TRANSITION": "status_transition",
    "EVENT_TASK_DISPATCHED": "task_dispatched",
    "EVENT_TOOL_GRANTED": "tool_granted",
    "EVENT_TOOL_REVOKED": "tool_revoked",
    "EVENT_TOOL_DENIED": "tool_denied",
    "EVENT_TOOL_CALLED": "tool_called",
    "EVENT_AGENT_CREATED": "agent_created",
    "EVENT_AGENT_RELEASED": "agent_released",
    "EVENT_REPLACEMENT_REQUESTED": "replacement_requested",
    "EVENT_REPLACEMENT_EXECUTING": "replacement_executing",
    "EVENT_REPLACEMENT_COMPLETED": "replacement_completed",
    "EVENT_REPLACEMENT_FAILED": "replacement_failed",
    "EVENT_DECOMPOSITION_ISSUE": "decomposition_issue",
    "EVENT_REPLAN": "replan",
    "EVENT_HUMAN_ESCALATION": "human_escalation",
    "EVENT_MESSAGE_ROUTED": "message_routed",
    "EVENT_MESSAGE_REJECTED": "message_rejected",
    "EVENT_TOKEN_USAGE": "token_usage",
    "EVENT_REVIEW_PARSE_ERROR": "review_parse_error",
    "EVENT_REVIEW_AGENT_OUTPUT": "review_agent_output",
    "EVENT_REVIEW_CONTRACT_REPAIR": "review_contract_repair",
    "EVENT_REVIEW_CONTRACT_ERROR": "review_contract_error",
    "EVENT_TIMEOUT_AUDIT": "timeout_audit",
    "EVENT_RESULT_REUSED": "result_reused",
    "EVENT_REUSE_REJECTED": "reuse_rejected",
}


def _assert_frozen_events_still_registered(
    registered: Iterable[str], frozen: Mapping[str, str]
) -> None:
    """断言冻结事件集合是已登记集合的子集（向后兼容判定）。

    合法演进只允许新增事件（``registered`` 为 ``frozen`` 的超集）；
    任何一个冻结事件被移除或改名都破坏向后兼容，断言失败并列出
    缺失清单。抽成 helper 便于用参数直接验证该语义，无需篡改全局
    常量 ``ALL_EVENT_TYPES``。
    """
    missing = set(frozen.values()) - set(registered)
    assert not missing, (
        f"破坏事件向后兼容：以下冻结事件类型被移除或改名：{sorted(missing)}"
    )


class TestEventNameBackwardCompat:
    """事件名向后兼容：旧事件必须继续存在，但允许新增事件。

    冻结集合是登记表的兼容子集（不再做长度/全集精确相等）；
    常量名 → 值的映射仍逐个精确冻结（改名即红灯）。
    """

    def test_frozen_event_names_all_still_registered(self):
        """全部 26 个冻结事件类型仍在登记表中（允许额外新增事件）。"""
        _assert_frozen_events_still_registered(
            ALL_EVENT_TYPES, _FROZEN_EVENT_CONSTANTS
        )

    @pytest.mark.parametrize("constant_name", sorted(_FROZEN_EVENT_CONSTANTS))
    def test_event_constant_values_frozen(self, constant_name):
        assert getattr(events_mod, constant_name) == _FROZEN_EVENT_CONSTANTS[constant_name]


class TestEventCompatHelperSemantics:
    """helper 语义自证：允许新增事件、拒绝缺失冻结事件。"""

    def test_helper_allows_extra_events(self):
        """登记表含新增事件（超集）时判定通过——合法演进不红灯。"""
        _assert_frozen_events_still_registered(
            set(_FROZEN_EVENT_CONSTANTS.values()) | {"brand_new_event_type"},
            _FROZEN_EVENT_CONSTANTS,
        )

    def test_helper_fails_when_frozen_event_missing(self):
        """任一冻结事件缺失即失败，且报错点名缺失的事件类型。"""
        crippled = set(_FROZEN_EVENT_CONSTANTS.values()) - {"tool_granted"}
        with pytest.raises(AssertionError, match="tool_granted"):
            _assert_frozen_events_still_registered(
                crippled, _FROZEN_EVENT_CONSTANTS
            )


# ---------------------------------------------------------------------------
# 安全治理不变量护栏
# ---------------------------------------------------------------------------


class TestSecurityInvariantGuards:
    def test_default_deny_and_exit_revocation_idempotent(self):
        """默认拒绝：无授权必拒；签发后可校验；任务级撤销后必拒且幂等。"""
        gateway = PermissionGateway(settings=Settings(_env_file=None))
        # 未签发任何授权：默认拒绝
        assert gateway.verify_grant("agent-1", "write_file", "T-1") is False

        grant = gateway.issue_grant(
            "T-1", "agent-1", ["write_file"], issued_by="supervisor"
        )
        assert gateway.verify_grant("agent-1", "write_file", "T-1") is True
        # 授权仅限本次任务：跨任务拒绝
        assert gateway.verify_grant("agent-1", "write_file", "T-其他") is False
        # 白名单外工具拒绝
        assert gateway.verify_grant("agent-1", "delete_all", "T-1") is False

        # 任意出口撤销（任务级统一收口）+ 幂等
        assert gateway.revoke_grants_for_task("T-1", reason="lifecycle_cleanup:test") == [
            grant.grant_id
        ]
        assert gateway.verify_grant("agent-1", "write_file", "T-1") is False
        assert gateway.revoke_grants_for_task("T-1") == []

    def test_passed_is_the_only_approval_signal(self):
        """passed 唯一语义信号：字段组合契约矛盾检测冻结。"""
        assert (
            _review_contract_contradiction(
                ReviewDecision(stage="plan_review", passed=True)
            )
            is None
        )
        assert (
            _review_contract_contradiction(
                ReviewDecision(
                    stage="plan_review", passed=True, required_fixes=["必须修复"]
                )
            )
            is not None
        )
        assert (
            _review_contract_contradiction(
                ReviewDecision(stage="plan_review", passed=False)
            )
            is not None
        )
        assert (
            _review_contract_contradiction(
                ReviewDecision(
                    stage="plan_review", passed=False, required_fixes=["必须修复"]
                )
            )
            is None
        )

    def test_star_topology_rejects_lateral_communication(self):
        """星型通信：子 Agent 之间禁止横向通信，主管↔子合法。"""
        router = SupervisorRouter()
        supervisor_id = router.supervisor_id
        assert router.validate_communication("sub-a", "sub-b") is False
        assert router.validate_communication(supervisor_id, "sub-a") is True
        assert router.validate_communication("sub-a", supervisor_id) is True

    def test_replacement_chain_depth_one_frozen(self):
        """替代链深度一：替换请求全局上限为 1（配置层硬约束冻结）。"""
        settings = Settings(_env_file=None)
        assert settings.max_replacement_requests == 1
        assert settings.max_agent_attempts == 3
        assert settings.max_replan_attempts == 3

    def test_reuse_from_excluded_from_semantic_fingerprint(self):
        """显式 reuse_from 是声明元数据：不改变语义指纹。"""
        declared = SubtaskSpec(
            subtask_id="n1",
            title="子任务",
            instructions="相同指令",
            role="code",
            reuse_from="s1",
        )
        undeclared = declared.model_copy(update={"reuse_from": None})
        assert spec_fingerprint(declared) == spec_fingerprint(undeclared)

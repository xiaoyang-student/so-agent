"""单元测试：显式复用声明的安全校验矩阵（任务 35）。

覆盖 ``_validate_reuse_declarations`` 的全部拒绝路径与放行例外，
以及持久化兼容与授权交互：

1. 来源校验：不存在（``source_not_found``）、失败
   （``source_not_successful``）、位于失败下游闭包
   （``source_poisoned_by_failure``）；
2. 指纹成分不兼容（``fingerprint_incompatible``）：instructions /
   role / expected_output 变化、依赖改名无映射、依赖增删；
3. 依赖未满足（``dependency_not_satisfied``）：同名依赖留有旧成功
   结果但未同时复用；旧依赖无成功结果时允许（仍会在执行前完成）；
4. artifacts：丢失（``artifact_missing``，含纯文本伪装路径）、
   越界（``artifact_outside_sandbox``）、路径穿越
   （``artifact_path_traversal``）；URL 跳过；沙箱内相对路径通过；
5. 工具范围扩大（``tool_scope_expanded``）；动态工具按 manifest
   稳定标识（tool_id）比较：改名同 id 放行、不同 id 拒绝；
6. 单一消费（``duplicate_reference``）；
7. ``reuse_rejected`` 事件：机器可读 reason，不携带完整输出；
8. SQLite 兼容：``reuse_from`` 序列化往返；旧库缺字段默认 None；
9. 授权交互：``_issue_grants`` 跳过复用任务（不留无用活跃授权），
   失败待重试任务不受影响。
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ExecutionResult,
    GeneratedToolManifest,
    PlanVersion,
    SubtaskSpec,
    TaskRequest,
)
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.result_store import SQLiteResultStore
from so_agent.runtime.events import (
    EVENT_REUSE_REJECTED,
    EVENT_RESULT_REUSED,
    EventLogger,
)
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import AgentRegistry
from so_agent.runtime.supervisor_router import SupervisorRouter
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.state_machine import WorkflowState
from so_agent.workflow_support import HumanEscalationHandler, WorkflowDeps


def make_spec(
    subtask_id: str,
    *,
    instructions: str = "完成通用工作",
    role: str = "code",
    expected_output: str = "",
    dependencies: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    reuse_from: str | None = None,
) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"子任务 {subtask_id}",
        instructions=instructions,
        role=role,
        expected_output=expected_output,
        dependencies=list(dependencies or []),
        allowed_tools=list(allowed_tools or []),
        reuse_from=reuse_from,
    )


def make_result(
    subtask_id: str,
    *,
    success: bool = True,
    output: str = "",
    artifacts: list[str] | None = None,
    error: str | None = None,
    metadata: dict | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        subtask_id=subtask_id,
        success=success,
        output=output or f"{subtask_id} 产出",
        artifacts=list(artifacts or []),
        error=error if error is not None else None if success else (error or "失败（模拟）"),
        metadata=dict(metadata or {}),
    )


def build_coordinator(tmp_path, old_plan=None):
    """装配最小规划协调器（真实依赖，不启动工作流）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="reuse-validation")
    settings = Settings(_env_file=None)
    events = EventLogger(context=context)
    registry = AgentRegistry(context=context, settings=settings)
    tool_registry = ToolRegistry(generated_tools_dir=tmp_path / "generated_tools")
    deps = WorkflowDeps(
        context=context,
        settings=settings,
        events=events,
        registry=registry,
        gateway=PermissionGateway(
            context=context,
            settings=settings,
            events=events,
            tool_registry=tool_registry,
        ),
        router=SupervisorRouter(registry=registry, events=events),
        tool_registry=tool_registry,
        escalation_handler=HumanEscalationHandler(),
        orchestrator=SimpleNamespace(name="stub-supervisor"),
        review_agent=SimpleNamespace(name="stub-reviewer"),
    )
    coordinator = PlanningCoordinator(WorkflowState(events=events), deps)
    coordinator._task_request = TaskRequest(
        task_id="reuse-validation-task", objective="测试目标"
    )
    coordinator._subtasks = [
        spec.model_copy(deep=True) for spec in (old_plan or [])
    ]
    return coordinator


def prepare_and_validate(
    tmp_path,
    *,
    old_plan: list[SubtaskSpec],
    old_results: dict[str, ExecutionResult],
    new_plan: list[SubtaskSpec],
):
    """推到重规划采纳后状态并执行校验，返回 (coordinator, reused)。"""
    coordinator = build_coordinator(tmp_path)
    coordinator._subtasks = [spec.model_copy(deep=True) for spec in old_plan]
    coordinator._results = {
        key: value.model_copy(deep=True)
        for key, value in old_results.items()
    }
    coordinator._plan_version = 1
    coordinator._plan_history = [
        [spec.model_copy(deep=True) for spec in old_plan]
    ]
    coordinator._archive_completed_results()
    coordinator._plan_version = 2
    coordinator._results = {}
    coordinator._subtasks = [spec.model_copy(deep=True) for spec in new_plan]
    reused = coordinator._validate_reuse_declarations()
    return coordinator, reused


def rejected_reasons(coordinator) -> list[tuple[str, str]]:
    """提取全部 reuse_rejected 事件的 (subtask_id, reason)。"""
    events = coordinator._events.get_events(event_type=EVENT_REUSE_REJECTED)
    return [
        (event["payload"]["subtask_id"], event["payload"]["reason"])
        for event in events
    ]


def register_generated(tool_registry, tool_id: str, tool_name: str) -> None:
    """注册一个已批准的动态生成工具（稳定标识为 tool_id）。"""
    manifest = GeneratedToolManifest(
        tool_id=tool_id,
        tool_name=tool_name,
        entry_file=f"{tool_name}.py",
        created_by_task="reuse-validation",
        review_status="approved",
    )
    tool_registry.register_generated_tool(manifest)


class TestSourceValidation:
    """校验 1-3：来源存在、成功、非污染。"""

    def test_source_not_found(self, tmp_path):
        """来源不在上一版计划（不存在的 id）→ source_not_found。"""
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1")},
            new_plan=[
                make_spec("n1", instructions="A", reuse_from="ghost")
            ],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [("n1", "source_not_found")]

    def test_source_without_archived_result_rejected(self, tmp_path):
        """来源在上一版计划中但无归档结果（未执行到）→ source_not_found。"""
        old_plan = [
            make_spec("s1", instructions="A"),
            make_spec("s2", instructions="B"),
        ]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1")},  # s2 未执行
            new_plan=[
                make_spec("n2", instructions="B", reuse_from="s2")
            ],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [("n2", "source_not_found")]

    def test_source_not_successful(self, tmp_path):
        """来源失败 → source_not_successful。"""
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1", success=False)},
            new_plan=[
                make_spec("n1", instructions="A", reuse_from="s1")
            ],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [("n1", "source_not_successful")]

    def test_source_poisoned_by_failure(self, tmp_path):
        """来源成功但位于失败子任务下游闭包 → source_poisoned_by_failure。"""
        old_plan = [
            make_spec("f", instructions="F"),
            make_spec("d", instructions="D", dependencies=["f"]),
        ]
        # 构造性输入：依赖 f 的 d 留有"成功记录"（真实执行中会因依赖
        # 失败被跳过；此处验证级联约束独立生效）
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={
                "f": make_result("f", success=False),
                "d": make_result("d"),
            },
            new_plan=[
                make_spec("n1", instructions="D", dependencies=["x2"], reuse_from="d")
            ],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [
            ("n1", "source_poisoned_by_failure")
        ]


class TestFingerprintCompatibility:
    """校验 4：指纹成分兼容（禁止文本相似度放宽）。"""

    def test_instructions_changed_rejected(self, tmp_path):
        old_plan = [make_spec("s1", instructions="实现报表模块")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1")},
            new_plan=[
                make_spec(
                    "n1",
                    instructions="改用图形界面实现报表（相似但不同）",
                    reuse_from="s1",
                )
            ],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [("n1", "fingerprint_incompatible")]

    def test_role_changed_rejected(self, tmp_path):
        old_plan = [make_spec("s1", instructions="A", role="code")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1")},
            new_plan=[
                make_spec("n1", instructions="A", role="review", reuse_from="s1")
            ],
        )

        assert rejected_reasons(coordinator) == [("n1", "fingerprint_incompatible")]

    def test_expected_output_changed_rejected(self, tmp_path):
        old_plan = [make_spec("s1", instructions="A", expected_output="R")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1")},
            new_plan=[
                make_spec(
                    "n1", instructions="A", expected_output="R2", reuse_from="s1"
                )
            ],
        )

        assert rejected_reasons(coordinator) == [("n1", "fingerprint_incompatible")]

    def test_dependency_rename_without_declaration_rejected(self, tmp_path):
        """旧依赖 x → 新依赖 y，但 y 未声明复用（无映射）→ 不兼容。"""
        old_plan = [
            make_spec("a", instructions="A", dependencies=["x"]),
            make_spec("x", instructions="X"),
        ]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"a": make_result("a"), "x": make_result("x")},
            new_plan=[
                make_spec(
                    "n1", instructions="A", dependencies=["y"], reuse_from="a"
                ),
                make_spec("y", instructions="X"),  # 同语义但未声明复用
            ],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [
            ("n1", "fingerprint_incompatible")
        ]

    def test_dependency_removed_rejected(self, tmp_path):
        """新任务删除了旧依赖 → 依赖集合不等价 → 不兼容。"""
        old_plan = [
            make_spec("a", instructions="A", dependencies=["x"]),
            make_spec("x", instructions="X"),
        ]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"a": make_result("a"), "x": make_result("x")},
            new_plan=[
                make_spec("n1", instructions="A", reuse_from="a"),
                make_spec("x", instructions="X", reuse_from="x"),
            ],
        )

        assert rejected_reasons(coordinator) == [
            ("n1", "fingerprint_incompatible")
        ]

    def test_dependency_added_rejected(self, tmp_path):
        """新任务新增了旧计划没有的依赖 → 不兼容。"""
        old_plan = [
            make_spec("a", instructions="A"),
            make_spec("x", instructions="X"),
        ]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"a": make_result("a")},
            new_plan=[
                make_spec(
                    "n1", instructions="A", dependencies=["x2"], reuse_from="a"
                ),
                make_spec("x2", instructions="X"),
            ],
        )

        assert rejected_reasons(coordinator) == [
            ("n1", "fingerprint_incompatible")
        ]


class TestDependencySatisfaction:
    """校验 5：依赖必须已经有效、同时复用，或仍会在执行前完成。"""

    def test_dependency_old_success_without_reuse_rejected(self, tmp_path):
        """同名依赖留有旧成功结果但未声明复用 → 依赖链断裂 → 拒绝。"""
        old_plan = [
            make_spec("x", instructions="X"),
            make_spec("a", instructions="A", dependencies=["x"]),
        ]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"x": make_result("x"), "a": make_result("a")},
            new_plan=[
                make_spec("x", instructions="X"),  # 同语义但不声明复用
                make_spec(
                    "n1", instructions="A", dependencies=["x"], reuse_from="a"
                ),
            ],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [
            ("n1", "dependency_not_satisfied")
        ]

    def test_dependency_without_archived_result_allowed(self, tmp_path):
        """旧依赖无归档结果（将重新执行，仍会在执行前完成）→ 放行。"""
        old_plan = [
            make_spec("x", instructions="X"),
            make_spec("a", instructions="A", dependencies=["x"]),
        ]
        # 构造性输入：a 留有成功记录而 x 未执行到（无归档结果）——验证
        # 依赖规则分支（旧依赖无成功结果时允许其重新执行）
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"a": make_result("a")},
            new_plan=[
                make_spec("x", instructions="X"),  # 重新执行
                make_spec(
                    "n1", instructions="A", dependencies=["x"], reuse_from="a"
                ),
            ],
        )

        assert set(reused) == {"n1"}
        assert rejected_reasons(coordinator) == []


class TestArtifactValidation:
    """校验 6：artifacts 本地路径存在、在沙箱内、无穿越。"""

    def test_artifact_missing_rejected(self, tmp_path):
        (tmp_path / "out").mkdir()
        # out/report.txt 未创建
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={
                "s1": make_result(
                    "s1", artifacts=[str(tmp_path / "out" / "report.txt")]
                )
            },
            new_plan=[make_spec("n1", instructions="A", reuse_from="s1")],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [("n1", "artifact_missing")]

    def test_artifact_text_disguised_as_path_rejected(self, tmp_path):
        """纯文本证据伪装为产物路径 → 无法通过存在性校验。"""
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={
                "s1": make_result("s1", artifacts=["步骤 1 的输出文本"])
            },
            new_plan=[make_spec("n1", instructions="A", reuse_from="s1")],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [("n1", "artifact_missing")]

    def test_artifact_outside_sandbox_rejected(self, tmp_path):
        """沙箱外的现有文件 → artifact_outside_sandbox。"""
        outside = tmp_path.parent / "outside_secret.txt"
        outside.write_text("secret", encoding="utf-8")
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1", artifacts=[str(outside)])},
            new_plan=[make_spec("n1", instructions="A", reuse_from="s1")],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [
            ("n1", "artifact_outside_sandbox")
        ]

    def test_artifact_path_traversal_rejected(self, tmp_path):
        """含 .. 逃逸沙箱的路径 → artifact_path_traversal。"""
        escape = tmp_path.parent / "escape.txt"
        escape.write_text("escaped", encoding="utf-8")
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={
                "s1": make_result(
                    "s1", artifacts=[str(tmp_path / ".." / "escape.txt")]
                )
            },
            new_plan=[make_spec("n1", instructions="A", reuse_from="s1")],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [
            ("n1", "artifact_path_traversal")
        ]

    def test_artifact_url_skips_local_validation(self, tmp_path):
        """URL 等非本地引用不作本地路径校验 → 放行。"""
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={
                "s1": make_result(
                    "s1", artifacts=["https://example.com/report"]
                )
            },
            new_plan=[make_spec("n1", instructions="A", reuse_from="s1")],
        )

        assert set(reused) == {"n1"}

    def test_artifact_relative_path_inside_sandbox_allowed(self, tmp_path):
        """沙箱内相对路径（产物约定形态）且存在 → 放行。"""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "report.txt").write_text("报告", encoding="utf-8")
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1", artifacts=["out/report.txt"])},
            new_plan=[make_spec("n1", instructions="A", reuse_from="s1")],
        )

        assert set(reused) == {"n1"}


class TestToolScopeValidation:
    """校验 7：新 spec 工具范围不得扩大旧 spec。"""

    def test_tool_scope_expanded_rejected(self, tmp_path):
        old_plan = [
            make_spec("s1", instructions="A", allowed_tools=["read_file"])
        ]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1")},
            new_plan=[
                make_spec(
                    "n1",
                    instructions="A",
                    allowed_tools=["read_file", "write_file"],
                    reuse_from="s1",
                )
            ],
        )

        assert reused == {}
        assert rejected_reasons(coordinator) == [("n1", "tool_scope_expanded")]

    def test_dynamic_tool_rename_same_tool_id_allowed(self, tmp_path):
        """动态工具改名但 manifest 稳定标识（tool_id）相同 → 放行。"""
        coordinator = build_coordinator(tmp_path)
        register_generated(coordinator._tool_registry, "T1", "tool_alpha")
        register_generated(coordinator._tool_registry, "T1", "tool_beta")
        old_plan = [
            make_spec("s1", instructions="A", allowed_tools=["tool_alpha"])
        ]
        coordinator._subtasks = [spec.model_copy(deep=True) for spec in old_plan]
        coordinator._results = {"s1": make_result("s1")}
        coordinator._plan_version = 1
        coordinator._plan_history = [
            [spec.model_copy(deep=True) for spec in old_plan]
        ]
        coordinator._archive_completed_results()
        coordinator._plan_version = 2
        coordinator._results = {}
        coordinator._subtasks = [
            make_spec(
                "n1", instructions="A", allowed_tools=["tool_beta"], reuse_from="s1"
            )
        ]

        reused = coordinator._validate_reuse_declarations()

        assert set(reused) == {"n1"}

    def test_dynamic_tool_different_tool_id_rejected(self, tmp_path):
        """动态工具不同名且 tool_id 不同 → 视为范围扩大 → 拒绝。"""
        coordinator = build_coordinator(tmp_path)
        register_generated(coordinator._tool_registry, "T1", "tool_alpha")
        register_generated(coordinator._tool_registry, "T2", "tool_gamma")
        old_plan = [
            make_spec("s1", instructions="A", allowed_tools=["tool_alpha"])
        ]
        coordinator._subtasks = [spec.model_copy(deep=True) for spec in old_plan]
        coordinator._results = {"s1": make_result("s1")}
        coordinator._plan_version = 1
        coordinator._plan_history = [
            [spec.model_copy(deep=True) for spec in old_plan]
        ]
        coordinator._archive_completed_results()
        coordinator._plan_version = 2
        coordinator._results = {}
        coordinator._subtasks = [
            make_spec(
                "n1", instructions="A", allowed_tools=["tool_gamma"], reuse_from="s1"
            )
        ]

        reused = coordinator._validate_reuse_declarations()

        assert reused == {}
        assert rejected_reasons(coordinator) == [("n1", "tool_scope_expanded")]


class TestSingleConsumption:
    """校验 8：同一旧结果不可被重复引用。"""

    def test_duplicate_reference_rejected(self, tmp_path):
        """两个新任务声明同一来源：先通过者消费，后者被拒。"""
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1", output="唯一产出")},
            new_plan=[
                make_spec("x", instructions="A", reuse_from="s1"),
                make_spec("y", instructions="A", reuse_from="s1"),
            ],
        )

        assert set(reused) == {"x"}
        assert reused["x"].output == "唯一产出"
        assert rejected_reasons(coordinator) == [("y", "duplicate_reference")]

    def test_rejected_declaration_does_not_consume_source(self, tmp_path):
        """首个声明因指纹不兼容被拒时不消费来源，合法声明仍可复用。"""
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={"s1": make_result("s1")},
            new_plan=[
                make_spec(
                    "bad", instructions="语义已变化", reuse_from="s1"
                ),
                make_spec("good", instructions="A", reuse_from="s1"),
            ],
        )

        assert set(reused) == {"good"}
        reasons = dict(rejected_reasons(coordinator))
        assert reasons["bad"] == "fingerprint_incompatible"
        assert "good" not in reasons


class TestReuseRejectedEvent:
    """reuse_rejected 事件：机器可读 reason，不泄露完整输出。"""

    def test_rejected_event_payload_shape(self, tmp_path):
        old_plan = [make_spec("s1", instructions="A")]
        coordinator, reused = prepare_and_validate(
            tmp_path,
            old_plan=old_plan,
            old_results={
                "s1": make_result(
                    "s1",
                    output="这份输出包含大段内容，绝不能出现在拒绝事件里",
                )
            },
            new_plan=[
                make_spec(
                    "n1", instructions="语义已变", reuse_from="s1"
                )
            ],
        )

        assert reused == {}
        events = coordinator._events.get_events(event_type=EVENT_REUSE_REJECTED)
        assert len(events) == 1
        payload = events[0]["payload"]
        assert payload["reason"] == "fingerprint_incompatible"
        assert payload["subtask_id"] == "n1"
        assert payload["source_subtask_id"] == "s1"
        # 不携带完整输出：payload 不含任何输出内容字段
        assert "output" not in payload
        assert "这份输出包含大段内容" not in json.dumps(payload, ensure_ascii=False)
        assert coordinator._events.get_events(event_type=EVENT_RESULT_REUSED) == []


class TestSqliteCompatibility:
    """持久化兼容：reuse_from 序列化往返与旧库缺字段。"""

    def test_plan_version_reuse_from_roundtrip(self, tmp_path):
        """含 reuse_from 声明的 specs 经 SQLite 往返保持不变。"""
        store = SQLiteResultStore(str(tmp_path / "store.db"))
        spec = make_spec("n1", instructions="A", reuse_from="s1")
        plan = PlanVersion(version=2, specs=[spec])

        store.save_plan_version(plan)
        loaded = store.get_plan_version(2)

        assert loaded is not None
        assert loaded.specs[0].reuse_from == "s1"
        store.close()

    def test_legacy_specs_without_reuse_field_defaults_none(self, tmp_path):
        """旧库 specs_json 不含 reuse_from 字段 → 反序列化默认 None。"""
        db_path = tmp_path / "legacy.db"
        store = SQLiteResultStore(str(db_path))
        legacy_spec = {
            "subtask_id": "s1",
            "title": "旧子任务",
            "instructions": "A",
            "role": "code",
        }
        store._conn.execute(
            "INSERT INTO plan_versions (version, specs_json, executions_json) "
            "VALUES (?, ?, ?)",
            (1, json.dumps([legacy_spec], ensure_ascii=False), "{}"),
        )
        store._conn.commit()

        loaded = store.get_plan_version(1)

        assert loaded is not None
        assert loaded.specs[0].reuse_from is None
        store.close()

    def test_archive_results_independent_of_store(self, tmp_path):
        """归档在内存 state 中，与 ResultStore 实现无关（InMemory 兼容）。"""
        old_plan = [make_spec("s1", instructions="A")]
        coordinator = build_coordinator(tmp_path)
        coordinator._plan_version = 1
        coordinator._results = {"s1": make_result("s1")}

        coordinator._archive_completed_results()
        coordinator._results.clear()

        assert coordinator._archive_results[(1, "s1")].success is True


class TestIssueGrantsInteraction:
    """授权交互：复用任务不签发真实授权，失败重试不受影响。"""

    def test_issue_grants_skips_reused_subtask(self, tmp_path):
        """复用任务（结果带来源标记）不获得真实授权。"""
        coordinator = build_coordinator(tmp_path)
        pending_spec = make_spec(
            "p1", instructions="A", allowed_tools=["write_file"]
        )
        reused_spec = make_spec(
            "r1", instructions="B", allowed_tools=["write_file"]
        )
        coordinator._subtasks = [pending_spec, reused_spec]
        coordinator._results = {
            "r1": make_result(
                "r1",
                metadata={
                    "reused_from": "old-r1",
                    "reused_from_plan_version": 1,
                },
            )
        }

        grants = coordinator._issue_grants()

        issued_ids = {spec.subtask_id for spec in coordinator._subtasks
                      if spec.subtask_id in coordinator._plan_grant_map}
        assert issued_ids == {"p1"}
        assert all(grant.agent_id.endswith(":p1:grant-slot") for grant in grants)

    def test_issue_grants_still_covers_failed_retry(self, tmp_path):
        """失败待重试任务（结果无来源标记）仍获得授权。"""
        coordinator = build_coordinator(tmp_path)
        failed_spec = make_spec(
            "f1", instructions="A", allowed_tools=["write_file"]
        )
        coordinator._subtasks = [failed_spec]
        coordinator._results = {
            "f1": make_result("f1", success=False)  # 无 reused_from 标记
        }

        coordinator._issue_grants()

        assert "f1" in coordinator._plan_grant_map

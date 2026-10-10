"""单元测试：跨计划版本的显式结果复用（任务 35）。

核心原则：LLM（主管）负责判断旧结果在新计划语境下是否可复用；
代码不凭同名或 fingerprint 自动作语义决定，只校验主管的显式
``reuse_from`` 声明是否满足安全硬约束。覆盖：

1. 无 reuse_from 声明时绝不复用——即使 id、名称与指纹完全相同
   （仅凭 fingerprint 的自动复用路径已删除）；
2. 合法声明（同 id / 改名）→ 复用：ExecutionResult 复制到新
   subtask_id，metadata 保留来源标记，记录 result_reused 事件；
3. ``reuse_from`` 不参与 spec_fingerprint（声明不得改变语义指纹）；
4. 依赖集合经显式声明形成的 id 映射后等价（安全兼容规则，
   非文本相似度）；链式复用经固定点迭代（声明顺序无关）；
5. 复用候选载荷（``_reuse_candidates_payload``）的最小充分信息、
   失败诊断与失败级联范围；首次规划不提供候选；
6. 归档（``_archive_completed_results``）固化历史来源，独立于
   ``results`` 存活；
7. ``_downstream_closure`` 兼容（默认当前计划 / 显式指定上一版计划）。

校验矩阵（拒绝路径）的正反例见 test_reuse_validation.py。
"""

from __future__ import annotations

from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.fingerprint import spec_fingerprint
from so_agent.models import ExecutionResult, SubtaskSpec, TaskRequest
from so_agent.planning_coordinator import PlanningCoordinator
from so_agent.runtime.events import EVENT_RESULT_REUSED, EventLogger
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
    """构造子任务规格；语义字段显式指定以控制指纹。"""
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"子任务 {subtask_id}",
        instructions=instructions,
        role=role,
        expected_output=expected_output,
        dependencies=list(dependencies or []),
        allowed_tools=list(allowed_tools or []),
        max_attempts=3,
        reuse_from=reuse_from,
    )


def make_result(
    subtask_id: str,
    *,
    success: bool = True,
    output: str = "",
    evidence: list[str] | None = None,
    artifacts: list[str] | None = None,
    error: str | None = None,
) -> ExecutionResult:
    """构造执行结果（默认成功）。"""
    return ExecutionResult(
        subtask_id=subtask_id,
        success=success,
        output=output or f"{subtask_id} 产出",
        evidence=list(evidence or []),
        artifacts=list(artifacts or []),
        error=error if error is not None else None
        if success
        else (error or f"{subtask_id} 失败（模拟）"),
    )


def build_coordinator(tmp_path, old_plan: list[SubtaskSpec]) -> PlanningCoordinator:
    """装配仅承载旧计划的最小规划协调器（真实依赖，不启动工作流）。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="reuse-unit")
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
    coordinator = PlanningCoordinator(
        WorkflowState(events=events), deps
    )
    coordinator._task_request = TaskRequest(
        task_id="reuse-unit-task", objective="测试目标"
    )
    coordinator._subtasks = [spec.model_copy(deep=True) for spec in old_plan]
    return coordinator


def prepare_replanning(
    coordinator: PlanningCoordinator,
    *,
    old_plan: list[SubtaskSpec],
    old_results: dict[str, ExecutionResult],
) -> PlanningCoordinator:
    """把协调器推到"重规划采纳后、校验前"的状态。

    模拟 ``_plan_and_review`` 的时序：承载旧计划与结果（版本 1）→
    归档 → 版本号递增到 2 → 清空 ``results``。调用方随后把新计划
    写入 ``_subtasks`` 并调用 ``_validate_reuse_declarations``。
    """
    coordinator._subtasks = [spec.model_copy(deep=True) for spec in old_plan]
    coordinator._results = {
        key: value.model_copy(deep=True)
        for key, value in old_results.items()
    }
    coordinator._plan_version = 1
    coordinator._plan_history = [[spec.model_copy(deep=True) for spec in old_plan]]
    coordinator._archive_completed_results()
    coordinator._plan_version = 2
    coordinator._results = {}
    return coordinator


def validate(
    coordinator: PlanningCoordinator, new_plan: list[SubtaskSpec]
) -> dict[str, ExecutionResult]:
    """写入新计划并执行显式复用声明校验（返回可并入 _results 的结果）。"""
    coordinator._subtasks = [spec.model_copy(deep=True) for spec in new_plan]
    return coordinator._validate_reuse_declarations()


class TestNoDeclarationNeverReuses:
    """核心原则：无声明时绝不复用（自动 fingerprint 复用路径已删除）。"""

    def test_no_declaration_same_id_same_fingerprint_not_reused(self, tmp_path):
        """id、名称与指纹完全相同但未声明 → 不复用（旧自动路径已停用）。"""
        plan = [make_spec("s1", instructions="实现报表模块")]
        coordinator = build_coordinator(tmp_path, plan)
        prepare_replanning(
            coordinator,
            old_plan=plan,
            old_results={"s1": make_result("s1", output="报表完成")},
        )

        reused = validate(coordinator, [make_spec("s1", instructions="实现报表模块")])

        assert reused == {}
        assert coordinator._events.get_events(event_type=EVENT_RESULT_REUSED) == []

    def test_blank_reuse_from_not_treated_as_declaration(self, tmp_path):
        """空白字符串的 reuse_from 视为无声明。"""
        plan = [make_spec("s1", instructions="实现报表模块")]
        coordinator = build_coordinator(tmp_path, plan)
        prepare_replanning(
            coordinator,
            old_plan=plan,
            old_results={"s1": make_result("s1")},
        )

        reused = validate(
            coordinator, [make_spec("s1", instructions="实现报表模块", reuse_from="  ")]
        )

        assert reused == {}

    def test_first_planning_without_history_rejects_any_declaration(
        self, tmp_path,
    ):
        """首次规划（无历史版本）：任何声明都因来源不存在被拒。"""
        plan = [make_spec("s1", instructions="实现报表模块")]
        coordinator = build_coordinator(tmp_path, [])
        coordinator._plan_version = 1  # 首次采纳后
        coordinator._plan_history = [plan]
        coordinator._results = {}

        reused = validate(
            coordinator,
            [make_spec("s1", instructions="实现报表模块", reuse_from="s1")],
        )

        assert reused == {}


class TestExplicitDeclarationReuse:
    """合法显式声明：复制结果、来源标记与事件。"""

    def test_valid_declaration_same_id_reused(self, tmp_path):
        """同 id 声明 + 语义一致 → 复用，内容原样保留并带来源标记。"""
        plan = [make_spec("s1", instructions="实现报表模块")]
        coordinator = build_coordinator(tmp_path, plan)
        prepare_replanning(
            coordinator,
            old_plan=plan,
            old_results={"s1": make_result("s1", output="报表完成")},
        )

        reused = validate(
            coordinator,
            [make_spec("s1", instructions="实现报表模块", reuse_from="s1")],
        )

        assert set(reused) == {"s1"}
        assert reused["s1"].success is True
        assert reused["s1"].output == "报表完成"
        assert reused["s1"].subtask_id == "s1"
        assert reused["s1"].metadata["reused_from"] == "s1"
        assert reused["s1"].metadata["reused_from_plan_version"] == 1
        events = coordinator._events.get_events(event_type=EVENT_RESULT_REUSED)
        assert len(events) == 1
        assert events[0]["payload"]["subtask_id"] == "s1"
        assert events[0]["payload"]["source_subtask_id"] == "s1"
        assert events[0]["payload"]["source_plan_version"] == 1

    def test_valid_declaration_with_rename_normalizes_subtask_id(self, tmp_path):
        """改名声明：语义一致 → 复用，结果 subtask_id 归一化为新编号。"""
        old_plan = [make_spec("old-1", instructions="实现报表模块")]
        coordinator = build_coordinator(tmp_path, old_plan)
        prepare_replanning(
            coordinator,
            old_plan=old_plan,
            old_results={"old-1": make_result("old-1", output="报表完成")},
        )

        reused = validate(
            coordinator,
            [make_spec("new-1", instructions="实现报表模块", reuse_from="old-1")],
        )

        assert set(reused) == {"new-1"}
        assert reused["new-1"].subtask_id == "new-1"
        assert reused["new-1"].output == "报表完成"
        assert reused["new-1"].metadata["reused_from"] == "old-1"

    def test_reused_result_is_independent_copy(self, tmp_path):
        """复制是独立副本：修改复用结果不影响归档来源。"""
        old_plan = [make_spec("s1", instructions="实现报表模块")]
        coordinator = build_coordinator(tmp_path, old_plan)
        prepare_replanning(
            coordinator,
            old_plan=old_plan,
            old_results={"s1": make_result("s1", output="原始产出")},
        )

        reused = validate(
            coordinator,
            [make_spec("s1", instructions="实现报表模块", reuse_from="s1")],
        )

        reused["s1"].output = "被篡改"
        assert coordinator._archive_results[(1, "s1")].output == "原始产出"

    def test_allowed_tools_subset_still_reused(self, tmp_path):
        """新 spec 收缩工具范围（未扩大）→ 复用放行。"""
        old_plan = [
            make_spec(
                "s1",
                instructions="实现报表模块",
                allowed_tools=["read_file", "write_file"],
            )
        ]
        coordinator = build_coordinator(tmp_path, old_plan)
        prepare_replanning(
            coordinator,
            old_plan=old_plan,
            old_results={"s1": make_result("s1")},
        )

        reused = validate(
            coordinator,
            [
                make_spec(
                    "s1",
                    instructions="实现报表模块",
                    allowed_tools=["read_file"],
                    reuse_from="s1",
                )
            ],
        )

        assert set(reused) == {"s1"}


class TestFingerprintContract:
    """reuse_from 是声明元数据：不得进入语义指纹。"""

    def test_reuse_from_does_not_change_spec_fingerprint(self):
        """填写/留空 reuse_from 的两个同语义 spec 指纹一致。"""
        base = make_spec("s1", instructions="实现报表模块")
        declared = make_spec(
            "s1", instructions="实现报表模块", reuse_from="old-s1"
        )

        assert spec_fingerprint(base) == spec_fingerprint(declared)

    def test_equivalent_specs_same_dependencies(self):
        """依赖一致 + 语义字段一致 → 指纹成分兼容。"""
        old = make_spec("a", instructions="A", dependencies=["x"])
        new = make_spec("b", instructions="A", dependencies=["x"])

        from so_agent.fingerprint import specs_fingerprint_equivalent

        assert specs_fingerprint_equivalent(new, old)
        assert specs_fingerprint_equivalent(new, old, {})

    def test_equivalent_specs_dependency_rename_via_map(self):
        """依赖经显式映射（旧 x → 新 y）后集合等价 → 兼容。"""
        from so_agent.fingerprint import specs_fingerprint_equivalent

        old = make_spec("a", instructions="A", dependencies=["x"])
        new = make_spec("b", instructions="A", dependencies=["y"])

        assert specs_fingerprint_equivalent(new, old, {"x": "y"})
        assert not specs_fingerprint_equivalent(new, old)
        assert not specs_fingerprint_equivalent(new, old, {"x": "z"})

    def test_equivalent_specs_rejects_semantic_drift(self):
        """语义字段（instructions/role/expected_output）变化 → 不兼容。"""
        from so_agent.fingerprint import specs_fingerprint_equivalent

        old = make_spec("a", instructions="A", role="code", expected_output="R")
        assert not specs_fingerprint_equivalent(
            make_spec("b", instructions="A2"), old
        )
        assert not specs_fingerprint_equivalent(
            make_spec("b", instructions="A", role="review"), old
        )
        assert not specs_fingerprint_equivalent(
            make_spec("b", instructions="A", expected_output="R2"), old
        )


class TestDependencyMappingReuse:
    """依赖经显式声明映射改名后仍可复用（固定点迭代，顺序无关）。"""

    def test_dependency_rename_with_mapping_both_reused(self, tmp_path):
        """改名 + 依赖同步改名：两条声明同时通过。"""
        old_plan = [
            make_spec("s1", instructions="生成原始数据"),
            make_spec("s2", instructions="清洗数据", dependencies=["s1"]),
        ]
        coordinator = build_coordinator(tmp_path, old_plan)
        prepare_replanning(
            coordinator,
            old_plan=old_plan,
            old_results={"s1": make_result("s1"), "s2": make_result("s2")},
        )

        reused = validate(
            coordinator,
            [
                make_spec("n1", instructions="生成原始数据", reuse_from="s1"),
                make_spec(
                    "n2",
                    instructions="清洗数据",
                    dependencies=["n1"],
                    reuse_from="s2",
                ),
            ],
        )

        assert set(reused) == {"n1", "n2"}
        assert reused["n2"].metadata["reused_from"] == "s2"

    def test_chain_reused_via_fixed_point_any_order(self, tmp_path):
        """依赖方先声明（顺序颠倒）：固定点迭代保证链式复用。"""
        old_plan = [
            make_spec("s1", instructions="生成原始数据"),
            make_spec("s2", instructions="清洗数据", dependencies=["s1"]),
        ]
        coordinator = build_coordinator(tmp_path, old_plan)
        prepare_replanning(
            coordinator,
            old_plan=old_plan,
            old_results={"s1": make_result("s1"), "s2": make_result("s2")},
        )

        reused = validate(
            coordinator,
            [
                make_spec(
                    "n2",
                    instructions="清洗数据",
                    dependencies=["n1"],
                    reuse_from="s2",
                ),
                make_spec("n1", instructions="生成原始数据", reuse_from="s1"),
            ],
        )

        assert set(reused) == {"n1", "n2"}


class TestReuseCandidatesPayload:
    """重规划输入的可复用候选载荷（最小充分信息）。"""

    def _setup(self, tmp_path, old_plan, old_results):
        coordinator = build_coordinator(tmp_path, old_plan)
        coordinator._subtasks = [spec.model_copy(deep=True) for spec in old_plan]
        coordinator._results = dict(old_results)
        coordinator._plan_version = 1
        coordinator._plan_history = [
            [spec.model_copy(deep=True) for spec in old_plan]
        ]
        coordinator._archive_completed_results()
        return coordinator

    def test_candidates_carry_minimal_sufficient_info(self, tmp_path):
        """候选条目含旧版本、id、语义字段、success、摘要、证据与产物。"""
        old_plan = [
            make_spec(
                "s1",
                instructions="生成原始数据",
                role="code",
                expected_output="数据集",
                allowed_tools=["read_file"],
            ),
            make_spec("s2", instructions="清洗数据", dependencies=["s1"]),
        ]
        coordinator = self._setup(
            tmp_path,
            old_plan,
            {
                "s1": make_result(
                    "s1",
                    output="原始数据已生成",
                    evidence=["命令输出 A"],
                    artifacts=[],
                ),
                "s2": make_result(
                    "s2", success=False, output="", error="清洗脚本报错"
                ),
            },
        )

        candidates, cascade = coordinator._reuse_candidates_payload()

        # 失败级联范围含失败项自身（s2 失败但无依赖它的任务，无下游）
        assert cascade == ["s2"]
        by_id = {entry["subtask_id"]: entry for entry in candidates}
        assert set(by_id) == {"s1", "s2"}
        s1 = by_id["s1"]
        assert s1["plan_version"] == 1
        assert s1["instructions"] == "生成原始数据"
        assert s1["role"] == "code"
        assert s1["expected_output"] == "数据集"
        assert s1["dependencies"] == []
        assert s1["allowed_tools"] == ["read_file"]
        assert s1["success"] is True
        assert s1["output_summary"] == "原始数据已生成"
        assert s1["evidence"] == ["命令输出 A"]
        assert s1["artifacts"] == []
        assert "failure_diagnosis" not in s1
        s2 = by_id["s2"]
        assert s2["success"] is False
        assert s2["failure_diagnosis"] == "清洗脚本报错"

    def test_candidates_output_summary_truncated(self, tmp_path):
        """输出摘要截断到 500 字符（不回传完整输出）。"""
        old_plan = [make_spec("s1", instructions="A")]
        coordinator = self._setup(
            tmp_path,
            old_plan,
            {"s1": make_result("s1", output="x" * 800)},
        )

        candidates, _ = coordinator._reuse_candidates_payload()

        assert len(candidates[0]["output_summary"]) == 500

    def test_cascade_scope_covers_failed_and_downstream(self, tmp_path):
        """失败级联范围 = 失败子任务及其全部下游（含多层级联）。"""
        old_plan = [
            make_spec("s1", instructions="生成原始数据"),
            make_spec("s2", instructions="清洗数据", dependencies=["s1"]),
            make_spec("s3", instructions="生成图表", dependencies=["s2"]),
            make_spec("t", instructions="撰写文档"),
        ]
        coordinator = self._setup(
            tmp_path,
            old_plan,
            {
                "s1": make_result("s1", success=False, error="数据源不可用"),
                "s2": make_result("s2"),
                "s3": make_result("s3"),
                "t": make_result("t"),
            },
        )

        candidates, cascade = coordinator._reuse_candidates_payload()

        assert cascade == ["s1", "s2", "s3"]
        by_id = {entry["subtask_id"]: entry for entry in candidates}
        assert by_id["s1"]["success"] is False
        assert by_id["t"]["success"] is True

    def test_first_planning_has_no_candidates(self, tmp_path):
        """首次规划（版本 0，无历史）→ 无候选、无级联范围。"""
        coordinator = build_coordinator(tmp_path, [])
        coordinator._plan_version = 0

        candidates, cascade = coordinator._reuse_candidates_payload()

        assert candidates == []
        assert cascade == []


class TestArchiveCompletedResults:
    """归档：重规划替换 results 前固化历史来源。"""

    def test_archive_records_all_results_with_version_key(self, tmp_path):
        """成功与失败结果都按 (版本, subtask_id) 归档。"""
        old_plan = [
            make_spec("s1", instructions="A"),
            make_spec("s2", instructions="B"),
        ]
        coordinator = build_coordinator(tmp_path, old_plan)
        coordinator._plan_version = 1
        coordinator._results = {
            "s1": make_result("s1", output="成功产出"),
            "s2": make_result("s2", success=False, error="失败"),
        }

        coordinator._archive_completed_results()

        assert set(coordinator._archive_results) == {(1, "s1"), (1, "s2")}
        assert coordinator._archive_results[(1, "s1")].output == "成功产出"
        assert coordinator._archive_results[(1, "s2")].success is False

    def test_archive_survives_results_clear(self, tmp_path):
        """清空 results（重规划采纳）后归档仍可查证来源。"""
        old_plan = [make_spec("s1", instructions="A")]
        coordinator = build_coordinator(tmp_path, old_plan)
        coordinator._plan_version = 1
        coordinator._results = {"s1": make_result("s1", output="产出")}

        coordinator._archive_completed_results()
        coordinator._results.clear()

        assert coordinator._archive_results[(1, "s1")].output == "产出"


class TestDownstreamClosure:
    """失败级联闭包：默认当前计划 / 显式指定上一版计划。"""

    def test_closure_propagates_transitively(self, tmp_path):
        """级联闭包沿依赖边传递（s1 → s2 → s3），独立分支不受影响。"""
        old_plan = [
            make_spec("s1", instructions="A"),
            make_spec("s2", instructions="B", dependencies=["s1"]),
            make_spec("s3", instructions="C", dependencies=["s2"]),
            make_spec("t", instructions="D"),
        ]
        coordinator = build_coordinator(tmp_path, old_plan)

        closure = coordinator._downstream_closure({"s1"})

        assert closure == {"s1", "s2", "s3"}

    def test_closure_with_explicit_specs_ignores_current_plan(self, tmp_path):
        """显式传入上一版计划时，闭包按该计划的依赖图计算。"""
        previous_plan = [
            make_spec("a", instructions="A"),
            make_spec("b", instructions="B", dependencies=["a"]),
        ]
        current_plan = [make_spec("a", instructions="A")]  # 无 b
        coordinator = build_coordinator(tmp_path, current_plan)

        closure = coordinator._downstream_closure({"a"}, previous_plan)

        assert closure == {"a", "b"}

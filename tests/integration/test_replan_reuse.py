"""集成测试：重规划后的显式结果复用（模拟 Runner，不消耗真实 API）。

场景（v1 → 重规划 → v2，任务 35 显式声明制）：

- v1 计划 = [s1, s2, s3(依赖 s1), s4]：s1/s3/s4 成功，s2 三次原始尝试 +
  一次替换（三次替代尝试）全部失败 → 触发任务分解问题重规划；
- v2 计划（显式声明）：
  * s1/s3 声明 ``reuse_from`` 指向旧版自身（语义一致）→ 校验通过复用；
  * s2 换方案（不声明）→ 重新执行；
  * n4 声明 ``reuse_from="s4"`` 但语义改变 → ``fingerprint_incompatible``
    拒绝并回退执行；
  * n5 声明 ``reuse_from="s2"``（失败来源）→ ``source_not_successful``
    拒绝并回退执行；
- 验证：
  1. 重规划载荷附带 ``reuse_candidates`` 最小充分信息（含失败项诊断）
     与 ``failure_cascade_scope``；首次规划不提供；
  2. 无声明时绝不复用：v2 与 v1 语义完全相同（id、名称、指纹一致）
     也全部重新执行；
  3. 复用项不进入 pending（未重跑）；候选授权照常参与计划评审
     （评审可见性保留），但计划评审通过后真实签发被过滤——不为
     已复用任务留下无用途活跃 grant；
  4. 被拒声明回退正常 pending 执行（不静默成功）；
  5. 复用结果原样进入最终结果集与最终评审输入，携带 ``reused_from``
     来源标记，评审不因复用而跳过；
  6. 每个计划版本（含 fingerprint 索引与 reuse_from 声明）持久化到
     ResultStore（SQLite 兼容）。
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import so_agent.workflow as workflow_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.fingerprint import spec_fingerprint
from so_agent.models import (
    ExecutionResult,
    ReviewDecision,
    SubtaskSpec,
    TaskStatus,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.result_store import SQLiteResultStore
from so_agent.runtime.events import (
    EVENT_REPLAN,
    EVENT_RESULT_REUSED,
    EVENT_REUSE_REJECTED,
    EVENT_TOOL_GRANTED,
)
from so_agent.tool_packages.review_agent.agent import ReviewInput, ReviewStage
from so_agent.workflow import WorkflowEngine


class FakeRunResult:
    """假 RunResult：只保留 final_output，供控制层结构与文本解析。"""

    def __init__(self, final_output):
        self.final_output = final_output


class FakeStreamedResult:
    """假 RunResultStreaming：首次取流式事件时求值协程工厂并回填输出。

    正常路径下取到假事件后迭代结束，返回对象（final_output 已就绪）；
    ``cancel`` 记录超时收尾路径。
    """

    def __init__(self, run_call):
        self._run_call = run_call
        self.final_output = None
        self.raw_responses = []
        self.cancelled = False

    def cancel(self, mode="immediate"):
        self.cancelled = True

    async def stream_events(self):
        result = await self._run_call()
        self.final_output = result.final_output
        self.raw_responses = getattr(result, "raw_responses", None) or []
        yield {"type": "fake_event"}


def install_fake_runner(
    monkeypatch, *, orchestrator, review_agent, supervisor, reviews
):
    """把 workflow 模块内的 Runner 替换为按 Agent 身份分派的假 Runner。

    主管 Agent 经 ``run_streamed``（run_agent_streamed 的流式入口），
    评审仍走 ``run``。
    """

    class FakeRunner:
        @classmethod
        async def run(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            if starting_agent is review_agent:
                review_input = ReviewInput.model_validate_json(input)
                return FakeRunResult(reviews(review_input))
            payload = json.loads(input)
            return FakeRunResult(supervisor(payload))

        @classmethod
        def run_streamed(cls, starting_agent, input=None, *, max_turns=None, **kwargs):
            return FakeStreamedResult(
                lambda: cls.run(
                    starting_agent, input=input, max_turns=max_turns, **kwargs
                )
            )

    monkeypatch.setattr(workflow_mod, "Runner", FakeRunner)


def make_spec(
    subtask_id: str,
    *,
    instructions: str,
    dependencies: list[str] | None = None,
    reuse_from: str | None = None,
) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"子任务 {subtask_id}",
        instructions=instructions,
        role="code",
        dependencies=list(dependencies or []),
        allowed_tools=["read_file"],
        max_attempts=3,
        reuse_from=reuse_from,
    )


def v1_plan() -> list[SubtaskSpec]:
    """首版计划：s2 为初版方案（将失败）。"""
    return [
        make_spec("s1", instructions="生成原始数据"),
        make_spec("s2", instructions="清洗数据（初版方案）"),
        make_spec("s3", instructions="生成图表", dependencies=["s1"]),
        make_spec("s4", instructions="生成附录"),
    ]


def v2_plan() -> list[SubtaskSpec]:
    """重规划计划（显式声明）：

    - s1/s3 声明复用旧版自身（语义一致，校验通过）；
    - s2 换方案（不声明，重新执行）；
    - n4 声明复用 s4 但语义改变（指纹不兼容，被拒回退）；
    - n5 声明复用失败的 s2（来源不成功，被拒回退）。
    """
    return [
        make_spec("s1", instructions="生成原始数据", reuse_from="s1"),
        make_spec("s2", instructions="清洗数据（改用备用数据源方案）"),
        make_spec("s3", instructions="生成图表", dependencies=["s1"], reuse_from="s3"),
        make_spec("n4", instructions="生成附录（换新方案）", reuse_from="s4"),
        make_spec("n5", instructions="清洗数据（第二备选）", reuse_from="s2"),
    ]


def v2_plan_silent() -> list[SubtaskSpec]:
    """重规划计划（无声明）：与 v1 语义完全相同（id、指纹一致）但不声明复用。"""
    return [
        make_spec("s1", instructions="生成原始数据"),
        make_spec("s2", instructions="清洗数据（初版方案）"),
        make_spec("s3", instructions="生成图表", dependencies=["s1"]),
        make_spec("s4", instructions="生成附录"),
    ]


class ReplanSupervisor:
    """按计划轮次返回不同拆分方案；指定版本的指定子任务持续失败。

    ``exec_counts`` 记录每个子任务的执行载荷次数（含失败尝试），
    输出文本形如 ``s1#1``（第 N 次执行的产出）便于断重复用来源。
    """

    def __init__(
        self,
        *,
        plans: list[list[SubtaskSpec]],
        fail_version: int = 1,
        fail_ids: set[str] | None = None,
    ):
        self.plans = plans
        self.fail_version = fail_version
        self.fail_ids = set(fail_ids or set())
        self.payloads: list[dict] = []
        self.exec_counts: dict[str, int] = {}

    @property
    def phases(self) -> list[str]:
        return [payload.get("phase") for payload in self.payloads]

    def __call__(self, payload: dict) -> OrchestratorOutput:
        self.payloads.append(payload)
        phase = payload.get("phase")
        if phase in ("planning", "replanning"):
            return self._plan_output(self._plan_index(), payload)
        if phase == "execute_subtask":
            return self._execution_output(payload)
        if phase == "aggregating":
            return OrchestratorOutput(
                task_id=str(payload.get("task_id") or "stub-task"),
                status=TaskStatus.AGGREGATING,
                final_result="聚合最终结果",
            )
        raise AssertionError(f"未预期的阶段载荷：{phase}")

    def _plan_index(self) -> int:
        plan_calls = sum(
            1 for item in self.payloads if item.get("phase") in ("planning", "replanning")
        )
        return min(plan_calls - 1, len(self.plans) - 1)

    def _plan_output(self, index: int, payload: dict) -> OrchestratorOutput:
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=(
                TaskStatus.REPLANNING
                if payload.get("phase") == "replanning"
                else TaskStatus.PLANNING
            ),
            plan_versions=[
                [spec.model_copy(deep=True) for spec in self.plans[index]]
            ],
        )

    def _execution_output(self, payload: dict) -> OrchestratorOutput:
        spec = payload["subtask"]
        subtask_id = spec["subtask_id"]
        self.exec_counts[subtask_id] = self.exec_counts.get(subtask_id, 0) + 1
        if subtask_id in self.fail_ids and spec["plan_version"] == self.fail_version:
            result = ExecutionResult(
                subtask_id=subtask_id,
                success=False,
                error=f"{subtask_id} 持续失败（模拟）",
            )
        else:
            result = ExecutionResult(
                subtask_id=subtask_id,
                success=True,
                output=f"{subtask_id}#{self.exec_counts[subtask_id]}",
            )
        return OrchestratorOutput(
            task_id=str(payload.get("task_id") or "stub-task"),
            status=TaskStatus.EXECUTING,
            final_result=result.model_dump_json(),
        )


class ReviewScript:
    """评审脚本：计划/授权评审与结果评审均通过（记录调用序列）。"""

    def __init__(self, *, plan_passed: bool = True, result_passed: bool = True):
        self.plan_passed = plan_passed
        self.result_passed = result_passed
        self.calls: list[ReviewInput] = []

    def __call__(self, review_input: ReviewInput) -> ReviewDecision:
        self.calls.append(review_input)
        if review_input.stage == ReviewStage.RESULT_REVIEW:
            return ReviewDecision(
                stage="result_review",
                passed=self.result_passed,
                issues=[] if self.result_passed else ["整体结果不满足验收标准"],
                required_fixes=[] if self.result_passed else ["修复整体结果后重新提交"],
                summary="结果评审（模拟）",
            )
        return ReviewDecision(
            stage="plan_review",
            passed=self.plan_passed,
            issues=[] if self.plan_passed else ["子任务拆分粒度过粗"],
            required_fixes=[] if self.plan_passed else ["重新拆分并补充依赖"],
            summary="计划评审（模拟）",
        )


def build_engine(tmp_path, monkeypatch, *, supervisor, reviews):
    """组装工作流引擎：SQLite 结果存储 + 假主管/评审 Agent。"""
    context = ProjectContext(sandbox_dir=tmp_path, project_name="replan-reuse")
    settings = Settings(_env_file=None)
    store = SQLiteResultStore(str(tmp_path / "result_store.db"))
    orchestrator = SimpleNamespace(name="fake-supervisor")
    review_agent = SimpleNamespace(name="fake-reviewer")
    install_fake_runner(
        monkeypatch,
        orchestrator=orchestrator,
        review_agent=review_agent,
        supervisor=supervisor,
        reviews=reviews,
    )
    engine = WorkflowEngine(
        context,
        settings=settings,
        orchestrator=orchestrator,
        review_agent=review_agent,
        result_store=store,
    )
    return context, engine, store


def exec_ids(supervisor: ReplanSupervisor, plan_version: int) -> list[str]:
    """提取某计划版本下全部子任务执行载荷的子任务编号（按发生顺序）。"""
    return [
        payload["subtask"]["subtask_id"]
        for payload in supervisor.payloads
        if payload.get("phase") == "execute_subtask"
        and payload["subtask"]["plan_version"] == plan_version
    ]


def granted_slot_ids(engine: WorkflowEngine) -> list[str]:
    """提取全部真实签发授权（EVENT_TOOL_GRANTED）的子任务槽位。

    proposal 候选不经过网关登记，不产生该事件——只有
    ``_issue_grants`` 的真实签发才会记录。
    """
    return [
        record["payload"]["agent_id"].split(":")[-2]
        for record in engine._events.events
        if record["event_type"] == EVENT_TOOL_GRANTED
    ]


def granted_slots_after_replan(engine: WorkflowEngine) -> set[str]:
    """提取重规划事件之后真实签发的授权槽位（即 v2 阶段的签发）。"""
    records = engine._events.events
    replan_index = next(
        index
        for index, record in enumerate(records)
        if record["event_type"] == EVENT_REPLAN
    )
    return {
        record["payload"]["agent_id"].split(":")[-2]
        for record in records[replan_index:]
        if record["event_type"] == EVENT_TOOL_GRANTED
    }


def event_payloads(engine: WorkflowEngine, event_type: str) -> list[dict]:
    """提取指定事件类型的 payload 列表（按发生顺序）。"""
    return [
        record["payload"]
        for record in engine._events.events
        if record["event_type"] == event_type
    ]


def review_call(reviews: ReviewScript, stage: ReviewStage) -> ReviewInput:
    return next(call for call in reviews.calls if call.stage == stage)


class TestExplicitResultReuse:
    async def test_explicit_declarations_reused_and_rejected_fallback(
        self, tmp_path, monkeypatch, capsys
    ):
        supervisor = ReplanSupervisor(
            plans=[v1_plan(), v2_plan()], fail_version=1, fail_ids={"s2"}
        )
        reviews = ReviewScript()
        context, engine, _ = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("生成数据报表", acceptance_criteria="通过验收")

        # 1) v1 执行：s2 同因两次原始尝试 + 两次替代尝试全部失败（抑制）
        v1_execs = exec_ids(supervisor, plan_version=1)
        assert sorted(set(v1_execs)) == ["s1", "s2", "s3", "s4"]
        assert v1_execs.count("s2") == 4
        assert v1_execs.count("s1") == 1
        assert v1_execs.count("s3") == 1
        assert v1_execs.count("s4") == 1

        # 2) 重规划载荷：可复用候选最小充分信息 + 失败级联范围
        replanning = [
            payload
            for payload in supervisor.payloads
            if payload.get("phase") == "replanning"
        ]
        assert len(replanning) == 1
        candidates = replanning[0]["reuse_candidates"]
        assert {item["subtask_id"] for item in candidates} == {
            "s1", "s2", "s3", "s4",
        }
        by_id = {item["subtask_id"]: item for item in candidates}
        # 成功候选：含语义字段与输出摘要，无失败诊断
        assert by_id["s1"]["plan_version"] == 1
        assert by_id["s1"]["success"] is True
        assert by_id["s1"]["instructions"] == "生成原始数据"
        assert by_id["s1"]["output_summary"] == "s1#1"
        assert "failure_diagnosis" not in by_id["s1"]
        # 失败候选：success=False 且含失败诊断（完整失败语境呈现）
        assert by_id["s2"]["success"] is False
        assert "持续失败" in by_id["s2"]["failure_diagnosis"]
        # 失败级联范围：失败项自身（s2 无下游）
        assert replanning[0]["failure_cascade_scope"] == ["s2"]
        # 已完成结果摘要照常附带（供主管参考）
        assert {item["subtask_id"] for item in replanning[0]["completed_results"]} == {
            "s1", "s3", "s4",
        }

        # 3) v2 阶段：复用项未重跑；被拒声明回退重新执行
        assert sorted(exec_ids(supervisor, plan_version=2)) == ["n4", "n5", "s2"]
        assert supervisor.exec_counts["s1"] == 1  # 复用，未重跑
        assert supervisor.exec_counts["s3"] == 1  # 复用，未重跑
        assert supervisor.exec_counts["s2"] == 5  # v1 四次失败（同因抑制）+ v2 一次成功
        assert supervisor.exec_counts["n4"] == 1  # 指纹不兼容被拒 → 回退执行
        assert supervisor.exec_counts["n5"] == 1  # 失败来源被拒 → 回退执行

        # 4) 复用结果原样进入最终结果集（v1 产出 + 来源标记）
        assert set(engine._results) == {"s1", "s2", "s3", "n4", "n5"}
        assert all(result.success for result in engine._results.values())
        assert engine._results["s1"].output == "s1#1"
        assert engine._results["s1"].metadata["reused_from"] == "s1"
        assert engine._results["s1"].metadata["reused_from_plan_version"] == 1
        assert engine._results["s3"].output == "s3#1"
        assert engine._results["s3"].metadata["reused_from"] == "s3"
        # 被拒回退与换方案任务是 v2 新产出，无来源标记
        assert engine._results["s2"].output == "s2#5"
        assert engine._results["n4"].output == "n4#1"
        assert engine._results["n5"].output == "n5#1"
        for key in ("s2", "n4", "n5"):
            assert "reused_from" not in (engine._results[key].metadata or {})

        # 5) 事件：result_reused × 2、reuse_rejected × 2（机器可读 reason）
        reused_events = event_payloads(engine, EVENT_RESULT_REUSED)
        assert {
            (item["subtask_id"], item["source_subtask_id"])
            for item in reused_events
        } == {("s1", "s1"), ("s3", "s3")}
        rejected = {
            (item["subtask_id"], item["reason"])
            for item in event_payloads(engine, EVENT_REUSE_REJECTED)
        }
        assert rejected == {("n4", "fingerprint_incompatible"), ("n5", "source_not_successful")}
        # 被拒事件不携带完整输出（防泄露）
        for item in event_payloads(engine, EVENT_REUSE_REJECTED):
            assert "output" not in item

        # 6) 授权：v2 真实签发只覆盖实际执行任务，复用项无活跃 grant
        assert granted_slots_after_replan(engine) == {"s2", "n4", "n5"}
        # 全量签发历史：s1/s3 仅在 v1 签发一次，v2 不再签发
        all_slots = granted_slot_ids(engine)
        assert all_slots.count("s1") == 1
        assert all_slots.count("s3") == 1
        # 候选授权照常覆盖全部任务（评审可见性：复用项候选参与最小化评审）
        auth_review = [
            call
            for call in reviews.calls
            if call.stage == ReviewStage.AUTHORIZATION_REVIEW
        ][-1]
        proposal_slots = {
            grant.agent_id.split(":")[-2] for grant in auth_review.tool_grants
        }
        assert proposal_slots == {"s1", "s2", "s3", "n4", "n5"}
        # 计划评审可见复用声明本身
        v2_plan_review = [
            call
            for call in reviews.calls
            if call.stage == ReviewStage.PLAN_REVIEW
        ][-1]
        declared = {
            spec.subtask_id: spec.reuse_from for spec in v2_plan_review.subtasks
        }
        assert declared["s1"] == "s1"
        assert declared["s3"] == "s3"
        assert declared["n4"] == "s4"
        assert declared["n5"] == "s2"
        assert declared["s2"] is None

        # 7) 最终评审：复用结果照常参与且可见来源标记（不因复用跳过）
        result_review = review_call(reviews, ReviewStage.RESULT_REVIEW)
        review_results = {
            item.subtask_id: item for item in result_review.execution_results
        }
        assert set(review_results) == {"s1", "s2", "s3", "n4", "n5"}
        assert review_results["s1"].metadata["reused_from"] == "s1"
        assert review_results["s1"].metadata["reused_from_plan_version"] == 1
        assert review_results["s3"].metadata["reused_from"] == "s3"
        assert "reused_from" not in (review_results["n4"].metadata or {})
        # 复用结果携带 v1 原始产出（评审对象是复用来源内容本身）
        assert review_results["s1"].output == "s1#1"

        # 8) 评审序列：v1/v2 各完整两阶段计划评审 + 一次结果评审
        assert [call.stage for call in reviews.calls] == [
            ReviewStage.PLAN_REVIEW,
            ReviewStage.AUTHORIZATION_REVIEW,
            ReviewStage.PLAN_REVIEW,
            ReviewStage.AUTHORIZATION_REVIEW,
            ReviewStage.RESULT_REVIEW,
        ]

        # 9) 会话明细：复用提示打印到 stderr
        stderr = capsys.readouterr().err
        assert "复用 2 个子任务的既有成功结果（s1, s3），跳过重新执行" in stderr
        assert "复用声明被拒：[n4] ← [s4]（原因 fingerprint_incompatible）" in stderr
        assert "复用声明被拒：[n5] ← [s2]（原因 source_not_successful）" in stderr

        # 10) 工作流正常完成
        assert output.status == TaskStatus.COMPLETED
        assert engine.plan_version == 2
        assert output.final_result == "聚合最终结果"

    async def test_no_declaration_never_reuses(
        self, tmp_path, monkeypatch, capsys
    ):
        """无 reuse_from 声明：即使 id、名称与指纹完全相同也绝不复用。"""
        supervisor = ReplanSupervisor(
            plans=[v1_plan(), v2_plan_silent()],
            fail_version=1,
            fail_ids={"s2"},
        )
        reviews = ReviewScript()
        context, engine, _ = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("生成数据报表", acceptance_criteria="通过验收")

        # v1：s2 同因四次失败触发重规划；v2：全部子任务重新执行
        assert exec_ids(supervisor, plan_version=1).count("s2") == 4
        v2_execs = exec_ids(supervisor, plan_version=2)
        assert sorted(set(v2_execs)) == ["s1", "s2", "s3", "s4"]
        assert v2_execs.count("s1") == 1
        assert v2_execs.count("s2") == 1
        assert v2_execs.count("s3") == 1
        assert v2_execs.count("s4") == 1
        assert supervisor.exec_counts["s1"] == 2
        assert supervisor.exec_counts["s3"] == 2
        assert supervisor.exec_counts["s4"] == 2
        assert supervisor.exec_counts["s2"] == 5

        # 重规划载荷照常提供可复用候选（主管未声明则不复用）
        replanning = [
            payload
            for payload in supervisor.payloads
            if payload.get("phase") == "replanning"
        ]
        assert {item["subtask_id"] for item in replanning[0]["reuse_candidates"]} == {
            "s1", "s2", "s3", "s4",
        }

        # 结果全部来自 v2 重跑产出，无来源标记
        assert set(engine._results) == {"s1", "s2", "s3", "s4"}
        assert all(result.success for result in engine._results.values())
        assert engine._results["s1"].output == "s1#2"
        assert engine._results["s3"].output == "s3#2"
        for result in engine._results.values():
            assert "reused_from" not in (result.metadata or {})

        # 无复用/拒绝事件；无复用提示
        assert event_payloads(engine, EVENT_RESULT_REUSED) == []
        assert event_payloads(engine, EVENT_REUSE_REJECTED) == []
        assert "跳过重新执行" not in capsys.readouterr().err

        # 最终评审输入覆盖全部 v2 新执行结果
        result_review = review_call(reviews, ReviewStage.RESULT_REVIEW)
        assert {item.subtask_id for item in result_review.execution_results} == {
            "s1", "s2", "s3", "s4",
        }

        # 正常完成
        assert output.status == TaskStatus.COMPLETED
        assert engine.plan_version == 2

    async def test_plan_versions_with_fingerprints_persisted(
        self, tmp_path, monkeypatch
    ):
        supervisor = ReplanSupervisor(
            plans=[v1_plan(), v2_plan()], fail_version=1, fail_ids={"s2"}
        )
        reviews = ReviewScript()
        context, engine, store = build_engine(
            tmp_path, monkeypatch, supervisor=supervisor, reviews=reviews
        )

        output = await engine.run("生成数据报表", acceptance_criteria="通过验收")

        assert output.status == TaskStatus.COMPLETED

        # 每个采纳的计划版本（含指纹索引）持久化到 ResultStore
        pv1 = store.get_plan_version(1)
        pv2 = store.get_plan_version(2)
        assert pv1 is not None and pv2 is not None
        assert {spec.subtask_id for spec in pv1.specs} == {"s1", "s2", "s3", "s4"}
        assert {spec.subtask_id for spec in pv2.specs} == {"s1", "s2", "s3", "n4", "n5"}
        assert set(pv1.fingerprints) == {"s1", "s2", "s3", "s4"}
        assert set(pv2.fingerprints) == {"s1", "s2", "s3", "n4", "n5"}

        # 语义不变的子任务指纹跨版本一致（复用声明的校验依据）
        assert pv1.fingerprints["s1"] == pv2.fingerprints["s1"]
        assert pv1.fingerprints["s3"] == pv2.fingerprints["s3"]
        assert pv1.fingerprints["s2"] != pv2.fingerprints["s2"]

        # 指纹与 spec 语义字段哈希一致（reuse_from 不入指纹）
        specs_v1 = {spec.subtask_id: spec for spec in pv1.specs}
        specs_v2 = {spec.subtask_id: spec for spec in pv2.specs}
        for subtask_id, fingerprint in pv1.fingerprints.items():
            assert fingerprint == spec_fingerprint(specs_v1[subtask_id])
        for subtask_id, fingerprint in pv2.fingerprints.items():
            assert fingerprint == spec_fingerprint(specs_v2[subtask_id])

        # 显式声明经 SQLite 往返保留；首版无声明
        assert specs_v1["s1"].reuse_from is None
        assert specs_v2["s1"].reuse_from == "s1"
        assert specs_v2["s3"].reuse_from == "s3"
        assert specs_v2["n4"].reuse_from == "s4"
        assert specs_v2["n5"].reuse_from == "s2"
        assert specs_v2["s2"].reuse_from is None

        # v1 为首次采纳（无触发问题）；v2 由执行失败重规划触发
        assert pv1.trigger_issue is None
        assert pv2.trigger_issue is not None
        assert pv2.trigger_issue.subtask_id == "s2"

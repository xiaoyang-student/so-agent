"""单元测试：增量补充机制（重试 / 替换 / 返工三处的历史产出注入）。

覆盖：
1. ``SubtaskDAGScheduler._run_attempts``：非首次尝试时，前次失败的
   output（> 20 字符）作为"已有部分产出"注入 instructions；空/过短
   输出不注入；多次重试始终使用最近一次的部分产出；超长产出截断到
   2000 字符；原始 SubtaskSpec 不被就地修改；
2. ``SubtaskDAGScheduler._execute_subtask`` 替换分支：主 Agent 失败
   但留有部分产出时，注入"前任产出参考"给替代 Agent；空/过短输出
   不注入；
3. ``AggregationCoordinator._build_rework_specs``：返工子任务注入
   已有产出（> 20 字符）与最近一次评审的 required_fixes；无已有产出
   （或无评审反馈）时 instructions 保持不变。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import AgentRecord, ExecutionResult, ReviewDecision, SubtaskSpec
from so_agent.runtime.scheduler import SubtaskDAGScheduler
from so_agent.workflow import WorkflowEngine


def subtask(
    subtask_id: str,
    *,
    deps: list[str] | None = None,
    max_attempts: int = 3,
) -> SubtaskSpec:
    """构造测试用子任务规格（instructions 为可辨识的固定文本）。"""
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=f"子任务 {subtask_id}",
        instructions=f"完成子任务 {subtask_id}",
        role="code",
        dependencies=deps or [],
        max_attempts=max_attempts,
    )


class TestRetryPartialOutputInjection:
    """C1：重试时传递前次部分产出。"""

    async def test_retry_injects_previous_partial_output(self):
        seen: list[str] = []
        original = subtask("s1")

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            seen.append(task_spec.instructions)
            if agent.attempt_count == 1:
                return ExecutionResult(
                    subtask_id=task_spec.subtask_id,
                    success=False,
                    output="前次部分产出：" + "细" * 30,
                    error="第一次失败",
                )
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(
            executor,
            settings=Settings(max_agent_attempts=2, max_replacement_requests=0),
        )
        results = await scheduler.schedule("t1", [original])

        assert results["s1"].success is True
        assert "[重试提示]" not in seen[0]
        assert "[重试提示]" in seen[1]
        assert "[已有部分产出]" in seen[1]
        assert "前次部分产出" in seen[1]
        # 注入发生在深拷贝上，原始 spec 不被就地修改
        assert original.instructions == "完成子任务 s1"

    @pytest.mark.parametrize("length,expect_injected", [(20, False), (21, True)])
    async def test_retry_output_length_threshold(self, length, expect_injected):
        seen: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            seen.append(task_spec.instructions)
            if agent.attempt_count == 1:
                return ExecutionResult(
                    subtask_id=task_spec.subtask_id,
                    success=False,
                    output="p" * length,
                    error="失败",
                )
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(
            executor,
            settings=Settings(max_agent_attempts=2, max_replacement_requests=0),
        )
        await scheduler.schedule("t1", [subtask("s1")])

        assert "[重试提示]" in seen[1]
        assert ("[已有部分产出]" in seen[1]) is expect_injected

    async def test_retry_empty_output_not_injected(self):
        seen: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            seen.append(task_spec.instructions)
            if agent.attempt_count == 1:
                return ExecutionResult(
                    subtask_id=task_spec.subtask_id, success=False, error="无产出失败"
                )
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(
            executor,
            settings=Settings(max_agent_attempts=2, max_replacement_requests=0),
        )
        await scheduler.schedule("t1", [subtask("s1")])

        assert "[重试提示]" in seen[1]
        assert "[已有部分产出]" not in seen[1]

    async def test_retry_uses_latest_partial_output(self):
        """连续失败时，每次重试注入的都是最近一次尝试的部分产出。

        错误文本指纹互异（不含纯数字差异，避免同因抑制提前终止）。
        """
        seen: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            seen.append(task_spec.instructions)
            if agent.attempt_count < 3:
                return ExecutionResult(
                    subtask_id=task_spec.subtask_id,
                    success=False,
                    output=f"第{agent.attempt_count}次的产出内容" + "甲" * 25,
                    error="失败alpha" if agent.attempt_count == 1 else "失败beta",
                )
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(
            executor,
            settings=Settings(max_agent_attempts=3, max_replacement_requests=0),
        )
        await scheduler.schedule("t1", [subtask("s1")])

        assert "第1次的产出内容" in seen[1]
        assert "第1次的产出内容" not in seen[2]
        assert "第2次的产出内容" in seen[2]

    async def test_retry_partial_output_truncated_to_2000(self):
        big = "长" * 3000
        seen: list[str] = []

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            seen.append(task_spec.instructions)
            if agent.attempt_count == 1:
                return ExecutionResult(
                    subtask_id=task_spec.subtask_id,
                    success=False,
                    output=big,
                    error="失败",
                )
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(
            executor,
            settings=Settings(max_agent_attempts=2, max_replacement_requests=0),
        )
        await scheduler.schedule("t1", [subtask("s1")])

        assert big[:2000] in seen[1]
        assert big[:2001] not in seen[1]


class TestReplacementPredecessorOutputInjection:
    """C2：替换 Agent 时传递前次产出。"""

    async def test_replacement_injects_predecessor_output(self):
        replacement_instructions: list[str] = []

        async def primary_executor(
            task_spec: SubtaskSpec, agent: AgentRecord
        ) -> ExecutionResult:
            return ExecutionResult(
                subtask_id=task_spec.subtask_id,
                success=False,
                output="主 Agent 的部分成果：" + "成" * 30,
                error="主 Agent 失败",
            )

        async def replacement_executor(
            task_spec: SubtaskSpec, agent: AgentRecord
        ) -> ExecutionResult:
            replacement_instructions.append(task_spec.instructions)
            return ExecutionResult(
                subtask_id=task_spec.subtask_id, success=True, output="替代完成"
            )

        scheduler = SubtaskDAGScheduler(
            primary_executor,
            replacement_executor=replacement_executor,
            settings=Settings(max_agent_attempts=1, max_replacement_requests=1),
        )
        results = await scheduler.schedule("t1", [subtask("s1")])

        assert results["s1"].success is True
        assert len(replacement_instructions) == 1
        assert "[替代 Agent 上下文]" in replacement_instructions[0]
        assert "[前任产出参考]" in replacement_instructions[0]
        assert "主 Agent 的部分成果" in replacement_instructions[0]

    @pytest.mark.parametrize("output,expect_injected", [("", False), ("r" * 20, False), ("r" * 21, True)])
    async def test_replacement_output_threshold(self, output, expect_injected):
        replacement_instructions: list[str] = []

        async def primary_executor(
            task_spec: SubtaskSpec, agent: AgentRecord
        ) -> ExecutionResult:
            return ExecutionResult(
                subtask_id=task_spec.subtask_id,
                success=False,
                output=output,
                error="主 Agent 失败",
            )

        async def replacement_executor(
            task_spec: SubtaskSpec, agent: AgentRecord
        ) -> ExecutionResult:
            replacement_instructions.append(task_spec.instructions)
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(
            primary_executor,
            replacement_executor=replacement_executor,
            settings=Settings(max_agent_attempts=1, max_replacement_requests=1),
        )
        await scheduler.schedule("t1", [subtask("s1")])

        assert "[替代 Agent 上下文]" in replacement_instructions[0]
        assert ("[前任产出参考]" in replacement_instructions[0]) is expect_injected

    async def test_replacement_truncates_predecessor_output_to_3000(self):
        big = "大" * 4000
        replacement_instructions: list[str] = []

        async def primary_executor(
            task_spec: SubtaskSpec, agent: AgentRecord
        ) -> ExecutionResult:
            return ExecutionResult(
                subtask_id=task_spec.subtask_id,
                success=False,
                output=big,
                error="主 Agent 失败",
            )

        async def replacement_executor(
            task_spec: SubtaskSpec, agent: AgentRecord
        ) -> ExecutionResult:
            replacement_instructions.append(task_spec.instructions)
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True)

        scheduler = SubtaskDAGScheduler(
            primary_executor,
            replacement_executor=replacement_executor,
            settings=Settings(max_agent_attempts=1, max_replacement_requests=1),
        )
        await scheduler.schedule("t1", [subtask("s1")])

        assert big[:3000] in replacement_instructions[0]
        assert big[:3001] not in replacement_instructions[0]


def build_engine(tmp_path) -> WorkflowEngine:
    """组装最小工作流引擎（与集成测试同构，供返工 spec 构造测试使用）。"""
    context = ProjectContext(
        sandbox_dir=tmp_path, project_name="incremental-supplement"
    )
    engine = WorkflowEngine(
        context,
        settings=Settings(_env_file=None),
        orchestrator=SimpleNamespace(name="fake-supervisor"),
        review_agent=SimpleNamespace(name="fake-reviewer"),
    )
    engine._task_request = SimpleNamespace(task_id="t1")
    return engine


class TestReworkSpecIncrementalSupplement:
    """C3：返工时传递已有产出 + 补充指令。"""

    def test_rework_specs_include_existing_output_and_review_fixes(self, tmp_path):
        engine = build_engine(tmp_path)
        engine._subtasks = [subtask("s1"), subtask("s2", deps=["s1"])]
        engine._results = {
            "s1": ExecutionResult(
                subtask_id="s1", success=True, output="旧产出：" + "内" * 30
            ),
        }
        engine._last_review = ReviewDecision(
            stage="result_review",
            passed=False,
            issues=["产出不完整"],
            required_fixes=["补充第 3 节数据", "校对结论"],
        )

        specs = engine._build_rework_specs({"s1", "s2"})

        assert [item.subtask_id for item in specs] == ["s1", "s2"]
        s1 = specs[0]
        assert "[已有产出]" in s1.instructions
        assert "旧产出" in s1.instructions
        assert "[评审要求]" in s1.instructions
        assert "补充第 3 节数据" in s1.instructions
        # 依赖裁剪语义不受注入影响：s1 在返工集合内，s2 保留依赖
        assert specs[1].dependencies == ["s1"]
        # 仅返工 s2 时依赖被裁剪为空
        only_s2 = engine._build_rework_specs({"s2"})
        assert only_s2[0].dependencies == []
        # 原始子任务规格不被就地修改
        assert engine._subtasks[0].instructions == "完成子任务 s1"

    def test_rework_without_existing_output_gets_only_review_fixes(self, tmp_path):
        engine = build_engine(tmp_path)
        engine._subtasks = [subtask("s1")]
        engine._results = {}
        engine._last_review = ReviewDecision(
            stage="result_review",
            passed=False,
            required_fixes=["重做并补充分析"],
        )

        specs = engine._build_rework_specs({"s1"})

        assert "[已有产出]" not in specs[0].instructions
        assert "[评审要求]" in specs[0].instructions
        assert "重做并补充分析" in specs[0].instructions

    def test_rework_short_existing_output_not_injected(self, tmp_path):
        engine = build_engine(tmp_path)
        engine._subtasks = [subtask("s1")]
        engine._results = {
            "s1": ExecutionResult(subtask_id="s1", success=True, output="短产出")
        }
        engine._last_review = None

        specs = engine._build_rework_specs({"s1"})

        assert "[已有产出]" not in specs[0].instructions
        assert "[评审要求]" not in specs[0].instructions
        assert specs[0].instructions == "完成子任务 s1"

    def test_rework_specs_unchanged_without_output_or_review(self, tmp_path):
        engine = build_engine(tmp_path)
        engine._subtasks = [subtask("s1")]
        engine._results = {}
        engine._last_review = None

        specs = engine._build_rework_specs({"s1"})

        assert specs[0].instructions == "完成子任务 s1"

    def test_rework_review_without_required_fixes_no_review_section(self, tmp_path):
        engine = build_engine(tmp_path)
        engine._subtasks = [subtask("s1")]
        engine._results = {}
        engine._last_review = ReviewDecision(
            stage="result_review",
            passed=False,
            issues=["仅记录问题，未给出必改项"],
        )

        specs = engine._build_rework_specs({"s1"})

        assert "[评审要求]" not in specs[0].instructions
        assert specs[0].instructions == "完成子任务 s1"

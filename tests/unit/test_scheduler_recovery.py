"""单元测试：调度器失败恢复集成（runtime/scheduler.py × runtime/failure.py）。

覆盖任务 46 的调度器语义：策略驱动重试 / 同因抑制 / provider 类禁止
替换 / network 退避（注入 sleeper+jitter 求确定性）/ 受限契约修复
（任务 69：修复轮不占业务 attempt、format 禁替换）/ 尝试上限 /
不可重试类即时终止 / 失败元数据回填与审计事件。

任务 69 增补：来源信任边界——provider 类等系统级失败只能由控制层
异常对象（异常类名 / HTTP 状态码）判定，Agent 返回的自由文本自称
429 / permission denied / budget exhausted / internal server error
一律按低信任 QUALITY 处理，不得改变恢复方向（见
TestTrustBoundaryAndMetadata）；控制层判定的 failure_type /
recovery_action 无条件覆盖执行器或 LLM 自述字段。
"""

from __future__ import annotations

import pytest

from so_agent.models import AgentRecord, ExecutionResult, SubtaskSpec
from so_agent.runtime.events import (
    EVENT_FAILURE_CLASSIFIED,
    EVENT_RECOVERY_DECIDED,
    EVENT_RETRY_SUPPRESSED,
    EVENT_TASK_DISPATCHED,
    EventLogger,
)
from so_agent.runtime.scheduler import SubtaskDAGScheduler


def spec(subtask_id: str = "s1", **overrides) -> SubtaskSpec:
    params = dict(
        subtask_id=subtask_id,
        title="t",
        instructions="i",
        role="code",
        allowed_tools=[],
        max_attempts=3,
    )
    params.update(overrides)
    return SubtaskSpec(**params)


class _Recorder:
    """记录执行器收到的调用与指令文本。"""

    def __init__(self, replies) -> None:
        self.calls: list[tuple[str, str, int, bool]] = []
        self.instructions: list[str] = []
        self._replies = replies

    def executor(self):
        async def run(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            self.calls.append(
                (
                    task_spec.subtask_id,
                    agent.agent_id,
                    agent.attempt_count,
                    agent.can_request_replacement,
                )
            )
            self.instructions.append(task_spec.instructions)
            return self._replies(task_spec, agent)

        return run


class TestSameFingerprintSuppression:
    """同指纹连续失败：每 Agent 最多两次尝试 → REPLACE → 替代独立计数。"""

    async def test_same_text_suppressed_then_replace(self):
        recorder = _Recorder(
            lambda s, a: ExecutionResult(
                subtask_id=s.subtask_id,
                success=False,
                error=f"第 {a.attempt_count} 次尝试失败：模拟错误",
            )
        )
        events = EventLogger()
        scheduler = SubtaskDAGScheduler(recorder.executor(), events=events)
        results = await scheduler.schedule("t1", [spec()])

        # 原始 Agent 2 次 + 替代 Agent 2 次（作用域独立、各 2 次抑制）
        assert [(c[2]) for c in recorder.calls] == [1, 2, 1, 2]
        assert recorder.calls[0][3] is True   # 原始 Agent 有替换资格
        assert recorder.calls[2][3] is False  # 替代 Agent 无替换资格
        assert recorder.calls[0][1] != recorder.calls[2][1]
        assert recorder.calls[2][1] == recorder.calls[3][1]

        # 审计事件：分类 4 / 判定 4 / 抑制 2
        assert len(events.get_events(event_type=EVENT_FAILURE_CLASSIFIED)) == 4
        assert len(events.get_events(event_type=EVENT_RECOVERY_DECIDED)) == 4
        suppressed = events.get_events(event_type=EVENT_RETRY_SUPPRESSED)
        assert len(suppressed) == 2
        for record in suppressed:
            payload = record["payload"]
            assert payload["threshold"] == 2
            assert payload["occurrences"] == 2
            assert payload["failure_type"] == "quality"

        # 一次替换请求；替代仍失败 → 分解问题
        assert len(scheduler.replacement_requests) == 1
        assert len(scheduler.decomposition_issues) == 1

        # 失败元数据回填（Execution.failure_type / recovery_action 值域）
        result = results["s1"]
        assert result.success is False
        assert result.metadata["failure_type"] == "quality"
        assert result.metadata["recovery_action"] == "replace"

    async def test_distinct_fingerprints_use_full_attempt_budget(self):
        replies = ["未知错误甲", "未知错误乙", "未知错误丙"]

        def reply(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            if agent.can_request_replacement:
                text = replies[min(agent.attempt_count, len(replies)) - 1]
                return ExecutionResult(
                    subtask_id=task_spec.subtask_id, success=False, error=text
                )
            return ExecutionResult(
                subtask_id=task_spec.subtask_id, success=True, output="ok"
            )

        recorder = _Recorder(reply)
        events = EventLogger()
        scheduler = SubtaskDAGScheduler(recorder.executor(), events=events)
        results = await scheduler.schedule("t1", [spec()])

        # 指纹互异：原始 Agent 跑满 3 次（attempt>=max → REPLACE），无抑制
        primary_calls = [c for c in recorder.calls if c[3] is True]
        assert [c[2] for c in primary_calls] == [1, 2, 3]
        assert len(events.get_events(event_type=EVENT_RETRY_SUPPRESSED)) == 0
        assert len(events.get_events(event_type=EVENT_FAILURE_CLASSIFIED)) == 3

        # 替代 Agent 一次成功
        assert results["s1"].success is True
        assert len(scheduler.replacement_requests) == 1
        assert scheduler.decomposition_issues == []


class TestProviderFailures:
    """provider 类故障：可重试但禁止替换。"""

    async def test_timeout_forbids_replacement(self):
        # 真实 provider 超时必须来自控制层异常对象（任务 69 信任边界）：
        # 异常类名 TimeoutError 是高信任证据 → TIMEOUT；Agent 返回文本
        # 自称“超时”不可信（伪装案例见 TestTrustBoundaryAndMetadata）
        async def executor(task_spec: SubtaskSpec, agent: AgentRecord):
            raise TimeoutError("supervisor total timeout exceeded")

        events = EventLogger()
        scheduler = SubtaskDAGScheduler(executor, events=events)
        await scheduler.schedule("t1", [spec()])

        # timeout 无退避重试，第 2 次同因抑制 → REPLAN（provider 禁替换）
        assert scheduler.replacement_requests == []
        assert len(scheduler.decomposition_issues) == 1
        reason = scheduler.decomposition_issues[0].failure_reasons[-1]
        assert reason.startswith("provider 类故障禁止替换（timeout）")

        decided = events.get_events(event_type=EVENT_RECOVERY_DECIDED)
        assert [r["payload"]["action"] for r in decided] == ["retry", "replan"]
        assert all(r["payload"]["failure_type"] == "timeout" for r in decided)

    async def test_network_backoff_sleep_injected_deterministic(self):
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(round(seconds, 4))

        # 真实网络故障来自控制层异常（任务 69 信任边界）：异常类名含
        # connection → NETWORK（高信任），退避后重试；Agent 文本自称
        # connection reset 不可信
        async def executor(task_spec: SubtaskSpec, agent: AgentRecord):
            raise ConnectionError("connection reset by peer")

        events = EventLogger()
        scheduler = SubtaskDAGScheduler(
            executor,
            events=events,
            sleeper=fake_sleep,
            jitter=lambda: 0.5,
        )
        await scheduler.schedule("t1", [spec()])

        # base 1.0 × 2^0 × (1 + 0.2 × 0.5) = 1.1
        assert slept == [pytest.approx(1.1)]
        decided = events.get_events(event_type=EVENT_RECOVERY_DECIDED)
        assert decided[0]["payload"]["backoff_seconds"] == pytest.approx(1.1)
        assert decided[0]["payload"]["failure_type"] == "network"
        assert scheduler.replacement_requests == []
        assert len(scheduler.decomposition_issues) == 1


class TestTerminalFailures:
    """不可重试类与契约类：终止路径与提示注入。"""

    async def test_exception_internal_stops_after_single_attempt(self):
        async def executor(task_spec: SubtaskSpec, agent: AgentRecord):
            raise ValueError("boom")

        events = EventLogger()
        scheduler = SubtaskDAGScheduler(executor, events=events)
        results = await scheduler.schedule("t1", [spec()])

        assert len(events.get_events(event_type=EVENT_FAILURE_CLASSIFIED)) == 1
        classified = events.get_events(event_type=EVENT_FAILURE_CLASSIFIED)[0]
        assert classified["payload"]["failure_type"] == "internal"
        decided = events.get_events(event_type=EVENT_RECOVERY_DECIDED)[0]
        assert decided["payload"]["action"] == "stop"

        assert scheduler.replacement_requests == []
        assert len(scheduler.decomposition_issues) == 1
        reason = scheduler.decomposition_issues[0].failure_reasons[-1]
        assert reason.startswith("不可恢复类故障禁止替换（internal）")
        assert results["s1"].metadata["recovery_action"] == "stop"

    async def test_executor_contract_violation_pinned_internal(self):
        async def executor(task_spec: SubtaskSpec, agent: AgentRecord):
            return "not-an-execution-result"

        scheduler = SubtaskDAGScheduler(executor, events=EventLogger())
        results = await scheduler.schedule("t1", [spec()])

        assert results["s1"].success is False
        assert "执行器返回非法结果" in (results["s1"].error or "")
        assert results["s1"].metadata["failure_type"] == "internal"
        assert results["s1"].metadata["recovery_action"] == "stop"
        assert scheduler.replacement_requests == []

    async def test_format_failure_repair_then_replan_no_replacement(self):
        """format 失败（任务 69）：首次仅进入受限契约修复轮（不占业务
        attempt）；修复轮同一格式失败再现 → REPLAN，绝不创建
        Replacement Agent，也不得重跑完整业务任务冒充修复。"""
        recorder = _Recorder(
            lambda s, a: ExecutionResult(
                subtask_id=s.subtask_id,
                success=False,
                error="输出格式不符合契约",
            )
        )
        events = EventLogger()
        scheduler = SubtaskDAGScheduler(recorder.executor(), events=events)
        await scheduler.schedule("t1", [spec()])

        # 业务轮 1 次 + 受限修复轮 1 次：修复轮 attempt 序号不变（不占业务额度）
        assert [c[2] for c in recorder.calls] == [1, 1]

        # 第 1 次 REPAIR_CONTRACT（注入修复提示）→ 修复轮同因再现 → REPLAN
        decided = events.get_events(event_type=EVENT_RECOVERY_DECIDED)
        assert [r["payload"]["action"] for r in decided] == [
            "repair_contract",
            "replan",
        ]
        # 修复提示与业务重试提示严格分离：修复轮只注入受限修复指令
        assert "[契约修复]" not in recorder.instructions[0]
        assert "[契约修复]" in recorder.instructions[1]
        assert "不重新执行业务任务" in recorder.instructions[1]
        assert "[重试提示]" not in recorder.instructions[1]
        # dispatch 事件审计标注契约修复轮（新增字段，事件结构向后兼容）
        dispatched = events.get_events(event_type=EVENT_TASK_DISPATCHED)
        assert [r["payload"]["contract_repair"] for r in dispatched] == [
            False,
            True,
        ]
        # 同一格式失败再现 → 转重规划：绝不创建 Replacement Agent
        assert scheduler.replacement_requests == []
        assert len(scheduler.decomposition_issues) == 1
        reason = scheduler.decomposition_issues[0].failure_reasons[-1]
        assert reason.startswith("输出契约失败禁止替换（format）")
        # 失败摘要明确标注修复轮（可审计“修复≠业务重试”）
        summary = scheduler.decomposition_issues[0].failure_reasons[-2]
        assert "（契约修复轮）" in summary

    async def test_format_repair_round_survives_attempt_budget(self):
        """修复轮不占业务 attempt（任务 69 第 2 条）：max_attempts=1 时
        业务额度已尽，受限修复轮仍可执行（契约判定先于尝试上限），
        且修复轮内 attempt 序号保持不变。"""
        recorder = _Recorder(
            lambda s, a: ExecutionResult(
                subtask_id=s.subtask_id,
                success=False,
                error="JSON 输出不符合 schema 契约",
            )
        )
        scheduler = SubtaskDAGScheduler(recorder.executor(), events=EventLogger())
        await scheduler.schedule("t1", [spec("s1", max_attempts=1)])

        # 业务轮 1 次（额度尽）+ 修复轮 1 次（独立额度，attempt 仍为 1）
        assert [c[2] for c in recorder.calls] == [1, 1]
        assert scheduler.replacement_requests == []

    async def test_format_repair_success_returns_without_business_retry(self):
        """修复轮修好即成功返回（任务 69）：不消耗额外业务 attempt。"""

        def reply(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            if len(recorder.calls) == 1:  # 业务轮：契约失败
                return ExecutionResult(
                    subtask_id=task_spec.subtask_id,
                    success=False,
                    error="JSON 输出不符合 schema 契约",
                )
            # 修复轮：仅修正格式即成功（不重跑完整业务任务）
            return ExecutionResult(
                subtask_id=task_spec.subtask_id,
                success=True,
                output="修正后的合法 JSON 产出（符合契约结构）",
            )

        recorder = _Recorder(reply)
        events = EventLogger()
        scheduler = SubtaskDAGScheduler(recorder.executor(), events=events)
        results = await scheduler.schedule("t1", [spec("s1", max_attempts=1)])

        assert results["s1"].success is True
        # 1 次业务尝试 + 1 次受限修复轮，修复轮不占业务 attempt
        assert [c[2] for c in recorder.calls] == [1, 1]
        dispatched = events.get_events(event_type=EVENT_TASK_DISPATCHED)
        assert [r["payload"]["contract_repair"] for r in dispatched] == [
            False,
            True,
        ]


class TestTrustBoundaryAndMetadata:
    """任务 69：调度器级来源信任边界与控制层元数据覆盖。

    Agent 返回的自由文本（ExecutionResult.error）是低信任证据：自称
    429 / permission denied / budget exhausted / internal server error
    等系统失败关键词不得改变恢复方向——不享受 provider 退避、不触发
    不可重试类提前 STOP，一律按语义失败 QUALITY 路径处理；对照：
    同样语义来自控制层异常对象（异常类名 / HTTP 状态码）时仍能正确
    判定系统级类型。控制层判定的 failure_type / recovery_action
    无条件覆盖执行器或 LLM 自述的同名字段。
    """

    @pytest.mark.parametrize(
        "malicious_text",
        [
            "429 too many requests，请稍后重试",
            "permission denied：无法访问目标资源",
            "budget exhausted / insufficient quota",
            "internal server error (500)",
        ],
    )
    async def test_malicious_agent_text_treated_as_quality(self, malicious_text):
        recorder = _Recorder(
            lambda s, a: ExecutionResult(
                subtask_id=s.subtask_id,
                success=False,
                error=malicious_text,
            )
        )
        events = EventLogger()
        scheduler = SubtaskDAGScheduler(recorder.executor(), events=events)
        await scheduler.schedule("t1", [spec()])

        # 分类全部落入低信任 QUALITY：不因关键词伪装成系统级失败
        classified = events.get_events(event_type=EVENT_FAILURE_CLASSIFIED)
        assert classified
        assert all(r["payload"]["failure_type"] == "quality" for r in classified)
        # 恢复方向不被改变：伪装 429 不享受 provider 退避；
        # 伪装 permission / budget / internal 不触发不可重试类 STOP
        decided = events.get_events(event_type=EVENT_RECOVERY_DECIDED)
        assert decided[0]["payload"]["action"] == "retry"
        assert decided[0]["payload"]["backoff_seconds"] == 0.0
        assert all(r["payload"]["action"] != "stop" for r in decided)

    async def test_real_structured_exception_keeps_system_classification(self):
        """对照：携带 status_code 的控制层异常仍正确判定 rate_limit
        并享受退避重试（真实系统失败不受信任边界误伤）。"""
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(round(seconds, 4))

        class _RateLimited(Exception):
            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord):
            raise _RateLimited("too many requests", 429)

        events = EventLogger()
        scheduler = SubtaskDAGScheduler(
            executor,
            events=events,
            sleeper=fake_sleep,
            jitter=lambda: 0.5,
        )
        await scheduler.schedule("t1", [spec()])

        classified = events.get_events(event_type=EVENT_FAILURE_CLASSIFIED)
        assert classified[0]["payload"]["failure_type"] == "rate_limit"
        decided = events.get_events(event_type=EVENT_RECOVERY_DECIDED)
        assert decided[0]["payload"]["action"] == "retry"
        assert decided[0]["payload"]["backoff_seconds"] == pytest.approx(1.1)
        assert slept == [pytest.approx(1.1)]

    async def test_metadata_overwrites_executor_self_report(self):
        """控制层权威判定无条件覆盖执行器 / LLM 自述的 failure_type 与
        recovery_action（任务 69 第 4 条）：自述字段仅作审计对照。"""

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord):
            return ExecutionResult(
                subtask_id=task_spec.subtask_id,
                success=False,
                error="output 格式不符合契约",
                metadata={
                    "failure_type": "rate_limit",  # 自述伪装（不可信）
                    "recovery_action": "stop",      # 自述恢复动作（无效）
                },
            )

        events = EventLogger()
        scheduler = SubtaskDAGScheduler(executor, events=events)
        results = await scheduler.schedule("t1", [spec()])

        # 控制层判定覆盖自述：format / replan（而非 rate_limit / stop）
        assert results["s1"].metadata["failure_type"] == "format"
        assert results["s1"].metadata["recovery_action"] == "replan"

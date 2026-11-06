"""单元测试：确定性并行度决策内核与调度器接入。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import AgentRecord, ExecutionResult, SubtaskSpec, TaskRequest
from so_agent.runtime.events import EVENT_RUNTIME_BREAKER_TRIPPED, EventLogger
from so_agent.runtime.parallelism import (
    MODE_FAN_OUT,
    MODE_LIMITED,
    MODE_PAIR,
    MODE_SERIAL,
    RuntimeBreaker,
    collapse_ready,
    compile_plan_shape,
    decide_parallelism,
    record_breaker_result,
)
from so_agent.runtime.scheduler import SubtaskDAGScheduler
from so_agent.runtime.usage import log_model_usage
from so_agent.workflow import WorkflowEngine


def spec(
    subtask_id: str,
    deps: list[str] | None = None,
    tools: list[str] | None = None,
    scope: list[str] | None = None,
) -> SubtaskSpec:
    return SubtaskSpec(
        subtask_id=subtask_id,
        title=subtask_id,
        instructions="do",
        role="analyst",
        dependencies=deps or [],
        allowed_tools=tools or [],
        write_scope=scope or [],
    )


async def _noop_sleep(_seconds: float) -> None:
    return None


class TestPlanShape:
    def test_linear_chain_has_width_one(self):
        specs = [spec("a"), spec("b", ["a"]), spec("c", ["b"])]
        shape = compile_plan_shape(specs)
        assert shape.valid
        assert shape.width == 1
        assert shape.depth == 3
        assert shape.parallel_fraction == 0.0

    def test_wide_plan_layers(self):
        specs = [
            spec("a"),
            spec("b"),
            spec("c"),
            spec("d", ["a", "b", "c"]),
        ]
        shape = compile_plan_shape(specs)
        assert shape.valid
        assert shape.width == 3
        assert shape.depth == 2
        assert shape.level_sizes == (3, 1)

    def test_cycle_marks_invalid(self):
        specs = [spec("a", ["b"]), spec("b", ["a"])]
        shape = compile_plan_shape(specs)
        assert not shape.valid
        assert any("环" in issue for issue in shape.issues)

    def test_missing_dependency_reported(self):
        shape = compile_plan_shape([spec("a", ["ghost"])])
        assert not shape.valid
        assert any("不存在" in issue for issue in shape.issues)

    def test_duplicate_ids_reported(self):
        shape = compile_plan_shape([spec("a"), spec("a")])
        assert not shape.valid
        assert any("重复" in issue for issue in shape.issues)

    def test_write_conflict_detected_between_parallel_writers(self):
        specs = [spec("a", tools=["write_file"]), spec("b", tools=["run_in_sandbox"])]
        shape = compile_plan_shape(specs)
        assert shape.write_conflicts
        assert set(shape.conflict_subtask_ids) == {"a", "b"}

    def test_read_only_parallel_has_no_conflict(self):
        specs = [spec("a", tools=["read_file"]), spec("b", tools=[])]
        shape = compile_plan_shape(specs)
        assert not shape.write_conflicts


class TestWriteScopeConflict:
    """写范围（write_scope）判定：双方都声明且互不重叠才允许并行。"""

    def test_disjoint_scopes_allow_parallel(self):
        specs = [
            spec("a", tools=["write_file"], scope=["out/a.csv"]),
            spec("b", tools=["write_file"], scope=["out/b.csv"]),
        ]
        shape = compile_plan_shape(specs)
        assert not shape.write_conflicts
        decision = decide_parallelism(specs, max_concurrency=3)
        assert decision.mode == MODE_PAIR
        assert decision.concurrency == 2

    def test_identical_path_conflicts(self):
        specs = [
            spec("a", tools=["write_file"], scope=["out/report.md"]),
            spec("b", tools=["write_file"], scope=["./out/report.md"]),
        ]
        shape = compile_plan_shape(specs)
        assert shape.write_conflicts
        assert set(shape.conflict_subtask_ids) == {"a", "b"}

    def test_directory_prefix_conflicts(self):
        specs = [
            spec("a", tools=["write_file"], scope=["out"]),
            spec("b", tools=["write_file"], scope=["out/report.md"]),
        ]
        assert compile_plan_shape(specs).write_conflicts

    def test_sibling_directories_do_not_conflict(self):
        specs = [
            spec("a", tools=["write_file"], scope=["out/alpha/x.csv"]),
            spec("b", tools=["write_file"], scope=["out/beta/y.csv"]),
        ]
        assert not compile_plan_shape(specs).write_conflicts

    def test_wildcard_overlap_conflicts(self):
        specs = [
            spec("a", tools=["write_file"], scope=["out/*.csv"]),
            spec("b", tools=["write_file"], scope=["out/report.csv"]),
        ]
        assert compile_plan_shape(specs).write_conflicts

    def test_wildcard_non_match_does_not_conflict(self):
        specs = [
            spec("a", tools=["write_file"], scope=["out/*.csv"]),
            spec("b", tools=["write_file"], scope=["out/report.md"]),
        ]
        assert not compile_plan_shape(specs).write_conflicts

    def test_undeclared_scope_conflicts_with_declared(self):
        specs = [
            spec("a", tools=["write_file"], scope=["out/a.csv"]),
            spec("b", tools=["write_file"]),
        ]
        shape = compile_plan_shape(specs)
        assert shape.write_conflicts
        assert "未声明" in shape.write_conflicts[0]

    def test_declared_scope_marks_read_only_tools_as_writer(self):
        """write_scope 优先于工具白名单：只读工具 + 同路径声明仍按写冲突处理。"""
        specs = [
            spec("a", tools=["read_file"], scope=["out/a.csv"]),
            spec("b", tools=["read_file"], scope=["out/a.csv"]),
        ]
        shape = compile_plan_shape(specs)
        assert shape.write_conflicts
        assert set(shape.conflict_subtask_ids) == {"a", "b"}

    def test_three_disjoint_writers_fan_out(self):
        specs = [
            spec(f"s{i}", tools=["write_file"], scope=[f"out/s{i}.md"])
            for i in range(3)
        ]
        decision = decide_parallelism(specs, max_concurrency=3)
        assert decision.mode == MODE_FAN_OUT
        assert decision.concurrency == 3
        assert decision.gates["g2_independence"]


class TestDecideParallelism:
    def test_three_independent_read_only_is_fan_out(self):
        specs = [spec(f"s{i}", tools=["read_file"]) for i in range(3)]
        decision = decide_parallelism(specs, max_concurrency=3)
        assert decision.mode == MODE_FAN_OUT
        assert decision.concurrency == 3
        assert decision.gates["g1_shape"] and decision.gates["g2_independence"]

    def test_three_independent_writers_become_serial(self):
        specs = [spec(f"s{i}", tools=["write_file"]) for i in range(3)]
        decision = decide_parallelism(specs, max_concurrency=3)
        assert decision.mode == MODE_SERIAL
        assert decision.concurrency == 1
        assert not decision.gates["g2_independence"]

    def test_linear_plan_serial(self):
        decision = decide_parallelism(
            [spec("a"), spec("b", ["a"]), spec("c", ["b"])], max_concurrency=3
        )
        assert decision.mode == MODE_SERIAL
        assert decision.concurrency == 1

    def test_pair_keeps_two_way_parallel(self):
        specs = [spec("a", tools=["read_file"]), spec("b", tools=["read_file"])]
        decision = decide_parallelism(specs, max_concurrency=3)
        assert decision.mode == MODE_PAIR
        assert decision.concurrency == 2

    def test_low_gain_wide_plan_limited(self):
        specs = [spec(f"w{i}") for i in range(3)]
        previous = "w0"
        for index in range(9):
            node = f"x{index}"
            specs.append(spec(node, [previous]))
            previous = node
        decision = decide_parallelism(specs, max_concurrency=3)
        assert decision.mode == MODE_LIMITED
        assert decision.concurrency == 2
        assert not decision.gates["g3_gain"]

    def test_concurrency_respects_cap(self):
        specs = [spec(f"s{i}", tools=["read_file"]) for i in range(5)]
        decision = decide_parallelism(specs, max_concurrency=2)
        assert decision.mode == MODE_FAN_OUT
        assert decision.concurrency == 2

    def test_invalid_plan_serial(self):
        decision = decide_parallelism([spec("a", ["b"]), spec("b", ["a"])])
        assert decision.mode == MODE_SERIAL
        assert not decision.gates["g1_shape"]


class TestObjectiveThresholds:
    """时间优先与 token 优先的门槛差异（任务 103）。"""

    @staticmethod
    def _moderately_parallel_specs() -> list[SubtaskSpec]:
        """层宽 3、并行占比 0.6：time_first 扇出，token_first 低并发。"""
        specs = [spec(f"w{i}", tools=["read_file"]) for i in range(3)]
        specs.append(spec("c0", ["w0", "w1", "w2"], tools=["read_file"]))
        specs.append(spec("c1", ["c0"], tools=["read_file"]))
        return specs

    def test_time_first_fans_out_where_token_first_limits(self):
        specs = self._moderately_parallel_specs()
        time_first = decide_parallelism(specs, objective="time_first")
        token_first = decide_parallelism(specs, objective="token_first")

        assert time_first.mode == MODE_FAN_OUT
        assert time_first.concurrency == 3
        assert token_first.mode == MODE_LIMITED
        assert token_first.concurrency == 2
        assert not token_first.gates["g3_gain"]
        assert token_first.gates["g3_speedup"]

    def test_time_first_rejects_tiny_gain_pair(self):
        specs = [
            spec("a", tools=["read_file"]),
            spec("b", tools=["read_file"]),
        ]
        previous = ["a", "b"]
        for index in range(20):
            node = f"c{index}"
            specs.append(spec(node, list(previous), tools=["read_file"]))
            previous = [node]

        time_first = decide_parallelism(specs, objective="time_first")
        token_first = decide_parallelism(specs, objective="token_first")

        assert time_first.mode == MODE_SERIAL
        assert not time_first.speedup_ok
        assert "时间收益" in time_first.reason
        assert token_first.mode == MODE_PAIR

    def test_min_speedup_override_allows_pair(self):
        specs = [
            spec("a", tools=["read_file"]),
            spec("b", tools=["read_file"]),
        ]
        previous = ["a", "b"]
        for index in range(20):
            node = f"c{index}"
            specs.append(spec(node, list(previous), tools=["read_file"]))
            previous = [node]

        decision = decide_parallelism(
            specs, objective="time_first", min_speedup=1.0
        )
        assert decision.mode == MODE_PAIR
        assert decision.gates["g3_speedup"]

    def test_fan_out_still_requires_independence_under_time_first(self):
        specs = [spec(f"w{i}", tools=["write_file"]) for i in range(3)]
        decision = decide_parallelism(specs, objective="time_first")
        assert decision.mode == MODE_SERIAL
        assert not decision.gates["g2_independence"]

    def test_payload_records_objective_and_thresholds(self):
        specs = self._moderately_parallel_specs()
        payload = decide_parallelism(
            specs, objective="token_first"
        ).as_event_payload()
        assert payload["objective"] == "token_first"
        assert payload["net_gain_threshold"] == 1.3
        assert payload["min_speedup"] == 1.0


class TestRuntimeBreaker:
    def test_provider_failures_trip(self):
        breaker = RuntimeBreaker(provider_failure_limit=2)
        assert breaker.record(success=False, failure_type="network") is False
        assert breaker.record(success=False, failure_type="timeout") is True
        assert "provider" in breaker.trip_reason

    def test_consecutive_failures_trip(self):
        breaker = RuntimeBreaker(consecutive_failure_limit=3)
        breaker.record(success=False, failure_type="quality")
        breaker.record(success=False, failure_type="quality")
        assert not breaker.tripped
        assert breaker.record(success=False, failure_type="quality")

    def test_success_breaks_consecutive_streak(self):
        breaker = RuntimeBreaker(consecutive_failure_limit=2)
        breaker.record(success=False, failure_type="quality")
        breaker.record(success=True)
        assert not breaker.record(success=False, failure_type="quality")

    def test_reset_clears_state(self):
        breaker = RuntimeBreaker(provider_failure_limit=1)
        breaker.record(success=False, failure_type="network")
        breaker.reset()
        assert not breaker.tripped

    def test_collapse_ready(self):
        breaker = RuntimeBreaker(consecutive_failure_limit=1)
        assert collapse_ready(["a", "b"], {}, breaker) == ["a", "b"]
        breaker.record(success=False)
        assert collapse_ready(["a", "b"], {}, breaker) == ["a"]
        assert collapse_ready(["a", "b"], {"t": "x"}, breaker) == []

    def test_record_breaker_result_only_true_on_transition(self):
        breaker = RuntimeBreaker(provider_failure_limit=1)
        result = ExecutionResult(
            subtask_id="a",
            success=False,
            metadata={"failure_type": "network"},
        )
        assert record_breaker_result(breaker, result) is True
        assert record_breaker_result(breaker, result) is False


class TestPlanningWiring:
    def test_decision_recorded_on_state_and_events(self, tmp_path):
        engine = WorkflowEngine(
            ProjectContext(sandbox_dir=tmp_path, project_name="parallel"),
            settings=Settings(_env_file=None),
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
        )
        engine._task_request = TaskRequest(task_id="t1", objective="目标")
        engine._subtasks = [spec(f"s{i}", tools=["read_file"]) for i in range(3)]

        decision = engine._decide_parallelism()

        assert decision.mode == MODE_FAN_OUT
        assert engine._parallelism_decision is decision
        assert isinstance(engine._runtime_breaker, RuntimeBreaker)
        records = engine._events.get_events(event_type="parallelism_decided")
        assert len(records) == 1
        assert records[0]["payload"]["mode"] == MODE_FAN_OUT
        assert records[0]["payload"]["concurrency"] == 3
        assert records[0]["payload"]["objective"] == "time_first"
        assert records[0]["payload"]["net_gain_threshold"] == 1.1
        assert records[0]["payload"]["min_speedup"] == 1.1


class TestNestedUsageAttribution:
    def test_log_model_usage_aggregates_responses(self):
        usage = SimpleNamespace(input_tokens=10, output_tokens=5, total_tokens=15)
        result = SimpleNamespace(
            raw_responses=[SimpleNamespace(usage=usage), SimpleNamespace(usage=usage)]
        )
        events = EventLogger()
        log_model_usage(events, result, "code_agent", subtask_id="s1")
        record = events.events[-1]
        assert record["event_type"] == "token_usage"
        assert record["payload"]["agent"] == "code_agent"
        assert record["payload"]["requests"] == 2
        assert record["payload"]["total_tokens"] == 30
        assert record["payload"]["subtask_id"] == "s1"


class TestSchedulerWiring:
    async def test_serial_concurrency_executes_one_at_a_time(self):
        active = 0
        peak = 0

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True, output="ok")

        scheduler = SubtaskDAGScheduler(executor)
        specs = [spec(f"s{i}", tools=["read_file"]) for i in range(3)]
        results = await scheduler.schedule("t1", specs, concurrency=1)
        assert all(item.success for item in results.values())
        assert peak == 1

    async def test_pre_tripped_breaker_forces_serial(self):
        active = 0
        peak = 0

        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return ExecutionResult(subtask_id=task_spec.subtask_id, success=True, output="ok")

        breaker = RuntimeBreaker(provider_failure_limit=1)
        breaker.record(success=False, failure_type="network")
        scheduler = SubtaskDAGScheduler(executor, sleeper=_noop_sleep, jitter=lambda: 0.0)
        specs = [spec(f"s{i}", tools=["read_file"]) for i in range(4)]
        results = await scheduler.schedule(
            "t1", specs, concurrency=4, breaker=breaker
        )
        assert all(item.success for item in results.values())
        assert peak == 1

    async def test_breaker_tripped_event_logged_once(self):
        async def executor(task_spec: SubtaskSpec, agent: AgentRecord) -> ExecutionResult:
            await asyncio.sleep(0.01)
            return ExecutionResult(
                subtask_id=task_spec.subtask_id,
                success=False,
                error="boom",
            )

        events = EventLogger()
        breaker = RuntimeBreaker(consecutive_failure_limit=1)
        scheduler = SubtaskDAGScheduler(
            executor, events=events, sleeper=_noop_sleep, jitter=lambda: 0.0
        )
        specs = [spec(f"s{i}", tools=["read_file"]) for i in range(3)]
        await scheduler.schedule("t1", specs, concurrency=3, breaker=breaker)
        assert breaker.tripped
        tripped = events.get_events(event_type=EVENT_RUNTIME_BREAKER_TRIPPED)
        assert len(tripped) == 1
        assert tripped[0]["payload"]["action"] == "collapse_to_serial"

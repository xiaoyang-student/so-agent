"""模型输入裁剪器：旧调用参数 / 旧工具输出回执化与预算丢弃。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agents.run_config import CallModelData, ModelInputData

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.runtime.context_pruner import (
    compact_model_input,
    prune_model_input,
)
from so_agent.tool_packages.code_agent.agent import _make_context_prune_filter


def _args(**kwargs) -> str:
    return json.dumps(kwargs, ensure_ascii=False)


def _call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def _output(call_id: str, output: str) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def _exec_output(stdout: str, *, exit_code: int = 0) -> str:
    return json.dumps(
        {
            "status": "ok",
            "execution_result": {
                "success": exit_code == 0,
                "output": stdout,
                "metadata": {"exit_code": exit_code, "run_dir": "runs/r1"},
            },
        },
        ensure_ascii=False,
    )


class TestArgumentPruning:
    def test_old_code_argument_is_replaced(self):
        items = [
            {"role": "user", "content": "写脚本"},
            _call("c1", "write_file", _args(path="a.py", content="x" * 2000)),
            _output("c1", json.dumps({"status": "ok", "path": "a.py"})),
            _call("c2", "write_file", _args(path="b.py", content="y" * 2000)),
            _output("c2", json.dumps({"status": "ok", "path": "b.py"})),
        ]

        pruned, stats = prune_model_input(items, keep_last_outputs=2)

        assert json.loads(pruned[1]["arguments"])["content"].startswith("[已省略")
        assert "a.py" in json.loads(pruned[1]["arguments"])["content"]
        assert stats.pruned_calls == 1
        # 最近一次调用参数必须原样保留
        assert pruned[3]["arguments"] == items[3]["arguments"]

    def test_small_arguments_are_kept(self):
        items = [
            _call("c1", "read_file", _args(path="a.py", start_line=3)),
        ]

        pruned, stats = prune_model_input(items)

        assert pruned[0]["arguments"] == items[0]["arguments"]
        assert stats.pruned_calls == 0


class TestOutputPruning:
    def test_old_exec_output_becomes_receipt(self):
        big_stdout = "line\n" * 5000
        items = [
            _call("c1", "run_in_sandbox", _args(code="print('x')", task_id="t1")),
            _output("c1", _exec_output(big_stdout)),
            _call("c2", "run_in_sandbox", _args(code="print('y')", task_id="t2")),
            _output("c2", _exec_output("fresh")),
            _call("c3", "read_file", _args(path="a.py")),
            _output("c3", json.dumps({"status": "ok", "content": "z"})),
            _call("c4", "read_file", _args(path="b.py")),
            _output("c4", json.dumps({"status": "ok", "content": "w"})),
        ]

        pruned, stats = prune_model_input(items, keep_last_outputs=2)

        # 最近两条输出与最近一次执行输出保持原文；更早的执行输出变回执
        assert pruned[5]["output"] == items[5]["output"]
        assert pruned[7]["output"] == items[7]["output"]
        assert pruned[3]["output"] == items[3]["output"]
        receipt = json.loads(pruned[1]["output"])
        assert receipt["success"] is True
        assert receipt["exit_code"] == 0
        assert receipt["run_dir"] == "runs/r1"
        assert "line" in receipt["output_tail"]
        assert len(pruned[1]["output"]) < len(items[1]["output"])
        assert stats.pruned_outputs == 1

    def test_error_output_keeps_error_text(self):
        items = [
            _call("c1", "run_in_sandbox", _args(code="boom")),
            _output(
                "c1",
                json.dumps({"status": "error", "error": "SyntaxError: bad"}),
            ),
            _call("c2", "run_in_sandbox", _args(code="print(1)")),
            _output("c2", _exec_output("1")),
            _call("c3", "read_file", _args(path="a.py")),
            _output("c3", json.dumps({"status": "ok", "content": "z"})),
            _call("c4", "read_file", _args(path="b.py")),
            _output("c4", json.dumps({"status": "ok", "content": "w"})),
        ]

        pruned, _ = prune_model_input(items, keep_last_outputs=2)

        receipt = json.loads(pruned[1]["output"])
        assert receipt["status"] == "error"
        assert "SyntaxError" in receipt["error"]

    def test_latest_exec_output_is_protected(self):
        # 执行输出即使较早也保留原文：下游验收需要真实 stdout
        items = [
            _call("c1", "run_in_sandbox", _args(code="print('keep')")),
            _output("c1", _exec_output("keep-me")),
            _call("c2", "read_file", _args(path="a.py")),
            _output("c2", json.dumps({"status": "ok", "content": "z"})),
            _call("c3", "read_file", _args(path="b.py")),
            _output("c3", json.dumps({"status": "ok", "content": "w"})),
        ]

        pruned, _ = prune_model_input(items, keep_last_outputs=1)

        assert pruned[1]["output"] == items[1]["output"]


class TestBudget:
    def test_over_budget_drops_pairs_without_orphans(self):
        items = [{"role": "user", "content": "task"}]
        for index in range(4):
            call_id = f"c{index}"
            items.append(_call(call_id, "read_file", _args(path=f"{index}.py")))
            items.append(_output(call_id, json.dumps({"status": "ok", "content": "a" * 2000})))

        pruned, stats = prune_model_input(items, max_chars=1000, keep_last_outputs=1)

        calls = {
            item["call_id"] for item in pruned if item.get("type") == "function_call"
        }
        outputs = {
            item["call_id"]
            for item in pruned
            if item.get("type") == "function_call_output"
        }
        assert calls == outputs
        assert stats.dropped_items > 0
        assert pruned[0] == items[0]

    def test_over_budget_keeps_reasoning_items(self):
        # 思考模式要求回传 reasoning_content：裁剪只能丢消息，不能丢推理项
        items = [
            {"type": "message", "role": "user", "content": "task"},
            {"type": "reasoning", "content": "r" * 2000},
            *[
                {"type": "message", "role": "assistant", "content": "x" * 2000}
                for _ in range(4)
            ],
            _call("c1", "read_file", _args(path="a.py")),
            _output("c1", json.dumps({"status": "ok", "content": "z"})),
        ]

        pruned, stats = prune_model_input(items, max_chars=500, keep_last_outputs=1)

        assert any(item.get("type") == "reasoning" for item in pruned)
        assert stats.dropped_items > 0

    def test_under_budget_keeps_every_item(self):
        items = [
            _call("c1", "read_file", _args(path="a.py")),
            _output("c1", json.dumps({"status": "ok", "content": "small"})),
        ]

        pruned, stats = prune_model_input(items, max_chars=100000)

        assert len(pruned) == len(items)
        assert stats.dropped_items == 0


class TestRobustness:
    def test_input_is_not_mutated(self):
        items = [
            _call("c1", "write_file", _args(path="a.py", content="x" * 2000)),
            _output("c1", _exec_output("y" * 5000)),
            _call("c2", "read_file", _args(path="b.py")),
            _output("c2", json.dumps({"status": "ok"})),
        ]
        snapshot = json.dumps(items, ensure_ascii=False)

        prune_model_input(items, max_chars=100, keep_last_outputs=1)

        assert json.dumps(items, ensure_ascii=False) == snapshot

    def test_unknown_and_non_dict_items_are_preserved(self):
        marker = object()
        items = [
            marker,
            {"type": "message", "role": "user", "content": "hi"},
            {"type": "future_kind", "payload": "x" * 5000},
        ]

        pruned, _ = prune_model_input(items, max_chars=10)

        assert pruned[0] is marker
        assert pruned[1] == items[1]
        assert pruned[2] == items[2]


class _FakeEventLogger:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.events: list[tuple[str, dict]] = []

    def log_event(self, event_type, payload=None):
        if self.fail:
            raise RuntimeError("audit down")
        self.events.append((event_type, payload or {}))


def _settings(tmp_path: Path, **overrides) -> Settings:
    return Settings(
        _env_file=None,
        tool_store_dir=str(tmp_path / "tools"),
        **overrides,
    )


def _call_data(items: list) -> CallModelData:
    return CallModelData(
        model_data=ModelInputData(input=items, instructions="sys"),
        agent=None,
        context=None,
    )


class TestBoundaryCompaction:
    @staticmethod
    def _task() -> dict:
        return {"role": "user", "content": "任务载荷"}

    @staticmethod
    def _pair(index: int, *, size: int = 800, name: str = "run_in_sandbox"):
        return (
            _call(f"c{index}", name, _args(path=f"f{index}.py")),
            _output(f"c{index}", _exec_output("x" * size)),
        )

    def test_append_only_until_high_water_mark(self):
        states: dict = {}
        items = [self._task()]
        for index in range(3):
            items.extend(self._pair(index, size=20))
        first, stats = compact_model_input(items, states, max_chars=10000)
        assert stats.compacted is False
        assert first == items

        items.extend(self._pair(3, size=20))
        second, stats = compact_model_input(items, states, max_chars=10000)
        assert stats.compacted is False
        assert second[: len(first)] == first

    def test_boundary_compaction_is_stable_after_folding(self):
        states: dict = {}
        items = [self._task()]
        for index in range(6):
            items.extend(self._pair(index, size=800))
        first, stats = compact_model_input(
            items, states, max_chars=4000, keep_last_outputs=1
        )
        assert stats.compacted is True
        assert stats.compactions == 1
        assert stats.pruned_outputs >= 1
        assert first[1]["role"] == "user"
        assert first[1]["content"].startswith("[上下文压缩摘要")
        assert stats.after_chars < stats.before_chars

        items.extend(self._pair(6, size=20))
        second, stats = compact_model_input(
            items, states, max_chars=4000, keep_last_outputs=1
        )
        assert stats.compacted is False
        assert second[: len(first)] == first

    def test_hysteresis_prevents_recompaction_every_turn(self):
        states: dict = {}
        items = [self._task()]
        for index in range(4):
            items.extend(self._pair(index, size=800))
        first, stats = compact_model_input(
            items, states, max_chars=1200, keep_last_outputs=1
        )
        assert stats.compactions == 1
        assert stats.after_chars > 1200  # 回执仍占篇幅，折叠后仍在阈值之上

        items.append(_call("r", "read_file", _args(path="r.py")))
        items.append(
            _output("r", json.dumps({"status": "ok", "content": "r"}))
        )
        second, stats = compact_model_input(
            items, states, max_chars=1200, keep_last_outputs=1
        )
        assert stats.compacted is False
        assert stats.compactions == 1
        assert second[: len(first)] == first

        for index in range(4, 6):
            items.extend(self._pair(index, size=800))
        _, stats = compact_model_input(
            items, states, max_chars=1200, keep_last_outputs=1
        )
        assert stats.compacted is True
        assert stats.compactions == 2

    def test_old_exec_output_does_not_pin_boundary(self):
        # 只有读取发生在执行之后时，早期执行输出不能被永久保护，
        # 否则边界停在原地、尾部无限增长。
        states: dict = {}
        items = [self._task()]
        items.extend(self._pair(0, size=3000))
        for index in range(3):
            items.append(
                _call(f"r{index}", "read_file", _args(path=f"{index}.py"))
            )
            items.append(
                _output(
                    f"r{index}",
                    json.dumps({"status": "ok", "content": "r" * 800}),
                )
            )

        compacted, stats = compact_model_input(
            items, states, max_chars=2500, keep_last_outputs=1
        )

        assert stats.compacted is True
        assert stats.pruned_outputs == 1
        outputs = {
            item.get("call_id"): item
            for item in compacted
            if item.get("type") == "function_call_output"
        }
        if "c0" in outputs:
            assert "output_tail" in json.loads(outputs["c0"]["output"])
        _assert_tool_messages_paired(compacted)

    def test_tail_growth_triggers_second_boundary(self):
        states: dict = {}
        items = [self._task()]
        for index in range(4):
            items.extend(self._pair(index, size=800))
        _, stats = compact_model_input(
            items, states, max_chars=1200, keep_last_outputs=1
        )
        assert stats.compactions == 1

        for index in range(4, 10):
            items.extend(self._pair(index, size=800))
        _, stats = compact_model_input(
            items, states, max_chars=1200, keep_last_outputs=1
        )
        assert stats.compacted is True
        assert stats.compactions == 2

    def test_read_outputs_stay_raw_and_latest_exec_is_protected(self):
        states: dict = {}
        items = [self._task()]
        items.extend(self._pair(0, size=6000))
        for index in range(3):
            items.append(_call(f"r{index}", "read_file", _args(path=f"{index}.py")))
            items.append(
                _output(
                    f"r{index}",
                    json.dumps({"status": "ok", "content": "r" * 900}),
                )
            )
        items.extend(self._pair(9, size=100))
        items.append(_call("r9", "read_file", _args(path="last.py")))
        items.append(
            _output(
                "r9", json.dumps({"status": "ok", "content": "z" * 900})
            )
        )
        compacted, stats = compact_model_input(
            items, states, max_chars=9000, keep_last_outputs=1
        )
        assert stats.compacted is True
        outputs = {
            item.get("call_id"): item
            for item in compacted
            if item.get("type") == "function_call_output"
        }
        # 读取结果原文保留（回执化会诱发反复回读）
        for index in range(3):
            assert json.loads(outputs[f"r{index}"]["output"])["content"] == "r" * 900
        # 最近一次执行输出保留原文，更早的执行输出已回执
        assert json.loads(outputs["c9"]["output"])["execution_result"]["output"] == "x" * 100
        assert "output_tail" in json.loads(outputs["c0"]["output"])

    def test_changed_first_item_resets_state(self):
        states: dict = {}
        items = [self._task()]
        for index in range(4):
            items.extend(self._pair(index, size=800))
        compact_model_input(items, states, max_chars=1200, keep_last_outputs=1)

        other = [{"role": "user", "content": "另一个任务"}]
        for index in range(4):
            other.extend(self._pair(index, size=800))
        output, stats = compact_model_input(
            other, states, max_chars=1200, keep_last_outputs=1
        )
        assert output[0]["content"] == "另一个任务"
        assert stats.before_chars == stats.before_chars
        # 两个运行各自独立计数，不存在串用上一条运行的边界
        assert stats.compactions == 1

    def test_zero_max_chars_keeps_legacy_receipts(self):
        states: dict = {}
        items = [self._task()]
        for index in range(4):
            items.extend(self._pair(index, size=800))
        compacted, stats = compact_model_input(
            items, states, max_chars=0, keep_last_outputs=1
        )
        assert stats.compacted is False
        assert any(
            item.get("type") == "function_call_output"
            and len(item.get("output", "")) < 800
            for item in compacted
        )


def _assert_tool_messages_paired(items: list) -> None:
    """模型输入必须成对：孤立 tool 消息会让 Chat Completions 直接 400。"""
    from agents.models.chatcmpl_converter import Converter

    messages = Converter.items_to_messages(items)
    pending: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            pending = {
                call["id"]
                for call in (message.get("tool_calls") or [])
            }
        elif message.get("role") == "tool":
            assert message.get("tool_call_id") in pending, message


class TestParallelCallIntegrity:
    def test_call_with_tail_output_is_not_dropped(self):
        # 并行调用输出乱序：a 的调用在边界前、输出落在受保护的 tail，
        # 压缩不能只删调用，否则尾部输出成为孤立 tool 消息。
        states: dict = {}
        items = [
            {"role": "user", "content": "任务载荷"},
            _call("a", "read_file", _args(payload="p" * 4000)),
            _call("b", "read_file", _args(path="b.py")),
            _output("b", json.dumps({"status": "ok", "content": "b" * 500})),
            _output("a", json.dumps({"status": "ok", "content": "a" * 500})),
        ]

        compacted, stats = compact_model_input(
            items, states, max_chars=1500, keep_last_outputs=1
        )

        assert stats.compacted is True
        calls = {
            item.get("call_id")
            for item in compacted
            if item.get("type") == "function_call"
        }
        outputs = {
            item.get("call_id")
            for item in compacted
            if item.get("type") == "function_call_output"
        }
        assert calls == outputs
        _assert_tool_messages_paired(compacted)

    def test_orphan_output_is_removed(self):
        # 防御性兜底：原始历史已缺配对时也不能把孤立输出送进模型。
        states: dict = {}
        items = [
            {"role": "user", "content": "任务载荷"},
            _call("a", "read_file", _args(path="a.py")),
            _output("a", json.dumps({"status": "ok", "content": "a" * 500})),
            _output("ghost", json.dumps({"status": "ok", "content": "g"})),
        ]

        compacted, stats = compact_model_input(
            items, states, max_chars=400, keep_last_outputs=1
        )

        assert stats.compacted is True
        assert all(
            item.get("call_id") != "ghost"
            for item in compacted
        )
        _assert_tool_messages_paired(compacted)


class TestRunConfigFilter:
    async def test_filter_returns_pruned_model_input(self, tmp_path):
        logger = _FakeEventLogger()
        context = ProjectContext(
            sandbox_dir=tmp_path, project_name="prune", event_logger=logger
        )
        prune_filter = _make_context_prune_filter(
            context,
            _settings(
                tmp_path,
                code_agent_context_keep_tool_outputs=1,
                code_agent_context_max_chars=500,
            ),
        )
        items = [
            _call("c1", "write_file", _args(path="a.py", content="x" * 2000)),
            _output("c1", json.dumps({"status": "ok", "path": "a.py"})),
            _call("c2", "read_file", _args(path="b.py")),
            _output("c2", json.dumps({"status": "ok", "content": "z"})),
        ]

        result = await prune_filter(_call_data(items))

        assert isinstance(result, ModelInputData)
        assert result.instructions == "sys"
        assert result.input[0]["arguments"] != items[0]["arguments"]
        assert logger.events
        event_type, payload = logger.events[0]
        assert event_type == "context_pruned"
        assert payload["saved_chars"] > 0
        assert "content" not in payload

    async def test_filter_logs_nothing_when_no_prune(self, tmp_path):
        logger = _FakeEventLogger()
        context = ProjectContext(
            sandbox_dir=tmp_path, project_name="prune", event_logger=logger
        )
        prune_filter = _make_context_prune_filter(context, _settings(tmp_path))
        items = [_call("c1", "read_file", _args(path="a.py"))]

        result = await prune_filter(_call_data(items))

        assert result.input == items
        assert logger.events == []

    async def test_audit_failure_does_not_break_filter(self, tmp_path):
        context = ProjectContext(
            sandbox_dir=tmp_path,
            project_name="prune",
            event_logger=_FakeEventLogger(fail=True),
        )
        prune_filter = _make_context_prune_filter(
            context,
            _settings(
                tmp_path,
                code_agent_context_keep_tool_outputs=1,
                code_agent_context_max_chars=500,
            ),
        )
        items = [
            _call("c1", "write_file", _args(path="a.py", content="x" * 2000)),
            _output("c1", json.dumps({"status": "ok"})),
            _call("c2", "read_file", _args(path="b.py")),
            _output("c2", json.dumps({"status": "ok"})),
        ]

        result = await prune_filter(_call_data(items))

        assert isinstance(result, ModelInputData)
        assert result.input[0]["arguments"].startswith('{"path": "a.py", "content": "[已省略')

"""单元测试：统一工具/Agent 响应信封 ToolResponse（B 组）。

覆盖三块：
1. ToolResponse 模型本身（创建、默认值、序列化、必填校验）；
2. ExecutionResult 继承 ToolResponse 的向后兼容性；
3. 沙箱执行结果落在 ToolResponse 信封上的字段映射（含 metadata）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_agent.config import Settings
from so_agent.models import ExecutionResult, SubtaskSpec, ToolResponse
from so_agent.runtime.sandbox import SandboxRunner


class TestToolResponseModel:
    """ToolResponse 基础契约。"""

    def test_minimal_construction_defaults(self):
        resp = ToolResponse(success=True)
        assert resp.success is True
        assert resp.output == ""
        assert resp.error is None
        assert resp.artifacts == []
        assert resp.metadata == {}

    def test_success_is_required(self):
        with pytest.raises(ValueError):
            ToolResponse()  # type: ignore[call-arg]

    def test_failure_construction(self):
        resp = ToolResponse(success=False, error="boom")
        assert resp.success is False
        assert resp.error == "boom"
        assert resp.output == ""

    def test_defaults_are_independent_between_instances(self):
        a = ToolResponse(success=True)
        b = ToolResponse(success=False)
        a.artifacts.append("/tmp/x")
        a.metadata["exit_code"] = 1
        assert b.artifacts == []
        assert b.metadata == {}

    def test_full_roundtrip_json(self):
        resp = ToolResponse(
            success=False,
            output="o",
            error="boom",
            artifacts=["/tmp/a.txt"],
            metadata={"exit_code": 3, "duration": 1.5},
        )
        restored = ToolResponse.model_validate_json(resp.model_dump_json())
        assert restored == resp

    def test_output_is_string_channel(self):
        """结构化结果须序列化为字符串放入 output（信封统一约定）。"""
        resp = ToolResponse(
            success=True, output=json.dumps({"k": 1}, ensure_ascii=False)
        )
        payload = json.loads(resp.model_dump_json())
        assert isinstance(payload["output"], str)
        assert json.loads(payload["output"]) == {"k": 1}

    def test_envelope_keys_in_serialization(self):
        payload = json.loads(ToolResponse(success=True).model_dump_json())
        assert set(payload) == {"success", "output", "error", "artifacts", "metadata"}


class TestExecutionResultInheritance:
    """ExecutionResult 继承 ToolResponse 的兼容性。"""

    def test_issubclass_and_isinstance(self):
        assert issubclass(ExecutionResult, ToolResponse)
        result = ExecutionResult(subtask_id="s1", success=True)
        assert isinstance(result, ToolResponse)

    def test_legacy_field_construction_unchanged(self):
        """既有字段（含默认值）与旧构造方式全部保持不变。"""
        result = ExecutionResult(
            subtask_id="s1",
            success=False,
            output="o",
            evidence=["e"],
            artifacts=["/tmp/x"],
            error="boom",
            duration=1.5,
        )
        assert result.subtask_id == "s1"
        assert result.success is False
        assert result.output == "o"
        assert result.evidence == ["e"]
        assert result.artifacts == ["/tmp/x"]
        assert result.error == "boom"
        assert result.duration == 1.5
        assert result.metadata == {}  # 继承新增字段的默认值

    def test_legacy_json_without_metadata_still_parses(self):
        """旧快照（无 metadata 键）仍可解析，metadata 回退为空字典。"""
        legacy = '{"subtask_id": "s1", "success": true, "output": "done"}'
        result = ExecutionResult.model_validate_json(legacy)
        assert result.subtask_id == "s1"
        assert result.metadata == {}

    def test_metadata_roundtrip_preserved(self):
        result = ExecutionResult(
            subtask_id="s1",
            success=True,
            metadata={"exit_code": 0, "token_usage": 42},
        )
        restored = ExecutionResult.model_validate_json(result.model_dump_json())
        assert restored == result
        assert restored.metadata["token_usage"] == 42

    def test_consumable_as_tool_response(self):
        """以 ToolResponse 语义消费 ExecutionResult：无需区分具体类型。"""

        def summarize(resp: ToolResponse) -> str:
            return resp.output if resp.success else (resp.error or "")

        result = ExecutionResult(subtask_id="s1", success=False, error="boom")
        assert summarize(result) == "boom"


class TestSandboxToolResponseMapping:
    """沙箱返回落在 ToolResponse 信封上的映射正确性。"""

    @staticmethod
    def make_runner(tmp_path: Path, *, timeout: float = 30.0) -> SandboxRunner:
        return SandboxRunner(
            sandbox_dir=tmp_path,
            settings=Settings(sandbox_timeout=timeout, sandbox_output_limit=1_048_576),
        )

    async def test_success_mapping_and_metadata(self, tmp_path):
        runner = self.make_runner(tmp_path)
        result = await runner.execute("print('ok')", "t1", "run1")
        assert isinstance(result, ToolResponse)
        assert result.success is True
        assert result.error is None
        assert "ok" in result.output  # stdout → output
        assert result.metadata["exit_code"] == 0
        assert result.metadata["timed_out"] is False
        assert result.metadata["duration_seconds"] >= 0.0

    async def test_failure_mapping_stderr_to_error(self, tmp_path):
        code = (
            "import sys\n"
            "print('partial-stdout')\n"
            "print('boom-message', file=sys.stderr)\n"
            "sys.exit(3)\n"
        )
        runner = self.make_runner(tmp_path)
        result = await runner.execute(code, "t1", "run1")
        assert result.success is False
        assert result.error is not None  # stderr 摘要 → error
        assert "boom-message" in result.error
        assert "partial-stdout" in result.output  # stdout → output（始终保留）
        assert result.metadata["exit_code"] == 3

    async def test_timeout_recorded_in_metadata(self, tmp_path):
        runner = self.make_runner(tmp_path, timeout=0.8)
        result = await runner.execute("import time\ntime.sleep(60)\n", "t1", "run1")
        assert result.success is False
        assert "执行超时" in (result.error or "")
        assert result.metadata["timed_out"] is True

    async def test_serialized_envelope_keys(self, tmp_path):
        """沙箱结果的 JSON 序列化包含完整信封字段。"""
        runner = self.make_runner(tmp_path)
        result = await runner.execute("print('ok')", "t1", "run1")
        payload = json.loads(result.model_dump_json())
        for key in (
            "success",
            "output",
            "error",
            "artifacts",
            "metadata",
            "subtask_id",
            "evidence",
            "duration",
        ):
            assert key in payload


class TestDynamicInstructionsContract:
    """B3：动态子 Agent 提示词与 ToolResponse 字段名对齐。"""

    def test_output_template_aligns_with_tool_response(self):
        from so_agent.tool_packages.subagent_creator.agent import (
            _compose_dynamic_instructions,
        )

        spec = SubtaskSpec(
            subtask_id="s1", title="演示", instructions="做一件事", role="code"
        )
        text = _compose_dynamic_instructions(spec, ["read_file"])
        assert "ToolResponse 信封" in text
        for field in (
            "subtask_id",
            "success",
            "output",
            "evidence",
            "artifacts",
            "error",
            "metadata",
        ):
            assert f'"{field}"' in text

    def test_failure_semantics_documented(self):
        from so_agent.tool_packages.subagent_creator.agent import (
            _compose_dynamic_instructions,
        )

        spec = SubtaskSpec(
            subtask_id="s2", title="演示", instructions="做一件事", role="code"
        )
        text = _compose_dynamic_instructions(spec, [])
        assert "success 设为 false" in text
        assert "（无工具授权）" in text

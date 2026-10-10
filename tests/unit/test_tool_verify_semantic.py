"""单元测试：沙箱验收的语义比对（``_verify_generated_tool`` / ``_semantic_matches``）。

锁定语义比对策略（相比精确 JSON 字符串比对 ``json.dumps(sort_keys)`` 全等放宽）：
- 数值类型按 ``math.isclose`` 以 1e-6 容差比较（bool 与数值显式区分，
  避免 ``1 == True`` 的跨型误判）；
- 字符串去除首尾空白后比较；
- dict 以期望值为键集基准：实际值多出的额外字段忽略，递归匹配；
- list 长度一致并逐元素递归匹配；
- 期望值确实不符（错误答案）仍然拒绝。
"""

from __future__ import annotations

from types import SimpleNamespace

import so_agent.tool_packages.code_agent.agent as code_agent_agent_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.execution_coordinator import _SEMANTIC_TOLERANCE, _semantic_matches
from so_agent.models import GeneratedToolManifest
from so_agent.workflow import WorkflowEngine


class TestSemanticMatches:
    """``_semantic_matches`` 的纯逻辑单元测试。"""

    def test_tolerance_constant_is_1e_6(self):
        assert _SEMANTIC_TOLERANCE == 1e-6

    def test_float_within_tolerance_passes(self):
        assert _semantic_matches(6.0000005, 6.0) is True
        assert _semantic_matches(6.0, 6.0000005) is True

    def test_float_out_of_tolerance_fails(self):
        assert _semantic_matches(6.001, 6.0) is False

    def test_int_float_equivalence(self):
        assert _semantic_matches(6, 6.0) is True
        assert _semantic_matches(6.0, 6) is True

    def test_bool_is_not_number(self):
        assert _semantic_matches(True, 1) is False
        assert _semantic_matches(1, True) is False
        assert _semantic_matches(True, True) is True
        assert _semantic_matches(False, False) is True

    def test_string_stripped(self):
        assert _semantic_matches("  hello  ", "hello") is True
        assert _semantic_matches("hello", " hello ") is True

    def test_string_mismatch_fails(self):
        assert _semantic_matches("hello", "world") is False
        # 类型不同不匹配（数字 vs 字符串）
        assert _semantic_matches(123, "123") is False

    def test_none(self):
        assert _semantic_matches(None, None) is True
        assert _semantic_matches(None, 0) is False

    def test_dict_extra_fields_ignored(self):
        actual = {"double": 6.0, "extra": "noise", "meta": {"k": 1}}
        assert _semantic_matches(actual, {"double": 6.0}) is True

    def test_dict_missing_key_fails(self):
        assert _semantic_matches({"other": 1}, {"double": 6.0}) is False

    def test_dict_recursive_match(self):
        actual = {"outer": {"value": 0.5000001}, "items": [1.0, " a ", None]}
        expected = {"outer": {"value": 0.5}, "items": [1, "a", None]}
        assert _semantic_matches(actual, expected) is True

    def test_list_length_mismatch_fails(self):
        assert _semantic_matches([1, 2], [1]) is False
        assert _semantic_matches([1], [1, 2]) is False

    def test_dict_vs_non_dict_fails(self):
        assert _semantic_matches([1], {"k": 1}) is False
        assert _semantic_matches("text", {"k": 1}) is False


class TestVerifyGeneratedToolSemantic:
    """沙箱验收端到端：容差/额外字段/空白差异不再误判为验收失败。"""

    def _engine(self, tmp_path):
        context = ProjectContext(sandbox_dir=tmp_path, project_name="verify-semantic")
        engine = WorkflowEngine(
            context,
            settings=Settings(_env_file=None),
            orchestrator=SimpleNamespace(name="fake-supervisor"),
            review_agent=SimpleNamespace(name="fake-reviewer"),
        )
        engine._task_request = SimpleNamespace(task_id="t1")
        return engine

    def _manifest_with(self, gen_dir, expected, source):
        gen_dir.mkdir(exist_ok=True)
        (gen_dir / "demo_tool.py").write_text(source, encoding="utf-8")
        return GeneratedToolManifest(
            tool_id="t",
            tool_name="demo_tool",
            entry_file="demo_tool.py",
            created_by_task="task",
            test_cases=[{"x": 3}],
            expected_outputs=expected,
        )

    async def test_float_tolerance_passes(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path)
        gen_dir = tmp_path / "gen"
        manifest = self._manifest_with(
            gen_dir,
            [{"double": 6.0000004}],
            "def run(x):\n    return {'double': x * 2.0}\n",
        )
        monkeypatch.setattr(code_agent_agent_mod, "GENERATED_TOOLS_DIR", gen_dir)
        verified, err = await engine._verify_generated_tool(manifest)
        assert err is None
        assert verified.verified is True

    async def test_extra_field_and_string_strip_pass(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path)
        gen_dir = tmp_path / "gen"
        manifest = self._manifest_with(
            gen_dir,
            [{"msg": "hello", "n": 6}],
            (
                "def run(x):\n"
                "    return {'msg': '  hello  ', 'n': 6, 'extra': 'noise'}\n"
            ),
        )
        monkeypatch.setattr(code_agent_agent_mod, "GENERATED_TOOLS_DIR", gen_dir)
        verified, err = await engine._verify_generated_tool(manifest)
        assert err is None
        assert verified.verified is True

    async def test_wrong_answer_still_fails(self, tmp_path, monkeypatch):
        engine = self._engine(tmp_path)
        gen_dir = tmp_path / "gen"
        manifest = self._manifest_with(
            gen_dir,
            [{"double": 999}],
            "def run(x):\n    return {'double': x * 2}\n",
        )
        monkeypatch.setattr(code_agent_agent_mod, "GENERATED_TOOLS_DIR", gen_dir)
        verified, err = await engine._verify_generated_tool(manifest)
        assert err is not None
        assert "不符" in err
        assert verified.verified is False

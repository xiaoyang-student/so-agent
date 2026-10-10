"""单元测试：SubtaskSpec 指纹计算（跨计划版本结果复用的匹配键）。

验证 spec_fingerprint 的两个核心契约：
- 确定性：同一规格恒定产出同一指纹（16 位十六进制）；
- 语义敏感性：影响执行结果的字段（instructions / role /
  allowed_tools / expected_output / dependencies）任一变化都改变指纹；
  元数据字段（subtask_id / plan_version / max_attempts / title）变化
  不影响指纹，且列表字段与顺序无关。
"""

from __future__ import annotations

import pytest

from so_agent.fingerprint import spec_fingerprint
from so_agent.models import SubtaskSpec


def make_spec(**overrides) -> SubtaskSpec:
    """构造基准 SubtaskSpec，可按字段覆盖。"""
    base = dict(
        subtask_id="s1",
        plan_version=1,
        title="统计数字",
        instructions="统计 1..100 的偶数之和",
        role="code",
        dependencies=[],
        allowed_tools=["write_file", "run_code"],
        expected_output="一个整数",
        max_attempts=3,
    )
    base.update(overrides)
    return SubtaskSpec(**base)


class TestFingerprintDeterminism:
    """确定性契约。"""

    def test_same_spec_same_fingerprint(self) -> None:
        assert spec_fingerprint(make_spec()) == spec_fingerprint(make_spec())

    def test_fingerprint_is_16_hex_chars(self) -> None:
        fp = spec_fingerprint(make_spec())
        assert len(fp) == 16
        int(fp, 16)  # 合法十六进制，否则抛 ValueError

    def test_empty_collections_and_defaults(self) -> None:
        """默认空列表与显式空列表指纹一致。"""
        explicit = make_spec(dependencies=[], allowed_tools=[])
        minimal = SubtaskSpec(
            subtask_id="s1",
            title="统计数字",
            instructions="统计 1..100 的偶数之和",
            role="code",
            expected_output="一个整数",
        )
        assert spec_fingerprint(explicit) == spec_fingerprint(minimal)


class TestFingerprintSensitivity:
    """语义字段敏感性：任一字段变化 → 指纹变化。"""

    def test_instructions_change(self) -> None:
        assert spec_fingerprint(make_spec()) != spec_fingerprint(
            make_spec(instructions="统计 1..100 的奇数之和")
        )

    def test_role_change(self) -> None:
        assert spec_fingerprint(make_spec()) != spec_fingerprint(
            make_spec(role="review")
        )

    def test_allowed_tools_change(self) -> None:
        assert spec_fingerprint(make_spec()) != spec_fingerprint(
            make_spec(allowed_tools=["write_file"])
        )

    def test_expected_output_change(self) -> None:
        assert spec_fingerprint(make_spec()) != spec_fingerprint(
            make_spec(expected_output="两个整数")
        )

    def test_dependencies_change(self) -> None:
        assert spec_fingerprint(make_spec()) != spec_fingerprint(
            make_spec(dependencies=["s0"])
        )


class TestFingerprintInsensitivity:
    """元数据不敏感性：元数据变化 → 指纹不变。"""

    def test_subtask_id_change(self) -> None:
        assert spec_fingerprint(make_spec()) == spec_fingerprint(
            make_spec(subtask_id="s999")
        )

    def test_max_attempts_change(self) -> None:
        assert spec_fingerprint(make_spec()) == spec_fingerprint(
            make_spec(max_attempts=5)
        )

    def test_plan_version_change(self) -> None:
        assert spec_fingerprint(make_spec()) == spec_fingerprint(
            make_spec(plan_version=2)
        )

    def test_title_change(self) -> None:
        assert spec_fingerprint(make_spec()) == spec_fingerprint(
            make_spec(title="另一个标题")
        )


class TestFingerprintOrderInsensitivity:
    """列表字段与顺序无关（排序后参与哈希）。"""

    def test_allowed_tools_order(self) -> None:
        assert spec_fingerprint(make_spec(allowed_tools=["a", "b"])) == \
            spec_fingerprint(make_spec(allowed_tools=["b", "a"]))

    def test_dependencies_order(self) -> None:
        assert spec_fingerprint(make_spec(dependencies=["s0", "s2"])) == \
            spec_fingerprint(make_spec(dependencies=["s2", "s0"]))


@pytest.mark.parametrize("field", ["subtask_id", "plan_version", "max_attempts", "title"])
def test_metadata_fields_not_in_fingerprint(field: str) -> None:
    """参数化回归：全部元数据字段统一断言不参与指纹。"""
    values = {
        "subtask_id": "s2",
        "plan_version": 9,
        "max_attempts": 7,
        "title": "变更标题",
    }
    assert spec_fingerprint(make_spec()) == spec_fingerprint(make_spec(**{field: values[field]}))

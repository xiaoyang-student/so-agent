"""工具意图升级的确定性判定：禁止性指令不得触发升级。"""

from __future__ import annotations

import pytest

from so_agent.router import _needs_tool_execution, _tool_intent_evidence


@pytest.mark.parametrize(
    ("objective", "acceptance", "expected"),
    [
        (
            "用两点说明 HTTP/1.1 与 HTTP/2 的主要差异，并从适用场景给出选择建议。"
            "不要写入任何文件，直接在回答中给出简短说明。",
            "",
            False,
        ),
        ("解释 Go 的 goroutine 如何执行", "", False),
        ("不要生成任何文件，直接回答", "", False),
        ("", "不要创建任何文件，直接回答", False),
        ("Do not write any file; answer directly", "", False),
        ("创建 web-smoke.txt，内容写 WEB_OK", "", True),
        ("不要修改原文件，在新目录生成报告", "", True),
        ("运行测试并修复失败用例", "", True),
        ("读取仓库结构并生成分析报告", "", True),
        ("", "报告文件必须存在且包含结论", True),
    ],
)
def test_tool_intent_detection(objective, acceptance, expected):
    assert _needs_tool_execution(objective, acceptance) is expected
    if expected:
        assert _tool_intent_evidence(objective, acceptance)
    else:
        assert _tool_intent_evidence(objective, acceptance) == ""

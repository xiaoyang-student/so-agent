"""子 Agent 输入最小化：动态提示词与代码 Agent 输入契约隔离。"""

from __future__ import annotations

from so_agent.models import SubtaskSpec
from so_agent.tool_packages.code_agent.agent import CodeAgentInput
from so_agent.tool_packages.subagent_creator.agent import (
    _compose_dynamic_instructions,
)


def _specs() -> tuple[SubtaskSpec, SubtaskSpec]:
    return (
        SubtaskSpec(
            subtask_id="s1",
            title="采集原始数据",
            instructions="只处理数据采集，不要生成报告",
            role="code",
            allowed_tools=["read_file"],
            expected_output="结构化数据文件",
        ),
        SubtaskSpec(
            subtask_id="s2",
            title="生成最终报告",
            instructions="只处理报告生成，不要重复采集数据",
            role="code",
            dependencies=["s1"],
            allowed_tools=["write_file"],
        ),
    )


def test_dynamic_subagent_prompt_contains_only_own_spec():
    own, sibling = _specs()

    prompt = _compose_dynamic_instructions(own, ["read_file"])

    assert own.subtask_id in prompt
    assert own.title in prompt
    assert own.instructions in prompt
    assert own.expected_output in prompt
    assert "read_file" in prompt
    assert sibling.subtask_id not in prompt
    assert sibling.title not in prompt
    assert sibling.instructions not in prompt
    assert "write_file" not in prompt


def test_code_agent_input_contract_is_minimal():
    # write_scope 是控制层下发的写时约束，不接受其他上下文注入
    assert set(CodeAgentInput.model_fields) == {
        "task_description",
        "code_content",
        "file_path",
        "action",
        "write_scope",
    }

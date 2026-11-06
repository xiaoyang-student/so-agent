"""子任务执行器路由（确定性判据）。

LLM 负责拆解与规格，但"由哪个执行器执行"由代码判定：

- 子任务声明需要的工具全部落在标准内置工具内（或未声明工具）时，
  由 ``code_agent`` 执行——它本身已具备这些工具，继续创建动态子
  Agent 只会额外支付一次创建器与嵌套 Agent 的模型开销；
- 只有声明了非标准工具（专用领域工具 / 经审批的动态生成工具）时，
  才交给 ``subagent_creator`` 动态创建子 Agent。

本模块是纯函数，不依赖业务状态，供执行协调器与子 Agent 创建器共用。
"""

from __future__ import annotations

from typing import Any, Iterable

STANDARD_BUILTIN_TOOLS: frozenset[str] = frozenset(
    {"read_file", "write_file", "run_in_sandbox"}
)

EXECUTOR_CODE_AGENT: str = "code_agent"
EXECUTOR_SUBAGENT_CREATOR: str = "subagent_creator"

REASON_STANDARD_TOOLS_ONLY: str = "standard_tools_only"
REASON_NON_STANDARD_TOOLS: str = "non_standard_tools"


def _normalized_names(allowed_tools: Iterable[Any] | None) -> list[str]:
    return [
        str(name).strip()
        for name in (allowed_tools or [])
        if str(name).strip()
    ]


def is_standard_only(allowed_tools: Iterable[Any] | None) -> bool:
    """声明工具是否全部为标准内置工具（未声明工具也视为标准）。"""
    return all(
        name in STANDARD_BUILTIN_TOOLS for name in _normalized_names(allowed_tools)
    )


def resolve_executor(allowed_tools: Iterable[Any] | None) -> dict[str, str]:
    """返回确定性路由结论（tool / reason），供主管执行期遵循与审计。"""
    if is_standard_only(allowed_tools):
        return {
            "tool": EXECUTOR_CODE_AGENT,
            "reason": REASON_STANDARD_TOOLS_ONLY,
        }
    return {
        "tool": EXECUTOR_SUBAGENT_CREATOR,
        "reason": REASON_NON_STANDARD_TOOLS,
    }

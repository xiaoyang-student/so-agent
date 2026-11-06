"""代码 Agent 构建模块（code_agent 工具包）。

职责：作为主 Agent 的固定工具入口之一，构造“代码执行 Agent（Code Agent）”：

- 读取本目录 ``prompt.md`` 作为系统提示词（instructions）；
- 通过 ``runtime.config_loader`` 加载本目录 ``api_config.yaml``，按其中
  ``api_key`` 直接值（或旧 ``api_key_env`` 环境变量回退）构造 OpenAI 兼容
  聊天模型（密钥值绝不写日志）；
- 绑定五个内部工具：``write_file`` / ``write_and_run`` / ``read_file`` /
  ``run_in_sandbox`` / ``write_generated_tool``，所有工具返回值统一为
  JSON 字符串；
- Agent 输出为纯文本 JSON（``output_type=None``，不依赖 SDK 结构化输出：
  千问等 OpenAI 兼容接口会忽略 ``response_format=json_schema`` 并把
  JSON 包裹在 Markdown 代码围栏中返回，故由调用方剥离围栏后按
  ``CodeAgentOutput`` 手动校验）；
- 对外同时提供 MCP 暴露契约与 MCP Server（``get_mcp_schema`` /
  ``get_mcp_server``），使本工具可按 MCP 规范被主管 Agent 与 MCPClient
  统一发现与调用。

对外接口::

    agent = create_code_agent(context)         # 构造 SDK Agent 实例
    tool = get_code_agent_as_tool(context)     # 转为主 Agent 可用的工具入口
    schema = get_mcp_schema()                  # MCP 暴露契约（MCPToolSchema）
    server = get_mcp_server(context)           # 包装为 MCP Server（统一调用）

全局约束（最高优先级）：
- 所有 Agent 之间的通信必须严格使用 JSON：工具返回值与 Agent 最终输出
  一律为可解析的 JSON（Pydantic 序列化），不允许自由文本协议；
- 文件读写全部限制在 ``ProjectContext.sandbox_dir`` 之内，路径穿越一律拒绝；
- 生成工具只写入 ``generated_tools`` 目录并登记候选清单（含 mcp_schema.json
  MCP 暴露契约），未经评审与主管 Agent 注册前不可被调用。
"""

from __future__ import annotations

import contextvars
import difflib
import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    FunctionTool,
    Model,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    RunResult,
    RunResultStreaming,
    function_tool,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.integrity import sha256_file
from so_agent.json_utils import extract_json_object
from so_agent.mcp.adapter import agent_to_mcp_server, truncate_output_text
from so_agent.mcp.protocol import MCPToolSchema
from so_agent.mcp.server import MCPToolServer
from so_agent.models import ExecutionResult, GeneratedToolManifest
from so_agent.runtime.approval_gate import ApprovalDenied, build_approval_gate
from so_agent.runtime.config_loader import (
    AgentAPIConfig,
    load_agent_api_config,
    resolve_api_config_path,
    resolve_model_http_timeout,
)
from so_agent.runtime.artifact_index import build_artifact_index
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.model_gateway import build_chat_model as build_chat_model_with_gateway
from so_agent.runtime.context_pruner import CompactState, compact_model_input
from so_agent.runtime.events import (
    EVENT_ACTION_REPLAYED,
    EVENT_CODE_AGENT_POSTCONDITION_FAILED,
    EVENT_CONTEXT_PRUNED,
    EVENT_READ_COLLAPSED,
)
from so_agent.runtime.sandbox import SandboxBackend, SandboxError, create_sandbox
from so_agent.runtime.tool_registry import ToolRegistry, ToolRegistryError
from so_agent.runtime.tool_store import (
    resolve_generated_registry_path as _resolve_store_registry_path,
)
from so_agent.runtime.tool_store import (
    resolve_generated_tools_dir as _resolve_store_generated_dir,
)
from so_agent.runtime.usage import make_usage_recorder
from so_agent.runtime.write_scope import path_allowed

# ---------------------------------------------------------------------------
# 包内路径（用 pathlib 定位同目录下的提示词与 API 配置）
# ---------------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROMPT_PATH: Path = PACKAGE_DIR / "prompt.md"
API_CONFIG_PATH: Path = PACKAGE_DIR / "api_config.yaml"

# 生成工具目录的旧默认位置（包内）；运行时默认改写到项目沙箱内
# （``Settings.generated_tools_dir``），避免写入安装目录导致的权限与
# 升级丢失问题。两个模块常量保留，作为测试/嵌入方的显式覆盖点。
_PACKAGE_GENERATED_TOOLS_DIR: Path = PACKAGE_DIR.parent / "generated_tools"
GENERATED_TOOLS_DIR: Path = _PACKAGE_GENERATED_TOOLS_DIR
GENERATED_REGISTRY_PATH: Path = GENERATED_TOOLS_DIR / "registry.json"

# 生成工具名称规范：小写字母开头，仅字母/数字/下划线，最长 64 字符
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# 当前 MCP 调用的写入范围：工具在构建时绑定，范围只能按请求传递，
# 因此用 ContextVar 而非闭包变量，避免并发子任务互相覆盖。
_CURRENT_WRITE_SCOPE: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "code_agent_write_scope", default=()
)

# 同一 Agent 运行内的读取台账：路径 → {sha256, ranges}。重复读取未变更
# 文件的已读窗口时只返回回执；台账必须按运行重置，否则新运行会误报
# “与上次读取相同”而模型并未见过该内容。None 表示不启用（直接调用
# build_code_tools 的嵌入方需显式 reset_read_ledger()）。
_READ_LEDGER: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = (
    contextvars.ContextVar("code_agent_read_ledger", default=None)
)

# 同一运行内的执行台账：代码哈希 → 上次结果与工作区指纹。相同代码且
# 工作区未变化时返回回执而不重复产生副作用；写入去重直接比对文件内容，
# 不依赖此台账。与读取台账同生命周期，按运行重置。
_ACTION_LEDGER: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = (
    contextvars.ContextVar("code_agent_action_ledger", default=None)
)


def reset_read_ledger() -> None:
    """为一次新的 Agent 运行重置读取与执行台账。"""
    _READ_LEDGER.set({})
    _ACTION_LEDGER.set({})


def _action_ledger() -> dict[str, Any] | None:
    return _ACTION_LEDGER.get()


def _sandbox_state_fingerprint(root: Path, limit: int = 2000) -> str:
    """沙箱工作区指纹：排除执行器自身的 task_* 输出目录。

    重复执行同一段代码前用它判断工作区是否发生变化；``task_*`` 目录
    每次执行都会新增日志，纳入指纹会让判断恒为“已变化”。
    """
    digest = hashlib.sha256()
    count = 0
    try:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if relative.parts and relative.parts[0].startswith("task_"):
                continue
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            digest.update(
                f"{relative.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
            )
            count += 1
            if count >= limit:
                break
    except OSError:
        return ""
    return digest.hexdigest()


def _log_action_replayed(
    context: ProjectContext,
    tool: str,
    target: str,
    repeats: int,
) -> None:
    event_logger = getattr(context, "event_logger", None)
    if event_logger is None:
        return
    try:
        event_logger.log_event(
            EVENT_ACTION_REPLAYED,
            {
                "agent": AGENT_NAME,
                "tool": tool,
                "target": target,
                "repeats": repeats,
            },
        )
    except Exception:
        return


def _range_covered(
    ranges: list[tuple[int, int]], start: int, end: int
) -> bool:
    return any(left <= start and end <= right for left, right in ranges)


def _merge_range(
    ranges: list[tuple[int, int]], start: int, end: int
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for left, right in sorted([*ranges, (start, end)]):
        if merged and left <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def _log_read_collapsed(
    context: ProjectContext,
    path: str,
    window: dict[str, Any],
    saved_chars: int,
) -> None:
    event_logger = getattr(context, "event_logger", None)
    if event_logger is None or saved_chars <= 0:
        return
    try:
        event_logger.log_event(
            EVENT_READ_COLLAPSED,
            {
                "agent": AGENT_NAME,
                "path": path,
                "start_line": window.get("start_line"),
                "end_line": window.get("end_line"),
                "saved_chars": saved_chars,
            },
        )
    except Exception:  # 审计写入失败不得影响工具结果
        pass


# 单次 diff 输出的最大行数（超出截断，避免工具返回值过大）
_DIFF_MAX_LINES = 200

# Agent 名称（与工具注册表 / 路由约定保持一致）
AGENT_NAME = "code_agent"

logger = get_logger("code_agent")

# 主管侧调用说明（同时用于 as_tool 入口与 MCP 暴露契约）
_TOOL_DESCRIPTION = (
    "代码执行 Agent：承接编码与执行类子任务（创建/修改文件、在沙箱中运行代码、"
    "写入生成工具候选），返回 CodeAgentOutput 结构的 JSON 结果。"
)

# 追加在 prompt.md 之后的输出契约说明（不改动原始提示词文件）
_EXTRA_INSTRUCTIONS = """
## 结构化输入（主 Agent 下发的调用契约）

主 Agent 会以如下字段（CodeAgentInput）下发调用：

- `task_description`：本次子任务要完成的编码/执行目标（必填）；
- `code_content`：待创建或修改的代码内容（可选）；
- `file_path`：涉及的文件路径（相对项目沙箱目录，可选）；
- `action`：动作类型，取值 `create` / `modify` / `review` / `execute` /
  `auto`（`auto` 表示按任务自行组合所需动作，被直派调用时的缺省值）。

调用时还可能附带 `sandbox_artifact_index` 字段：它是沙箱根目录当前
相关文件的清单（仅文件名、大小与修改时间，不含内容），用于确认依赖
产物的真实名称；它不是任务要求，不需要对它做任何处理。

## 回执与历史

- `read_file` 默认一次返回整份文件；仅超大文件才返回行窗口（含
  `next_start`），此时才需要带 `start_line` 继续读取。
- 重复读取文件中已经读过的同一窗口时，工具只返回 `unchanged: true`
  回执而不重复正文；确需再看原始内容时传 `force=true`。
- `run_in_sandbox` / `write_and_run` 返回结构化执行回执：退出码、
  stdout、耗时与 `run_dir`；完整输出已落盘，不要重复粘贴到正文。
- 控制层会把较早轮次的代码参数与执行输出替换为回执以控制上下文。
  需要某份文件内容时用 `read_file` 重新读取一次，不要凭记忆重写
  整份文件，也不要因为历史被省略而重复执行已完成的动作。

## 最终输出（结构化强约束，覆盖上文任何较宽松的输出描述）

最终回复必须是 CodeAgentOutput 结构的纯 JSON 文本——不包裹 Markdown
代码围栏，不附加任何解释性文字，字段如下：

- `success`：bool，本次动作是否成功完成；
- `action_performed`：str，实际执行的动作描述（与 action 对应或说明为何未执行）；
- `file_path`：str 或 null，涉及的沙箱内相对路径；
- `code_diff`：str，代码变更的 unified diff 文本（无变更时为空字符串）；
- `execution_result`：ExecutionResult 结构或 null，当 action 为 execute 时
  内嵌 `run_in_sandbox` 工具返回的完整结果对象；
- `error`：str 或 null，失败原因（成功时为 null）；
- `summary`：str，完成情况摘要（只写关键结论与自检结果）；
- `evidence`：list[str]，关键证据，逐条写命令、输出摘要或产物路径；
  没有证据时给空数组（不要用 `null`）。

硬性纪律：

1. 文件读写在项目沙箱目录内进行，必须调用 `write_file` / `read_file` 工具；
2. 代码运行必须调用 `run_in_sandbox` 工具，禁止声称“已运行”而未真实调用；
   脚本的工作目录就是项目沙箱根目录，相对路径与 `write_file` / `read_file`
   一致，不要自行拼接 `task_*` 或 `run_*` 前缀；
2.1 写脚本后立刻执行同一份脚本时，优先一次调用 `write_and_run`（写入 +
   执行在同一轮完成）。不要先 `write_file` 再单独 `run_in_sandbox`：
   两次往返不会带来额外信息，只会放大上下文；
2.2 不要为“确认已写入 / 已执行 / 已完成”单独发起一轮无新信息的调用；
   需要确认就 `read_file` 或直接运行，并在同一轮内完成下一步；
3. 生成新工具候选必须调用 `write_generated_tool` 写入 generated_tools 目录，
   不得写入其他位置，也不得自行注册或授权；
4. 一切证据（命令、输出、路径）必须真实可复核，禁止编造；
5. `code_diff` 只写变更摘要（涉及路径、增删行数），不要回显完整代码或
   完整 diff——代码本身已经写入文件，正文与 diff 不要重复同一份内容；
6. `action=execute` 时 `summary` 只写退出码、结论与产物路径，不要粘贴
   完整 stdout/stderr 或数据行；
7. `action=auto` 时自行组合所需动作：声明了的产物路径必须真实存在且
   非空，做了执行就必须内嵌真实的 `execution_result`。
"""


def _budget_phrase(value: Any, unit: str) -> str:
    """把预算配置格式化为提示词短语（<=0 表示不限制）。"""
    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = 0
    return f"≤{limit}{unit}" if limit > 0 else "不限制"


def _output_budget_instructions(settings: Settings | None) -> str:
    """生成叙述性字段的输出预算段落（与运行时截断配置保持一致）。"""
    def _value(name: str, default: int) -> Any:
        return getattr(settings, name, default) if settings is not None else default

    summary = _budget_phrase(_value("code_agent_summary_max_chars", 400), " 字符")
    action = _budget_phrase(_value("code_agent_action_max_chars", 160), " 字符")
    item = _budget_phrase(
        _value("code_agent_evidence_item_max_chars", 240), " 字符"
    )
    items = _budget_phrase(_value("code_agent_evidence_max_items", 10), " 条")
    write_bytes = _budget_phrase(
        _value("code_agent_write_max_bytes", 16384), " 字节"
    )
    return f"""
## 输出预算（超出部分由控制层截断；写入超限会被工具拒绝）

- `action_performed` {action}；
- `summary` {summary}；
- `evidence` 最多 {items}，每条 {item}；
- `execution_result` 原样保留 `run_in_sandbox` 的结果对象，不要附加日志；
- 单次 `write_file` / `write_and_run` 的 `content` {write_bytes}：超限时
  先写第一部分，再用 `append=true` 追加剩余内容，最后执行。

## 脚本复用

输入中出现 `reusable_scripts` 时，表示控制层已把此前验证通过、哈希一致的
脚本复制进沙箱。先 `read_file` 核对其输入格式与本次数据一致；一致就直接
`run_in_sandbox`（或 `write_and_run` 只做必要修改），不要重写整份脚本。
不一致或不适用时忽略它，按任务重新生成。
"""


class CodeAction(str, Enum):
    """代码 Agent 可执行的动作类型。"""

    CREATE = "create"    # 创建新文件/新代码
    MODIFY = "modify"    # 修改既有文件/代码
    REVIEW = "review"    # 静态审阅（只读核对，不落盘）
    EXECUTE = "execute"  # 在沙箱中执行代码并收集结果
    AUTO = "auto"        # 由 Agent 按任务自行组合动作（直派路线的缺省值）


class CodeAgentInput(BaseModel):
    """代码 Agent 的输入契约（主 Agent 下发）。"""

    task_description: str = Field(description="本次子任务的编码/执行目标")
    code_content: str | None = Field(default=None, description="待创建或修改的代码内容")
    file_path: str | None = Field(default=None, description="涉及的文件路径（相对项目沙箱目录）")
    action: CodeAction = Field(
        description="动作类型：create/modify/review/execute/auto"
    )
    write_scope: list[str] = Field(
        default_factory=list,
        description=(
            "控制层下发的写入范围（相对项目沙箱目录）；非空时 write_file / "
            "write_and_run 只能写入其中的路径"
        ),
    )


class CodeAgentOutput(BaseModel):
    """代码 Agent 的输出契约（结构化 JSON，禁止自由文本）。"""

    success: bool = Field(description="本次动作是否成功完成")
    action_performed: str = Field(description="实际执行的动作描述")
    file_path: str | None = Field(default=None, description="涉及的文件路径（沙箱内相对路径）")
    code_diff: str = Field(default="", description="代码变更的 unified diff 文本；无变更时为空字符串")
    execution_result: ExecutionResult | None = Field(
        default=None, description="沙箱执行结果；action=execute 时内嵌 run_in_sandbox 的返回值"
    )
    error: str | None = Field(default=None, description="失败原因；成功时为 None")
    summary: str = Field(default="", description="完成情况摘要")
    evidence: list[str] = Field(
        default_factory=list,
        description="关键证据（命令/输出摘要/产物路径），每条一行",
    )


# ---------------------------------------------------------------------------
# 通用小工具（JSON 序列化 / 上下文解析 / 路径防护）
# ---------------------------------------------------------------------------
def _json(payload: Any) -> str:
    """将载荷序列化为 JSON 字符串（工具返回值的统一格式）。"""
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    return json.dumps(payload, ensure_ascii=False, default=str)


def _ok(**fields: Any) -> str:
    """构造成功形态的 JSON 工具返回值。"""
    return _json({"status": "ok", **fields})


def _error(message: str, **fields: Any) -> str:
    """构造失败形态的 JSON 工具返回值。"""
    return _json({"status": "error", "error": message, **fields})


def _resolve_context(context: ProjectContext | None) -> ProjectContext:
    """返回生效的 ProjectContext；未提供时创建默认实例（cwd/sandbox）。"""
    if context is not None:
        return context
    return ProjectContext(sandbox_dir=Path.cwd() / "sandbox", project_name="default")


def _resolve_settings(context: ProjectContext, settings: Settings | None) -> Settings:
    """返回生效配置：显式参数 > context.config > 全局配置。"""
    if settings is not None:
        return settings
    return context.config if context.config is not None else get_settings()


def _sandbox_root(context: ProjectContext) -> Path:
    """返回项目沙箱根目录（绝对路径、已规范化）。"""
    return Path(context.sandbox_dir).expanduser().resolve()


def resolve_generated_tools_dir(
    context: ProjectContext | None = None,
    settings: Settings | None = None,
) -> Path:
    """解析生成工具目录。

    优先级：模块常量被显式覆盖（测试/嵌入方 monkeypatch）> 旧配置项
    ``generated_tools_dir`` 被显式设置（相对沙箱根目录，兼容旧部署）>
    工具库 ``{tool_store_dir}/generated``（与沙箱生命周期解耦）。
    """
    if GENERATED_TOOLS_DIR != _PACKAGE_GENERATED_TOOLS_DIR:
        return GENERATED_TOOLS_DIR
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    configured = str(resolved_settings.generated_tools_dir or "").strip()
    if configured and configured != "generated_tools":
        path = Path(configured).expanduser()
        return path if path.is_absolute() else Path(resolved_context.sandbox_dir) / path
    return _resolve_store_generated_dir(resolved_settings)


def resolve_generated_registry_path(
    context: ProjectContext | None = None,
    settings: Settings | None = None,
) -> Path:
    """解析生成工具清单路径（与工具目录同源，除非被显式覆盖）。"""
    if GENERATED_REGISTRY_PATH != _PACKAGE_GENERATED_TOOLS_DIR / "registry.json":
        return GENERATED_REGISTRY_PATH
    if GENERATED_TOOLS_DIR != _PACKAGE_GENERATED_TOOLS_DIR:
        return GENERATED_TOOLS_DIR / "registry.json"
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    configured = str(resolved_settings.generated_tools_dir or "").strip()
    if configured and configured != "generated_tools":
        path = Path(configured).expanduser()
        base = path if path.is_absolute() else Path(resolved_context.sandbox_dir) / path
        return base / "registry.json"
    return _resolve_store_registry_path(resolved_settings)


def _is_within(root: Path, target: Path) -> bool:
    """判断 target 是否位于 root 之内（Windows 大小写不敏感比较）。"""
    root_text = os.path.normcase(str(root))
    target_text = os.path.normcase(str(target))
    return target_text == root_text or target_text.startswith(root_text + os.sep)


def _resolve_in_sandbox(root: Path, relative_path: str) -> Path:
    """将相对路径解析为沙箱内的绝对路径并做越界防护。

    Raises:
        ValueError: 路径为空、非法，或解析结果超出沙箱根目录时。
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("path 不能为空")
    candidate = Path(relative_path)
    target = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not _is_within(root, target):
        raise ValueError(f"路径越界：{relative_path!r} 不在项目沙箱目录 {root} 之内")
    return target


def _build_unified_diff(
    old_text: str,
    new_text: str,
    rel_path: str,
    *,
    max_lines: int = _DIFF_MAX_LINES,
) -> str:
    """生成 unified diff 文本；超过行数上限时截断并标注。"""
    diff_lines = list(
        difflib.unified_diff(
            old_text.splitlines(),
            new_text.splitlines(),
            fromfile=f"a/{rel_path}",
            tofile=f"b/{rel_path}",
            lineterm="",
        )
    )
    if not diff_lines:
        return ""
    limit = int(max_lines or 0)
    if limit > 0 and len(diff_lines) > limit:
        omitted = len(diff_lines) - limit
        diff_lines = diff_lines[:limit] + [
            f"...（diff 共 {len(diff_lines)} 行，省略后 {omitted} 行；"
            "完整内容已写入文件）"
        ]
    return "\n".join(diff_lines)


def _read_window(
    text: str,
    *,
    start_line: int,
    max_lines: int,
    max_chars: int,
) -> dict[str, Any]:
    """按行窗口读取文本；字符预算在行边界处生效。"""
    lines = text.splitlines()
    total_lines = len(lines)
    start = max(1, int(start_line))
    if total_lines == 0:
        return {
            "content": "",
            "start_line": start,
            "end_line": start - 1,
            "total_lines": 0,
            "next_start": None,
        }
    if start > total_lines:
        return {
            "content": "",
            "start_line": start,
            "end_line": total_lines,
            "total_lines": total_lines,
            "next_start": None,
        }
    limit = int(max_lines or 0)
    end = total_lines if limit <= 0 else min(total_lines, start - 1 + limit)
    char_limit = int(max_chars or 0)
    selected: list[str] = []
    used = 0
    for index in range(start - 1, end):
        line = lines[index]
        cost = len(line) + 1
        if char_limit > 0 and used + cost > char_limit:
            if selected:
                end = index
                break
            # 单行就超过预算：截断该行，避免整段超长 JSON 绕过上限
            selected.append(line[:char_limit] + "…（单行超长，已截断）")
            end = index + 1
            break
        selected.append(line)
        used += cost
    else:
        end = min(end, start - 1 + len(selected))
    next_start = end + 1 if end < total_lines else None
    return {
        "content": "\n".join(selected),
        "start_line": start,
        "end_line": end,
        "total_lines": total_lines,
        "next_start": next_start,
    }


def _load_prompt(settings: Settings | None = None) -> str:
    """读取本目录 prompt.md 作为系统提示词（含输出契约补充说明）。"""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return (
        f"{text}\n\n{_EXTRA_INSTRUCTIONS}"
        f"{_output_budget_instructions(settings)}"
    )


def _build_chat_model(config: AgentAPIConfig) -> Model:
    """根据 api_config.yaml 构造模型；配置了 fallbacks 时启用故障转移。"""
    return build_chat_model_with_gateway(
        config,
        config_path=resolve_api_config_path(API_CONFIG_PATH),
        client_factory=AsyncOpenAI,
    )


async def _extract_output_json(
    result: RunResult | RunResultStreaming, *, max_chars: int = 0
) -> str:
    """``as_tool`` 输出提取器：保证工具返回值始终为 JSON 字符串。

    Agent 以纯文本返回（``output_type=None``），字符串输出先剥离
    Markdown 代码围栏提取 JSON 对象；无法结构化时回退为
    ``{"output": ...}`` 文本包装。``max_chars`` 对返回 JSON 中的
    字符串字段做截断，避免嵌套 Agent 的超长产出放大父级上下文。
    """
    output = result.final_output
    if isinstance(output, BaseModel):
        text = output.model_dump_json()
    elif isinstance(output, (dict, list)):
        text = json.dumps(output, ensure_ascii=False, default=str)
    elif isinstance(output, str):
        obj = extract_json_object(output)
        if obj is not None:
            text = json.dumps(obj, ensure_ascii=False)
        else:
            text = json.dumps(
                {"output": str(output)}, ensure_ascii=False
            )
    else:
        text = json.dumps(
            {"output": "" if output is None else str(output)},
            ensure_ascii=False,
        )
    return truncate_output_text(text, max_chars)


def _coerce_output_dict(output: Any) -> dict[str, Any] | None:
    """把 Agent 输出统一为 dict（BaseModel / dict / JSON 文本）。"""
    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    if isinstance(output, dict):
        return dict(output)
    if isinstance(output, str):
        obj = extract_json_object(output)
        if isinstance(obj, dict):
            return obj
    return None


def _apply_postcondition(
    output: Any, payload: dict[str, Any], root: Path
) -> tuple[Any, str | None]:
    """按真实文件系统/执行结果校验 code_agent 的自述成功。

    Returns:
        ``(可能被翻转的输出, 失败原因)``；满足后置条件或无法结构化时
        返回原输出与 None（不猜测、不二次解释自由文本）。
    """
    data = _coerce_output_dict(output)
    if data is None:
        return output, None
    if data.get("success") is not True:
        return output, None
    action = str(payload.get("action") or "").strip().lower()
    failure: str | None = None
    if action in (
        CodeAction.CREATE.value,
        CodeAction.MODIFY.value,
        CodeAction.AUTO.value,
    ):
        raw_path = data.get("file_path") or payload.get("file_path")
        if not raw_path:
            if action != CodeAction.AUTO.value:
                failure = f"后置校验失败：action={action} 但未声明 file_path"
        else:
            # auto 动作的 file_path 可能列出多个“涉及文件”（只读核对时
            # 常见逗号/顿号并列）；逐个校验，避免把整个列表当成一个文件名
            declared = (
                [
                    part.strip()
                    for part in re.split(r"[,，、;；]", str(raw_path))
                    if part.strip()
                ]
                if action == CodeAction.AUTO.value
                else [str(raw_path)]
            ) or [str(raw_path)]
            for path in declared:
                try:
                    target = _resolve_in_sandbox(root, path)
                except ValueError as exc:
                    failure = f"后置校验失败：file_path 非法（{exc}）"
                    break
                if not target.is_file():
                    failure = (
                        "后置校验失败：声明写入的文件在沙箱中不存在："
                        f"{path}"
                    )
                    break
                if target.stat().st_size == 0:
                    failure = f"后置校验失败：声明写入的文件为空：{path}"
                    break
    if failure is None and action in (
        CodeAction.EXECUTE.value,
        CodeAction.AUTO.value,
    ):
        nested = data.get("execution_result")
        if action == CodeAction.EXECUTE.value and not isinstance(nested, dict):
            failure = "后置校验失败：action=execute 但缺少 execution_result"
        elif isinstance(nested, dict) and nested.get("success") is False:
            failure = (
                "后置校验失败：沙箱执行结果为失败："
                f"{nested.get('error') or '未提供错误信息'}"
            )
    if failure is None:
        return output, None
    summary = str(data.get("summary") or "").strip()
    data.update(
        success=False,
        error=failure,
        summary=f"{summary}；{failure}" if summary else failure,
    )
    return data, failure


def _truncate_field(text: str, limit: int) -> str:
    """按字符预算截断叙述字段，保留可识别的截断标记。"""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…（已截断）"


def _output_budget_limits(settings: Any) -> dict[str, int] | None:
    """从配置读取输出预算；全部关闭时返回 None。"""
    budget = {
        "summary": int(
            getattr(settings, "code_agent_summary_max_chars", 0) or 0
        ),
        "action": int(
            getattr(settings, "code_agent_action_max_chars", 0) or 0
        ),
        "evidence_item": int(
            getattr(settings, "code_agent_evidence_item_max_chars", 0) or 0
        ),
        "evidence_items": int(
            getattr(settings, "code_agent_evidence_max_items", 0) or 0
        ),
    }
    return budget if any(value > 0 for value in budget.values()) else None


def _trim_narrative_fields(
    data: dict[str, Any], budgets: dict[str, int]
) -> bool:
    """截断 summary / action_performed / evidence；返回是否有改动。"""
    changed = False
    for field, key in (("summary", "summary"), ("action_performed", "action")):
        value = data.get(field)
        limit = budgets.get(key, 0)
        if isinstance(value, str) and limit > 0 and len(value) > limit:
            data[field] = _truncate_field(value, limit)
            changed = True
    evidence = data.get("evidence")
    if isinstance(evidence, list) and evidence:
        item_limit = budgets.get("evidence_item", 0)
        max_items = budgets.get("evidence_items", 0)
        trimmed = [
            _truncate_field(item, item_limit)
            if isinstance(item, str)
            else item
            for item in evidence
        ]
        if max_items > 0:
            trimmed = trimmed[:max_items]
        if trimmed != evidence:
            data["evidence"] = trimmed
            changed = True
    return changed


def _apply_output_governance(
    output: Any,
    payload: dict[str, Any],
    root: Path,
    *,
    suppress_diff: bool,
    budgets: dict[str, int] | None = None,
) -> Any:
    """输出治理：压缩 diff 回显并截断叙述字段。

    代码本身已写入沙箱文件，重复回显整份 diff 只会放大父级上下文；
    模型自述的正文同样进入上级上下文，按配置做确定性截断。
    无任何改动时返回原对象（保持调用方持有的身份语义）。
    """
    if not suppress_diff and not budgets:
        return output
    data = _coerce_output_dict(output)
    if data is None:
        return output
    changed = False
    if suppress_diff:
        diff = data.get("code_diff")
        if isinstance(diff, str) and diff.strip():
            added = sum(
                1
                for line in diff.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            )
            removed = sum(
                1
                for line in diff.splitlines()
                if line.startswith("-") and not line.startswith("---")
            )
            rel_path = str(
                data.get("file_path")
                or payload.get("file_path")
                or "（未声明路径）"
            )
            size = 0
            try:
                target = _resolve_in_sandbox(root, rel_path)
            except ValueError:
                target = None
            if target is not None and target.is_file():
                size = target.stat().st_size
            data["code_diff"] = (
                f"[已省略 diff 回显] {rel_path}：+{added}/-{removed} 行，"
                f"{size} 字节（变更已写入文件）"
            )
            changed = True
    if budgets and _trim_narrative_fields(data, budgets):
        changed = True
    return data if changed else output


def _make_postcondition_validator(
    context: ProjectContext,
    root: Path,
    settings: Settings,
):
    """构造 code_agent 的 MCP 输出校验器（后置校验 + 输出治理）。"""
    postcondition_enabled = getattr(
        settings, "code_agent_postcondition_enabled", True
    )
    suppress_diff = getattr(settings, "code_agent_suppress_diff_echo", True)
    budgets = _output_budget_limits(settings)
    if not postcondition_enabled and not suppress_diff and not budgets:
        return None

    def _validator(output: Any, payload: dict[str, Any]) -> Any:
        validated = output
        if postcondition_enabled:
            validated, failure = _apply_postcondition(output, payload, root)
            if failure is not None:
                logger.warning(
                    "code_agent 后置校验失败（action=%s）：%s",
                    payload.get("action"),
                    failure,
                )
                event_logger = getattr(context, "event_logger", None)
                if event_logger is not None:
                    try:
                        event_logger.log_event(
                            EVENT_CODE_AGENT_POSTCONDITION_FAILED,
                            {
                                "action": payload.get("action"),
                                "file_path": payload.get("file_path"),
                                "reason": failure,
                            },
                        )
                    except Exception:  # 审计写入失败不得影响校验结论
                        pass
        return _apply_output_governance(
            validated,
            payload,
            root,
            suppress_diff=suppress_diff,
            budgets=budgets,
        )

    return _validator


def _make_artifact_index_decorator(
    context: ProjectContext,
    settings: Settings,
):
    """构造调用时注入沙箱产物索引的输入装饰器（未启用时返回 None）。"""
    if not getattr(settings, "artifact_index_in_prompt", True):
        return None
    root = _sandbox_root(context)
    max_entries = int(
        getattr(settings, "artifact_index_max_entries", 20) or 0
    )
    if max_entries <= 0:
        return None

    def _decorate(data: dict[str, Any]) -> dict[str, Any] | None:
        # 有写入范围时只列相关文件；没有范围时按最近修改取前 N 条，
        # 避免整个沙箱目录随任务推进无限进入上下文。
        scope = [str(item) for item in (data.get("write_scope") or []) if str(item)]
        index = build_artifact_index(
            root,
            max_entries=max_entries,
            includes=scope or ["*"],
        )
        if not index:
            return None
        return {**data, "sandbox_artifact_index": index}

    return _decorate


def _make_input_decorator(context: ProjectContext, settings: Settings):
    """调用时绑定本请求的写入范围，并按需注入沙箱产物索引。"""
    artifact_decorator = _make_artifact_index_decorator(context, settings)

    def _decorate(data: dict[str, Any]) -> dict[str, Any] | None:
        # 每次调用都必须重设：同一 Server 会被多个子任务复用
        scope = data.get("write_scope") or []
        _CURRENT_WRITE_SCOPE.set(tuple(str(item) for item in scope))
        # 每个 MCP 调用对应一次新的 Agent 运行，读取台账必须从空开始，
        # 否则复用同一 Server 的子任务会把上一轮的读取误判为已读。
        reset_read_ledger()
        if artifact_decorator is None:
            return None
        return artifact_decorator(data)

    return _decorate


def _make_context_prune_filter(context: ProjectContext, settings: Settings):
    """构造模型调用前的确定性输入压缩器（异步，供 RunConfig 使用）。"""
    from agents.run_config import ModelInputData

    max_chars = int(
        getattr(settings, "code_agent_context_max_chars", 0) or 0
    )
    keep_outputs = int(
        getattr(settings, "code_agent_context_keep_tool_outputs", 2) or 2
    )
    event_logger = getattr(context, "event_logger", None)
    states: dict[str, CompactState] = {}

    async def _filter(data: Any) -> Any:
        model_input = data.model_data
        pruned, stats = compact_model_input(
            list(model_input.input),
            states,
            max_chars=max_chars,
            keep_last_outputs=keep_outputs,
        )
        if event_logger is not None and (
            stats.compacted
            or stats.pruned_calls
            or stats.pruned_outputs
            or stats.dropped_items
        ):
            try:
                event_logger.log_event(
                    EVENT_CONTEXT_PRUNED,
                    {
                        "agent": AGENT_NAME,
                        "before_chars": stats.before_chars,
                        "after_chars": stats.after_chars,
                        "saved_chars": stats.saved_chars,
                        "pruned_calls": stats.pruned_calls,
                        "pruned_outputs": stats.pruned_outputs,
                        "dropped_items": stats.dropped_items,
                        "compacted": stats.compacted,
                        "compactions": stats.compactions,
                    },
                )
            except Exception:  # 审计写入失败不得影响模型调用
                pass
        return ModelInputData(
            input=pruned, instructions=model_input.instructions
        )

    return _filter


def _write_scope_violation(path: str) -> str | None:
    """返回越界写入的错误文本；未声明范围或路径合法时返回 None。"""
    scope = _CURRENT_WRITE_SCOPE.get()
    if not scope or path_allowed(list(scope), path):
        return None
    return (
        f"写入被拒绝：{path} 不在本子任务声明的写入范围内"
        f"（允许：{'、'.join(scope)}）"
    )


def _python_syntax_error(
    root: Path,
    path: str,
    content: str,
    *,
    append: bool,
) -> dict[str, Any] | None:
    """对 .py 目标做写前语法检查；返回错误回执字段或 None。

    检查的是写入后的完整文件（append 时拼接已有内容），避免半截脚本
    落盘后还要靠一次执行失败才发现语法错误。
    """
    if not str(path).lower().endswith(".py"):
        return None
    full = content
    if append:
        try:
            target = _resolve_in_sandbox(root, path)
            if target.is_file():
                full = (
                    target.read_text(encoding="utf-8", errors="replace")
                    + content
                )
        except (ValueError, OSError):
            return None
    try:
        compile(full, str(path), "exec")
    except SyntaxError as exc:
        detail = (exc.text or "").strip()
        return {
            "error": (
                f"Python 语法检查未通过：{exc.msg}"
                f"（第 {exc.lineno or 0} 行，第 {exc.offset or 0} 列）"
                + (f"：{detail}" if detail else "")
            ),
            "syntax_error": True,
            "line": int(exc.lineno or 0),
        }
    return None


def _write_text_file(
    root: Path,
    path: str,
    content: str,
    *,
    append: bool,
    max_bytes: int,
    diff_max_lines: int = _DIFF_MAX_LINES,
) -> dict[str, Any]:
    """在沙箱内写入或追加文本，返回结构化写入结果。

    单次内容超过 ``max_bytes`` 时拒绝而不是截断：截断会静默破坏脚本，
    拒绝并提示分段写入才能让模型用 append 完成续写。
    """
    encoded = content.encode("utf-8")
    limit = int(max_bytes or 0)
    if limit > 0 and len(encoded) > limit:
        raise ValueError(
            f"content 超过单次写入上限 {limit} 字节（实际 {len(encoded)} 字节）；"
            "请分段写入：先写第一部分，再用 append=true 追加剩余内容"
        )
    target = _resolve_in_sandbox(root, path)
    existed = target.is_file()
    old_text = (
        target.read_text(encoding="utf-8", errors="replace") if existed else ""
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    if append and existed:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
    else:
        target.write_text(content, encoding="utf-8")
    new_text = target.read_text(encoding="utf-8", errors="replace")
    rel_path = target.relative_to(root).as_posix()
    diff_text = (
        _build_unified_diff(
            old_text, new_text, rel_path, max_lines=diff_max_lines
        )
        if existed
        else f"（新建文件 {rel_path}，{len(new_text.splitlines())} 行）"
    )
    return {
        "path": rel_path,
        "created": not existed,
        "appended": bool(append and existed),
        "bytes": len(encoded),
        "total_bytes": len(new_text.encode("utf-8")),
        "diff": diff_text,
    }


# ---------------------------------------------------------------------------
# 内部工具构建
# ---------------------------------------------------------------------------
def build_code_tools(
    context: ProjectContext | None = None,
    *,
    sandbox_runner: SandboxBackend | None = None,
    settings: Settings | None = None,
) -> dict[str, FunctionTool]:
    """构造代码 Agent 的全部内部工具（以工具名索引的字典）。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        sandbox_runner: 沙箱执行器；未提供时按 context/settings 新建。
        settings: 生效配置；未提供时取 context.config。

    Returns:
        ``{工具名: FunctionTool}``，包含 write_file / write_and_run /
        read_file / run_in_sandbox / write_generated_tool 五个工具。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    sandbox = sandbox_runner or create_sandbox(
        context=resolved_context, settings=resolved_settings
    )
    sandbox_root = _sandbox_root(resolved_context)
    tool_output_limit = int(
        getattr(resolved_settings, "tool_output_max_chars", 0) or 0
    )
    write_max_bytes = int(
        getattr(resolved_settings, "code_agent_write_max_bytes", 0) or 0
    )
    read_max_lines = int(
        getattr(resolved_settings, "code_agent_read_max_lines", 0) or 0
    )
    read_max_chars = int(
        getattr(resolved_settings, "code_agent_read_max_chars", 0) or 0
    )
    read_dedup = bool(
        getattr(resolved_settings, "code_agent_read_dedup", True)
    )
    action_dedup = bool(
        getattr(resolved_settings, "code_agent_action_dedup", True)
    )
    tool_diff_max_lines = int(
        getattr(resolved_settings, "code_agent_tool_diff_max_lines", 0) or 0
    )
    approval_gate = build_approval_gate(
        resolved_settings, context=resolved_context
    )

    @function_tool
    def write_file(
        ctx: RunContextWrapper[Any],
        path: str,
        content: str,
        append: bool = False,
    ) -> str:
        """在项目沙箱目录内写入一个文本文件（默认覆盖，可追加）。

        Args:
            path: 相对项目沙箱根目录的文件路径；禁止沙箱外路径。
            content: 要写入的文本内容（UTF-8）。
            append: 为 true 且文件已存在时追加到末尾；文件不存在时等同于新建。

        Returns:
            JSON 字符串：成功为 ``{"status": "ok", "path", "created",
            "appended", "bytes", "total_bytes", "diff"}``；失败为
            ``{"status": "error", "error"}``。``.py`` 语法检查不通过时
            不落盘，错误回执另带 ``syntax_error: true`` 与 ``line``。
        """
        violation = _write_scope_violation(path)
        if violation is not None:
            return _error(violation, path=path)
        syntax = _python_syntax_error(
            sandbox_root, path, content, append=append
        )
        if syntax is not None:
            return _error(
                syntax["error"],
                path=path,
                syntax_error=True,
                line=syntax["line"],
            )
        encoded = content.encode("utf-8")
        if action_dedup:
            # 以文件现状判定是否为无变化写入，避免重复副作用；不依赖台账，
            # 因此跨续跑分段、跨工具调用都成立。
            try:
                target = _resolve_in_sandbox(sandbox_root, path)
                if target.is_file():
                    current = target.read_bytes()
                    no_op = (
                        current.endswith(encoded) if append else current == encoded
                    )
                    if no_op:
                        _log_action_replayed(
                            resolved_context, "write_file", path, 1
                        )
                        return _ok(
                            path=path,
                            unchanged=True,
                            bytes=len(encoded),
                            total_bytes=len(current),
                            note="文件内容已是目标内容，本次未重复写盘",
                        )
            except OSError:
                pass
        try:
            return _ok(
                **_write_text_file(
                    sandbox_root,
                    path,
                    content,
                    append=append,
                    max_bytes=write_max_bytes,
                    diff_max_lines=tool_diff_max_lines,
                )
            )
        except (ValueError, OSError) as exc:
            return _error(f"写入失败：{type(exc).__name__}: {exc}", path=path)

    @function_tool
    async def write_and_run(
        ctx: RunContextWrapper[Any],
        path: str,
        content: str,
        task_id: str,
        append: bool = False,
    ) -> str:
        """写入脚本并立即在同一沙箱进程内执行它（一次调用完成写入+运行）。

        适用场景：脚本已经写好、下一步就是对同一份内容执行。相比先
        ``write_file`` 再 ``run_in_sandbox``，省去一轮模型往返与一次
        完整上下文重发。执行的是文件写入后的完整内容（append 时包含
        追加前的已有部分）。

        Args:
            path: 相对项目沙箱根目录的脚本路径（建议 .py）。
            content: 本次要写入的完整脚本内容（UTF-8）。
            task_id: 任务编号；只允许字母/数字/下划线/连字符（长度 1-64）。
            append: 为 true 且文件已存在时追加后再执行。

        Returns:
            JSON 字符串：成功为 ``{"status": "ok", "path", "created",
            "appended", "bytes", "total_bytes", "execution_result"}``；
            失败为 ``{"status": "error", "error"}``。
        """
        try:
            await approval_gate.authorize(
                "run_in_sandbox",
                {"code": content, "task_id": task_id},
                task_id=task_id,
            )
        except ApprovalDenied as exc:
            return _error(str(exc), task_id=task_id, approval_required=True)
        violation = _write_scope_violation(path)
        if violation is not None:
            return _error(violation, path=path)
        syntax = _python_syntax_error(
            sandbox_root, path, content, append=append
        )
        if syntax is not None:
            return _error(
                syntax["error"],
                path=path,
                task_id=task_id,
                syntax_error=True,
                line=syntax["line"],
            )
        try:
            written = _write_text_file(
                sandbox_root,
                path,
                content,
                append=append,
                max_bytes=write_max_bytes,
                diff_max_lines=tool_diff_max_lines,
            )
            target = _resolve_in_sandbox(sandbox_root, path)
            code = target.read_text(encoding="utf-8", errors="replace")
            run_id = f"code-{uuid.uuid4().hex[:10]}"
            result = await sandbox.execute(
                code, task_id, run_id, workdir=sandbox_root
            )
            return truncate_output_text(
                _ok(
                    **written,
                    execution_result=json.loads(result.model_dump_json()),
                ),
                tool_output_limit,
            )
        except SandboxError as exc:
            return _error(f"沙箱拒绝执行：{exc}", task_id=task_id)
        except (ValueError, OSError) as exc:
            return _error(f"写入失败：{type(exc).__name__}: {exc}", path=path)

    @function_tool
    def read_file(
        ctx: RunContextWrapper[Any],
        path: str,
        start_line: int = 1,
        max_lines: int = 0,
        force: bool = False,
    ) -> str:
        """按行窗口读取项目沙箱目录内的文本文件（默认读取前若干行）。

        同一运行内重复读取未变更文件的已读窗口时只返回回执
        （``unchanged: true``），不再重复正文；确需重读时传 ``force=true``。

        Args:
            path: 相对项目沙箱根目录的文件路径；禁止沙箱外路径。
            start_line: 起始行号（从 1 开始）。
            max_lines: 本次最多读取的行数；0 表示使用默认窗口。
            force: 为 true 时忽略读取台账，强制返回正文。

        Returns:
            JSON 字符串：成功为 ``{"status": "ok", "path", "content",
            "start_line", "end_line", "total_lines", "next_start", "bytes"}``；
            重复读取为不含 ``content`` 的 ``unchanged`` 回执；
            失败为 ``{"status": "error", "error"}``。
        """
        try:
            target = _resolve_in_sandbox(sandbox_root, path)
            if not target.is_file():
                return _error(f"文件不存在：{path}", path=path)
            encoded = target.read_bytes()
            text = encoded.decode("utf-8", errors="replace")
            rel_path = target.relative_to(sandbox_root).as_posix()
            window = _read_window(
                text,
                start_line=start_line,
                max_lines=max_lines or read_max_lines,
                max_chars=read_max_chars,
            )
            # 被输出上限截断的窗口不记入台账：模型只看到片段，
            # 再读必须重新返回正文，否则会误回报「已读」。
            recordable = (
                tool_output_limit <= 0
                or len(window["content"]) <= tool_output_limit
            )
            if (
                read_dedup
                and not force
                and recordable
                and window["end_line"] >= window["start_line"]
            ):
                ledger = _READ_LEDGER.get()
                if ledger is not None:
                    digest = hashlib.sha256(encoded).hexdigest()[:16]
                    entry = ledger.get(rel_path)
                    if entry is None or entry["sha256"] != digest:
                        # 内容变化视为新内容：旧窗口的「已读」不再成立
                        entry = {"sha256": digest, "ranges": []}
                        ledger[rel_path] = entry
                    if _range_covered(
                        entry["ranges"],
                        window["start_line"],
                        window["end_line"],
                    ):
                        saved = len(window["content"])
                        _log_read_collapsed(
                            resolved_context, rel_path, window, saved
                        )
                        return truncate_output_text(
                            _json(
                                {
                                    "status": "ok",
                                    "path": rel_path,
                                    "bytes": len(encoded),
                                    "unchanged": True,
                                    "content_sha256": digest,
                                    **{
                                        key: value
                                        for key, value in window.items()
                                        if key != "content"
                                    },
                                    "note": (
                                        "该窗口与本次运行中已读取的内容完全相同，"
                                        "正文未重复返回；确需重读请传 force=true"
                                    ),
                                }
                            ),
                            tool_output_limit,
                        )
                    entry["ranges"] = _merge_range(
                        entry["ranges"],
                        window["start_line"],
                        window["end_line"],
                    )
            return truncate_output_text(
                _ok(
                    path=rel_path,
                    bytes=len(encoded),
                    **window,
                ),
                tool_output_limit,
            )
        except (ValueError, OSError) as exc:
            return _error(f"读取失败：{type(exc).__name__}: {exc}", path=path)

    @function_tool
    async def run_in_sandbox(ctx: RunContextWrapper[Any], code: str, task_id: str) -> str:
        """在验证型沙箱中以独立子进程运行一段 Python 代码并返回结构化结果。

        脚本工作目录为项目沙箱根目录（与 ``read_file`` / ``write_file``
        的相对路径基准一致），因此代码中的相对路径可直接引用已写入的
        产物；stdout、stderr 与元数据落盘在
        ``sandbox/task_{task_id}/run_{run_id}/``，超时（默认
        sandbox_timeout）后强制终止。

        Args:
            code: 待执行的 Python 源码（完整可运行脚本）。
            task_id: 任务编号；只允许字母/数字/下划线/连字符（长度 1-64）。

        Returns:
            JSON 字符串：ExecutionResult 的完整序列化结果；沙箱拒绝执行时
            返回 ``{"status": "error", "error"}``。
        """
        try:
            await approval_gate.authorize(
                "run_in_sandbox",
                {"code": code, "task_id": task_id},
                task_id=task_id,
            )
        except ApprovalDenied as exc:
            return _error(str(exc), task_id=task_id, approval_required=True)
        ledger = _action_ledger() if action_dedup else None
        run_key = hashlib.sha256(f"{task_id}\n{code}".encode("utf-8")).hexdigest()
        if ledger is not None:
            entry = ledger.get(run_key)
            if entry is not None and _sandbox_state_fingerprint(
                sandbox_root
            ) == entry.get("fingerprint"):
                repeats = int(entry.get("repeats", 1))
                if repeats >= 2:
                    return _error(
                        "相同代码已执行过且沙箱工作区未变化；"
                        "请修改代码或说明需要重跑的原因",
                        task_id=task_id,
                        code_hash=run_key[:12],
                        duplicate_execution=True,
                    )
                entry["repeats"] = repeats + 1
                _log_action_replayed(
                    resolved_context, "run_in_sandbox", run_key[:12], repeats + 1
                )
                replayed = dict(entry.get("result") or {})
                replayed.update(
                    {
                        "replayed": True,
                        "note": "相同代码且工作区未变化，返回上次执行结果，未重复执行",
                    }
                )
                return truncate_output_text(
                    json.dumps(replayed, ensure_ascii=False),
                    tool_output_limit,
                )
        try:
            run_id = f"code-{uuid.uuid4().hex[:10]}"
            result = await sandbox.execute(
                code, task_id, run_id, workdir=sandbox_root
            )
            text = truncate_output_text(
                result.model_dump_json(), tool_output_limit
            )
            if ledger is not None:
                try:
                    parsed = json.loads(result.model_dump_json())
                except (json.JSONDecodeError, TypeError):
                    parsed = None
                if isinstance(parsed, dict):
                    ledger[run_key] = {
                        "fingerprint": _sandbox_state_fingerprint(sandbox_root),
                        "result": parsed,
                        "repeats": 1,
                    }
            return text
        except SandboxError as exc:
            return _error(f"沙箱拒绝执行：{exc}", task_id=task_id)

    @function_tool
    def write_generated_tool(
        ctx: RunContextWrapper[Any],
        tool_name: str,
        code: str,
        manifest_data: str,
    ) -> str:
        """将新工具候选代码写入 generated_tools 目录并登记候选清单（registry.json）。

        候选工具写入后仅为“待评审”状态，未经评审通过并由主管 Agent 注册前
        不可被任何 Agent 调用；同时生成 MCP 暴露契约
        （``{tool_name}.mcp_schema.json``，MCPToolSchema 的 JSON 描述）。

        Args:
            tool_name: 工具名称；小写字母开头，仅字母/数字/下划线，最长 64 字符。
            code: 工具实现的完整 Python 源码（将写入 ``{tool_name}.py``）。
            manifest_data: 清单元数据的 JSON 文本，可含 description /
                input_schema / output_schema / mcp_schema / created_by_task /
                version / tool_id 等字段；``created_by_task`` 建议填写所属任务编号。
                还应提供 ``test_cases``（验收测试用例数组，每项为入口函数
                ``run()`` 的关键字参数对象）与 ``expected_outputs``（与之一一
                对应的期望返回值，须为 JSON 可序列化），供注册前沙箱验收比对。

        Returns:
            JSON 字符串：成功为 ``{"status": "ok", "code_path",
            "mcp_schema_path", "manifest", "note"}``；失败为
            ``{"status": "error", "error"}``。
        """
        try:
            if not isinstance(tool_name, str) or not _TOOL_NAME_PATTERN.match(tool_name):
                return _error(
                    "tool_name 非法：只允许小写字母开头、由字母/数字/下划线组成、"
                    "长度不超过 64 的 snake_case 名称",
                    tool_name=tool_name,
                )
            if not isinstance(code, str) or not code.strip():
                return _error("code 不能为空", tool_name=tool_name)
            if isinstance(manifest_data, str):
                payload = json.loads(manifest_data) if manifest_data.strip() else {}
            else:
                payload = manifest_data
            if not isinstance(payload, dict):
                return _error("manifest_data 必须是 JSON 对象文本", tool_name=tool_name)

            generated_dir = resolve_generated_tools_dir(
                resolved_context, resolved_settings
            )
            registry_path = resolve_generated_registry_path(
                resolved_context, resolved_settings
            )
            generated_dir.mkdir(parents=True, exist_ok=True)
            entry_file = f"{tool_name}.py"
            code_path = generated_dir / entry_file
            code_path.write_text(code, encoding="utf-8")

            # MCP 暴露契约（MCPToolSchema 形态）：随候选代码一并落盘；
            # payload.mcp_schema 可覆盖默认推断（name / description / inputSchema）。
            manifest_input_schema = payload.get("input_schema") or {}
            if not isinstance(manifest_input_schema, dict):
                manifest_input_schema = {}
            raw_mcp = payload.get("mcp_schema")
            raw_mcp = raw_mcp if isinstance(raw_mcp, dict) else {}
            mcp_input_schema = raw_mcp.get("inputSchema") or manifest_input_schema
            if not isinstance(mcp_input_schema, dict) or not mcp_input_schema:
                mcp_input_schema = {"type": "object", "properties": {}}
            mcp_schema = {
                "name": str(raw_mcp.get("name") or tool_name),
                "description": str(
                    raw_mcp.get("description") or payload.get("description") or ""
                ),
                "inputSchema": dict(mcp_input_schema),
            }
            mcp_schema_path = generated_dir / f"{tool_name}.mcp_schema.json"
            mcp_schema_path.write_text(
                json.dumps(mcp_schema, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            raw_test_cases = payload.get("test_cases")
            test_cases = raw_test_cases if isinstance(raw_test_cases, list) else []
            raw_expected = payload.get("expected_outputs")
            expected_outputs = raw_expected if isinstance(raw_expected, list) else []
            manifest = GeneratedToolManifest(
                tool_id=str(payload.get("tool_id") or f"tool-{uuid.uuid4().hex[:12]}"),
                tool_name=tool_name,
                entry_file=entry_file,
                description=str(payload.get("description") or ""),
                input_schema=manifest_input_schema,
                output_schema=payload.get("output_schema") or {},
                created_by_task=str(payload.get("created_by_task") or "unknown"),
                review_status="pending",
                version=str(payload.get("version") or "0.1.0"),
                mcp_schema=mcp_schema,
                test_cases=test_cases,
                expected_outputs=expected_outputs,
                verified=False,
                content_hash=sha256_file(code_path),
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            registry = ToolRegistry(generated_tools_dir=generated_dir)
            if registry_path.is_file():
                registry.load_generated_registry(registry_path)
            registry.register_generated_tool(manifest)
            registry.save_generated_registry(registry_path)
            return _ok(
                tool_name=tool_name,
                code_path=str(code_path),
                mcp_schema_path=str(mcp_schema_path),
                review_status=manifest.review_status,
                note=(
                    "候选工具与 MCP 契约（mcp_schema.json）已写入 generated_tools"
                    "（review_status=pending）；须经评审通过并由主管 Agent 注册后"
                    "才可进入授权候选集合。"
                ),
            )
        except (ToolRegistryError, ValueError, OSError) as exc:
            return _error(f"写入生成工具失败：{type(exc).__name__}: {exc}", tool_name=tool_name)

    return {
        "write_file": write_file,
        "write_and_run": write_and_run,
        "read_file": read_file,
        "run_in_sandbox": run_in_sandbox,
        "write_generated_tool": write_generated_tool,
    }


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def create_code_agent(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
    sandbox_runner: SandboxBackend | None = None,
) -> Agent[Any]:
    """构造配置完毕的代码 Agent（SDK Agent 实例）。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 模型实例或名称覆盖；未提供时按本包 api_config.yaml 构造
            （密钥优先取 api_key 直接值，回退 api_key_env 环境变量，
            均缺失时抛 ConfigLoadError）。
        sandbox_runner: 沙箱执行器覆盖；未提供时按 context/settings 新建。

    Returns:
        ``Agent`` 实例：name="code_agent"，instructions 取自 prompt.md，
        不设置 output_type（None，纯文本返回）：输出由调用方剥离
        Markdown 围栏后按 ``CodeAgentOutput`` 手动校验（MCP 适配层
        内置该容错路径）。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    resolved_model = model or _build_chat_model(
        load_agent_api_config(resolve_api_config_path(API_CONFIG_PATH))
    )
    tools = build_code_tools(
        resolved_context, sandbox_runner=sandbox_runner, settings=resolved_settings
    )
    # 单次模型输出上限：0 表示不限制（默认），基准验证后再按需开启
    max_output_tokens = int(
        getattr(resolved_settings, "code_agent_max_output_tokens", 0) or 0
    )
    agent_kwargs: dict[str, Any] = {
        "name": AGENT_NAME,
        "instructions": _load_prompt(resolved_settings),
        "tools": list(tools.values()),
        "model": resolved_model,
        # output_type 保持 None（纯文本返回）：千问等兼容接口忽略
        # response_format=json_schema 时 SDK 严格校验会抛 ModelBehaviorError，
        # 改由调用方手动剥离围栏 + CodeAgentOutput 校验。
    }
    if max_output_tokens > 0:
        agent_kwargs["model_settings"] = ModelSettings(
            max_tokens=max_output_tokens
        )
    return Agent(**agent_kwargs)


def get_code_agent_as_tool(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
    sandbox_runner: SandboxBackend | None = None,
) -> FunctionTool:
    """将代码 Agent 转为主 Agent 可直接调用的工具入口（Agent.as_tool）。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 模型实例或名称覆盖；未提供时按本包 api_config.yaml 构造。
        sandbox_runner: 沙箱执行器覆盖；未提供时按 context/settings 新建。

    Returns:
        ``FunctionTool``：工具名 "code_agent"，返回值经输出提取器规范化为
        结构化 JSON 字符串。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    agent = create_code_agent(
        resolved_context,
        settings=resolved_settings,
        model=model,
        sandbox_runner=sandbox_runner,
    )
    output_limit = int(
        getattr(resolved_settings, "code_agent_output_max_chars", 0) or 0
    )
    return agent.as_tool(
        tool_name=AGENT_NAME,
        tool_description=_TOOL_DESCRIPTION,
        custom_output_extractor=lambda result: _extract_output_json(
            result, max_chars=output_limit
        ),
        max_turns=resolved_settings.max_model_turns,
    )


def get_mcp_schema() -> MCPToolSchema:
    """返回代码 Agent 的 MCP 暴露契约（MCPToolSchema）。

    inputSchema 取自 ``CodeAgentInput.model_json_schema()``，供 MCP
    tools/list 发现与主管侧工具构造统一使用。
    """
    return MCPToolSchema(
        name=AGENT_NAME,
        description=_TOOL_DESCRIPTION,
        inputSchema=CodeAgentInput.model_json_schema(),
    )


def get_mcp_server(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
    sandbox_runner: SandboxBackend | None = None,
) -> MCPToolServer:
    """将代码 Agent 包装为 MCP Server（供主管与 MCPClient 统一调用）。

    与 ``get_code_agent_as_tool`` 等价的 MCP 暴露形态：调用时按
    ``CodeAgentInput`` 校验参数、运行 Agent，并按 ``CodeAgentOutput``
    提取结构化 JSON 文本。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 模型实例或名称覆盖；未提供时按本包 api_config.yaml 构造。
        sandbox_runner: 沙箱执行器覆盖；未提供时按 context/settings 新建。

    Returns:
        ``MCPToolServer``：name="code_agent"，含单个工具的 MCP Server。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    agent = create_code_agent(
        resolved_context,
        settings=resolved_settings,
        model=model,
        sandbox_runner=sandbox_runner,
    )
    run_config = None
    if getattr(resolved_settings, "code_agent_context_prune", False):
        from agents import RunConfig

        run_config = RunConfig(
            call_model_input_filter=_make_context_prune_filter(
                resolved_context, resolved_settings
            )
        )
    return agent_to_mcp_server(
        agent,
        AGENT_NAME,
        _TOOL_DESCRIPTION,
        CodeAgentInput,
        CodeAgentOutput,
        max_turns=resolved_settings.max_model_turns,
        usage_recorder=make_usage_recorder(resolved_context, "code_agent"),
        output_max_chars=int(
            getattr(resolved_settings, "code_agent_output_max_chars", 0) or 0
        ),
        settings=resolved_settings,
        activity_hook=resolved_context.report_activity,
        result_validator=_make_postcondition_validator(
            resolved_context,
            _sandbox_root(resolved_context),
            resolved_settings,
        ),
        input_decorator=_make_input_decorator(
            resolved_context, resolved_settings
        ),
        run_config=run_config,
        event_sink=getattr(
            getattr(resolved_context, "event_logger", None), "log_event", None
        ),
        turn_extension=bool(
            getattr(resolved_settings, "code_agent_turn_extension", True)
        ),
    )

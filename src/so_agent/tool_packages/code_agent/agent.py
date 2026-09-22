"""代码 Agent 构建模块（code_agent 工具包）。

职责：作为主 Agent 的固定工具入口之一，构造“代码执行 Agent（Code Agent）”：

- 读取本目录 ``prompt.md`` 作为系统提示词（instructions）；
- 通过 ``runtime.config_loader`` 加载本目录 ``api_config.yaml``，按其中
  ``api_key`` 直接值（或旧 ``api_key_env`` 环境变量回退）构造 OpenAI 兼容
  聊天模型（密钥值绝不写日志）；
- 绑定四个内部工具：``write_file`` / ``read_file`` / ``run_in_sandbox`` /
  ``write_generated_tool``，所有工具返回值统一为 JSON 字符串；
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

import difflib
import json
import os
import re
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    FunctionTool,
    Model,
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
from so_agent.json_utils import extract_json_object
from so_agent.mcp.adapter import agent_to_mcp_server
from so_agent.mcp.protocol import MCPToolSchema
from so_agent.mcp.server import MCPToolServer
from so_agent.models import ExecutionResult, GeneratedToolManifest
from so_agent.runtime.config_loader import AgentAPIConfig, load_agent_api_config
from so_agent.runtime.sandbox import SandboxError, SandboxRunner
from so_agent.runtime.tool_registry import ToolRegistry, ToolRegistryError

# ---------------------------------------------------------------------------
# 包内路径（用 pathlib 定位同目录下的提示词与 API 配置）
# ---------------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROMPT_PATH: Path = PACKAGE_DIR / "prompt.md"
API_CONFIG_PATH: Path = PACKAGE_DIR / "api_config.yaml"

# generated_tools 目录与其候选工具清单（位于 tool_packages 之下）
GENERATED_TOOLS_DIR: Path = PACKAGE_DIR.parent / "generated_tools"
GENERATED_REGISTRY_PATH: Path = GENERATED_TOOLS_DIR / "registry.json"

# 生成工具名称规范：小写字母开头，仅字母/数字/下划线，最长 64 字符
_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# 单次 diff 输出的最大行数（超出截断，避免工具返回值过大）
_DIFF_MAX_LINES = 200

# Agent 名称（与工具注册表 / 路由约定保持一致）
AGENT_NAME = "code_agent"

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
- `action`：动作类型，取值 `create` / `modify` / `review` / `execute`。

## 最终输出（结构化强约束，覆盖上文输出格式段落）

最终回复必须是 CodeAgentOutput 结构的纯 JSON 文本——不包裹 Markdown
代码围栏，不附加任何解释性文字，字段如下：

- `success`：bool，本次动作是否成功完成；
- `action_performed`：str，实际执行的动作描述（与 action 对应或说明为何未执行）；
- `file_path`：str 或 null，涉及的沙箱内相对路径；
- `code_diff`：str，代码变更的 unified diff 文本（无变更时为空字符串）；
- `execution_result`：ExecutionResult 结构或 null，当 action 为 execute 时
  内嵌 `run_in_sandbox` 工具返回的完整结果对象；
- `error`：str 或 null，失败原因（成功时为 null）；
- `summary`：str，完成情况摘要（含关键结论与自检结果）。

硬性纪律：

1. 文件读写在项目沙箱目录内进行，必须调用 `write_file` / `read_file` 工具；
2. 代码运行必须调用 `run_in_sandbox` 工具，禁止声称“已运行”而未真实调用；
3. 生成新工具候选必须调用 `write_generated_tool` 写入 generated_tools 目录，
   不得写入其他位置，也不得自行注册或授权；
4. 一切证据（命令、输出、路径）必须真实可复核，禁止编造。
"""


class CodeAction(str, Enum):
    """代码 Agent 可执行的动作类型。"""

    CREATE = "create"    # 创建新文件/新代码
    MODIFY = "modify"    # 修改既有文件/代码
    REVIEW = "review"    # 静态审阅（只读核对，不落盘）
    EXECUTE = "execute"  # 在沙箱中执行代码并收集结果


class CodeAgentInput(BaseModel):
    """代码 Agent 的输入契约（主 Agent 下发）。"""

    task_description: str = Field(description="本次子任务的编码/执行目标")
    code_content: str | None = Field(default=None, description="待创建或修改的代码内容")
    file_path: str | None = Field(default=None, description="涉及的文件路径（相对项目沙箱目录）")
    action: CodeAction = Field(description="动作类型：create/modify/review/execute")


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


def _build_unified_diff(old_text: str, new_text: str, rel_path: str) -> str:
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
    if len(diff_lines) > _DIFF_MAX_LINES:
        diff_lines = diff_lines[: _DIFF_MAX_LINES] + [
            f"...（diff 超过 {_DIFF_MAX_LINES} 行，已截断）"
        ]
    return "\n".join(diff_lines)


def _load_prompt() -> str:
    """读取本目录 prompt.md 作为系统提示词（含输出契约补充说明）。"""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return f"{text}\n\n{_EXTRA_INSTRUCTIONS}"


def _build_chat_model(config: AgentAPIConfig) -> Model:
    """根据 api_config.yaml 构造 OpenAI 兼容聊天模型客户端。

    密钥优先取 ``config.api_key``（api_config.yaml 中的直接密钥值）；
    未配置时回退从 ``config.api_key_env`` 指定的环境变量解析；两者均
    不可用时抛出 ``ConfigLoadError``（明确失败，不静默降级）。

    Raises:
        ConfigLoadError: api_key 为空且 api_key_env 对应环境变量未设置或为空时。
    """
    api_key = config.resolve_api_key()
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.base_url or None,
        timeout=config.timeout,
        max_retries=config.retry,
    )
    return OpenAIChatCompletionsModel(model=config.model, openai_client=client)


async def _extract_output_json(result: RunResult | RunResultStreaming) -> str:
    """``as_tool`` 输出提取器：保证工具返回值始终为 JSON 字符串。

    Agent 以纯文本返回（``output_type=None``），字符串输出先剥离
    Markdown 代码围栏提取 JSON 对象；无法结构化时回退为
    ``{"output": ...}`` 文本包装。
    """
    output = result.final_output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False, default=str)
    if isinstance(output, str):
        obj = extract_json_object(output)
        if obj is not None:
            return json.dumps(obj, ensure_ascii=False)
    return json.dumps({"output": "" if output is None else str(output)}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 内部工具构建
# ---------------------------------------------------------------------------
def build_code_tools(
    context: ProjectContext | None = None,
    *,
    sandbox_runner: SandboxRunner | None = None,
    settings: Settings | None = None,
) -> dict[str, FunctionTool]:
    """构造代码 Agent 的全部内部工具（以工具名索引的字典）。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        sandbox_runner: 沙箱执行器；未提供时按 context/settings 新建。
        settings: 生效配置；未提供时取 context.config。

    Returns:
        ``{工具名: FunctionTool}``，包含 write_file / read_file /
        run_in_sandbox / write_generated_tool 四个工具。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    sandbox = sandbox_runner or SandboxRunner(
        context=resolved_context, settings=resolved_settings
    )
    sandbox_root = _sandbox_root(resolved_context)

    @function_tool
    def write_file(ctx: RunContextWrapper[Any], path: str, content: str) -> str:
        """在项目沙箱目录内写入（覆盖）一个文本文件。

        Args:
            path: 相对项目沙箱根目录的文件路径；禁止沙箱外路径。
            content: 要写入的完整文本内容（UTF-8）。

        Returns:
            JSON 字符串：成功为 ``{"status": "ok", "path", "created", "bytes",
            "diff"}``；失败为 ``{"status": "error", "error"}``。
        """
        try:
            target = _resolve_in_sandbox(sandbox_root, path)
            existed = target.is_file()
            old_text = ""
            if existed:
                old_text = target.read_text(encoding="utf-8", errors="replace")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            rel_path = target.relative_to(sandbox_root).as_posix()
            diff_text = (
                _build_unified_diff(old_text, content, rel_path)
                if existed
                else f"（新建文件 {rel_path}，{len(content.splitlines())} 行）"
            )
            return _ok(
                path=rel_path,
                created=not existed,
                bytes=len(content.encode("utf-8")),
                diff=diff_text,
            )
        except (ValueError, OSError) as exc:
            return _error(f"写入失败：{type(exc).__name__}: {exc}", path=path)

    @function_tool
    def read_file(ctx: RunContextWrapper[Any], path: str) -> str:
        """读取项目沙箱目录内的一个文本文件。

        Args:
            path: 相对项目沙箱根目录的文件路径；禁止沙箱外路径。

        Returns:
            JSON 字符串：成功为 ``{"status": "ok", "path", "content", "lines",
            "bytes"}``；失败为 ``{"status": "error", "error"}``。
        """
        try:
            target = _resolve_in_sandbox(sandbox_root, path)
            if not target.is_file():
                return _error(f"文件不存在：{path}", path=path)
            text = target.read_text(encoding="utf-8", errors="replace")
            return _ok(
                path=target.relative_to(sandbox_root).as_posix(),
                content=text,
                lines=len(text.splitlines()),
                bytes=len(text.encode("utf-8")),
            )
        except (ValueError, OSError) as exc:
            return _error(f"读取失败：{type(exc).__name__}: {exc}", path=path)

    @function_tool
    async def run_in_sandbox(ctx: RunContextWrapper[Any], code: str, task_id: str) -> str:
        """在验证型沙箱中以独立子进程运行一段 Python 代码并返回结构化结果。

        执行目录为 ``sandbox/task_{task_id}/run_{run_id}/``；stdout、stderr 与
        元数据均落盘留痕，超时（默认 sandbox_timeout）后强制终止。

        Args:
            code: 待执行的 Python 源码（完整可运行脚本）。
            task_id: 任务编号；只允许字母/数字/下划线/连字符（长度 1-64）。

        Returns:
            JSON 字符串：ExecutionResult 的完整序列化结果；沙箱拒绝执行时
            返回 ``{"status": "error", "error"}``。
        """
        try:
            run_id = f"code-{uuid.uuid4().hex[:10]}"
            result = await sandbox.execute(code, task_id, run_id)
            return result.model_dump_json()
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

            GENERATED_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
            entry_file = f"{tool_name}.py"
            code_path = GENERATED_TOOLS_DIR / entry_file
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
            mcp_schema_path = GENERATED_TOOLS_DIR / f"{tool_name}.mcp_schema.json"
            mcp_schema_path.write_text(
                json.dumps(mcp_schema, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

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
            )
            registry = ToolRegistry()
            if GENERATED_REGISTRY_PATH.is_file():
                registry.load_generated_registry(GENERATED_REGISTRY_PATH)
            registry.register_generated_tool(manifest)
            registry.save_generated_registry(GENERATED_REGISTRY_PATH)
            return _ok(
                code_path=str(code_path),
                mcp_schema_path=str(mcp_schema_path),
                manifest=manifest.model_dump(mode="json"),
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
    sandbox_runner: SandboxRunner | None = None,
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
    resolved_model = model or _build_chat_model(load_agent_api_config(API_CONFIG_PATH))
    tools = build_code_tools(
        resolved_context, sandbox_runner=sandbox_runner, settings=resolved_settings
    )
    return Agent(
        name=AGENT_NAME,
        instructions=_load_prompt(),
        tools=list(tools.values()),
        model=resolved_model,
        # output_type 保持 None（纯文本返回）：千问等兼容接口忽略
        # response_format=json_schema 时 SDK 严格校验会抛 ModelBehaviorError，
        # 改由调用方手动剥离围栏 + CodeAgentOutput 校验。
    )


def get_code_agent_as_tool(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
    sandbox_runner: SandboxRunner | None = None,
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
    return agent.as_tool(
        tool_name=AGENT_NAME,
        tool_description=_TOOL_DESCRIPTION,
        custom_output_extractor=_extract_output_json,
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
    sandbox_runner: SandboxRunner | None = None,
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
    return agent_to_mcp_server(
        agent,
        AGENT_NAME,
        _TOOL_DESCRIPTION,
        CodeAgentInput,
        CodeAgentOutput,
        max_turns=resolved_settings.max_model_turns,
    )

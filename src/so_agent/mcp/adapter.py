"""MCP 适配器：Agent / OpenAI Agents SDK 与 MCP 之间的双向转换。

提供三个适配函数：

- ``agent_to_mcp_server(agent, tool_name, description, input_model, output_model)``：
  把一个 OpenAI Agents SDK 的 ``Agent`` 包装为 ``MCPToolServer``——
  inputSchema 取 ``input_model.model_json_schema()``；调用时先按
  ``input_model`` 校验参数，再以 JSON 文本作为输入运行 Agent，
  输出统一提取为 JSON 文本（``output_model`` 用于结构化校验与序列化）。
- ``mcp_server_to_agent_tool(server, tool_name)``：把 MCP Server 上的一个
  工具反向包装为 SDK 的 ``FunctionTool``，使主管 Agent 可以通过标准
  SDK 调用（工具名、描述、参数 schema 均取自 MCP 暴露契约）。
- ``generated_tool_to_mcp_server(tool_path, manifest)``：把
  generated_tools 中的动态生成工具包装为 ``MCPToolServer``（惰性加载
  工具源码模块并调用其 ``run(**arguments)`` 入口）。

全局约束：适配器不引入业务层依赖（models 除外）；所有返回值均为
JSON 文本，保持"Agent 间通信严格 JSON"。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from agents import FunctionTool, Runner
from agents.exceptions import MaxTurnsExceeded
from agents.tool_context import ToolContext
from pydantic import BaseModel, ValidationError

from so_agent.config import Settings, get_settings
from so_agent.json_utils import extract_json_object
from so_agent.mcp.protocol import (
    METHOD_TOOLS_CALL,
    MCPProtocolError,
    MCPRequest,
    MCPToolSchema,
    extract_result_text,
)
from so_agent.mcp.server import MCPServer, MCPToolServer, stringify
from so_agent.models import GeneratedToolManifest
from so_agent.turn_budget import current_model_turn_override

if TYPE_CHECKING:
    from so_agent.runtime.sandbox import SandboxBackend

# 生成工具模块加载命名空间前缀（避免与业务模块重名）
_GENERATED_MODULE_PREFIX = "so_agent_generated_tool_"

# 生成工具入口函数候选名（按序尝试）
_GENERATED_ENTRY_CANDIDATES = ("run", "main", "execute")


class MCPAdapterError(Exception):
    """MCP 适配器领域错误：契约不匹配、入口缺失等。"""


def _populate_missing_tool_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """为缺省参数 schema 补全最小合法形态（空对象参数）。"""
    if not isinstance(schema, dict) or not schema:
        return {"type": "object", "properties": {}}
    return dict(schema)


def _schema_from_manifest(manifest: GeneratedToolManifest) -> MCPToolSchema:
    """从生成工具清单构造 MCP 暴露契约（宽容解析 mcp_schema 字段）。"""
    raw = dict(manifest.mcp_schema or {})
    input_schema = raw.get("inputSchema")
    if not isinstance(input_schema, dict) or not input_schema:
        input_schema = manifest.input_schema
    return MCPToolSchema(
        name=manifest.tool_name,
        description=str(raw.get("description") or manifest.description or ""),
        inputSchema=_populate_missing_tool_schema(input_schema),
    )


def _truncate_json_strings(value: Any, limit: int) -> Any:
    """递归截断 JSON 中的超长字符串（保留结构，避免破坏契约）。"""
    if limit <= 0:
        return value
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + f"\n...[已截断，原文 {len(value)} 字符]"
    if isinstance(value, dict):
        return {key: _truncate_json_strings(item, limit) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_json_strings(item, limit) for item in value]
    return value


def truncate_output_text(text: str, limit: int) -> str:
    """按字符上限截断 JSON 文本中的字符串字段（非法 JSON 直接截断）。"""
    if limit <= 0 or len(text) <= limit:
        return text
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text[:limit] + f"\n...[已截断，原文 {len(text)} 字符]"
    return json.dumps(
        _truncate_json_strings(payload, limit), ensure_ascii=False
    )


def _build_agent_input(
    payload_data: dict[str, Any],
    input_decorator: Callable[[dict[str, Any]], dict[str, Any] | None] | None,
    extra_note: str | None,
) -> str:
    """构造嵌套 Agent 输入：按需重新注入运行时信息并追加续轮说明。

    续轮发生在同一次工具调用内部，重跑前必须重新执行 ``input_decorator``，
    否则注入的沙箱产物索引仍是中断前的旧快照。
    """
    decorated = input_decorator(payload_data) if input_decorator is not None else None
    base = decorated if isinstance(decorated, dict) else payload_data
    if extra_note:
        base = dict(base)
        description = base.get("task_description")
        if isinstance(description, str):
            base["task_description"] = description + extra_note
        else:
            base["continuation"] = extra_note
    return json.dumps(base, ensure_ascii=False, default=str)


def _continuation_note(
    progress: dict[str, Any],
    previous_limit: int,
    new_limit: int,
    *,
    resumed: bool = False,
) -> str:
    """构造续轮说明：说明中断原因、已完成的工具动作与"不要重做"约束。

    ``resumed=True`` 表示同一会话历史已保留（SDK Session 续跑），
    此时不需要提示模型重新读取沙箱文件，只需禁止重复动作。
    """
    tools = [str(name) for name in (progress.get("successful_tools") or [])[:6]]
    lines = [
        "",
        "",
        f"[续轮] 上一轮达到回合上限（{previous_limit}）被中断，"
        f"本次上限已放宽到 {new_limit} 轮。",
    ]
    if resumed:
        lines.append(
            "对话历史与工具结果均已保留：不要重复已完成的读取、写入或执行，"
            "直接从中断处继续，补齐剩余步骤并按约定返回结构化结果。"
        )
    else:
        lines.append(
            "沙箱中可能已存在上一轮写好的文件：先用 read_file 确认内容，"
            "能复用就复用，不要从头重做。"
        )
    if tools:
        lines.append(f"上一轮已成功调用的工具：{'、'.join(tools)}。")
    preview = str(progress.get("last_success_preview") or "").strip()
    if preview:
        lines.append(f"最近一次工具输出摘要：{preview[:200]}")
    if not resumed:
        lines.append("请从中断处继续，补齐剩余步骤并按约定返回结构化结果。")
    return "\n".join(lines)


def _record_usage_safely(
    usage_recorder: Callable[[Any], None] | None, result: Any
) -> None:
    """补记一次（可能失败的）运行的用量；失败不影响主流程。"""
    if usage_recorder is None or result is None:
        return
    try:
        usage_recorder(result)
    except Exception:  # 用量上报失败不得影响后续续轮
        pass


def _emit_turn_extension(
    event_sink: Callable[[str, dict[str, Any]], None] | None,
    tool_name: str,
    previous_limit: int,
    new_limit: int,
    extension_count: int,
    progress: dict[str, Any],
    *,
    mode: str = "restart",
) -> None:
    """记录一次回合续期；写入失败不影响续轮。"""
    if event_sink is None:
        return
    # 延迟导入：runtime 包初始化会反向依赖本模块，模块级导入形成环
    from so_agent.runtime.events import EVENT_TURN_BUDGET_EXTENDED

    try:
        event_sink(
            EVENT_TURN_BUDGET_EXTENDED,
            {
                "tool_name": tool_name,
                "previous_max_turns": previous_limit,
                "new_max_turns": new_limit,
                "extension_count": extension_count,
                "tool_successes": int(progress.get("tool_successes", 0) or 0),
                "tool_failures": int(progress.get("tool_failures", 0) or 0),
                "mode": mode,
            },
        )
    except Exception:
        return


def _make_continuation_session() -> Any | None:
    """创建单次工具调用内的内存会话；SDK 不可用时返回 None。

    会话由 SDK 在每回合结束与运行开始时落盘，因此回合触顶后带着同一
    会话续跑即可保留工具结果，不必把整份载荷重新发一遍。
    """
    try:
        from agents.memory import SQLiteSession

        return SQLiteSession(f"code-agent-{uuid.uuid4().hex}")
    except Exception:
        return None


async def _session_has_items(session: Any) -> bool:
    """判断续跑会话里是否已有可复用的历史（没有则回退完整重建）。"""
    if session is None:
        return False
    try:
        items = await session.get_items()
        return bool(items)
    except Exception:
        return False


def _output_to_text(output: Any, output_model: type[BaseModel] | None = None) -> str:
    """把 Agent 最终输出规范化为 JSON 文本。

    Agent 均以 ``output_type=None`` 纯文本返回（兼容忽略
    ``response_format=json_schema`` 的 OpenAI 兼容接口），字符串输出
    先剥离 Markdown 代码围栏提取 JSON 对象，再按 ``output_model``
    手动校验并序列化；无法结构化时回退为 ``{"output": ...}`` 包装。
    """
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if output_model is not None and isinstance(output, dict):
        try:
            return output_model.model_validate(output).model_dump_json()
        except ValidationError:
            pass
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False, default=str)
    if isinstance(output, str) and output_model is not None:
        obj = extract_json_object(output)
        if obj is not None:
            try:
                return output_model.model_validate(obj).model_dump_json()
            except ValidationError:
                pass
    return json.dumps(
        {"output": "" if output is None else str(output)}, ensure_ascii=False
    )


def agent_to_mcp_server(
    agent: Any,
    tool_name: str,
    description: str,
    input_model: type[BaseModel],
    output_model: type[BaseModel],
    *,
    max_turns: int | None = None,
    usage_recorder: Callable[[Any], None] | None = None,
    output_max_chars: int = 0,
    settings: Settings | None = None,
    activity_hook: Callable[[str], None] | None = None,
    result_validator: Callable[[Any, dict[str, Any]], Any] | None = None,
    input_decorator: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    run_config: Any = None,
    event_sink: Callable[[str, dict[str, Any]], None] | None = None,
    turn_extension: bool = False,
) -> MCPToolServer:
    """把 OpenAI Agents SDK 的 Agent 包装为 MCP Server（单工具）。

    嵌套 Agent 以流式方式运行（任务 80）：保留首事件 / 模型空闲 /
    工具执行三类存活窗口，默认不设单次调用总预算；``activity_hook``
    把嵌套流事件上报给上层 pipeline 看门狗续时，避免父级在子 Agent
    正常工作时误判“无进展”。

    Args:
        agent: SDK ``Agent`` 实例（或其等价物，含 ``name`` 属性）。
        tool_name: MCP 暴露的工具名（同时作为服务器名）。
        description: 工具用途说明（写入 MCPToolSchema）。
        input_model: 输入契约模型（Pydantic）；其 JSON Schema 作为
            inputSchema，调用时用于参数校验。
        output_model: 输出契约模型（Pydantic）；用于把 Agent 输出校验并
            序列化为结构化 JSON 文本。
        max_turns: 运行 Agent 的最大模型回合数；None 时取生效配置的
            ``max_model_turns``。
        usage_recorder: 可选的 token 用量回调；嵌套 Agent 的模型调用
            发生在工具内部，控制层拿不到 ``RunResult``，由本回调上报。
        output_max_chars: 回传 JSON 中单个字符串字段的字符上限（<=0
            不截断）；防止嵌套 Agent 的超长产出放大父级上下文。
        settings: 生效配置；None 时取进程内全局配置。
        activity_hook: 可选活动回调；嵌套 Agent 的流事件到达时调用，
            用于上层 pipeline 看门狗续时。
        result_validator: 可选输出后置校验器，接收
            ``(agent 原始输出, 输入参数字典)``，返回替换后的输出；
            返回 None 表示保持原输出。用于把确定性验收（文件是否真实
            落地、执行结果是否真实存在）放在工具调用内部，避免模型
            自述成功穿透到主管上下文。
        input_decorator: 可选输入装饰器，接收已校验的输入字典，返回
            增强后的字典（返回 None 表示不修改）。用于在调用时刻注入
            运行时信息（如沙箱产物索引），不改变 MCP 暴露的参数契约。
        run_config: 可选的 SDK ``RunConfig``（如模型输入裁剪过滤器）；
            None 表示使用 SDK 默认配置。
        event_sink: 可选的事件写入回调 ``(event_type, payload)``，用于
            记录回合续期等工具内部治理事件；None 表示不记录。
        turn_extension: 是否启用回合续期（默认关闭）：达到回合上限时，
            若本轮已有成功的工具调用，则在同一工具调用内放宽上限继续，
            而不是直接把 ``MaxTurnsExceeded`` 抛给上层整体重派。

    Returns:
        ``MCPToolServer``：含单个工具的 MCP Server，handler 为 async。

    Raises:
        MCPAdapterError: input_model / output_model 类型非法时。
    """
    for label, model in (("input_model", input_model), ("output_model", output_model)):
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise MCPAdapterError(f"{label} 必须是 Pydantic BaseModel 子类")

    server = MCPToolServer(name=tool_name)
    schema = MCPToolSchema(
        name=tool_name,
        description=description,
        inputSchema=input_model.model_json_schema(),
    )

    async def _handler(arguments: dict[str, Any]) -> str:
        """校验参数 → 流式运行 Agent → 输出提取为 JSON 文本。"""
        try:
            payload = input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ValueError(
                f"参数不符合 {input_model.__name__} 契约：{exc}"
            ) from exc
        resolved_settings = settings if settings is not None else get_settings()
        # 延迟导入：workflow_support 依赖 runtime.tool_registry，而后者
        # 依赖本适配器；模块级导入会形成循环。
        from so_agent.workflow_support import (
            make_tool_timeout_resolver,
            run_agent_streamed,
        )

        payload_data = payload.model_dump(mode="json")
        agent_input = _build_agent_input(payload_data, input_decorator, None)

        base_turns = int(
            max_turns
            if max_turns is not None
            else resolved_settings.max_model_turns
        )
        # 控制层对合并步骤的作用域覆盖只能抬高上限，不能降低
        effective_turns = max(base_turns, current_model_turn_override() or 0)
        extension_max = 0
        extension_budget = 0
        if turn_extension:
            extension_budget = int(
                getattr(resolved_settings, "code_agent_turn_extension_budget", 0) or 0
            )
            extension_max = int(
                getattr(resolved_settings, "code_agent_turn_extension_max", 0) or 0
            )
            if extension_budget <= 0:
                extension_max = 0
        turn_limit = effective_turns
        progress: dict[str, Any] = {}
        extensions = 0
        session = _make_continuation_session() if extension_max else None
        while True:
            try:
                result = await run_agent_streamed(
                    agent,
                    input=agent_input,
                    max_turns=turn_limit,
                    first_event_timeout=float(resolved_settings.first_event_timeout),
                    model_idle_timeout=float(resolved_settings.model_idle_timeout),
                    tool_execution_timeout=float(resolved_settings.tool_execution_timeout),
                    subagent_tool_execution_timeout=float(
                        resolved_settings.subagent_tool_execution_timeout
                    ),
                    agent_total_timeout=float(resolved_settings.agent_total_timeout),
                    tool_timeout_resolver=make_tool_timeout_resolver(resolved_settings),
                    activity_hook=activity_hook,
                    runner=Runner,
                    run_config=run_config,
                    progress_state=progress,
                    session=session,
                )
                break
            except MaxTurnsExceeded as exc:
                # 失败运行已产生的 token 仍要入账（部分响应挂在异常上），
                # 否则失败尝试会成为预算与台账的盲区
                _record_usage_safely(
                    usage_recorder, getattr(exc, "partial_result", None)
                )
                if (
                    extensions >= extension_max
                    or not progress.get("tool_successes")
                ):
                    raise
                extensions += 1
                previous_limit = turn_limit
                turn_limit += extension_budget
                resumed = await _session_has_items(session)
                note = _continuation_note(
                    progress, previous_limit, turn_limit, resumed=resumed
                )
                agent_input = (
                    note
                    if resumed
                    else _build_agent_input(payload_data, input_decorator, note)
                )
                _emit_turn_extension(
                    event_sink,
                    tool_name,
                    previous_limit,
                    turn_limit,
                    extensions,
                    progress,
                    mode="session" if resumed else "restart",
                )
        if usage_recorder is not None:
            try:
                usage_recorder(result)
            except Exception:  # 用量上报失败不得影响工具结果
                pass
        final_output = result.final_output
        if result_validator is not None:
            validated = result_validator(
                final_output, payload.model_dump(mode="json")
            )
            if validated is not None:
                final_output = validated
        text = _output_to_text(final_output, output_model)
        return truncate_output_text(text, output_max_chars)

    server.register_tool(schema, _handler)
    return server


def mcp_server_to_agent_tool(
    server: MCPServer,
    tool_name: str,
    *,
    before_invoke: Callable[[str, dict[str, Any]], str | None] | None = None,
) -> FunctionTool:
    """把 MCP Server 上的一个工具包装为 SDK 的 FunctionTool。

    主管 Agent 通过该 FunctionTool 以标准 SDK 方式调用 MCP 工具；
    调用内部构造 tools/call 请求并交给服务器路由，返回值统一为 JSON 文本。

    Args:
        server: MCP Server 实例。
        tool_name: 已登记的工具名。
        before_invoke: 可选的调用前守卫；返回非 None 的 JSON 文本时
            直接作为工具结果返回，不进入 MCP 服务器（用于控制层按
            当前阶段确定性拒绝调用，而非依赖提示词约束）。

    Returns:
        ``FunctionTool``：name/description/params_json_schema 取自工具的
        MCP 暴露契约；``on_invoke_tool`` 为 async 调用桥。

    Raises:
        MCPAdapterError: 工具未在服务器登记时。
    """
    schema = server.get_tool_schema(tool_name)
    if schema is None:
        raise MCPAdapterError(
            f"服务器 {server.name!r} 未登记工具 {tool_name!r}，无法转换"
        )

    async def _on_invoke(ctx: ToolContext[Any], args_json: str) -> str:
        """SDK 调用桥：JSON 参数 → MCP tools/call → JSON 文本结果。"""
        try:
            arguments = json.loads(args_json) if args_json and args_json.strip() else {}
        except json.JSONDecodeError as exc:
            return json.dumps(
                {"status": "error", "error": f"工具参数不是合法 JSON：{exc}"},
                ensure_ascii=False,
            )
        if not isinstance(arguments, dict):
            return json.dumps(
                {"status": "error", "error": "工具参数必须是 JSON 对象"},
                ensure_ascii=False,
            )
        if before_invoke is not None:
            rejection = before_invoke(tool_name, arguments)
            if rejection is not None:
                return rejection
        request = MCPRequest(
            method=METHOD_TOOLS_CALL,
            params={"name": tool_name, "arguments": arguments},
            id=f"agent-tool-{tool_name}",
        )
        response = await server.handle_request(request)
        if response.error is not None:
            return json.dumps(
                {
                    "status": "error",
                    "error": f"[{response.error.code}] {response.error.message}",
                },
                ensure_ascii=False,
            )
        return extract_result_text(response)

    return FunctionTool(
        name=tool_name,
        description=schema.description,
        params_json_schema=_populate_missing_tool_schema(schema.inputSchema),
        on_invoke_tool=_on_invoke,
        strict_json_schema=False,
    )


def _is_within(root: Path, target: Path) -> bool:
    """判断 target 是否位于 root 之内（Windows 大小写不敏感）。"""
    root_text = os.path.normcase(str(root))
    target_text = os.path.normcase(str(target))
    return target_text == root_text or target_text.startswith(root_text + os.sep)


def _ensure_within(root: str | Path, target: str | Path, tool_name: str) -> None:
    """校验生成工具入口位于根目录内，否则抛 MCPAdapterError（防路径穿越）。"""
    root_path = Path(root).expanduser().resolve()
    target_path = Path(target).expanduser().resolve()
    if not _is_within(root_path, target_path):
        raise MCPAdapterError(
            f"生成工具入口越界：{tool_name!r} 的入口 {target_path} "
            f"不在生成工具目录 {root_path} 之内（拒绝加载）"
        )


def _load_generated_tool_module(tool_path: str | Path, module_name: str) -> Any:
    """惰性加载生成工具源码模块（每次调用重新加载，保证源码更新即时生效）。

    Raises:
        MCPAdapterError: 入口文件不存在或无法加载时。
    """
    path = Path(tool_path)
    if not path.is_file():
        raise MCPAdapterError(f"生成工具入口文件不存在：{path}")
    safe_name = f"{_GENERATED_MODULE_PREFIX}{module_name}"
    spec = importlib.util.spec_from_file_location(safe_name, path)
    if spec is None or spec.loader is None:
        raise MCPAdapterError(f"生成工具无法构造模块加载器：{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[safe_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise MCPAdapterError(
            f"生成工具加载失败：{path}（{type(exc).__name__}: {exc}）"
        ) from exc
    return module


def _resolve_generated_entry(module: Any, tool_name: str) -> Any:
    """在生成工具模块中解析入口函数（run → 与工具同名 → main → execute）。"""
    for candidate in (tool_name, *_GENERATED_ENTRY_CANDIDATES):
        func = getattr(module, candidate, None)
        if callable(func):
            return func
    return None


def generated_tool_to_mcp_server(
    tool_path: str | Path,
    manifest: GeneratedToolManifest,
    *,
    sandbox: "SandboxBackend | None" = None,
    generated_tools_root: str | Path | None = None,
    on_use: Callable[[str, bool], None] | None = None,
) -> MCPToolServer:
    """把动态生成工具包装为 MCP Server（单工具）。

    工具在**沙箱子进程**执行（而非主进程 importlib 加载），返回值强制 JSON。

    Args:
        tool_path: 生成工具入口源码文件路径（``generated_tools/{name}.py``）。
        manifest: 生成工具清单（提供名称、描述与参数 schema）。
        sandbox: 沙箱执行器；未提供时按默认配置惰性构造。
        generated_tools_root: 生成工具根目录；提供时校验 ``tool_path`` 位于
            其内（防路径穿越加载任意源码）。
        on_use: 调用结束回调 ``(tool_name, success)``，用于使用统计与
            生命周期维护；回调异常不影响工具执行结果。

    Returns:
        ``MCPToolServer``：handler 在沙箱子进程执行入口函数
        （``run(**arguments)``），返回值为 JSON 文本。

    Raises:
        MCPAdapterError: manifest 类型非法或 ``tool_path`` 越界时。
    """
    if not isinstance(manifest, GeneratedToolManifest):
        raise MCPAdapterError(
            f"manifest 必须是 GeneratedToolManifest，实际为 {type(manifest).__name__}"
        )
    if generated_tools_root is not None:
        _ensure_within(generated_tools_root, tool_path, manifest.tool_name)

    server = MCPToolServer(name=manifest.tool_name)
    schema = _schema_from_manifest(manifest)

    async def _handler(arguments: dict[str, Any]) -> str:
        """在沙箱子进程执行入口函数（返回值强制 JSON 文本）。

        加载失败/缺入口/执行失败抛 ``GeneratedToolExecutionError``（非
        ValueError 子类），由 MCP Server 映射为 INTERNAL_ERROR（服务器侧
        工具错误，而非参数校验失败）。
        """
        # 惰性导入：避免 mcp.adapter ↔ runtime 的模块级循环依赖
        from so_agent.integrity import file_matches_hash
        from so_agent.runtime.generated_tool_runner import (
            GeneratedToolExecutionError,
            run_generated_tool_in_sandbox,
        )
        from so_agent.runtime.sandbox import create_sandbox

        if manifest.content_hash and not file_matches_hash(
            tool_path, manifest.content_hash
        ):
            raise GeneratedToolExecutionError(
                f"生成工具 {manifest.tool_name!r} 源码哈希不匹配（可能被篡改），拒绝执行"
            )

        runner = sandbox or create_sandbox()
        from so_agent.runtime.tool_store import stage_tool_for_sandbox

        execution_path = await stage_tool_for_sandbox(runner, tool_path)
        try:
            result = await run_generated_tool_in_sandbox(
                runner, execution_path, manifest.tool_name, arguments
            )
        except Exception:
            if on_use is not None:
                try:
                    on_use(manifest.tool_name, False)
                except Exception:
                    pass
            raise
        if on_use is not None:
            try:
                on_use(manifest.tool_name, True)
            except Exception:
                pass
        return json.dumps(result, ensure_ascii=False)

    server.register_tool(schema, _handler)
    return server


# 供外部构造最小合法 schema 的便捷引用
__all__ = [
    "MCPAdapterError",
    "MCPProtocolError",
    "agent_to_mcp_server",
    "generated_tool_to_mcp_server",
    "mcp_server_to_agent_tool",
]

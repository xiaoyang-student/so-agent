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
import sys
from pathlib import Path
from typing import Any

from agents import FunctionTool, Runner
from agents.tool_context import ToolContext
from pydantic import BaseModel, ValidationError

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
) -> MCPToolServer:
    """把 OpenAI Agents SDK 的 Agent 包装为 MCP Server（单工具）。

    Args:
        agent: SDK ``Agent`` 实例（或其等价物，含 ``name`` 属性）。
        tool_name: MCP 暴露的工具名（同时作为服务器名）。
        description: 工具用途说明（写入 MCPToolSchema）。
        input_model: 输入契约模型（Pydantic）；其 JSON Schema 作为
            inputSchema，调用时用于参数校验。
        output_model: 输出契约模型（Pydantic）；用于把 Agent 输出校验并
            序列化为结构化 JSON 文本。
        max_turns: 运行 Agent 的最大模型回合数；None 表示使用 SDK 默认。

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
        """校验参数 → 运行 Agent → 输出提取为 JSON 文本。"""
        try:
            payload = input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ValueError(
                f"参数不符合 {input_model.__name__} 契约：{exc}"
            ) from exc
        run_kwargs: dict[str, Any] = {}
        if max_turns is not None:
            run_kwargs["max_turns"] = int(max_turns)
        result = await Runner.run(agent, input=payload.model_dump_json(), **run_kwargs)
        return _output_to_text(result.final_output, output_model)

    server.register_tool(schema, _handler)
    return server


def mcp_server_to_agent_tool(server: MCPServer, tool_name: str) -> FunctionTool:
    """把 MCP Server 上的一个工具包装为 SDK 的 FunctionTool。

    主管 Agent 通过该 FunctionTool 以标准 SDK 方式调用 MCP 工具；
    调用内部构造 tools/call 请求并交给服务器路由，返回值统一为 JSON 文本。

    Args:
        server: MCP Server 实例。
        tool_name: 已登记的工具名。

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
    tool_path: str | Path, manifest: GeneratedToolManifest
) -> MCPToolServer:
    """把动态生成工具包装为 MCP Server（单工具）。

    Args:
        tool_path: 生成工具入口源码文件路径（``generated_tools/{name}.py``）。
        manifest: 生成工具清单（提供名称、描述与参数 schema）。

    Returns:
        ``MCPToolServer``：handler 惰性加载源码模块并调用其入口函数
        （``run(**arguments)``），返回值规范化为 JSON 文本。

    Raises:
        MCPAdapterError: manifest 类型非法时。
    """
    if not isinstance(manifest, GeneratedToolManifest):
        raise MCPAdapterError(
            f"manifest 必须是 GeneratedToolManifest，实际为 {type(manifest).__name__}"
        )

    server = MCPToolServer(name=manifest.tool_name)
    schema = _schema_from_manifest(manifest)

    def _handler(arguments: dict[str, Any]) -> str:
        """加载源码并调用入口函数（同步；生成工具为纯计算函数）。"""
        module = _load_generated_tool_module(tool_path, manifest.tool_name)
        func = _resolve_generated_entry(module, manifest.tool_name)
        if func is None:
            raise MCPAdapterError(
                f"生成工具 {manifest.tool_name!r} 缺少可调用入口"
                f"（需定义 run(...) 等函数）：{tool_path}"
            )
        try:
            outcome = func(**arguments)
        except TypeError as exc:
            raise ValueError(f"生成工具参数不匹配：{exc}") from exc
        return stringify(outcome)

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

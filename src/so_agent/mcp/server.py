"""MCP Server 基类（工具注册、发现与请求路由）。

职责与行为约定：

- ``register_tool(schema, handler)``：登记一个工具（MCPToolSchema + 处理函数）；
  handler 签名为 ``handler(arguments: dict) -> Any``，支持同步与 async；
- ``list_tools()``：返回全部已登记工具的 MCPToolSchema 列表；
- ``handle_request(request)``：按 JSON-RPC 方法路由（tools/list、tools/call），
  统一返回 MCPResponse；handler 抛 ValueError → INVALID_PARAMS，
  其他异常 → INTERNAL_ERROR；handler 返回值被包装为 MCP 标准
  content 文本块（BaseModel/dict/list 自动 JSON 序列化）；
- ``handle_request_sync``：同步便捷入口（无事件循环时可用）。

``MCPToolServer`` 是面向"具体 Agent 工具/生成工具"的 MCP Server 子类。

全局约束：handler 的返回值必须是可 JSON 序列化的（字符串优先），
保证所有 Agent 之间通信严格为 JSON。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from so_agent.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    MCPError,
    MCPProtocolError,
    MCPRequest,
    MCPResponse,
    MCPToolSchema,
    text_content,
)

# 工具处理函数：接收 arguments（JSON 对象），返回任意可序列化结果；
# 允许是同步函数或 async 函数。
MCPHandler = Callable[[dict[str, Any]], Any]


def stringify(payload: Any) -> str:
    """把 handler 返回值规范化为 JSON 文本。

    - Pydantic 模型 → ``model_dump_json``；
    - 字符串 → 原样返回（视为已序列化的 JSON 文本）；
    - 其他对象 → ``json.dumps``（ensure_ascii=False）。
    """
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, default=str)


class MCPServer:
    """MCP Server 基类（进程内实现）。"""

    def __init__(self, name: str) -> None:
        """初始化服务器。

        Args:
            name: 服务器名称（需唯一，供客户端按名寻址）。

        Raises:
            MCPProtocolError: 名称为空时。
        """
        if not isinstance(name, str) or not name.strip():
            raise MCPProtocolError("MCP Server 名称不能为空")
        self._name = name.strip()
        self._tools: dict[str, tuple[MCPToolSchema, MCPHandler]] = {}

    @property
    def name(self) -> str:
        """返回服务器名称。"""
        return self._name

    # ------------------------------------------------------------------
    # 注册与发现
    # ------------------------------------------------------------------
    def register_tool(self, schema: MCPToolSchema, handler: MCPHandler) -> None:
        """登记（或覆盖）一个工具。

        Args:
            schema: 工具的 MCP 暴露契约。
            handler: 处理函数（同步或 async），签名 ``handler(arguments)``。

        Raises:
            MCPProtocolError: schema 类型非法或 handler 不可调用时。
        """
        if not isinstance(schema, MCPToolSchema):
            raise MCPProtocolError(
                f"schema 必须是 MCPToolSchema，实际为 {type(schema).__name__}"
            )
        if not callable(handler):
            raise MCPProtocolError("handler 必须可调用")
        self._tools[schema.name] = (schema, handler)

    def has_tool(self, tool_name: str) -> bool:
        """判断工具是否已登记。"""
        return isinstance(tool_name, str) and tool_name in self._tools

    def get_tool_schema(self, tool_name: str) -> MCPToolSchema | None:
        """按名称返回工具的 MCP 暴露契约；不存在返回 None。"""
        entry = self._tools.get(tool_name)
        return entry[0] if entry is not None else None

    def list_tools(self) -> list[MCPToolSchema]:
        """返回全部已登记工具的 MCPToolSchema 列表（按登记顺序）。"""
        return [schema for schema, _handler in self._tools.values()]

    # ------------------------------------------------------------------
    # 请求路由
    # ------------------------------------------------------------------
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        """处理一条 MCP 请求（async 入口，支持 async handler）。

        Args:
            request: MCP 请求（JSON-RPC 2.0）。

        Returns:
            MCPResponse：成功携带 result（tools/list → {"tools": [...]}；
            tools/call → content 文本块）；失败携带标准错误码。
        """
        if not isinstance(request, MCPRequest):
            raise MCPProtocolError(
                f"request 必须是 MCPRequest，实际为 {type(request).__name__}"
            )
        if request.method == METHOD_TOOLS_LIST:
            return MCPResponse(
                id=request.id,
                result={
                    "tools": [
                        schema.model_dump(mode="json") for schema in self.list_tools()
                    ]
                },
            )
        if request.method == METHOD_TOOLS_CALL:
            return await self._handle_call(request)
        return MCPResponse(
            id=request.id,
            error=MCPError(
                code=METHOD_NOT_FOUND,
                message=f"不支持的方法：{request.method!r}（服务器 {self._name}）",
            ),
        )

    async def _handle_call(self, request: MCPRequest) -> MCPResponse:
        """路由 tools/call：参数校验 → 调用 handler → 包装结果。"""
        params = request.params if isinstance(request.params, dict) else {}
        tool_name = params.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=INVALID_PARAMS,
                    message="tools/call 缺少参数 name（工具名）",
                ),
            )
        entry = self._tools.get(tool_name.strip())
        if entry is None:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=METHOD_NOT_FOUND,
                    message=f"工具不存在：{tool_name!r}（服务器 {self._name}）",
                ),
            )
        _schema, handler = entry
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=INVALID_PARAMS,
                    message="tools/call 的 arguments 必须是 JSON 对象",
                ),
            )

        try:
            outcome = handler(dict(arguments))
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except ValueError as exc:
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=INVALID_PARAMS,
                    message=f"工具 {tool_name!r} 参数校验失败：{exc}",
                ),
            )
        except Exception as exc:  # handler 内部异常按服务端错误上报
            return MCPResponse(
                id=request.id,
                error=MCPError(
                    code=INTERNAL_ERROR,
                    message=f"工具 {tool_name!r} 执行异常：{type(exc).__name__}: {exc}",
                ),
            )

        if isinstance(outcome, MCPResponse):
            return outcome.model_copy(update={"id": request.id})
        return MCPResponse(id=request.id, result=text_content(stringify(outcome)))

    def handle_request_sync(self, request: MCPRequest) -> MCPResponse:
        """同步处理一条 MCP 请求（当前无事件循环时可用）。

        Raises:
            MCPProtocolError: 当前线程已有事件循环在运行时（应改用
                ``await handle_request(...)``）。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.handle_request(request))
        raise MCPProtocolError(
            "当前已有事件循环在运行，请使用 await server.handle_request(...)"
        )


class MCPToolServer(MCPServer):
    """面向具体 Agent 工具 / 生成工具的 MCP Server。"""


# 注册与调用辅助（供适配器复用）
def mcp_error_response(
    code: int, message: str, *, request_id: str | int | None = None
) -> MCPResponse:
    """构造携带标准错误的 MCPResponse（便捷函数）。"""
    return MCPResponse(id=request_id, error=MCPError(code=code, message=message))

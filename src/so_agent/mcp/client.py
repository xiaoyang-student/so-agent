"""MCP Client（进程内直连实现，首版不走网络）。

职责与行为约定：

- ``connect(server)``：注册一个 MCPServer（按 server.name 索引）；
- ``list_tools(server_name=None)``：发现工具——指定服务器时返回该服务器的
  MCPToolSchema 列表；不指定时聚合全部已连接服务器的工具（按服务器名排序）；
- ``call_tool(server_name, tool_name, params)``：调用工具（内部直接函数调用，
  构造 MCPRequest 并交给目标服务器路由），返回 MCPResponse；
- ``call_tool_sync``：同步便捷入口（无事件循环时可用）。

全局约束：客户端不持有工具实现，只做寻址与协议封装；所有调用结果为
JSON 兼容结构（MCPResponse）。
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from so_agent.mcp.protocol import (
    METHOD_TOOLS_CALL,
    MCPProtocolError,
    MCPRequest,
    MCPResponse,
    MCPToolSchema,
)
from so_agent.mcp.server import MCPServer


class MCPClientError(Exception):
    """MCP 客户端领域错误：服务器未连接等。"""


class MCPClient:
    """MCP 客户端（进程内直连多个 MCPServer）。"""

    def __init__(self) -> None:
        """初始化客户端（空连接集合）。"""
        self._servers: dict[str, MCPServer] = {}

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------
    def connect(self, server: MCPServer) -> None:
        """注册（或覆盖）一个 MCP Server（按其名称索引）。

        Raises:
            MCPClientError: server 类型非法时。
        """
        if not isinstance(server, MCPServer):
            raise MCPClientError(
                f"server 必须是 MCPServer，实际为 {type(server).__name__}"
            )
        self._servers[server.name] = server

    def disconnect(self, server_name: str) -> bool:
        """断开一个 MCP Server；未连接时返回 False。"""
        return self._servers.pop(server_name, None) is not None

    def get_server(self, server_name: str) -> MCPServer | None:
        """按名称返回已连接的服务器；未连接返回 None。"""
        return self._servers.get(server_name)

    def list_servers(self) -> list[str]:
        """返回全部已连接服务器名称（按字典序）。"""
        return sorted(self._servers)

    # ------------------------------------------------------------------
    # 工具发现与调用
    # ------------------------------------------------------------------
    def list_tools(self, server_name: str | None = None) -> list[MCPToolSchema]:
        """发现工具。

        Args:
            server_name: 指定服务器名称时仅返回该服务器的工具；
                为 None 时聚合全部已连接服务器的工具（按服务器名排序）。

        Raises:
            MCPClientError: 指定的服务器未连接时。
        """
        if server_name is not None:
            return self._require(server_name).list_tools()
        schemas: list[MCPToolSchema] = []
        for name in sorted(self._servers):
            schemas.extend(self._servers[name].list_tools())
        return schemas

    async def call_tool(
        self, server_name: str, tool_name: str, params: dict[str, Any]
    ) -> MCPResponse:
        """调用一个工具（内部直连服务器路由，不走网络）。

        Args:
            server_name: 目标服务器名称（connect 时注册的名称）。
            tool_name: 工具名称。
            params: 工具参数（JSON 对象，作为 arguments 传给 handler）。

        Returns:
            MCPResponse：成功携带 content 文本块结果；失败携带标准错误码
            （工具不存在 → METHOD_NOT_FOUND；参数非法 → INVALID_PARAMS）。

        Raises:
            MCPClientError: 目标服务器未连接时。
        """
        server = self._require(server_name)
        request = MCPRequest(
            method=METHOD_TOOLS_CALL,
            params={"name": tool_name, "arguments": dict(params or {})},
            id=f"call-{uuid.uuid4().hex[:12]}",
        )
        return await server.handle_request(request)

    def call_tool_sync(
        self, server_name: str, tool_name: str, params: dict[str, Any]
    ) -> MCPResponse:
        """同步调用一个工具（当前无事件循环时可用）。

        Raises:
            MCPClientError: 目标服务器未连接时。
            MCPProtocolError: 当前线程已有事件循环在运行时。
        """
        server = self._require(server_name)
        request = MCPRequest(
            method=METHOD_TOOLS_CALL,
            params={"name": tool_name, "arguments": dict(params or {})},
            id=f"call-{uuid.uuid4().hex[:12]}",
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(server.handle_request(request))
        raise MCPProtocolError(
            "当前已有事件循环在运行，请使用 await client.call_tool(...)"
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------
    def _require(self, server_name: str) -> MCPServer:
        """返回指定服务器；未连接时抛出 MCPClientError。"""
        server = self._servers.get(server_name)
        if server is None:
            raise MCPClientError(
                f"MCP Server 未连接：{server_name!r}（已连接：{self.list_servers()}）"
            )
        return server

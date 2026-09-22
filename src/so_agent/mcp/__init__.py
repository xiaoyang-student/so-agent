"""MCP（Model Context Protocol）基础设施包。

包含四层：

- ``protocol``：协议数据契约（MCPToolSchema / MCPRequest / MCPResponse /
  MCPError 与标准错误码常量）；
- ``server``：MCP Server 基类（注册/发现/请求路由，支持 async handler）
  与 ``MCPToolServer``；
- ``client``：MCP Client（进程内直连，工具发现与调用）；
- ``adapter``：双向适配器（Agent ↔ MCP Server ↔ SDK FunctionTool、
  生成工具 → MCP Server）。

全局约束：本包为叶子层，只依赖 pydantic / OpenAI Agents SDK / models，
不得引入 business 层（orchestrator / workflow / runtime）依赖，避免循环导入。
"""

from so_agent.mcp.adapter import (
    MCPAdapterError,
    agent_to_mcp_server,
    generated_tool_to_mcp_server,
    mcp_server_to_agent_tool,
)
from so_agent.mcp.client import MCPClient, MCPClientError
from so_agent.mcp.protocol import (
    CONTENT_TYPE_TEXT,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    JSONRPC_VERSION,
    METHOD_NOT_FOUND,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    PARSE_ERROR,
    MCPError,
    MCPProtocolError,
    MCPRequest,
    MCPResponse,
    MCPToolSchema,
    extract_result_text,
    text_content,
)
from so_agent.mcp.server import MCPServer, MCPToolServer, stringify

__all__ = [
    # protocol
    "CONTENT_TYPE_TEXT",
    "JSONRPC_VERSION",
    "PARSE_ERROR",
    "METHOD_NOT_FOUND",
    "INVALID_PARAMS",
    "INTERNAL_ERROR",
    "METHOD_TOOLS_LIST",
    "METHOD_TOOLS_CALL",
    "MCPProtocolError",
    "MCPToolSchema",
    "MCPRequest",
    "MCPError",
    "MCPResponse",
    "text_content",
    "extract_result_text",
    # server
    "MCPServer",
    "MCPToolServer",
    "stringify",
    # client
    "MCPClient",
    "MCPClientError",
    # adapter
    "MCPAdapterError",
    "agent_to_mcp_server",
    "mcp_server_to_agent_tool",
    "generated_tool_to_mcp_server",
]

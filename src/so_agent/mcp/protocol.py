"""MCP（Model Context Protocol）协议核心。

本模块定义框架内统一的 MCP 数据契约（JSON-RPC 2.0 子集）：

- ``MCPToolSchema``：工具暴露契约（name / description / inputSchema）；
- ``MCPRequest`` / ``MCPResponse`` / ``MCPError``：一次工具调用的
  请求、响应与错误结构（``jsonrpc`` 固定为 "2.0"）；
- 标准 JSON-RPC 2.0 错误码常量（PARSE_ERROR / METHOD_NOT_FOUND /
  INVALID_PARAMS / INTERNAL_ERROR）与 MCP 方法名（tools/list、tools/call）。

全局约束：所有结构都是纯数据（Pydantic 模型），可作为 JSON 在 Agent
之间传递与审计；本模块只依赖 pydantic，不得引入业务层依赖。
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

# JSON-RPC 2.0 协议版本标识
JSONRPC_VERSION = "2.0"

# 标准 JSON-RPC 2.0 错误码
PARSE_ERROR = -32700      # 请求不是合法 JSON / 结构无法解析
METHOD_NOT_FOUND = -32601  # 方法或工具不存在
INVALID_PARAMS = -32602    # 参数缺失或非法
INTERNAL_ERROR = -32603    # 服务端内部错误（handler 异常等）

# MCP 方法名
METHOD_TOOLS_LIST = "tools/list"
METHOD_TOOLS_CALL = "tools/call"

# tools/call 结果内容块类型（首版仅支持 text）
CONTENT_TYPE_TEXT = "text"


class MCPProtocolError(Exception):
    """MCP 协议领域错误：结构非法、方法不支持等。"""


class MCPToolSchema(BaseModel):
    """MCP 工具暴露契约（可被 tools/list 发现的元数据）。"""

    name: str = Field(description="工具名称（在所属服务器内唯一）")
    description: str = Field(default="", description="工具用途说明")
    inputSchema: dict[str, Any] = Field(
        default_factory=dict, description="输入参数的 JSON Schema"
    )


class MCPRequest(BaseModel):
    """MCP 请求（JSON-RPC 2.0 形态）。"""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    method: str = Field(description="方法名：tools/list 或 tools/call")
    params: dict[str, Any] = Field(default_factory=dict, description="方法参数")
    id: str | int | None = Field(default=None, description="请求标识（回传于响应）")


class MCPError(BaseModel):
    """MCP 错误结构（JSON-RPC 2.0 error 对象）。"""

    code: int = Field(description="标准错误码（如 -32601）")
    message: str = Field(description="错误说明")
    data: Any | None = Field(default=None, description="附加错误数据（可选）")


class MCPResponse(BaseModel):
    """MCP 响应（JSON-RPC 2.0 形态；result 与 error 至多其一）。"""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    result: Any | None = Field(default=None, description="成功结果")
    error: MCPError | None = Field(default=None, description="错误信息")
    id: str | int | None = Field(default=None, description="对应请求的标识")

    @property
    def ok(self) -> bool:
        """响应是否成功（无 error 即视为成功）。"""
        return self.error is None


def text_content(text: str, *, is_error: bool = False) -> dict[str, Any]:
    """构造 MCP tools/call 的标准结果载荷（content 文本块数组）。"""
    return {
        "content": [{"type": CONTENT_TYPE_TEXT, "text": text}],
        "isError": is_error,
    }


def extract_result_text(response: MCPResponse) -> str:
    """从 MCP 响应中提取文本结果。

    依次支持：content 文本块数组 → 顶层 text 字段 → 字符串结果 →
    其他结果对象（JSON 序列化）。响应携带 error 时返回空字符串。
    """
    if response.error is not None:
        return ""
    result = response.result
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list):
            parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == CONTENT_TYPE_TEXT
            ]
            if parts:
                return "\n".join(parts)
        text = result.get("text")
        if isinstance(text, str):
            return text
    return json.dumps(result, ensure_ascii=False, default=str)

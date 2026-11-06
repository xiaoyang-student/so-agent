"""单元测试：MCP 协议核心（protocol.py）。

覆盖：
- 标准错误码常量与 JSON-RPC 2.0 版本标识；
- MCPToolSchema / MCPRequest / MCPResponse / MCPError 的序列化与反序列化；
- text_content 结果载荷构造与 extract_result_text 各提取分支。
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from so_agent.mcp.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    JSONRPC_VERSION,
    METHOD_NOT_FOUND,
    METHOD_TOOLS_CALL,
    METHOD_TOOLS_LIST,
    PARSE_ERROR,
    MCPError,
    MCPRequest,
    MCPResponse,
    MCPToolSchema,
    extract_result_text,
    text_content,
)


class TestConstants:
    def test_error_codes_are_standard_jsonrpc(self):
        assert PARSE_ERROR == -32700
        assert METHOD_NOT_FOUND == -32601
        assert INVALID_PARAMS == -32602
        assert INTERNAL_ERROR == -32603

    def test_jsonrpc_version_and_methods(self):
        assert JSONRPC_VERSION == "2.0"
        assert METHOD_TOOLS_LIST == "tools/list"
        assert METHOD_TOOLS_CALL == "tools/call"


class TestToolSchema:
    def test_defaults(self):
        schema = MCPToolSchema(name="demo")
        assert schema.description == ""
        assert schema.inputSchema == {}

    def test_json_roundtrip(self):
        schema = MCPToolSchema(
            name="read_file",
            description="读取文件",
            inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        raw = schema.model_dump_json()
        payload = json.loads(raw)
        assert payload["name"] == "read_file"
        assert payload["inputSchema"]["required"] == ["path"]

        restored = MCPToolSchema.model_validate_json(raw)
        assert restored == schema


class TestRequestResponse:
    def test_request_defaults(self):
        request = MCPRequest(method=METHOD_TOOLS_LIST)
        assert request.jsonrpc == "2.0"
        assert request.params == {}
        assert request.id is None

    def test_request_roundtrip(self):
        request = MCPRequest(
            method=METHOD_TOOLS_CALL,
            params={"name": "csv_report", "arguments": {"rows": [1, 2]}},
            id=42,
        )
        restored = MCPRequest.model_validate_json(request.model_dump_json())
        assert restored == request
        assert restored.id == 42

    def test_request_rejects_wrong_jsonrpc(self):
        with pytest.raises(ValidationError):
            MCPRequest.model_validate(
                {"jsonrpc": "1.0", "method": METHOD_TOOLS_LIST}
            )

    def test_error_data_optional(self):
        error = MCPError(code=METHOD_NOT_FOUND, message="工具不存在")
        assert error.data is None
        with_data = MCPError(code=INVALID_PARAMS, message="bad", data={"k": 1})
        assert with_data.data == {"k": 1}

    def test_response_ok_flag(self):
        success = MCPResponse(id=1, result={"tools": []})
        assert success.ok is True
        failure = MCPResponse(id=1, error=MCPError(code=INTERNAL_ERROR, message="x"))
        assert failure.ok is False

    def test_response_roundtrip(self):
        response = MCPResponse(
            id="call-1",
            error=MCPError(code=METHOD_NOT_FOUND, message="不存在", data=[1]),
        )
        restored = MCPResponse.model_validate_json(response.model_dump_json())
        assert restored == response
        assert restored.error is not None


class TestTextContent:
    def test_shape(self):
        payload = text_content("hello", is_error=False)
        assert payload == {
            "content": [{"type": "text", "text": "hello"}],
            "isError": False,
        }

    def test_is_error_flag(self):
        payload = text_content("boom", is_error=True)
        assert payload["isError"] is True


class TestExtractResultText:
    def test_error_response_returns_empty(self):
        response = MCPResponse(
            id=1, error=MCPError(code=INVALID_PARAMS, message="bad")
        )
        assert extract_result_text(response) == ""

    def test_none_result_returns_empty(self):
        assert extract_result_text(MCPResponse(id=1)) == ""

    def test_plain_string_result(self):
        assert extract_result_text(MCPResponse(id=1, result="直接文本")) == "直接文本"

    def test_content_text_blocks_joined(self):
        response = MCPResponse(
            id=1,
            result={
                "content": [
                    {"type": "text", "text": "第一段"},
                    {"type": "image", "data": "ignored"},
                    {"type": "text", "text": "第二段"},
                ],
                "isError": False,
            },
        )
        assert extract_result_text(response) == "第一段\n第二段"

    def test_dict_text_field(self):
        response = MCPResponse(id=1, result={"text": "顶层文本"})
        assert extract_result_text(response) == "顶层文本"

    def test_other_result_json_serialized(self):
        response = MCPResponse(id=1, result={"a": "中文", "b": 2})
        text = extract_result_text(response)
        assert text == json.dumps({"a": "中文", "b": 2}, ensure_ascii=False)

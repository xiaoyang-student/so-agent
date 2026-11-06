"""工具输出截断：限制嵌套 Agent 产出对父级上下文的放大。"""

from __future__ import annotations

import json
from types import SimpleNamespace

from so_agent.mcp.adapter import truncate_output_text
from so_agent.tool_packages.code_agent.agent import _extract_output_json


class TestTruncateOutputText:
    def test_non_positive_limit_keeps_text(self):
        text = '{"output": "' + "长" * 500 + '"}'
        assert truncate_output_text(text, 0) == text
        assert truncate_output_text(text, -1) == text

    def test_short_text_kept_unchanged(self):
        text = '{"output": "短内容"}'
        assert truncate_output_text(text, 1000) == text

    def test_plain_text_truncated_with_marker(self):
        text = "x" * 200
        truncated = truncate_output_text(text, 50)
        assert truncated.startswith("x" * 50)
        assert "已截断" in truncated
        assert "200" in truncated

    def test_json_strings_truncated_preserving_structure(self):
        payload = {
            "status": "ok",
            "output": "y" * 300,
            "items": [{"text": "z" * 300}],
        }
        truncated = truncate_output_text(
            json.dumps(payload, ensure_ascii=False), 40
        )
        data = json.loads(truncated)
        # 结构与短字段保持不变，仅超长字符串被截断
        assert data["status"] == "ok"
        assert data["output"].startswith("y" * 40)
        assert "已截断" in data["output"]
        assert data["items"][0]["text"].startswith("z" * 40)
        assert "已截断" in data["items"][0]["text"]

    def test_json_under_limit_kept_unchanged(self):
        payload = {"status": "ok", "output": "足够短"}
        text = json.dumps(payload, ensure_ascii=False)
        assert truncate_output_text(text, 1000) == text


class TestCodeAgentOutputExtraction:
    async def test_long_json_output_truncated(self):
        result = SimpleNamespace(
            final_output='{"output": "' + "a" * 500 + '"}'
        )
        text = await _extract_output_json(result, max_chars=60)
        data = json.loads(text)
        assert data["output"].startswith("a" * 60)
        assert "已截断" in data["output"]

    async def test_plain_text_output_wrapped_and_truncated(self):
        result = SimpleNamespace(final_output="自由文本" * 200)
        text = await _extract_output_json(result, max_chars=30)
        data = json.loads(text)
        assert data["output"].startswith("自由文本")
        assert "已截断" in data["output"]

    async def test_zero_limit_keeps_full_output(self):
        raw = '{"output": "' + "a" * 500 + '"}'
        result = SimpleNamespace(final_output=raw)
        text = await _extract_output_json(result, max_chars=0)
        assert json.loads(text)["output"] == "a" * 500

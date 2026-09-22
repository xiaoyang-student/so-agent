"""JSON 文本容错提取工具。

背景：千问（Qwen）等 OpenAI 兼容接口会忽略 ``response_format=json_schema``
约束，把内容正确的 JSON 用 Markdown 代码围栏（````` ``json ... `` ````
`````）包裹后以纯文本返回，导致严格 JSON 解析直接失败。本模块提供
统一的围栏剥离与 JSON 对象提取，供控制层（workflow）、MCP 适配器
与各工具包的"纯文本返回 + 手动校验"输出解析路径共用。

所有 Agent 的 ``output_type`` 均为 None（不依赖 SDK 结构化输出），
输出统一走：剥离围栏 → 提取 JSON → Pydantic 手动校验。
"""

from __future__ import annotations

import json
import re
from typing import Any

# Markdown 代码围栏起始行（``` 或 ```json 等语言标记）
_FENCE_PREFIX = re.compile(r"^```[a-zA-Z0-9_-]*\s*")


def strip_code_fences(text: str) -> str:
    """去除 Markdown 代码围栏（````` ``json ... `` `````），返回内部文本。

    仅当文本以围栏起始时剥离；非围栏文本原样返回（仅去首尾空白）。
    非字符串输入按空文本处理（None）或转为字符串。
    """
    if not isinstance(text, str):
        return "" if text is None else str(text)
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = _FENCE_PREFIX.sub("", candidate)
        if candidate.endswith("```"):
            candidate = candidate[:-3].rstrip()
    return candidate


def extract_json_object(text: str) -> dict[str, Any] | None:
    """从文本中提取第一个 JSON 对象（容错解析）。

    依次尝试：整体解析（自动剥离 Markdown 代码围栏）→ 首个 ``{`` 到
    最后一个 ``}`` 的贪心子串解析。

    Returns:
        解析出的 JSON 对象；无法提取或顶层不是对象时为 None。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidate = strip_code_fences(text)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None

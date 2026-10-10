"""JSON 文本容错提取工具。

背景：千问（Qwen）等 OpenAI 兼容接口会忽略 ``response_format=json_schema``
约束，把内容正确的 JSON 用 Markdown 代码围栏（````` ``json ... `` ````
`````）包裹后以纯文本返回，或输出带轻微语法瑕疵（截断、尾逗号、
缺引号等）的 JSON，导致严格 JSON 解析直接失败。本模块提供统一的
围栏剥离、JSON 对象提取与 json-repair 宽容修复，供控制层（workflow）、
MCP 适配器与各工具包的"纯文本返回 + 手动校验"输出解析路径共用。

所有 Agent 的 ``output_type`` 均为 None（不依赖 SDK 结构化输出），
输出统一走：剥离围栏 → 提取 JSON（严格解析失败时 json-repair 修复）
→ Pydantic 手动校验。
"""

from __future__ import annotations

import json
import re
from typing import Any

import json_repair

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
    最后一个 ``}`` 的贪心子串解析 → ``json_repair.loads`` 宽容修复
    （截断、尾逗号、缺引号等语法瑕疵）。

    Returns:
        解析出的 JSON 对象；无法提取或顶层不是对象时为 None。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    candidate = strip_code_fences(text)
    parsed: Any = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end > start:
            try:
                parsed = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                parsed = None
    if isinstance(parsed, dict):
        return parsed
    if parsed is not None:
        return None  # 合法 JSON 但顶层不是对象（数组/标量）：与旧行为一致拒绝
    # 最终 fallback：json-repair 宽容修复；其结果可能为 list/str 等非对象
    # 形态（如纯文本修复为空字符串），仅在为 dict 时采用。
    try:
        repaired = json_repair.loads(candidate)
    except Exception:  # 修复失败与解析失败同等处理（返回 None）
        return None
    return repaired if isinstance(repaired, dict) else None

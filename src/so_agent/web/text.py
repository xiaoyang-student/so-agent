"""Web 控制台界面文案的中英双语支持。"""

from __future__ import annotations


def normalize_lang(value: str | None) -> str:
    """把查询参数或请求头归一化为 ``zh`` / ``en``。"""
    text = str(value or "").strip().lower()
    return "en" if text.startswith("en") else "zh"


def tr(lang: str, zh: str, en: str) -> str:
    return en if lang == "en" else zh

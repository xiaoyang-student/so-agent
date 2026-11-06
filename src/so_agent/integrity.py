"""文件内容完整性校验（SHA-256）。

生成工具在落盘时记录源码哈希，注册与执行前重新计算并比对，防止
工具源码在评审通过后被篡改（注册表只保存元数据，不能信任磁盘现状）。
"""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path


def sha256_text(text: str) -> str:
    """返回 UTF-8 文本的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """返回文件内容的 SHA-256 十六进制摘要（分块读取）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_matches_hash(path: str | Path, expected: str) -> bool:
    """判断文件内容是否与期望哈希一致（常量时间比较）。"""
    if not expected:
        return False
    try:
        actual = sha256_file(path)
    except OSError:
        return False
    return hmac.compare_digest(actual, expected)

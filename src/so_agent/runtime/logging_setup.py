"""统一日志配置。

运行时代码不再直接 ``print`` 到 stderr，而是写入 ``so_agent`` 命名空间
的 logger；入口（CLI / Web）调用 :func:`configure_logging` 决定级别与
输出目标。未配置时日志仍可被 pytest 的 caplog 或调用方自己的 handler
捕获，不会污染标准输出（CLI 的 stdout 必须保持纯 JSON）。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "so_agent"
_DEFAULT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_HANDLER_MARK = "_so_agent_handler"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: str | Path | None = None,
    force: bool = False,
) -> None:
    """配置 ``so_agent`` logger（幂等）。

    Args:
        level: 日志级别（默认 INFO）。
        log_file: 可选文件路径；提供时追加文件 handler（UTF-8）。
        force: 为 True 时重建 handler（测试/重复初始化场景）。
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = True
    if force:
        for handler in list(logger.handlers):
            if getattr(handler, _HANDLER_MARK, False):
                logger.removeHandler(handler)
                handler.close()

    has_own_handler = any(
        getattr(handler, _HANDLER_MARK, False) for handler in logger.handlers
    )
    if not has_own_handler:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        setattr(stream, _HANDLER_MARK, True)
        logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        existing = {
            getattr(handler, "baseFilename", None) for handler in logger.handlers
        }
        if str(path.resolve()) not in existing:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
            setattr(file_handler, _HANDLER_MARK, True)
            logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """返回 ``so_agent.<name>`` 命名空间下的 logger。"""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")

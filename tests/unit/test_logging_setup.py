"""单元测试：统一日志配置。"""

from __future__ import annotations

import logging

from so_agent.runtime.logging_setup import (
    LOGGER_NAME,
    configure_logging,
    get_logger,
)


def _flush_all() -> None:
    for handler in logging.getLogger(LOGGER_NAME).handlers:
        handler.flush()


def test_file_handler_writes_messages(tmp_path):
    log_file = tmp_path / "logs" / "agent.log"
    try:
        configure_logging(level=logging.INFO, log_file=log_file, force=True)
        get_logger("test").info("hello %s", "world")
        _flush_all()
        content = log_file.read_text(encoding="utf-8")
        assert "hello world" in content
        assert "so_agent.test" in content
    finally:
        configure_logging(force=True)


def test_configure_is_idempotent(tmp_path):
    try:
        configure_logging(level=logging.INFO, force=True)
        first = len(logging.getLogger(LOGGER_NAME).handlers)
        configure_logging(level=logging.DEBUG)
        configure_logging(level=logging.INFO)
        assert len(logging.getLogger(LOGGER_NAME).handlers) == first
        assert logging.getLogger(LOGGER_NAME).level == logging.INFO
    finally:
        configure_logging(force=True)


def test_get_logger_namespace():
    assert get_logger("workflow").name == "so_agent.workflow"

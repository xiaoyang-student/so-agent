"""工具调用前的高风险动作审批门。

启用后，命中风险模式的工具调用不会直接执行：先按
``(task_id, tool, 参数摘要)`` 幂等落盘一条审批记录，经 Web 控制台或
通知 outbox/Webhook 送达后决定；批准才放行，拒绝或超时按失败返回。

默认关闭（``high_risk_approval_enabled=False``）。开启时 fail-closed：
无法落盘、无法等待、超时未决一律视为拒绝。审批记录只保存参数摘要与
命中标签，不保存工具参数原文，避免密钥或代码内容进入审批视图。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from so_agent.config import Settings, get_settings
from so_agent.runtime.approvals import (
    APPROVED,
    ApprovalRequest,
    ApprovalStore,
    encode_summary,
    resolve_approval_store_path,
)
from so_agent.runtime.logging_setup import get_logger

logger = get_logger("runtime.approval_gate")

# 内置风险模式：(标签, 正则)。只标记明确的高风险能力，不追求穷尽。
_DEFAULT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("process_spawn", r"\bos\.(?:system|popen)\b"),
    ("subprocess", r"\bsubprocess\b"),
    ("delete_tree", r"\bshutil\.rmtree\b"),
    ("network_socket", r"\bsocket\."),
    ("http_client", r"\b(?:requests|httpx|urllib\.request|http\.client|aiohttp)\b"),
    ("shell_download", r"\b(?:curl|wget)\s"),
    ("shell_delete", r"\brm\s+-rf\b"),
    ("shell_privilege", r"\b(?:chmod\s+777|sudo)\b"),
    ("git_push", r"\bgit\s+push\b"),
    ("disk_format", r"\b(?:mkfs|format\s+[a-z]:)\b"),
)


class ApprovalDenied(Exception):
    """高风险动作未获批准（拒绝 / 超时 / 记录异常）。"""

    def __init__(self, message: str, *, request_id: str = "") -> None:
        super().__init__(message)
        self.request_id = request_id


@dataclass(frozen=True)
class RiskAssessment:
    risk: str
    summary: str


def _utc_iso(ts: float | None = None) -> str:
    moment = datetime.fromtimestamp(
        ts if ts is not None else datetime.now(timezone.utc).timestamp(),
        timezone.utc,
    )
    return moment.isoformat()


class HighRiskApprovalGate:
    """按配置评估并拦截高风险工具调用。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        context: Any = None,
        store: ApprovalStore | None = None,
        notifier: Any = None,
    ) -> None:
        self._settings = settings or (
            context.config if context is not None else get_settings()
        )
        self._context = context
        base = self._store_base(context)
        self._store_path = resolve_approval_store_path(
            self._settings, base_dir=base
        )
        self._store: ApprovalStore | None = store
        self._notifier = notifier
        self._patterns = self._compile_patterns(
            getattr(self._settings, "high_risk_approval_patterns", "")
        )

    @property
    def store(self) -> ApprovalStore:
        """审批存储按需创建（默认关闭时不产生数据库文件）。"""
        if self._store is None:
            self._store = ApprovalStore(self._store_path)
        return self._store

    @property
    def enabled(self) -> bool:
        return bool(getattr(self._settings, "high_risk_approval_enabled", False))

    def assess(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> RiskAssessment | None:
        """命中风险模式时返回评估结果；未命中返回 None。"""
        text = "\n".join(
            value for value in arguments.values() if isinstance(value, str)
        )
        labels = sorted(
            {label for label, pattern in self._patterns if pattern.search(text)}
        )
        if not labels:
            return None
        digest = hashlib.sha256(
            encode_summary(arguments).encode("utf-8")
        ).hexdigest()[:16]
        return RiskAssessment(
            risk=",".join(labels),
            summary=(
                f"工具 {tool_name} 命中高风险模式：{', '.join(labels)}；"
                f"参数摘要 {digest}，字符数 {len(text)}"
            ),
        )

    async def authorize(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        task_id: str = "",
    ) -> None:
        """高风险调用需批准；拒绝或超时抛 ApprovalDenied。"""
        if not self.enabled:
            return
        assessment = self.assess(tool_name, arguments)
        if assessment is None:
            return
        task = (task_id or "unknown").strip() or "unknown"
        timeout = float(
            getattr(self._settings, "high_risk_approval_timeout_seconds", 300.0)
        )
        poll = float(
            getattr(self._settings, "high_risk_approval_poll_seconds", 1.0)
        )
        key = self._idempotency_key(task, tool_name, arguments)
        expires_ts = datetime.now(timezone.utc).timestamp() + max(0.0, timeout)
        store = self.store
        request, created = store.request(
            idempotency_key=key,
            task_id=task,
            tool_name=tool_name,
            risk=assessment.risk,
            summary=assessment.summary,
            expires_ts=expires_ts,
        )
        self._emit("tool_approval_requested", request, created=created)
        if created:
            await self._notify(request)
        store = self.store
        if request.status == APPROVED:
            self._emit("tool_approval_granted", request, reused=True)
            return
        if not request.pending:
            self._emit("tool_approval_denied", request, reused=True)
            raise ApprovalDenied(
                self._denial_message(request), request_id=request.id
            )
        decided = await store.wait(
            request.id, timeout=timeout, poll_seconds=poll
        )
        if decided.status == APPROVED:
            self._emit("tool_approval_granted", decided, reused=False)
            return
        self._emit("tool_approval_denied", decided, reused=False)
        raise ApprovalDenied(
            self._denial_message(decided), request_id=decided.id
        )

    def decide(
        self,
        request_id: str,
        *,
        approved: bool,
        decided_by: str = "web",
        reason: str = "",
    ) -> ApprovalRequest:
        """供 Web 控制台调用：批准或拒绝一条待决记录。"""
        return self._store.decide(
            request_id,
            approved=approved,
            decided_by=decided_by,
            reason=reason,
        )

    async def _notify(self, request: ApprovalRequest) -> Any:
        if self._notifier is None:
            try:
                from so_agent.runtime.notifications import (
                    build_escalation_notifier,
                )

                self._notifier = build_escalation_notifier(self._settings)
            except Exception as exc:  # 通知构造失败不阻断审批落盘
                logger.warning("审批通知构造失败：%s", exc)
                return None
        payload = {
            "event": "tool_approval_required",
            "event_id": request.id,
            "task_id": request.task_id,
            "approval_id": request.id,
            "tool_name": request.tool_name,
            "risk": request.risk,
            "summary": request.summary,
            "status": request.status,
            "expires_at": _utc_iso(request.expires_ts),
        }
        try:
            return await self._notifier.notify_payload(
                payload, event_id=request.id
            )
        except Exception as exc:
            logger.warning("审批通知投递失败：%s", exc)
            return None

    def _emit(self, event: str, request: ApprovalRequest, **extra: Any) -> None:
        logger_fn = getattr(self._context, "event_logger", None)
        if logger_fn is None:
            return
        payload = {
            "approval_id": request.id,
            "task_id": request.task_id,
            "tool_name": request.tool_name,
            "risk": request.risk,
            "status": request.status,
            **extra,
        }
        try:
            logger_fn.log_event(event, payload)
        except Exception:
            return

    @staticmethod
    def _denial_message(request: ApprovalRequest) -> str:
        if request.status == APPROVED:
            return ""
        reason = f"：{request.reason}" if request.reason else ""
        return (
            f"高风险动作未获批准（状态 {request.status}{reason}）；"
            f"审批编号 {request.id}。请调整方案或联系人工批准后重试。"
        )

    @staticmethod
    def _idempotency_key(
        task_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> str:
        basis = f"{task_id}|{tool_name}|{encode_summary(arguments)}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    @staticmethod
    def _store_base(context: Any) -> Path | None:
        if context is None:
            return None
        task_store_dir = getattr(context, "task_store_dir", None)
        if task_store_dir is not None:
            return Path(task_store_dir)
        sandbox_dir = getattr(context, "sandbox_dir", None)
        return Path(sandbox_dir) if sandbox_dir is not None else None

    @staticmethod
    def _compile_patterns(extra: str) -> tuple[tuple[str, re.Pattern[str]], ...]:
        patterns: list[tuple[str, re.Pattern[str]]] = [
            (label, re.compile(regex, re.IGNORECASE))
            for label, regex in _DEFAULT_PATTERNS
        ]
        for index, raw in enumerate(
            item.strip() for item in (extra or "").split(";") if item.strip()
        ):
            try:
                patterns.append(
                    (f"custom_{index + 1}", re.compile(raw, re.IGNORECASE))
                )
            except re.error as exc:
                raise ValueError(f"high_risk_approval_patterns 正则非法：{raw!r}") from exc
        return tuple(patterns)


def build_approval_gate(
    settings: Settings | None = None, *, context: Any = None
) -> HighRiskApprovalGate:
    """构造审批门（供工具闭包持有，避免每次调用重建存储）。"""
    return HighRiskApprovalGate(settings, context=context)


__all__ = [
    "ApprovalDenied",
    "HighRiskApprovalGate",
    "RiskAssessment",
    "build_approval_gate",
]

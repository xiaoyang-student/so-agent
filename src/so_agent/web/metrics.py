"""运行指标快照（事件聚合、工具与 MCP 目录统计）。"""

from __future__ import annotations

from datetime import datetime
from datetime import timezone
from so_agent.mcp.directory import scan_mcp_directory
from so_agent.runtime.metrics import aggregate_events
from so_agent.runtime.metrics import merge_event_summaries
from so_agent.runtime.sandbox import available_sandbox_backends
from so_agent.runtime.sandbox_container import detect_container_runtime
from so_agent.runtime.sandbox_health import resolve_backend_name
from so_agent.runtime.tool_management import load_generated_manifests
from so_agent.runtime.tool_management import summarize_manifests
from so_agent.runtime.tool_store import resolve_mcp_servers_dir
from typing import Any
from urllib.parse import urlparse

from so_agent.web.common import (
    logger,
)


class MetricsMixin:

    def metrics_snapshot(self) -> dict[str, Any]:
        """聚合控制台运行指标（任务、事件、token、工具库、MCP、沙箱）。"""
        tasks = list(self._tasks.values())
        status_counts: dict[str, int] = {}
        violations = 0
        snapshot_tokens = 0
        snapshot_per_agent: dict[str, int] = {}
        for task in tasks:
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
            engine = task.engine
            if engine is None:
                continue
            try:
                snapshot = engine.token_budget_snapshot() or {}
            except Exception:
                snapshot = {}
            snapshot_tokens += int(snapshot.get("total_tokens") or 0)
            for agent_id, tokens in (snapshot.get("per_agent") or {}).items():
                snapshot_per_agent[str(agent_id)] = snapshot_per_agent.get(
                    str(agent_id), 0
                ) + int(tokens or 0)
            violations += len(snapshot.get("violations") or [])

        # token_usage 事件覆盖所有模型调用（含直答/分类等非工作流路径），
        # 优先作为统一口径；事件缺失时才回退到引擎预算快照，避免重复累计。
        total_tokens = 0
        total_input_tokens = 0
        cached_tokens = 0
        cache_write_tokens = 0
        per_agent: dict[str, int] = {}
        for task in tasks:
            task_total = 0
            task_agents: dict[str, int] = {}
            for record in task.context.event_log:
                if record.get("event_type") != "token_usage":
                    continue
                payload = record.get("payload") or {}
                tokens = int(payload.get("total_tokens") or 0)
                task_total += tokens
                total_input_tokens += int(payload.get("input_tokens") or 0)
                cached_tokens += int(payload.get("cached_tokens") or 0)
                cache_write_tokens += int(
                    payload.get("cache_write_tokens") or 0
                )
                agent = str(payload.get("agent") or "unknown")
                task_agents[agent] = task_agents.get(agent, 0) + tokens
            if task_total == 0 and task.restored_tokens:
                task_total = int(task.restored_tokens.get("total_tokens") or 0)
                total_input_tokens += int(
                    task.restored_tokens.get("input_tokens") or 0
                )
                cached_tokens += int(
                    task.restored_tokens.get("cached_tokens") or 0
                )
                cache_write_tokens += int(
                    task.restored_tokens.get("cache_write_tokens") or 0
                )
                task_agents = {
                    str(agent): int(value or 0)
                    for agent, value in (
                        task.restored_tokens.get("by_agent") or {}
                    ).items()
                }
            total_tokens += task_total
            for agent, tokens in task_agents.items():
                per_agent[agent] = per_agent.get(agent, 0) + tokens
        if total_tokens == 0:
            total_tokens = snapshot_tokens
            per_agent = snapshot_per_agent

        events = aggregate_events(task.context.event_log for task in tasks)
        restored_events = [
            task.restored_tokens.get("events")
            for task in tasks
            if not task.context.event_log
            and isinstance(task.restored_tokens.get("events"), dict)
        ]
        if restored_events:
            events = merge_event_summaries(events, *restored_events)
        manifests = load_generated_manifests(self._settings)
        tool_summary = summarize_manifests(manifests)

        try:
            mcp_entries = scan_mcp_directory(
                resolve_mcp_servers_dir(self._settings)
            )
        except Exception as exc:
            logger.warning("MCP 目录扫描失败：%s", exc)
            mcp_entries = []
        webhook = (self._settings.escalation_webhook_url or "").strip()
        webhook_host = urlparse(webhook).hostname or "" if webhook else ""

        now = datetime.now(timezone.utc)
        resolved_backend = resolve_backend_name(self._settings)
        runtime = (
            detect_container_runtime(resolved_backend)
            if resolved_backend in {"docker", "podman"}
            else None
        )
        return {
            "generated_at": now.isoformat(),
            "uptime_seconds": int((now - self._started_at).total_seconds()),
            "sandbox": {
                "backend": self._settings.sandbox_backend,
                "resolved_backend": resolved_backend,
                "available_backends": list(available_sandbox_backends()),
                "isolation": (
                    "local-validation-only"
                    if resolved_backend == "local"
                    else "external-backend"
                ),
                "require_isolation": bool(
                    self._settings.sandbox_require_isolation
                ),
                "allow_local_fallback": bool(
                    self._settings.sandbox_allow_local_fallback
                ),
                "container_runtime": runtime.name if runtime else None,
                "container_runtime_version": runtime.version if runtime else "",
                "container_image": self._settings.sandbox_container_image,
            },
            "persistent_scheduler": {
                "enabled": bool(self._settings.web_scheduler_enabled),
                "running": self.scheduler_running(),
            },
            "tasks": {
                "total": len(tasks),
                "by_status": dict(sorted(status_counts.items())),
                "active": sum(
                    1 for task in tasks if task.status in {"queued", "running"}
                ),
                "human_required": status_counts.get("human_required", 0),
            },
            "tokens": {
                "total_tokens": total_tokens,
                "input_tokens": total_input_tokens,
                "cached_tokens": cached_tokens,
                "cache_write_tokens": cache_write_tokens,
                "cache_hit_ratio": (
                    round(cached_tokens / total_input_tokens, 4)
                    if total_input_tokens > 0
                    else 0.0
                ),
                "by_agent": dict(sorted(per_agent.items())),
                "violations": violations,
            },
            "events": events,
            "tools": tool_summary,
            "mcp": {
                "servers": len(mcp_entries),
                "enabled": sum(1 for entry in mcp_entries if entry.enabled),
                "errors": sum(1 for entry in mcp_entries if entry.error),
            },
            "escalation": {
                "webhook_configured": bool(webhook),
                "webhook_host": webhook_host,
                "outbox": self.notification_outbox_stats(),
            },
        }


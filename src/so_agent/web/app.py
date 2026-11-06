"""Web 控制台 HTTP 应用（FastAPI）。

提供单页问答界面、任务状态/阶段/计划 DAG/时间线/产物查询，以及
产物下载。默认仅绑定 ``127.0.0.1``，不开放跨域；如确需在局域网
访问，必须设置 ``SO_AGENT_WEB_AUTH_TOKEN``，并通过 ``/?token=...``
首次访问写入 HttpOnly Cookie；API 不接受查询串令牌。

安全约束：
- 任务目标/约束/验收标准有长度上限；
- 产物读取限制在任务沙箱内，拒绝目录穿越、符号链接与内部目录；
- 下载统一设置 ``nosniff``；HTML/SVG 等可执行内容仅允许附件下载，
  不允许同源内联渲染；页面使用严格 CSP 且不加载任何外部资源。
"""

from __future__ import annotations

import hmac
import ipaddress
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# 与 CLI 一致：在导入 SDK 之前禁用 trace export
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from so_agent.config import Settings, get_settings
from so_agent.mcp.directory import MCPDirectoryWatcher, scan_mcp_directory
from so_agent.models import HumanGuidance
from so_agent.runtime.instance_lock import InstanceLock, InstanceLockError
from so_agent.runtime.approvals import (
    ApprovalStore,
    resolve_approval_store_path,
)
from so_agent.runtime.logging_setup import configure_logging, get_logger
from so_agent.runtime.metrics import render_prometheus
from so_agent.runtime.tool_management import (
    ToolAdminError,
    clear_tool_quarantine,
    deprecate_tool,
    load_generated_manifests,
    manifest_view,
    set_tool_pinned,
    summarize_manifests,
)
from so_agent.runtime.tool_store import (
    resolve_generated_registry_path,
    resolve_mcp_servers_dir,
)
from so_agent.web.manager import (
    MAX_OBJECTIVE_CHARS,
    ArtifactError,
    GuidanceStateError,
    TaskManager,
)
from so_agent.web.text import normalize_lang, tr

logger = get_logger("web.app")

STATIC_DIR = Path(__file__).resolve().parent / "static"
INDEX_PATH = STATIC_DIR / "index.html"
FONT_DIR = STATIC_DIR / "fonts"
FONT_FILES = ("host-grotesk-400.woff2", "atkinson-mono-400.woff2")


def _display_path(path: str | Path) -> str:
    """界面提示用路径：项目内显示相对路径，项目外保留绝对路径。"""
    target = Path(path).expanduser().resolve()
    try:
        return target.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(target)


class TaskCreateRequest(BaseModel):
    objective: str = Field(min_length=1, max_length=MAX_OBJECTIVE_CHARS)
    constraints: str = Field(default="", max_length=MAX_OBJECTIVE_CHARS)
    acceptance_criteria: str = Field(default="", max_length=MAX_OBJECTIVE_CHARS)


class ToolPinRequest(BaseModel):
    pinned: bool = True


class ApprovalDecisionRequest(BaseModel):
    approved: bool
    reason: str = Field(default="", max_length=500)


def _security_headers(response: Any, *, html: bool = False) -> Any:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    if html:
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
            "connect-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'",
        )
    return response


def create_app(
    settings: Settings | None = None,
    *,
    base_dir: Path | None = None,
) -> FastAPI:
    """构造 Web 应用；``base_dir`` 为 Web 任务沙箱根目录。"""
    resolved_settings = settings or get_settings()
    from so_agent.plugins import PluginError, load_plugins

    try:
        load_plugins(resolved_settings)
    except PluginError as exc:
        raise RuntimeError(f"插件加载失败：{exc}") from exc
    from so_agent.runtime.orchestration_runtime import (
        OrchestrationRuntimeError,
        ensure_runtime_available,
    )

    try:
        ensure_runtime_available(resolved_settings.orchestration_runtime)
    except OrchestrationRuntimeError as exc:
        raise RuntimeError(str(exc)) from exc
    root = (
        Path(base_dir).expanduser()
        if base_dir is not None
        else Path.cwd() / resolved_settings.web_sandbox_dir
    )
    manager = TaskManager(root, resolved_settings)
    instance_lock = InstanceLock(manager.base_dir / "web.lock")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            instance_lock.acquire()
        except InstanceLockError as exc:
            logger.error("Web 控制台启动被拒绝：%s", exc)
            raise
        logger.info("Web 控制台已启动，任务沙箱：%s", manager.base_dir)
        watcher: MCPDirectoryWatcher | None = None
        watch_seconds = int(resolved_settings.mcp_directory_watch_seconds)
        if watch_seconds > 0:
            watcher = MCPDirectoryWatcher(
                resolve_mcp_servers_dir(resolved_settings),
                interval_seconds=watch_seconds,
            )
            watcher.start()
        try:
            manager.start_scheduler()
        except Exception as exc:  # 调度器失败不阻塞控制台使用
            logger.warning("Web 内置调度器启动失败：%s", exc)
        try:
            await manager.start_notification_outbox()
        except Exception as exc:  # outbox 失败不阻塞控制台使用
            logger.warning("通知 outbox 启动失败：%s", exc)
        try:
            manager.resume_interrupted()
        except Exception as exc:  # 恢复失败不阻塞控制台使用
            logger.warning("Web 重启任务恢复启动失败：%s", exc)
        try:
            yield
        finally:
            if watcher is not None:
                watcher.stop()
            await manager.shutdown()
            instance_lock.release()
            logger.info("Web 控制台已关闭")

    app = FastAPI(
        title="so-agent Web 控制台",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.manager = manager

    def _request_authenticated(request: Request) -> bool:
        """请求是否携带有效令牌（未配置令牌时不视为已鉴权）。"""
        token = resolved_settings.web_auth_token.strip()
        if not token:
            return False
        provided = (
            request.headers.get("x-web-token")
            or request.cookies.get("so_agent_web_token")
            or ""
        )
        return bool(provided) and hmac.compare_digest(
            provided.encode("utf-8"), token.encode("utf-8")
        )

    def require_auth(request: Request) -> None:
        token = resolved_settings.web_auth_token.strip()
        if not token:
            return
        # 不接受查询串令牌：URL 会进入访问日志与浏览器历史，仅保留
        # 请求头与登录引导设置的 HttpOnly Cookie 两种传递方式。
        if not _request_authenticated(request):
            raise HTTPException(status_code=401, detail="需要访问令牌")

    @app.middleware("http")
    async def _headers(request: Request, call_next):
        response = await call_next(request)
        return _security_headers(
            response, html=request.url.path in {"/", "/index.html"}
        )

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, token: str | None = None) -> HTMLResponse:
        if not INDEX_PATH.is_file():
            raise HTTPException(status_code=500, detail="前端页面缺失")
        forwarded_proto = request.headers.get("x-forwarded-proto", "")
        request_is_https = request.url.scheme == "https" or (
            forwarded_proto.split(",")[0].strip() == "https"
        )
        response = HTMLResponse(
            content=INDEX_PATH.read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )
        expected = resolved_settings.web_auth_token.strip()
        if token and expected and hmac.compare_digest(
            token.encode("utf-8"), expected.encode("utf-8")
        ):
            response.set_cookie(
                "so_agent_web_token",
                token,
                httponly=True,
                samesite="strict",
                secure=request_is_https,
                max_age=12 * 3600,
            )
        _security_headers(response, html=True)
        return response

    @app.get("/static/fonts/{filename}", include_in_schema=False)
    def font_asset(filename: str) -> FileResponse:
        # 白名单只暴露这两份自托管字体；避免用 StaticFiles 挂载整个目录，
        # 否则 /static/index.html 会绕过首页的 CSP 头。
        if filename not in FONT_FILES:
            raise HTTPException(status_code=404, detail="Not found")
        path = FONT_DIR / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(
            path,
            media_type="font/woff2",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        """存活探针：配置令牌后未鉴权只返回最小信息；鉴权或未配置令牌时附带运维详情。

        未配置令牌的部署仅允许回环访问，等价于本地已授权，返回旧版字段；
        配置令牌后，携带有效令牌的请求同样返回运维详情。运维详情与
        ``/api/metrics`` 同源，保留旧版 ``tasks`` / ``sandbox_backend`` /
        ``token_budget`` 字段以兼容既有监控。
        """
        payload: dict[str, Any] = {
            "status": "ok",
            "auth_required": bool(resolved_settings.web_auth_token.strip()),
        }
        # 未配置令牌时仅允许回环访问，等价于已授权（与 require_auth 一致），
        # 保留旧版运维字段；配置令牌后必须携带有效令牌才返回详情。
        if payload["auth_required"] and not _request_authenticated(request):
            return payload
        metrics = manager.metrics_snapshot()
        payload["tasks"] = metrics["tasks"]
        payload["sandbox"] = metrics["sandbox"]
        payload["sandbox_backend"] = metrics["sandbox"]["backend"]
        payload["available_sandbox_backends"] = metrics["sandbox"][
            "available_backends"
        ]
        payload["token_budget"] = metrics["tokens"]
        return payload

    @app.get("/api/metrics")
    def metrics(_: None = Depends(require_auth)) -> dict[str, Any]:
        """运行指标 JSON（任务、事件、token、工具库、MCP、沙箱）。"""
        return manager.metrics_snapshot()

    approval_root = manager.base_dir / "persistent"

    def _approval_store() -> ApprovalStore | None:
        path = resolve_approval_store_path(
            resolved_settings, base_dir=approval_root
        )
        return ApprovalStore(path) if path.exists() else None

    @app.get("/api/approvals")
    def list_approvals(
        task_id: str = "",
        status: str = "pending",
        _: None = Depends(require_auth),
    ) -> dict[str, Any]:
        """高风险工具审批记录列表（默认仅待决）。"""
        store = _approval_store()
        if store is None:
            return {"approvals": []}
        items = store.list(
            status=status.strip() or None,
            task_id=task_id.strip() or None,
        )
        return {"approvals": [item.to_dict() for item in items]}

    @app.get("/api/approvals/{approval_id}")
    def get_approval(
        approval_id: str, _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        store = _approval_store()
        record = store.get(approval_id) if store is not None else None
        if record is None:
            raise HTTPException(status_code=404, detail="审批记录不存在")
        return {"approval": record.to_dict()}

    @app.post("/api/approvals/{approval_id}")
    def decide_approval(
        approval_id: str,
        payload: ApprovalDecisionRequest,
        _: None = Depends(require_auth),
    ) -> dict[str, Any]:
        """批准或拒绝一条待决高风险动作；已处理记录返回 409。"""
        store = _approval_store()
        if store is None or store.get(approval_id) is None:
            raise HTTPException(status_code=404, detail="审批记录不存在")
        try:
            record = store.decide(
                approval_id,
                approved=payload.approved,
                decided_by="web",
                reason=payload.reason,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        logger.info(
            "高风险动作审批：%s -> %s（%s）",
            approval_id,
            record.status,
            record.tool_name,
        )
        return {"approval": record.to_dict()}

    @app.get("/api/results/latest")
    def latest_result(
        lang: str = "zh", _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        """最近一次任务结果（latest.md / latest.json 的同源载荷）。"""
        payload = manager.latest_result()
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail=tr(
                    normalize_lang(lang),
                    "尚无任务结果",
                    "No task result yet",
                ),
            )
        return {
            "payload": payload,
            "markdown": manager.latest_result_markdown() or "",
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus_metrics(_: None = Depends(require_auth)) -> PlainTextResponse:
        """Prometheus 文本格式指标；配置令牌时请通过 ``X-Web-Token`` 头抓取。"""
        return PlainTextResponse(
            render_prometheus(manager.metrics_snapshot()),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    def _mcp_directory_payload(lang: str = "zh") -> dict[str, Any]:
        server_dir = resolve_mcp_servers_dir(resolved_settings)
        entries = scan_mcp_directory(server_dir)
        return {
            "server_dir": _display_path(server_dir),
            "entries": [
                {
                    "name": entry.name,
                    "directory": _display_path(entry.directory),
                    "enabled": entry.enabled,
                    "tools": (
                        entry.config.allowed_tools if entry.config is not None else []
                    ),
                    "error": entry.error,
                }
                for entry in entries
            ],
            "note": tr(
                lang,
                "复制或修改 Server 后，下一个任务开始时自动生效；本接口仅刷新校验结果。",
                "Copied or edited servers take effect when the next task starts; "
                "this endpoint only refreshes the validation result.",
            ),
        }

    @app.get("/api/tools/catalog")
    def tool_catalog(
        lang: str = "zh", _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        return _mcp_directory_payload(normalize_lang(lang))

    @app.post("/api/tools/reload")
    def reload_tools(
        lang: str = "zh", _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        return _mcp_directory_payload(normalize_lang(lang))

    @app.get("/api/tools/generated")
    def generated_tools(_: None = Depends(require_auth)) -> dict[str, Any]:
        """生成工具库清单与状态摘要（不返回源码与测试用例）。"""
        manifests = load_generated_manifests(resolved_settings)
        return {
            "registry": _display_path(
                resolve_generated_registry_path(resolved_settings)
            ),
            "summary": summarize_manifests(manifests),
            "tools": [manifest_view(item) for item in manifests],
        }

    def _tool_admin_error(exc: ToolAdminError) -> HTTPException:
        status = 404 if "不存在" in str(exc) else 409
        return HTTPException(status_code=status, detail=str(exc))

    @app.post("/api/tools/generated/{tool_name}/pin")
    def pin_generated_tool(
        tool_name: str,
        payload: ToolPinRequest,
        _: None = Depends(require_auth),
    ) -> dict[str, Any]:
        try:
            manifest = set_tool_pinned(
                tool_name, payload.pinned, resolved_settings
            )
        except ToolAdminError as exc:
            raise _tool_admin_error(exc) from exc
        return {"updated": True, "tool": manifest_view(manifest)}

    @app.post("/api/tools/generated/{tool_name}/clear-quarantine")
    def clear_generated_tool_quarantine(
        tool_name: str, _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        try:
            manifest = clear_tool_quarantine(tool_name, resolved_settings)
        except ToolAdminError as exc:
            raise _tool_admin_error(exc) from exc
        return {"updated": True, "tool": manifest_view(manifest)}

    @app.post("/api/tools/generated/{tool_name}/deprecate")
    def deprecate_generated_tool(
        tool_name: str, _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        try:
            manifest = deprecate_tool(tool_name, resolved_settings)
        except ToolAdminError as exc:
            raise _tool_admin_error(exc) from exc
        return {"updated": True, "tool": manifest_view(manifest)}

    @app.post("/api/tasks", status_code=202)
    async def create_task(
        payload: TaskCreateRequest,
        lang: str = "zh",
        _: None = Depends(require_auth),
    ) -> dict[str, Any]:
        normalized = normalize_lang(lang)
        try:
            task = manager.submit(
                payload.objective,
                payload.constraints,
                payload.acceptance_criteria,
                lang=normalized,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        return {
            "task_id": task.task_id,
            "status": task.status,
            "created_at": task.created_at,
        }

    @app.get("/api/tasks")
    def list_tasks(_: None = Depends(require_auth)) -> dict[str, Any]:
        return {"tasks": manager.list_tasks()}

    @app.get("/api/tasks/{task_id}")
    def get_task(
        task_id: str, lang: str = "zh", _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        normalized = normalize_lang(lang)
        snapshot = manager.snapshot(task_id, normalized)
        if snapshot is None:
            raise HTTPException(
                status_code=404,
                detail=tr(normalized, "任务不存在", "Task not found"),
            )
        return snapshot

    @app.get("/api/scheduled-tasks")
    def list_scheduled_tasks(_: None = Depends(require_auth)) -> dict[str, Any]:
        return {
            "tasks": manager.list_scheduled_tasks(),
            "policy": {
                "misfire": resolved_settings.scheduler_misfire_policy,
                "grace_seconds": resolved_settings.scheduler_misfire_grace_seconds,
                "max_catch_up_runs": resolved_settings.scheduler_max_catch_up_runs,
                "max_retries": resolved_settings.scheduler_task_max_retries,
            },
        }

    @app.get("/api/scheduled-tasks/{task_id}")
    def get_scheduled_task(
        task_id: str, lang: str = "zh", _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        normalized = normalize_lang(lang)
        snapshot = manager.get_scheduled_task(task_id)
        if snapshot is None:
            raise HTTPException(
                status_code=404,
                detail=tr(normalized, "调度任务不存在", "Scheduled task not found"),
            )
        return snapshot

    @app.post("/api/tasks/{task_id}/cancel")
    async def cancel_task(
        task_id: str, lang: str = "zh", _: None = Depends(require_auth)
    ) -> dict[str, Any]:
        normalized = normalize_lang(lang)
        if manager.get_task(task_id) is None:
            raise HTTPException(
                status_code=404,
                detail=tr(normalized, "任务不存在", "Task not found"),
            )
        return {"cancelled": manager.cancel(task_id)}

    @app.post("/api/tasks/{task_id}/guidance", status_code=202)
    async def submit_guidance(
        task_id: str,
        guidance: HumanGuidance,
        lang: str = "zh",
        _: None = Depends(require_auth),
    ) -> dict[str, Any]:
        """人工兜底后提交指导：指定回退子任务、处理说明与补充信息。"""
        normalized = normalize_lang(lang)
        try:
            accepted = manager.submit_guidance(task_id, guidance, lang=normalized)
        except GuidanceStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not accepted:
            raise HTTPException(
                status_code=404,
                detail=tr(normalized, "任务不存在", "Task not found"),
            )
        return {
            "accepted": True,
            "task_id": task_id,
            "action": guidance.action,
        }

    @app.get("/api/tasks/{task_id}/artifacts/{artifact_path:path}")
    def download_artifact(
        task_id: str,
        artifact_path: str,
        inline: bool = False,
        lang: str = "zh",
        _: None = Depends(require_auth),
    ) -> FileResponse:
        normalized = normalize_lang(lang)
        task = manager.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=404,
                detail=tr(normalized, "任务不存在", "Task not found"),
            )
        try:
            path = manager.resolve_artifact(task, artifact_path, lang=normalized)
        except ArtifactError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        media_type = manager.artifact_media_type(path)
        if inline:
            if not _inline_safe(media_type):
                raise HTTPException(
                    status_code=400,
                    detail=tr(
                        normalized,
                        "该类型仅支持附件下载",
                        "This file type can only be downloaded as an attachment",
                    ),
                )
            media_type = _inline_media_type(media_type)
        headers = {
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        }
        return FileResponse(
            path,
            media_type=media_type,
            filename=path.name,
            content_disposition_type="inline" if inline else "attachment",
            headers=headers,
        )

    return app


def _inline_safe(media_type: str) -> bool:
    # 文本类最终都会降级为 text/plain，故 text/* 整体可内联
    if media_type.startswith("text/"):
        return True
    if media_type in {"application/json", "application/pdf"}:
        return True
    return media_type.startswith("image/") and media_type != "image/svg+xml"


def _inline_media_type(media_type: str) -> str:
    # 文本类统一降级为 text/plain，避免 HTML/JS 在同源内联执行
    if media_type.startswith("text/"):
        return "text/plain; charset=utf-8"
    return media_type


def _is_loopback_host(host: str) -> bool:
    name = host.strip().strip("[]").lower()
    if name in {"localhost", "ip6-localhost"}:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> int:
    """``so-agent-web`` 命令入口。"""
    import argparse

    try:
        import uvicorn
    except ImportError:
        print(
            "缺少 Web 依赖，请先安装：pip install -e \".[web]\"",
            flush=True,
        )
        return 2

    from so_agent.main import _load_config_overrides, _resolve_settings

    parser = argparse.ArgumentParser(
        prog="so-agent-web",
        description="so-agent Web 控制台：提交任务、查看分解流程图与最终产物。",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="监听地址（默认 127.0.0.1；非回环地址必须配置访问令牌）",
    )
    parser.add_argument("--port", type=int, default=8765, help="监听端口（默认 8765）")
    parser.add_argument(
        "--sandbox",
        default=None,
        help="Web 任务沙箱根目录（默认：cwd/<web_sandbox_dir>）。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="运行时配置文件路径（YAML/JSON，覆盖 Settings 默认字段）。",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="访问令牌（覆盖 SO_AGENT_WEB_AUTH_TOKEN；为空则仅本机访问）。",
    )
    args = parser.parse_args(argv)

    try:
        overrides = _load_config_overrides(args.config)
        settings = _resolve_settings(overrides)
    except Exception as exc:
        print(f"[配置错误] {type(exc).__name__}: {exc}", flush=True)
        return 2
    if args.token is not None:
        settings = settings.model_copy(update={"web_auth_token": args.token})
    if not _is_loopback_host(args.host) and not settings.web_auth_token.strip():
        print(
            "[安全错误] 监听非回环地址必须配置访问令牌："
            "设置 SO_AGENT_WEB_AUTH_TOKEN 或传入 --token。",
            flush=True,
        )
        return 2

    configure_logging()
    base_dir = (
        Path(args.sandbox).expanduser()
        if args.sandbox
        else Path.cwd() / settings.web_sandbox_dir
    )
    app = create_app(settings=settings, base_dir=base_dir)
    print(
        f"so-agent Web 控制台：http://{args.host}:{args.port} "
        f"（沙箱：{base_dir}）",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

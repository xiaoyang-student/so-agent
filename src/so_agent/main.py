"""命令行入口（CLI）。

从命令行提交任务目标，组装运行时组件，经四象限路由器分类后
运行对应工作流，最终把 ``OrchestratorOutput`` 以结构化 JSON 打印到标准输出。

用法示例::

    python -m so_agent.main "为目标目录实现一个 CLI 统计工具" ^
        --constraints "仅使用标准库；不得访问沙箱目录之外" ^
        --acceptance-criteria "提供可运行脚本；包含自我验证" ^
        --config runtime_config.yaml

说明：
- ``--daemon`` 启动持久化任务调度守护进程（后台轮询并执行到期任务，
  此时无需提供任务目标）；
- ``--config`` 指向 YAML / JSON 配置文件（键值映射），用于覆盖
  ``Settings`` 的默认字段（例如 max_concurrency、max_replan_attempts、
  sandbox_timeout 等）；文件不存在或字段非法时以退出码 2 结束；
- 任务终态为 ``completed`` 时退出码为 0，其余终态（human_required /
  failed / cancelled）退出码为 1，便于脚本集成与自动化断言；
- 标准输出仅打印最终结果 JSON；运行状态摘要打印到标准错误。
"""

from __future__ import annotations

import os

# 必须在任何 so_agent 导入（间接导入 OpenAI Agents SDK）之前设置：
# 禁用 SDK 的 trace export，消除无 Tracing API key 时的导出警告。
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _load_secrets() -> None:
    """从项目根目录的 secrets.yaml 加载密钥到环境变量。

    查找顺序：当前工作目录 → main.py 所在包的上三级目录（项目根）。
    文件不存在时静默跳过（允许纯环境变量模式）。
    """
    import sys

    candidates = [
        Path.cwd() / "secrets.yaml",
        Path(__file__).resolve().parent.parent.parent / "secrets.yaml",
    ]
    for path in candidates:
        if path.is_file():
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(key, str) and isinstance(value, str):
                            os.environ.setdefault(key, value)
                    print(f"[so-agent] 已从 {path} 加载密钥配置", file=sys.stderr)
            except Exception:
                pass
            return


_load_secrets()

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.models import TaskStatus
from so_agent.router import TaskRouter
from so_agent.runtime.config_loader import ConfigLoadError
from so_agent.runtime.logging_setup import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="so-agent",
        description=(
            "多 Agent 编排框架 CLI：提交任务目标，经四象限路由器分类后"
            "运行对应工作流（路由分类 → 对应路径执行），"
            "并输出结构化 JSON 结果。"
        ),
    )
    parser.add_argument(
        "objective",
        nargs="?",
        default=None,
        help="任务目标（期望达成什么）；省略时可从标准输入（管道）读取。",
    )
    parser.add_argument(
        "-c",
        "--constraints",
        default="",
        help="约束条件文本（可选；支持换行或分号分隔多条）。",
    )
    parser.add_argument(
        "-a",
        "--acceptance-criteria",
        default="",
        help="验收标准文本（可选；支持换行或分号分隔多条）。",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "运行时配置文件路径（YAML/JSON，覆盖 Settings 默认字段，"
            "如 max_concurrency、max_replan_attempts、model_timeout 等）。"
        ),
    )
    parser.add_argument(
        "--sandbox",
        default=None,
        help="沙箱根目录（默认：当前工作目录下的 sandbox）。",
    )
    parser.add_argument(
        "--project-name",
        default="so-agent",
        help="项目名称（用于标识任务空间，默认 so-agent）。",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        default=False,
        help="启动持久化任务调度守护进程（后台轮询并执行到期任务）。",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="TASK_ID",
        help="从人工兜底检查点恢复任务（无需提供任务目标）。",
    )
    parser.add_argument(
        "--guidance-action",
        choices=("resume", "rollback", "abort"),
        default="resume",
        help="恢复动作：resume 直接继续、rollback 回退后重跑、abort 保持人工态。",
    )
    parser.add_argument(
        "--rollback-to",
        action="append",
        default=[],
        metavar="SUBTASK_ID",
        help="回退目标子任务 ID（可重复；rollback 时为空则回退全部失败子任务）。",
    )
    parser.add_argument(
        "--instructions",
        default="",
        help="给回退子任务的补充指导文本。",
    )
    parser.add_argument(
        "--additional-info",
        default="",
        help="恢复时提供给子任务的补充信息。",
    )
    return parser


def _load_config_overrides(path: str | None) -> dict[str, Any]:
    """读取 ``--config`` 配置文件并返回 Settings 覆盖字典。

    Args:
        path: 配置文件路径（YAML/JSON）；None 表示不覆盖。

    Returns:
        覆盖 Settings 字段的键值映射；未提供文件时为空字典。

    Raises:
        FileNotFoundError: 配置文件不存在时。
        ValueError: 文件内容不是键值映射（对象）时。
        yaml.YAMLError: YAML 解析失败时。
    """
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, dict):
        raise ValueError("配置文件内容必须是键值映射（YAML/JSON 对象）")
    return data


def _resolve_settings(overrides: dict[str, Any]) -> Settings:
    """构造生效配置：默认全局实例，叠加配置文件覆盖项。

    Args:
        overrides: 配置文件提供的字段覆盖（优先级高于环境变量与默认值）。

    Returns:
        Settings：生效配置实例。
    """
    if not overrides:
        return get_settings()
    return Settings(**overrides)


def _resolve_objective(
    parser: argparse.ArgumentParser, raw: str | None
) -> str:
    """解析任务目标：优先位置参数，其次从标准输入（管道）读取。

    Raises:
        SystemExit: 无法获得非空任务目标时（由 parser.error 触发）。
    """
    objective = (raw or "").strip()
    if not objective and not sys.stdin.isatty():
        objective = (sys.stdin.read() or "").strip()
    if not objective:
        parser.error("必须提供任务目标（位置参数 objective 或标准输入）")
    return objective


def _build_context(args: argparse.Namespace, settings: Settings) -> ProjectContext:
    """组装项目共享上下文（沙箱目录、项目名与生效配置）。"""
    sandbox_dir = (
        Path(args.sandbox).expanduser().resolve()
        if args.sandbox
        else Path.cwd() / "sandbox"
    )
    return ProjectContext(
        sandbox_dir=sandbox_dir,
        project_name=str(args.project_name),
        config=settings,
    )


def _configure_stdout() -> None:
    """在支持的平台上把标准输出切到 UTF-8，避免中文 JSON 乱码。"""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # 环境不支持时保持原编码
            pass


async def _resume_from_checkpoint(
    args: argparse.Namespace,
    context: ProjectContext,
    settings: Settings,
    log_file: Path,
) -> int:
    """读取人工兜底检查点，应用指导并继续执行。"""
    from so_agent.models import HumanGuidance
    from so_agent.runtime.checkpoints import (
        CheckpointError,
        delete_checkpoint,
        load_checkpoint,
    )
    from so_agent.state_machine import HumanGuidanceError
    from so_agent.workflow import WorkflowEngine, WorkflowError

    try:
        payload = load_checkpoint(context.sandbox_dir, str(args.resume))
        engine = WorkflowEngine(context, settings=settings, log_file=log_file)
        engine.restore_checkpoint(payload)
    except (CheckpointError, WorkflowError, ValueError) as exc:
        print(f"[检查点错误] {exc}", file=sys.stderr)
        return 2

    guidance = HumanGuidance(
        action=args.guidance_action,
        rollback_to=list(args.rollback_to or []),
        instructions=str(args.instructions or ""),
        additional_info=str(args.additional_info or ""),
    )
    try:
        output = await engine.resume_with_guidance(guidance)
    except HumanGuidanceError as exc:
        print(f"[指导无效] {exc}", file=sys.stderr)
        return 2
    except WorkflowError as exc:
        print(f"[恢复失败] {exc}", file=sys.stderr)
        return 1

    if output.status is TaskStatus.COMPLETED_PROVISIONAL:
        # CLI 进程退出会取消后台评审；续跑路径与路由层一致，等待收口
        final = await engine.wait_for_provisional()
        output = final or output

    if output.status != TaskStatus.HUMAN_REQUIRED:
        try:
            delete_checkpoint(context.sandbox_dir, str(args.resume))
        except CheckpointError:
            pass
    _configure_stdout()
    print(output.model_dump_json(indent=2))
    print(
        f"[so-agent] 任务 {output.task_id} 终态：{output.status.value}",
        file=sys.stderr,
    )
    return (
        0
        if output.status in {TaskStatus.COMPLETED, TaskStatus.SCHEDULED}
        else 1
    )


async def _run_daemon(scheduler, settings) -> None:
    """运行调度守护进程，并在启用时并行排空通知 outbox。"""
    worker = None
    if bool(settings.notification_outbox_enabled) and (
        settings.escalation_webhook_url or ""
    ).strip():
        from so_agent.runtime.logging_setup import get_logger
        from so_agent.runtime.notification_outbox import (
            NotificationOutboxWorker,
        )

        try:
            worker = NotificationOutboxWorker(settings)
            await worker.start()
        except Exception as exc:
            get_logger("main").warning("通知 outbox 启动失败：%s", exc)
            worker = None
    try:
        await scheduler.start()
    finally:
        if worker is not None:
            await worker.stop()


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口：解析参数 → 组装运行时 → 路由分发（或调度守护进程）
    → 打印 JSON 结果。

    Args:
        argv: 命令行参数列表；None 表示取 ``sys.argv[1:]``。

    Returns:
        进程退出码：0 表示任务完成；1 表示其它终态；2 表示参数/配置错误；
        130 表示用户中断（Ctrl+C）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.daemon and args.resume:
        parser.error("--daemon 与 --resume 不能同时使用")
    if args.resume and args.objective:
        parser.error("--resume 模式不需要任务目标")

    # 加载配置（daemon 与正常模式都需要）
    try:
        overrides = _load_config_overrides(args.config)
        settings = _resolve_settings(overrides)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pydantic 校验失败等
        print(f"[配置错误] 配置项非法：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    configure_logging()
    try:
        from so_agent.plugins import PluginError, load_plugins

        load_plugins(settings)
    except PluginError as exc:
        print(f"[插件错误] {exc}", file=sys.stderr)
        return 2
    try:
        from so_agent.runtime.orchestration_runtime import (
            OrchestrationRuntimeError,
            ensure_runtime_available,
        )

        ensure_runtime_available(settings.orchestration_runtime)
    except OrchestrationRuntimeError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    context = _build_context(args, settings)
    log_file = Path.cwd() / "logs" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # --daemon 模式：启动调度守护进程（不执行新任务，无需任务目标）
    if args.daemon:
        from so_agent.persistence import TaskStore, resolve_task_store_path
        from so_agent.task_scheduler import PeriodicTaskPoller

        store = TaskStore(resolve_task_store_path(context, settings))
        scheduler = PeriodicTaskPoller(store, context, settings)
        try:
            asyncio.run(_run_daemon(scheduler, settings))
        except KeyboardInterrupt:
            scheduler.stop()
            print("[中断] 调度器被用户中断", file=sys.stderr)
            return 130
        return 0

    # 断点续跑：读取人工兜底检查点并按指导恢复
    if args.resume:
        try:
            return asyncio.run(
                _resume_from_checkpoint(args, context, settings, log_file)
            )
        except KeyboardInterrupt:
            print("[中断] 恢复执行被用户中断", file=sys.stderr)
            return 130

    # 正常模式：解析任务目标，经 TaskRouter 分类并路由执行
    objective = _resolve_objective(parser, args.objective)
    router = TaskRouter(context, settings, log_file=log_file)

    try:
        output = asyncio.run(
            router.route(objective, args.constraints, args.acceptance_criteria)
        )
    except ConfigLoadError as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[中断] 任务被用户中断", file=sys.stderr)
        return 130

    _configure_stdout()
    print(output.model_dump_json(indent=2))
    print(
        f"[so-agent] 任务 {output.task_id} 终态：{output.status.value}",
        file=sys.stderr,
    )
    return (
        0
        if output.status in {TaskStatus.COMPLETED, TaskStatus.SCHEDULED}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())

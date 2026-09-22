"""命令行入口（CLI）。

从命令行提交任务目标，组装运行时组件并运行完整编排工作流，
最终把 ``OrchestratorOutput`` 以结构化 JSON 打印到标准输出。

用法示例::

    python -m so_agent.main "为目标目录实现一个 CLI 统计工具" ^
        --constraints "仅使用标准库；不得访问沙箱目录之外" ^
        --acceptance-criteria "提供可运行脚本；包含自我验证" ^
        --config runtime_config.yaml

说明：
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
from pathlib import Path
from typing import Any

import yaml

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.models import TaskStatus
from so_agent.workflow import WorkflowEngine


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="so-agent",
        description=(
            "多 Agent 编排框架 CLI：提交任务目标，运行完整编排工作流"
            "（拆解 → 计划评审 → 执行 → 失败升级 → 汇总 → 结果评审），"
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


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口：解析参数 → 组装运行时 → 运行工作流 → 打印 JSON 结果。

    Args:
        argv: 命令行参数列表；None 表示取 ``sys.argv[1:]``。

    Returns:
        进程退出码：0 表示任务完成；1 表示其它终态；2 表示参数/配置错误；
        130 表示用户中断（Ctrl+C）。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    objective = _resolve_objective(parser, args.objective)

    try:
        overrides = _load_config_overrides(args.config)
        settings = _resolve_settings(overrides)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"[配置错误] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pydantic 校验失败等
        print(f"[配置错误] 配置项非法：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    context = _build_context(args, settings)
    engine = WorkflowEngine(context, settings=settings)

    try:
        output = asyncio.run(
            engine.run(objective, args.constraints, args.acceptance_criteria)
        )
    except KeyboardInterrupt:
        print("[中断] 任务被用户中断", file=sys.stderr)
        return 130

    _configure_stdout()
    print(output.model_dump_json(indent=2))
    print(
        f"[so-agent] 任务 {output.task_id} 终态：{output.status.value}",
        file=sys.stderr,
    )
    return 0 if output.status == TaskStatus.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""单文件轻量快通道（P3）。

仅处理“把一个文本文件写进沙箱”的明确任务：一次轻量模型调用生成正文，
控制层做路径包含、原子写入与大小/关键内容校验，跳过规划与执行主管
调用。判定必须保守——出现多文件、运行/安装/网络等动作、路径越界或
任何不确定因素时返回 ``applied=False``，由调用方回退完整工作流。

默认关闭（``fast_path_single_file_enabled=false``）。开启后仍只覆盖
严格匹配的纯写入任务，不改变其他任务的工具权限或执行路径。
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from openai import AsyncOpenAI

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.json_utils import extract_json_object, strip_code_fences
from so_agent.models import TaskStatus
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.config_loader import (
    load_agent_api_config,
    resolve_api_config_path,
    resolve_model_http_timeout,
)
from so_agent.runtime.events import (
    EVENT_FAST_PATH,
    EventLogger,
    shared_event_logger,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.model_gateway import build_openai_client
from so_agent.runtime.telemetry import async_span_scope, set_span_attributes
from so_agent.runtime.usage import record_completion_usage

API_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "api_config.yaml"

logger = get_logger("path.single_file")

_SAFE_EXTENSION_NAMES: tuple[str, ...] = (
    "txt", "md", "markdown", "json", "jsonl", "csv", "tsv", "log", "html?",
    "xml", "ya?ml", "toml", "ini", "cfg", "py", "js", "ts", "css", "sql",
    "sh", "ps1", "bat", "java", "go", "rs", "c", "h", "cpp", "hpp", "rb",
    "php", "tex", "rst",
)
# 长扩展名优先，否则 jsonl/cpp/markdown 会被 json/c/md 截断匹配
_SAFE_EXTENSIONS = "|".join(
    sorted(_SAFE_EXTENSION_NAMES, key=len, reverse=True)
)
_FILE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_\-./\\])"
    r"((?:[A-Za-z0-9_\-\u4e00-\u9fff]+[./\\])*"
    r"[A-Za-z0-9_\-\u4e00-\u9fff]+\.(?:"
    + _SAFE_EXTENSIONS
    + r"))(?![A-Za-z0-9_\-])",
    re.IGNORECASE,
)
_WRITE_INTENT_RE = re.compile(r"(创建|新建|写入|写到|写进|写|保存|生成|输出)")
_NEGATION_RE = re.compile(
    r"(不要|无需|不需要|禁止|严禁|不得|不能|不可|避免|"
    r"\b(?:do not|don't|without|avoid)\b)",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，,。；;!?！？\n])")
_FORBIDDEN_ACTION_RE = re.compile(
    r"(运行|执行|测试|安装|部署|下载|上传|删除|重命名|移动|复制|压缩|解压|"
    r"监控|定时|每天|每周|调用|爬取|访问|联网|网络|发送|https?://|"
    r"\b(?:run|execute|test|install|deploy|download|upload|delete|rename|"
    r"move|copy|zip|unzip|monitor|schedule|call|fetch|request)\b)",
    re.IGNORECASE,
)
_MULTI_FILE_RE = re.compile(
    r"(多个|每个|所有|分别|目录|文件夹|压缩包|"
    r"两个文件|三个文件|这些文件|多个文件|其余文件)"
)
_REQUIRED_TEXT_RE = re.compile(
    r"(?:包含|内容(?:为|写|是)|写入)\s*[「“\"']?([^\s「」“”\"'，。；;,]{1,40})"
)
_RESERVED_NAMES = re.compile(
    r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$", re.IGNORECASE
)

CONTENT_SYSTEM_PROMPT = (
    "你是单文件写入执行器。根据用户目标生成且仅生成一个文件的完整正文。\n"
    "只输出一个 JSON 对象：{\"content\": \"<文件正文>\"}。\n"
    "content 必须是文件最终正文，不要输出 Markdown 代码围栏、解释、"
    "前后缀或工具调用标记。\n"
    "严格遵守用户约束与验收标准；无法在不违反约束的情况下完成时，"
    "输出 {\"content\": \"\"}。"
)


@dataclass(frozen=True)
class SingleFileTask:
    """通过严格判定后的单文件写入任务。"""

    path: str
    objective: str
    constraints: str
    acceptance_criteria: str
    required_contains: tuple[str, ...] = ()


@dataclass
class FastPathOutcome:
    """快通道结果；``output`` 为 None 表示应由调用方回退完整工作流。"""

    applied: bool
    output: OrchestratorOutput | None = None
    reason: str = ""
    target: str = ""


def _normalize_path(raw: str) -> str | None:
    candidate = raw.strip().replace("\\", "/")
    if not candidate or candidate.startswith("/") or ":" in candidate:
        return None
    parts = [part for part in candidate.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        return None
    if any(part.startswith(".") or len(part) > 80 for part in parts):
        return None
    name = parts[-1]
    if name.endswith((" ", ".")) or _RESERVED_NAMES.match(name):
        return None
    suffix = PurePosixPath(name).suffix.lower().lstrip(".")
    if not re.fullmatch(_SAFE_EXTENSIONS, suffix):
        return None
    normalized = "/".join(parts)
    if len(normalized) > 200:
        return None
    return normalized


def _paths_in(text: str) -> set[str]:
    paths: set[str] = set()
    for raw in _FILE_TOKEN_RE.findall(text or ""):
        normalized = _normalize_path(raw)
        if normalized is not None:
            paths.add(normalized)
    return paths


def _positive_write_intent(text: str) -> bool:
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        if _NEGATION_RE.search(clause):
            continue
        if _WRITE_INTENT_RE.search(clause):
            return True
    return False


def _required_literals(text: str) -> tuple[str, ...]:
    literals: list[str] = []
    for raw in _REQUIRED_TEXT_RE.findall(text or ""):
        value = raw.strip()
        if (
            value
            and not re.search(r"[{}\[\]()<>]", value)
            and not re.search(r"[与和及]", value)
            and re.fullmatch(r"[A-Za-z0-9_.:/\-]+", value)
        ):
            literals.append(value)
    return tuple(dict.fromkeys(literals))


def detect_single_file_task(
    objective: str,
    constraints: str = "",
    acceptance_criteria: str = "",
) -> SingleFileTask | None:
    """严格判定是否属于“唯一文本文件纯写入”任务；不确定一律返回 None。"""
    objective = objective or ""
    constraints = constraints or ""
    acceptance_criteria = acceptance_criteria or ""

    objective_paths = _paths_in(objective)
    if len(objective_paths) != 1:
        return None
    target = next(iter(objective_paths))
    all_paths = objective_paths | _paths_in(constraints) | _paths_in(
        acceptance_criteria
    )
    if all_paths != {target}:
        return None
    if not _positive_write_intent(objective):
        return None
    combined = "\n".join((objective, constraints, acceptance_criteria))
    if _FORBIDDEN_ACTION_RE.search(combined):
        return None
    if _MULTI_FILE_RE.search(combined):
        return None
    return SingleFileTask(
        path=target,
        objective=objective,
        constraints=constraints,
        acceptance_criteria=acceptance_criteria,
        required_contains=_required_literals(acceptance_criteria)
        or _required_literals(objective),
    )


def _content_prompt(task: SingleFileTask) -> str:
    parts = [
        f"任务目标：{task.objective.strip()}",
        f"目标文件：{task.path}",
    ]
    if task.constraints.strip():
        parts.append(f"约束条件：{task.constraints.strip()}")
    if task.acceptance_criteria.strip():
        parts.append(f"验收标准：{task.acceptance_criteria.strip()}")
    parts.append("请只生成该文件的正文，并按要求输出 JSON。")
    return "\n\n".join(parts)


def _extract_content(raw: str) -> str:
    """取出 JSON 中的正文；保留模型给出的原始空白（仅剥离围栏）。"""
    payload = extract_json_object(raw)
    if not isinstance(payload, dict):
        return ""
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        return ""
    if content.lstrip().startswith("```"):
        return strip_code_fences(content)
    return content


def _write_text_atomic(root: Path, relative: str, content: str) -> Path:
    root_resolved = root.resolve()
    target = (root_resolved / relative).resolve()
    target.relative_to(root_resolved)

    current = root_resolved
    for part in PurePosixPath(relative).parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise OSError(f"拒绝写入符号链接路径：{current}")
    target.parent.mkdir(parents=True, exist_ok=True)

    target = (root_resolved / relative).resolve()
    target.relative_to(root_resolved)
    if target.is_symlink():
        raise OSError(f"拒绝覆盖符号链接：{target}")

    fd, temporary = tempfile.mkstemp(
        prefix=".fast-path-", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target


def _emit_event(
    context: ProjectContext,
    *,
    status: str,
    task_id: str,
    target: str = "",
    reason: str = "",
    chars: int = 0,
) -> None:
    try:
        events = context.event_logger
        if not isinstance(events, EventLogger):
            events = shared_event_logger(context)
        events.log_event(
            EVENT_FAST_PATH,
            {
                "task_id": task_id,
                "status": status,
                "target": target,
                "reason": reason,
                "content_chars": chars,
            },
        )
    except Exception:
        return


async def maybe_run_single_file_fast_path(
    context: ProjectContext,
    settings: Settings,
    objective: str,
    constraints: str,
    acceptance_criteria: str,
    *,
    task_id: str = "",
) -> FastPathOutcome:
    """按配置尝试单文件快通道；未启用/不匹配/失败均返回未应用。"""
    if not bool(getattr(settings, "fast_path_single_file_enabled", False)):
        return FastPathOutcome(applied=False, reason="disabled")
    task = detect_single_file_task(
        objective, constraints, acceptance_criteria
    )
    if task is None:
        return FastPathOutcome(applied=False, reason="not_applicable")
    _emit_event(
        context, status="hit", task_id=task_id, target=task.path
    )
    logger.info("单文件快通道命中：%s", task.path)

    api_cfg = load_agent_api_config(
        resolve_api_config_path(API_CONFIG_PATH), resolve_secrets=True
    )
    http_timeout = resolve_model_http_timeout(api_cfg)
    client = build_openai_client(
        api_cfg,
        config_path=resolve_api_config_path(API_CONFIG_PATH),
        client_factory=AsyncOpenAI,
        http_timeout=http_timeout,
    )
    request: dict[str, object] = {
        "model": api_cfg.model,
        "messages": [
            {"role": "system", "content": CONTENT_SYSTEM_PROMPT},
            {"role": "user", "content": _content_prompt(task)},
        ],
        "timeout": http_timeout,
    }
    max_tokens = int(getattr(settings, "fast_path_max_output_tokens", 0))
    if max_tokens > 0:
        request["max_tokens"] = max_tokens
    try:
        async with async_span_scope(
            "so_agent.model.call",
            {"task.type": "fast_path_single_file", "model.name": api_cfg.model},
        ) as span:
            response = await client.chat.completions.create(**request)
            raw = (response.choices[0].message.content or "").strip()
            record_completion_usage(
                context, response, "fast_path_single_file"
            )
            set_span_attributes(span, {"answer.chars": len(raw)})
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _emit_event(
            context,
            status="fallback",
            task_id=task_id,
            target=task.path,
            reason=reason[:200],
        )
        logger.warning("单文件快通道模型调用失败，回退完整工作流：%s", reason)
        return FastPathOutcome(applied=False, reason="model_error", target=task.path)
    finally:
        await client.close()

    content = _extract_content(raw)
    if not content:
        _emit_event(
            context,
            status="fallback",
            task_id=task_id,
            target=task.path,
            reason="empty_content",
        )
        return FastPathOutcome(
            applied=False, reason="empty_content", target=task.path
        )
    lowered = content.lower()
    if "<tool_call" in lowered or "<function" in lowered:
        _emit_event(
            context,
            status="fallback",
            task_id=task_id,
            target=task.path,
            reason="tool_markup",
        )
        return FastPathOutcome(
            applied=False, reason="tool_markup", target=task.path
        )
    size = len(content.encode("utf-8"))
    max_bytes = int(getattr(settings, "fast_path_max_file_bytes", 0) or 0)
    if max_bytes > 0 and size > max_bytes:
        _emit_event(
            context,
            status="fallback",
            task_id=task_id,
            target=task.path,
            reason="too_large",
            chars=size,
        )
        return FastPathOutcome(
            applied=False, reason="too_large", target=task.path
        )
    missing = [item for item in task.required_contains if item not in content]
    if missing:
        _emit_event(
            context,
            status="fallback",
            task_id=task_id,
            target=task.path,
            reason="required_text_missing",
        )
        return FastPathOutcome(
            applied=False,
            reason="required_text_missing",
            target=task.path,
        )
    try:
        _write_text_atomic(Path(context.sandbox_dir), task.path, content)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _emit_event(
            context,
            status="fallback",
            task_id=task_id,
            target=task.path,
            reason=reason[:200],
        )
        logger.warning("单文件快通道写入失败，回退完整工作流：%s", reason)
        return FastPathOutcome(
            applied=False, reason="write_error", target=task.path
        )

    _emit_event(
        context,
        status="completed",
        task_id=task_id,
        target=task.path,
        chars=size,
    )
    output = OrchestratorOutput(
        task_id=task_id or uuid.uuid4().hex,
        status=TaskStatus.COMPLETED,
        final_result=f"已创建 `{task.path}`（{size} 字节）。",
        review_summary="执行路径：fast_path_single_file；确定性写入与关键内容校验已通过",
    )
    return FastPathOutcome(
        applied=True, output=output, reason="completed", target=task.path
    )

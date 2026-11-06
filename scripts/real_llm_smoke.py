"""真实 LLM 冒烟验收。

前置条件（缺一不可）：
1. 已从 ``.example`` 复制并填写各级 ``api_config.yaml``；
2. 密钥可通过 ``secrets.yaml``（项目根目录）或 ``api_key_env`` 环境变量提供。

用法：
    python scripts/real_llm_smoke.py
    python scripts/real_llm_smoke.py --skip-web --timeout 600

脚本会依次执行：简单问答、完整工作流、Web API 端到端（提交→轮询→下载产物），
并把实测证据写入 ``sandbox/real_llm_evidence.json``。任何一步未达成都以非零码
退出；不打印、不落盘任何密钥。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from so_agent.main import _load_secrets, _resolve_settings  # noqa: E402
from so_agent.models import TaskStatus  # noqa: E402
from so_agent.runtime.config_loader import (  # noqa: E402
    ConfigLoadError,
    load_agent_api_config,
    resolve_api_config_path,
)

SUPERVISOR_CONFIG = PROJECT_ROOT / "src" / "so_agent" / "api_config.yaml"
TOOL_CONFIGS = [
    PROJECT_ROOT / "src" / "so_agent" / "tool_packages" / name / "api_config.yaml"
    for name in ("code_agent", "subagent_creator", "review_agent")
]


def _check_configs() -> tuple[bool, list[str]]:
    """校验全部 API 配置存在且密钥可解析。"""
    problems: list[str] = []
    paths = [SUPERVISOR_CONFIG, *TOOL_CONFIGS]
    for default in paths:
        path = resolve_api_config_path(default)
        if not Path(path).is_file():
            problems.append(f"缺少配置文件：{path}（可从同名 .example 复制）")
            continue
        try:
            load_agent_api_config(path, resolve_secrets=True)
        except ConfigLoadError as exc:
            problems.append(f"{path}：{exc}")
    return not problems, problems


def _model_name() -> str | None:
    """读取生效模型名（仅模型标识，不解析、不输出密钥）。"""
    try:
        config = load_agent_api_config(
            resolve_api_config_path(SUPERVISOR_CONFIG), resolve_secrets=False
        )
    except ConfigLoadError:
        return None
    return config.model


def _snapshot_dirs(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    )


async def _scenario_direct(settings: Any, sandbox: Path) -> dict[str, Any]:
    from so_agent.context import ProjectContext
    from so_agent.router import TaskRouter

    context = ProjectContext(
        sandbox_dir=sandbox,
        project_name="real-llm-direct",
        config=settings,
    )
    router = TaskRouter(context, settings, log_file=sandbox / "direct.log")
    started = time.perf_counter()
    output = await router.route(
        "请用一句话回答：水的化学式是什么？",
        "只回答一句，不要展开",
        "答案包含 H2O",
    )
    return {
        "status": output.status.value,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "output_chars": len(output.final_result or ""),
        "output_preview": (output.final_result or "")[:200],
        "passed": output.status == TaskStatus.COMPLETED and bool(output.final_result),
    }


async def _scenario_workflow(settings: Any, sandbox: Path) -> dict[str, Any]:
    from so_agent.context import ProjectContext
    from so_agent.workflow import WorkflowEngine

    context = ProjectContext(
        sandbox_dir=sandbox,
        project_name="real-llm-workflow",
        config=settings,
    )
    engine = WorkflowEngine(context, settings=settings, log_file=sandbox / "workflow.log")
    started = time.perf_counter()
    output = await engine.run(
        "在沙箱目录创建 smoke.md，写入一行 SMOKE_OK，然后给出简短说明。",
        "只允许在沙箱目录内读写",
        "smoke.md 存在且内容包含 SMOKE_OK",
    )
    artifacts = _snapshot_dirs(sandbox)
    smoke = sandbox / "smoke.md"
    content_ok = smoke.is_file() and "SMOKE_OK" in smoke.read_text(
        encoding="utf-8", errors="replace"
    )
    return {
        "status": output.status.value,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "output_chars": len(output.final_result or ""),
        "output_preview": (output.final_result or "")[:200],
        "artifacts": artifacts,
        "acceptance_ok": content_ok,
        "passed": output.status == TaskStatus.COMPLETED and content_ok,
    }


def _scenario_web(settings: Any, base_dir: Path, timeout: float) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from so_agent.web.app import create_app

    app = create_app(settings=settings, base_dir=base_dir)
    started = time.perf_counter()
    with TestClient(app) as client:
        response = client.post(
            "/api/tasks",
            json={
                "objective": "创建 web-smoke.txt，内容写 WEB_OK，然后回答已完成。",
                "acceptance_criteria": "web-smoke.txt 存在且内容包含 WEB_OK",
            },
        )
        response.raise_for_status()
        task_id = response.json()["task_id"]
        deadline = started + timeout
        detail: dict[str, Any] = {}
        while time.perf_counter() < deadline:
            detail = client.get(f"/api/tasks/{task_id}").json()
            if detail.get("status") in {
                "completed",
                "failed",
                "cancelled",
                "human_required",
            }:
                break
            time.sleep(2)
        else:
            client.post(f"/api/tasks/{task_id}/cancel")

        names = [item["name"] for item in detail.get("artifacts") or []]
        content_ok = False
        if "web-smoke.txt" in names:
            download = client.get(
                f"/api/tasks/{task_id}/artifacts/web-smoke.txt"
            )
            content_ok = download.status_code == 200 and "WEB_OK" in download.text
        route = detail.get("route") or {}
        journey = detail.get("journey") or {}
        journey_nodes = journey.get("nodes") or []
        selected_branch = next(
            (
                branch.get("type")
                for branch in route.get("branches") or []
                if branch.get("selected")
            ),
            None,
        )
        tool_calls = detail.get("tool_calls") or []
        dynamic_agents = detail.get("dynamic_agents") or []
        generated_tools = detail.get("generated_tools") or []
        visualization_ok = bool(
            route.get("available")
            and selected_branch
            and journey_nodes
            and any(node.get("kind") == "branch" for node in journey_nodes)
            and any(node.get("kind") == "subtask" for node in journey_nodes)
        )
        # 终态任务不应残留进行中的工具调用（工具异常 / 超时必须收口）
        tool_calls_settled = not any(
            call.get("status") == "running" for call in tool_calls
        )
        return {
            "status": detail.get("status"),
            "task_type": (detail.get("classification") or {}).get("task_type"),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "artifacts": names,
            "acceptance_ok": content_ok,
            "output_preview": (detail.get("final_result") or "")[:300],
            "timeline_events": len(detail.get("timeline") or []),
            "route_method": route.get("method"),
            "route_effective_task_type": route.get("effective_task_type"),
            "selected_branch": selected_branch,
            "route_steps": [
                step.get("stage") for step in route.get("steps") or []
            ],
            "tool_calls": len(tool_calls),
            "tool_names": sorted({str(call.get("tool")) for call in tool_calls}),
            "dynamic_agents": len(dynamic_agents),
            "generated_tools": [
                tool.get("tool_name") for tool in generated_tools
            ],
            "journey_nodes": len(journey_nodes),
            "journey_node_kinds": sorted(
                {str(node.get("kind")) for node in journey_nodes}
            ),
            "journey_tool_calls": sum(
                len(node.get("tool_calls") or []) for node in journey_nodes
            ),
            "visualization_ok": visualization_ok,
            "tool_calls_settled": tool_calls_settled,
            "passed": (
                detail.get("status") == "completed"
                and content_ok
                and visualization_ok
                and tool_calls_settled
            ),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真实 LLM 冒烟验收")
    parser.add_argument("--skip-web", action="store_true", help="跳过 Web 端到端")
    parser.add_argument("--skip-direct", action="store_true", help="跳过简单问答")
    parser.add_argument("--skip-workflow", action="store_true", help="跳过完整工作流")
    parser.add_argument("--timeout", type=float, default=900.0, help="单场景超时秒数")
    parser.add_argument(
        "--evidence",
        default=str(PROJECT_ROOT / "sandbox" / "real_llm_evidence.json"),
        help="证据文件输出路径",
    )
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    _load_secrets()
    ok, problems = _check_configs()
    if not ok:
        print("[配置未就绪] 真实 LLM 验收无法开始：", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        return 2

    settings = _resolve_settings({})
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = PROJECT_ROOT / "sandbox" / f"real-llm-{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    scenarios: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    async def run_async(name: str, coro: Any) -> None:
        try:
            result = await asyncio.wait_for(coro, timeout=args.timeout)
        except asyncio.TimeoutError:
            result = {"passed": False, "error": f"超时（>{args.timeout}s）"}
        except Exception as exc:  # 记录真实异常类型，不掩盖失败
            result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
        scenarios[name] = result
        if not result.get("passed"):
            failures.append(name)
        print(f"[{name}] {'PASS' if result.get('passed') else 'FAIL'} {result}")

    if not args.skip_direct:
        asyncio.run(
            run_async(
                "direct_answer",
                _scenario_direct(settings, root / "direct"),
            )
        )
    if not args.skip_workflow:
        asyncio.run(
            run_async(
                "full_workflow",
                _scenario_workflow(settings, root / "workflow"),
            )
        )
    if not args.skip_web:
        try:
            scenarios["web_e2e"] = _scenario_web(settings, root / "web", args.timeout)
        except Exception as exc:
            scenarios["web_e2e"] = {
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        if not scenarios["web_e2e"].get("passed"):
            failures.append("web_e2e")
        print(
            f"[web_e2e] {'PASS' if scenarios['web_e2e'].get('passed') else 'FAIL'} "
            f"{scenarios['web_e2e']}"
        )

    evidence = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": _model_name(),
        "scenarios": scenarios,
        "failures": failures,
    }
    evidence_path = Path(args.evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"证据已写入：{evidence_path}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""原始模型基线：同一提示词、无工具单轮调用，用于与 SO-Agent 对照。

AI Studio 网页需要登录态；本脚本通过同一 OpenAI 兼容接口调用同一模型，
给出可复核的“裸模型”时间与 token 基线。它不会创建文件、不会执行代码，
产物仅为文本回答，因此与 SO-Agent 的完成效果差异可直接观察。

用法：
    python scripts/bench_raw.py --runs 2

证据写入 ``sandbox/bench-raw-<时间戳>/evidence.json``。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from agents import Agent, OpenAIChatCompletionsModel, Runner  # noqa: E402
from agents.tracing import (  # noqa: E402
    set_trace_processors,
    set_tracing_disabled,
)
from openai import AsyncOpenAI  # noqa: E402

from bench_probe import RAW_APPENDIX, TASKS, UsageCollector  # noqa: E402
from so_agent.main import _load_secrets  # noqa: E402
from so_agent.runtime.config_loader import (  # noqa: E402
    load_agent_api_config,
    resolve_api_config_path,
)

SUPERVISOR_CONFIG = PROJECT_ROOT / "src" / "so_agent" / "api_config.yaml"

INSTRUCTIONS = (
    "你是小米 MiMo 模型。直接完成用户交给你的任务，给出你能够提供的全部内容。"
    "你没有文件系统或代码执行工具，不要声称已经创建文件或运行了代码。"
)


def _build_model() -> OpenAIChatCompletionsModel:
    config = load_agent_api_config(
        resolve_api_config_path(SUPERVISOR_CONFIG), resolve_secrets=True
    )
    api_key = config.resolve_api_key()
    base_url = config.base_url or "https://api.xiaomimimo.com/v1"
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=config.timeout or 180.0,
    )
    return OpenAIChatCompletionsModel(model=config.model, openai_client=client)


async def _run_once(task_name: str, run_dir: Path) -> dict[str, Any]:
    task = TASKS[task_name]
    prompt = (
        f"{task['objective']}\n\n约束条件：{task['constraints']}\n"
        f"验收标准：{task['acceptance_criteria']}"
    )
    prompt += RAW_APPENDIX.get(task_name, "")
    if task_name in RAW_APPENDIX:
        prompt += (
            "\n\n务必在回答末尾输出一个 ```json 代码块，内容为完整的 metrics.json；"
            "同时输出完整脚本与报告正文。"
        )
    collector = UsageCollector()
    set_trace_processors([collector])
    agent = Agent(name="raw_mimo", instructions=INSTRUCTIONS, model=_build_model())
    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    error = None
    output_text = ""
    try:
        result = await asyncio.wait_for(Runner.run(agent, input=prompt), timeout=600.0)
        output_text = result.final_output or ""
    except asyncio.TimeoutError:
        error = "timeout>600s"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = round(time.perf_counter() - started, 2)
    ended_at = datetime.now().astimezone().isoformat()

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "answer.md").write_text(output_text, encoding="utf-8")
    usage = collector.summary()
    evidence = {
        "task": task_name,
        "prompt": prompt,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed,
        "error": error,
        "output_chars": len(output_text),
        "output_preview": output_text[:2000],
        "usage": usage,
        "artifacts": ["answer.md"],
    }
    (run_dir / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="原始模型基线")
    parser.add_argument("--task", choices=sorted(TASKS), default="sales")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    _load_secrets()
    set_tracing_disabled(False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = PROJECT_ROOT / "sandbox" / f"bench-raw-{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for index in range(1, max(1, args.runs) + 1):
        run_dir = root / f"run-{index}"
        print(f"[raw {index}] 开始 -> {run_dir}", file=sys.stderr)
        evidence = asyncio.run(_run_once(args.task, run_dir))
        runs.append(evidence)
        totals = evidence["usage"]["totals"]
        print(
            f"[raw {index}] {evidence['elapsed_seconds']}s "
            f"tokens={totals['total_tokens']} "
            f"in={totals['input_tokens']} out={totals['output_tokens']} "
            f"chars={evidence['output_chars']} error={evidence['error']}",
            file=sys.stderr,
        )

    elapsed = [run["elapsed_seconds"] for run in runs]
    tokens = [run["usage"]["totals"]["total_tokens"] for run in runs]
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "task": args.task,
        "runs": [
            {
                "elapsed_seconds": run["elapsed_seconds"],
                "output_chars": run["output_chars"],
                "error": run["error"],
                "usage": run["usage"]["totals"],
                "by_agent": run["usage"]["by_agent"],
            }
            for run in runs
        ],
        "elapsed_avg": round(sum(elapsed) / len(elapsed), 3),
        "tokens_avg": round(sum(tokens) / len(tokens), 1),
        "evidence_dir": str(root),
    }
    (root / "evidence.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"证据已写入：{root / 'evidence.json'}", file=sys.stderr)
    return 0 if all(not run["error"] for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())

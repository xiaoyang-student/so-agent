"""固定质量回归集运行入口（确定性评分）。

用法：
    python scripts/quality_regression.py --dry-run
    python scripts/quality_regression.py --recorded sandbox/quality-recorded
    python scripts/quality_regression.py --case qa_water --timeout 300
    python scripts/quality_regression.py --category instruction_following

- 默认对固定用例逐条调用真实路由（需要可用的 ``api_config.yaml`` 与密钥）；
- ``--recorded`` 不调用模型，只对已有输出 ``<id>.txt`` 与产物目录
  ``<id>/`` 重新评分，适合 CI 与优化前后对比；
- 证据写入 ``sandbox/quality_evidence.json``，不打印、不落盘任何密钥。
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
from so_agent.quality import (  # noqa: E402
    QualityCase,
    QualityCaseError,
    load_quality_cases,
    score_output,
)
from so_agent.runtime.config_loader import (  # noqa: E402
    ConfigLoadError,
    load_agent_api_config,
    resolve_api_config_path,
)

CASES_PATH = PROJECT_ROOT / "tests" / "baseline" / "quality_cases.json"
SUPERVISOR_CONFIG = PROJECT_ROOT / "src" / "so_agent" / "api_config.yaml"


def _check_configs() -> tuple[bool, list[str]]:
    path = resolve_api_config_path(SUPERVISOR_CONFIG)
    if not Path(path).is_file():
        return False, [f"缺少配置文件：{path}（可从同名 .example 复制）"]
    try:
        load_agent_api_config(path, resolve_secrets=True)
    except ConfigLoadError as exc:
        return False, [f"{path}：{exc}"]
    return True, []


def _model_name() -> str | None:
    try:
        config = load_agent_api_config(
            resolve_api_config_path(SUPERVISOR_CONFIG), resolve_secrets=False
        )
    except ConfigLoadError:
        return None
    return config.model


def _select_cases(
    cases: list[QualityCase],
    *,
    selected_ids: list[str],
    category: str,
) -> list[QualityCase]:
    selected = cases
    if category:
        selected = [case for case in selected if case.category == category]
    if selected_ids:
        wanted = set(selected_ids)
        selected = [case for case in selected if case.case_id in wanted]
        missing = sorted(wanted - {case.case_id for case in selected})
        if missing:
            raise QualityCaseError(f"未知用例 id：{missing}")
    return selected


async def _run_case(
    case: QualityCase,
    settings: Any,
    sandbox: Path,
    timeout: float,
) -> dict[str, Any]:
    from so_agent.context import ProjectContext
    from so_agent.router import TaskRouter

    sandbox.mkdir(parents=True, exist_ok=True)
    context = ProjectContext(
        sandbox_dir=sandbox,
        project_name=f"quality-{case.case_id}",
        config=settings,
    )
    router = TaskRouter(context, settings, log_file=sandbox / "quality.log")
    started = time.perf_counter()
    try:
        output = await asyncio.wait_for(
            router.route(
                case.objective, case.constraints, case.acceptance_criteria
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {
            "case_id": case.case_id,
            "category": case.category,
            "passed": False,
            "error": f"超时（>{timeout}s）",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "checks": [],
        }
    except Exception as exc:  # 记录真实异常，不掩盖失败
        return {
            "case_id": case.case_id,
            "category": case.category,
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "checks": [],
        }
    score = score_output(
        case, output.final_result or "", sandbox_dir=sandbox
    )
    return {
        "case_id": case.case_id,
        "category": case.category,
        "status": output.status.value,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "output_chars": score.output_chars,
        "output_preview": (output.final_result or "")[:200],
        "score": score.to_dict(),
        "passed": score.passed and output.status.value == "completed",
    }


def _run_recorded(
    cases: list[QualityCase], root: Path
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        text_path = root / f"{case.case_id}.txt"
        sandbox = root / case.case_id
        if not text_path.is_file():
            results.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "passed": False,
                    "error": f"缺少记录输出：{text_path}",
                    "checks": [],
                }
            )
            continue
        output = text_path.read_text(encoding="utf-8", errors="replace")
        score = score_output(case, output, sandbox_dir=sandbox)
        item = score.to_dict()
        item.update(
            {
                "case_id": case.case_id,
                "category": case.category,
                "passed": score.passed,
                "output_preview": output[:200],
            }
        )
        results.append(item)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="固定质量回归集与指令遵循评分")
    parser.add_argument("--dry-run", action="store_true", help="只校验用例集")
    parser.add_argument(
        "--recorded",
        default="",
        help="回放目录：<id>.txt 为输出，<id>/ 为产物目录（不调用模型）",
    )
    parser.add_argument(
        "--case", action="append", default=[], help="只运行指定用例 id（可重复）"
    )
    parser.add_argument("--category", default="", help="只运行指定类别")
    parser.add_argument("--timeout", type=float, default=900.0, help="单用例超时秒数")
    parser.add_argument(
        "--evidence",
        default=str(PROJECT_ROOT / "sandbox" / "quality_evidence.json"),
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

    try:
        all_cases = load_quality_cases(CASES_PATH)
        cases = _select_cases(
            all_cases, selected_ids=args.case, category=args.category
        )
    except QualityCaseError as exc:
        print(f"[用例错误] {exc}", file=sys.stderr)
        return 2

    categories: dict[str, int] = {}
    for case in all_cases:
        categories[case.category] = categories.get(case.category, 0) + 1
    print(
        f"固定用例 {len(all_cases)} 条，类别 {categories}，本次运行 {len(cases)} 条"
    )
    if args.dry_run:
        return 0
    if not cases:
        print("[用例错误] 过滤后没有可运行用例", file=sys.stderr)
        return 2

    evidence: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(CASES_PATH.relative_to(PROJECT_ROOT)),
        "dataset_total": len(all_cases),
        "model": _model_name(),
        "mode": "recorded" if args.recorded else "real_llm",
    }
    if args.recorded:
        results = _run_recorded(cases, Path(args.recorded))
    else:
        _load_secrets()
        ok, problems = _check_configs()
        if not ok:
            print("[配置未就绪] 质量回归无法开始：", file=sys.stderr)
            for item in problems:
                print(f"  - {item}", file=sys.stderr)
            return 2
        settings = _resolve_settings({})
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = PROJECT_ROOT / "sandbox" / f"quality-{stamp}"
        results = []
        for case in cases:
            result = asyncio.run(
                _run_case(case, settings, root / case.case_id, args.timeout)
            )
            results.append(result)
            tag = "PASS" if result.get("passed") else "FAIL"
            print(
                f"[{case.case_id}] {tag} "
                f"score={result.get('score', {}).get('score', 0)} "
                f"elapsed={result.get('elapsed_seconds', '?')}s"
            )

    evidence["results"] = results
    passed = sum(1 for item in results if item.get("passed"))
    evidence["summary"] = {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "failed_cases": [
            item["case_id"] for item in results if not item.get("passed")
        ],
    }
    evidence_path = Path(args.evidence)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"证据已写入：{evidence_path}", file=sys.stderr)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

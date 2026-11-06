"""把无文件系统渠道（原生 API / 网页）的文本回答还原为可校验产物。

原生 API 与网页只能输出文本。为了用同一套确定性校验器评分，本脚本从回答
中提取最后的 metrics.json 代码块、最长的 Python 代码块与报告正文，落盘后
调用 ``bench_probe._validate_incident`` 判定。

用法：
    python scripts/extract_incident_artifacts.py --answer answer.md --out sandbox/x/run-1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import bench_probe as bp  # noqa: E402

REQUIRED = (
    "total_requests",
    "error_count",
    "error_rate",
    "top_error_endpoint",
    "worst_5min_window",
    "p95_latency_ms",
)


def _fenced(text: str, pattern: str) -> list[str]:
    return re.findall(r"```" + pattern + r"\s*\n(.*?)```", text, re.S)


def _pick_metrics(answer: str) -> dict | None:
    found: list[dict] = []
    for block in re.findall(r"```[a-zA-Z]*\s*\n(.*?)```", answer, re.S):
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and all(key in data for key in REQUIRED):
            found.append(data)
    if not found:
        for match in re.finditer(r"\{[^{}]*\"p95_latency_ms\"[^{}]*\}", answer, re.S):
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if all(key in data for key in REQUIRED):
                found.append(data)
    return found[-1] if found else None


def extract(answer: str, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "access.log").write_text(bp.INCIDENT_LOG, encoding="utf-8")

    metrics = _pick_metrics(answer)
    if metrics is not None:
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    scripts = _fenced(answer, "python") + _fenced(answer, "py")
    if scripts:
        (out_dir / "analyze_incident.py").write_text(
            max(scripts, key=len), encoding="utf-8"
        )

    reports = _fenced(answer, "markdown") + _fenced(answer, "md")
    report = max(reports, key=len) if reports else answer
    (out_dir / "incident_report.md").write_text(report, encoding="utf-8")

    validation = bp._validate_incident(out_dir)
    validation["extracted_metrics"] = metrics is not None
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="还原并校验 incident 产物")
    parser.add_argument("--answer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    answer = Path(args.answer).read_text(encoding="utf-8", errors="replace")
    validation = extract(answer, Path(args.out))
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

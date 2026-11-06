"""固定质量回归集与指令遵循评分（确定性，不调用模型）。

用途：把一组固定的业务输入与可机检的评分规则固化下来，每次优化后
复跑同一套用例，作为“省 token 但准确性可控”的证据。评分只做确定性
文本/产物检查，不引入模型裁判，因此同一输入与输出永远得到同一分数。

检查项覆盖两类目标：
- 正确性：必须出现的关键值、禁止出现的错误结论；
- 指令遵循：字数、句数、语言、JSON 结构、禁用 Markdown 等。

用例格式见 ``tests/baseline/quality_cases.json``；运行入口见
``scripts/quality_regression.py``。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from so_agent.json_utils import extract_json_object, strip_code_fences

CHECK_KINDS: frozenset[str] = frozenset(
    {
        "contains",
        "contains_any",
        "not_contains",
        "regex",
        "max_chars",
        "min_chars",
        "max_sentences",
        "json_object",
        "language",
        "no_code_fence",
    }
)

_CATEGORIES: frozenset[str] = frozenset(
    {
        "direct_qa",
        "deep_reasoning",
        "single_file",
        "multi_file",
        "instruction_following",
    }
)

_SENTENCE_SPLIT = re.compile(r"[。！？!?]+|(?<!\d)\.(?!\d)")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


class QualityCaseError(ValueError):
    """质量用例文件非法。"""


@dataclass(frozen=True)
class QualityCheck:
    """单条确定性检查。"""

    kind: str
    value: Any = None
    label: str = ""
    weight: float = 1.0


@dataclass(frozen=True)
class ArtifactCheck:
    """产物检查：路径相对沙箱根目录，禁止越界。"""

    path: str
    contains: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    min_bytes: int = 1
    weight: float = 1.0


@dataclass(frozen=True)
class QualityCase:
    """一个固定回归用例。"""

    case_id: str
    category: str
    objective: str
    constraints: str = ""
    acceptance_criteria: str = ""
    expected_task_type: str = ""
    checks: tuple[QualityCheck, ...] = ()
    artifacts: tuple[ArtifactCheck, ...] = ()
    min_score: float = 1.0


@dataclass
class CheckResult:
    name: str
    passed: bool
    weight: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "weight": self.weight,
            "detail": self.detail,
        }


@dataclass
class QualityScore:
    case_id: str
    score: float
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    output_chars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "score": round(self.score, 4),
            "passed": self.passed,
            "output_chars": self.output_chars,
            "checks": [item.to_dict() for item in self.checks],
        }


def _count_sentences(text: str) -> int:
    """按中英文句末标点计数；空白与纯标点不计入。"""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text)]
    return len([part for part in parts if part])


def _json_payload(text: str) -> Any:
    """解析输出中的 JSON 对象；无法解析返回 None。"""
    candidate = strip_code_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return extract_json_object(candidate)


def _check_one(check: QualityCheck, text: str) -> CheckResult:
    kind = check.kind
    value = check.value
    name = check.label or kind
    passed = False
    detail = ""

    if kind == "contains":
        needles = [str(item) for item in (value or [])]
        missing = [item for item in needles if item not in text]
        passed = not missing
        detail = f"缺少：{missing}" if missing else "全部命中"
    elif kind == "contains_any":
        needles = [str(item) for item in (value or [])]
        matched = [item for item in needles if item in text]
        passed = bool(matched)
        detail = f"命中：{matched}" if matched else f"均未命中：{needles}"
    elif kind == "not_contains":
        needles = [str(item) for item in (value or [])]
        found = [item for item in needles if item in text]
        passed = not found
        detail = f"出现禁止内容：{found}" if found else "未出现禁止内容"
    elif kind == "regex":
        try:
            passed = re.search(str(value), text, re.MULTILINE) is not None
        except re.error as exc:
            passed = False
            detail = f"正则非法：{exc}"
        else:
            detail = "命中" if passed else f"未命中：{value}"
    elif kind == "max_chars":
        limit = int(value)
        passed = len(text) <= limit
        detail = f"{len(text)}/{limit} 字符"
    elif kind == "min_chars":
        limit = int(value)
        passed = len(text) >= limit
        detail = f"{len(text)}/{limit} 字符"
    elif kind == "max_sentences":
        limit = int(value)
        count = _count_sentences(text)
        passed = count <= limit
        detail = f"{count}/{limit} 句"
    elif kind == "json_object":
        payload = _json_payload(text)
        config = value if isinstance(value, dict) else {}
        required = [str(item) for item in config.get("required", [])]
        equals = config.get("equals") or {}
        allow_extra = bool(config.get("allow_extra", False))
        if not isinstance(payload, dict):
            detail = "输出不是 JSON 对象"
        else:
            missing = [key for key in required if key not in payload]
            mismatched = [
                key
                for key, expected in equals.items()
                if payload.get(key) != expected
            ]
            extra = (
                [key for key in payload if key not in required]
                if not allow_extra
                else []
            )
            passed = not missing and not mismatched and not extra
            detail = (
                f"缺少字段：{missing}；值不符：{mismatched}；多余字段：{extra}"
                if not passed
                else "JSON 结构符合"
            )
    elif kind == "language":
        language = str(value).lower()
        cjk = len(_CJK_RE.findall(text))
        latin = len(_LATIN_RE.findall(text))
        if language == "zh":
            passed = cjk >= 2 and cjk >= latin
            detail = f"中文 {cjk} / 拉丁 {latin}"
        elif language == "en":
            passed = latin >= 2 and cjk == 0
            detail = f"英文 {latin} / 中文 {cjk}"
        else:
            detail = f"未知语言：{language}"
    elif kind == "no_code_fence":
        passed = "```" not in text
        detail = "包含代码围栏" if not passed else "无代码围栏"
    else:  # pragma: no cover - 加载阶段已拦截未知类型
        detail = f"未知检查类型：{kind}"

    return CheckResult(
        name=name, passed=passed, weight=float(check.weight), detail=detail
    )


def _artifact_check_one(check: ArtifactCheck, root: Path) -> CheckResult:
    name = f"artifact:{check.path}"
    root_resolved = root.resolve()
    target = (root_resolved / check.path).resolve()
    try:
        target.relative_to(root_resolved)
    except ValueError:
        return CheckResult(
            name=name,
            passed=False,
            weight=float(check.weight),
            detail="产物路径越界",
        )
    if not target.is_file():
        return CheckResult(
            name=name,
            passed=False,
            weight=float(check.weight),
            detail="产物不存在",
        )
    try:
        raw = target.read_bytes()
    except OSError as exc:
        return CheckResult(
            name=name,
            passed=False,
            weight=float(check.weight),
            detail=f"产物不可读：{exc}",
        )
    if len(raw) < int(check.min_bytes):
        return CheckResult(
            name=name,
            passed=False,
            weight=float(check.weight),
            detail=f"产物过小：{len(raw)}/{check.min_bytes} 字节",
        )
    text = raw.decode("utf-8", errors="replace")
    missing = [item for item in check.contains if item not in text]
    found = [item for item in check.forbidden if item in text]
    passed = not missing and not found
    detail = (
        "产物符合"
        if passed
        else f"缺少：{missing}；出现禁止内容：{found}"
    )
    return CheckResult(
        name=name, passed=passed, weight=float(check.weight), detail=detail
    )


def score_output(
    case: QualityCase,
    output: str,
    *,
    sandbox_dir: Path | None = None,
) -> QualityScore:
    """按固定规则评分；未提供沙箱目录时忽略产物检查。"""
    text = output or ""
    results: list[CheckResult] = []
    if case.checks:
        results.extend(_check_one(check, text) for check in case.checks)
    if sandbox_dir is not None:
        results.extend(
            _artifact_check_one(check, sandbox_dir) for check in case.artifacts
        )
    if not results:
        results.append(
            CheckResult(name="non_empty", passed=bool(text.strip()), weight=1.0)
        )
    total = sum(item.weight for item in results)
    earned = sum(item.weight for item in results if item.passed)
    score = earned / total if total > 0 else 0.0
    return QualityScore(
        case_id=case.case_id,
        score=score,
        passed=score >= case.min_score,
        checks=results,
        output_chars=len(text),
    )


def _parse_check(raw: Any, case_id: str) -> QualityCheck:
    if not isinstance(raw, dict):
        raise QualityCaseError(f"用例 {case_id} 的检查项必须是对象")
    kind = str(raw.get("kind") or "")
    if kind not in CHECK_KINDS:
        raise QualityCaseError(f"用例 {case_id} 含未知检查类型：{kind!r}")
    weight = float(raw.get("weight", 1.0))
    if weight <= 0:
        raise QualityCaseError(f"用例 {case_id} 的检查权重必须为正数")
    return QualityCheck(
        kind=kind,
        value=raw.get("value"),
        label=str(raw.get("label") or ""),
        weight=weight,
    )


def _parse_artifact(raw: Any, case_id: str) -> ArtifactCheck:
    if not isinstance(raw, dict):
        raise QualityCaseError(f"用例 {case_id} 的产物检查必须是对象")
    path = str(raw.get("path") or "").strip().replace("\\", "/")
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise QualityCaseError(f"用例 {case_id} 的产物路径非法：{path!r}")
    return ArtifactCheck(
        path=path,
        contains=tuple(str(item) for item in raw.get("contains", [])),
        forbidden=tuple(str(item) for item in raw.get("forbidden", [])),
        min_bytes=int(raw.get("min_bytes", 1)),
        weight=float(raw.get("weight", 1.0)),
    )


def parse_quality_case(raw: Any, index: int = 0) -> QualityCase:
    if not isinstance(raw, dict):
        raise QualityCaseError(f"第 {index} 个用例必须是对象")
    case_id = str(raw.get("id") or "").strip()
    if not case_id:
        raise QualityCaseError(f"第 {index} 个用例缺少 id")
    category = str(raw.get("category") or "").strip()
    if category not in _CATEGORIES:
        raise QualityCaseError(
            f"用例 {case_id} 的 category 非法：{category!r}"
        )
    objective = str(raw.get("objective") or "").strip()
    if not objective:
        raise QualityCaseError(f"用例 {case_id} 缺少 objective")
    checks = tuple(
        _parse_check(item, case_id) for item in raw.get("checks", [])
    )
    artifacts = tuple(
        _parse_artifact(item, case_id) for item in raw.get("artifacts", [])
    )
    if not checks and not artifacts:
        raise QualityCaseError(f"用例 {case_id} 没有任何检查项")
    min_score = float(raw.get("min_score", 1.0))
    if not 0 < min_score <= 1:
        raise QualityCaseError(f"用例 {case_id} 的 min_score 必须在 (0, 1]")
    return QualityCase(
        case_id=case_id,
        category=category,
        objective=objective,
        constraints=str(raw.get("constraints") or ""),
        acceptance_criteria=str(raw.get("acceptance_criteria") or ""),
        expected_task_type=str(raw.get("expected_task_type") or ""),
        checks=checks,
        artifacts=artifacts,
        min_score=min_score,
    )


def load_quality_cases(path: str | Path) -> list[QualityCase]:
    """加载并校验固定用例文件；重复 id 或非法字段直接失败。"""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityCaseError(f"质量用例文件读取失败：{source}（{exc}）") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise QualityCaseError(f"质量用例文件格式非法：{source}")
    cases = [
        parse_quality_case(item, index)
        for index, item in enumerate(payload["cases"])
    ]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.case_id in seen:
            duplicates.add(case.case_id)
        seen.add(case.case_id)
    if duplicates:
        raise QualityCaseError(f"质量用例 id 重复：{sorted(duplicates)}")
    return cases


def summarize_scores(scores: Iterable[QualityScore]) -> dict[str, Any]:
    """汇总评分结果（供脚本证据与 CI 断言使用）。"""
    items = list(scores)
    failed = [item.case_id for item in items if not item.passed]
    return {
        "total": len(items),
        "passed": len(items) - len(failed),
        "failed": len(failed),
        "failed_cases": failed,
        "average_score": (
            round(sum(item.score for item in items) / len(items), 4)
            if items
            else 0.0
        ),
    }

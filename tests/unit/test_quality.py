"""固定质量回归集与指令遵循评分的确定性测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from so_agent.quality import (
    ArtifactCheck,
    QualityCase,
    QualityCaseError,
    QualityCheck,
    load_quality_cases,
    parse_quality_case,
    score_output,
)

CASES_PATH = (
    Path(__file__).resolve().parent.parent
    / "baseline"
    / "quality_cases.json"
)


def _case(*checks: QualityCheck, **kwargs) -> QualityCase:
    return QualityCase(
        case_id=kwargs.pop("case_id", "case"),
        category=kwargs.pop("category", "direct_qa"),
        objective=kwargs.pop("objective", "目标"),
        checks=checks,
        **kwargs,
    )


def test_fixed_dataset_covers_five_categories():
    cases = load_quality_cases(CASES_PATH)
    assert len(cases) >= 14
    assert {case.category for case in cases} == {
        "direct_qa",
        "deep_reasoning",
        "single_file",
        "multi_file",
        "instruction_following",
    }
    assert len({case.case_id for case in cases}) == len(cases)
    assert any(case.artifacts for case in cases)
    assert any(
        check.kind == "max_sentences"
        for case in cases
        for check in case.checks
    )


def test_contains_and_forbidden_terms():
    case = _case(
        QualityCheck("contains", ["关键值"]),
        QualityCheck("not_contains", ["错误结论"]),
    )
    passed = score_output(case, "这里包含关键值")
    assert passed.passed
    assert passed.score == 1.0
    failed = score_output(case, "这里包含关键值与错误结论")
    assert not failed.passed
    assert failed.score == 0.5


def test_contains_any_regex_and_length_limits():
    case = _case(
        QualityCheck("contains_any", ["42"]),
        QualityCheck("regex", r"^\s*42\s*$"),
        QualityCheck("max_chars", 4),
        QualityCheck("min_chars", 2),
    )
    assert score_output(case, "42").passed
    result = score_output(case, "答案是 42")
    assert not result.passed
    assert {item.name: item.passed for item in result.checks} == {
        "contains_any": True,
        "regex": False,
        "max_chars": False,
        "min_chars": True,
    }


def test_max_sentences_counts_chinese_and_ignores_decimals():
    one = _case(QualityCheck("max_sentences", 1))
    assert score_output(one, "圆周率约为 3.14。").passed
    assert not score_output(one, "第一句。第二句。").passed


def test_json_object_structure_and_no_extra_fields():
    case = _case(
        QualityCheck(
            "json_object",
            {
                "required": ["name", "value"],
                "equals": {"name": "so-agent", "value": 1},
                "allow_extra": False,
            },
        ),
        QualityCheck("no_code_fence"),
    )
    assert score_output(case, '{"name": "so-agent", "value": 1}').passed
    assert not score_output(
        case, '{"name": "so-agent", "value": 1, "extra": true}'
    ).passed
    fenced = score_output(
        case, '```json\n{"name": "so-agent", "value": 1}\n```'
    )
    assert not fenced.passed


def test_language_checks():
    zh = _case(QualityCheck("language", "zh"))
    assert score_output(zh, "这是中文回答").passed
    assert not score_output(zh, "This is an English answer").passed
    en = _case(QualityCheck("language", "en"))
    assert score_output(en, "This is an English answer").passed
    assert not score_output(en, "这是中文回答").passed


def test_artifact_checks(tmp_path: Path):
    (tmp_path / "note.md").write_text("FAST_PATH_OK\n", encoding="utf-8")
    case = _case(
        artifacts=(
            ArtifactCheck("note.md", contains=("FAST_PATH_OK",), min_bytes=5),
        )
    )
    assert score_output(case, "", sandbox_dir=tmp_path).passed
    assert not score_output(case, "", sandbox_dir=tmp_path / "missing").passed

    unsafe = _case(artifacts=(ArtifactCheck("../outside.txt"),))
    result = score_output(unsafe, "", sandbox_dir=tmp_path)
    assert not result.passed
    assert "越界" in result.checks[0].detail


def test_min_score_allows_partial_pass():
    case = _case(
        QualityCheck("contains", ["必选"]),
        QualityCheck("contains", ["可选"], weight=3),
        min_score=0.7,
    )
    score = score_output(case, "必选")
    assert score.score == 0.25
    assert not score.passed
    relaxed = _case(
        QualityCheck("contains", ["必选"]),
        QualityCheck("contains", ["可选"], weight=3),
        min_score=0.2,
    )
    assert score_output(relaxed, "必选").passed


def test_parse_quality_case_rejects_bad_input():
    with pytest.raises(QualityCaseError):
        parse_quality_case(
            {
                "id": "x",
                "category": "direct_qa",
                "objective": "目标",
                "checks": [{"kind": "unknown", "value": 1}],
            }
        )
    with pytest.raises(QualityCaseError):
        parse_quality_case(
            {
                "id": "x",
                "category": "direct_qa",
                "objective": "目标",
                "checks": [{"kind": "contains", "value": ["x"]}],
                "artifacts": [{"path": "../escape.txt"}],
            }
        )


def test_loader_rejects_duplicate_ids(tmp_path: Path):
    payload = {
        "version": 1,
        "cases": [
            {
                "id": "dup",
                "category": "direct_qa",
                "objective": "目标",
                "checks": [{"kind": "contains", "value": ["x"]}],
            },
            {
                "id": "dup",
                "category": "direct_qa",
                "objective": "目标",
                "checks": [{"kind": "contains", "value": ["x"]}],
            },
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QualityCaseError):
        load_quality_cases(path)

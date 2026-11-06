"""可机检验收规则：只读沙箱文件与执行结果，不调用模型。

规划器在 ``SubtaskSpec.verification`` 中声明规则，执行器在子任务返回
后确定性运行。任一规则失败即视为验收失败（由调用方翻转结果），使
低风险任务可以依赖客观证据而不是语义评审。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from so_agent.json_utils import extract_json_object
from so_agent.models import ExecutionResult, SubtaskSpec, VerificationRule


def _parse_json_text(text: Any) -> tuple[Any, str | None]:
    """严格解析优先，失败后从文本中提取 JSON 对象。"""
    if not isinstance(text, str) or not text.strip():
        return None, "内容不是合法 JSON"
    try:
        return json.loads(text), None
    except (json.JSONDecodeError, TypeError):
        pass
    repaired = extract_json_object(text)
    if repaired is not None:
        return repaired, None
    return None, "内容不是合法 JSON"


def _resolve_in_root(root: Path, relative_path: str) -> Path:
    """解析沙箱内相对路径并做越界防护。

    Raises:
        ValueError: 路径为空或解析结果超出沙箱根目录时。
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("path 不能为空")
    candidate = Path(relative_path)
    target = (
        candidate.expanduser().resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    root_text = os.path.normcase(str(root))
    target_text = os.path.normcase(str(target))
    if not (
        target_text == root_text or target_text.startswith(root_text + os.sep)
    ):
        raise ValueError(f"路径越界：{relative_path!r} 不在沙箱 {root} 之内")
    return target


def _load_json_value(
    rule: VerificationRule, result: ExecutionResult, root: Path
) -> tuple[Any, str | None]:
    """读取规则作用对象（沙箱文件或执行输出）并解析为 JSON。"""
    if rule.path:
        try:
            target = _resolve_in_root(root, rule.path)
        except ValueError as exc:
            return None, str(exc)
        if not target.is_file():
            return None, f"文件不存在：{rule.path}"
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"文件读取失败：{rule.path}（{exc}）"
        return _parse_json_text(text)
    # 无 path 时按“执行输出”校验：先看结果正文，再看 code_agent
    # execution_result.output（脚本打印的 JSON 常落在这里，而正文是摘要）。
    candidates: list[Any] = [result.output or ""]
    execution_output = (result.metadata or {}).get("execution_output")
    if isinstance(execution_output, str):
        candidates.append(execution_output)
    error = "内容不是合法 JSON"
    for text in candidates:
        parsed, failure = _parse_json_text(text)
        if failure is None:
            return parsed, None
        error = failure
    return None, error


def _dig(data: Any, field: str) -> tuple[Any, bool]:
    """按点分路径读取 JSON 字段；返回 ``(值, 是否命中)``。"""
    current = data
    for part in field.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None, False
    return current, True


def _exit_code_in(data: dict[str, Any]) -> Any:
    """在常见嵌套位置查找 exit_code。

    code_agent 把沙箱退出码放在 ``execution_result.metadata`` 内，
    与顶层的 ``execution_result.exit_code`` 是两种真实存在的形状。
    """
    if "exit_code" in data:
        return data.get("exit_code")
    for key in ("execution_result", "metadata"):
        nested = data.get(key)
        if isinstance(nested, dict):
            found = _exit_code_in(nested)
            if found is not None:
                return found
    return None


def _extract_exit_code(result: ExecutionResult) -> Any:
    """从结果元数据或输出 JSON 中提取 exit_code。"""
    metadata = result.metadata or {}
    found = _exit_code_in(metadata)
    if found is not None:
        return found
    payload: Any = result.output
    for _ in range(3):
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(payload, dict):
            return None
        found = _exit_code_in(payload)
        if found is not None:
            return found
        payload = payload.get("output")
    return None


def evaluate_rule(
    rule: VerificationRule, result: ExecutionResult, root: Path
) -> str | None:
    """运行单条规则；通过返回 None，失败返回原因文本。"""
    root = Path(root).expanduser().resolve()
    if rule.kind in ("file_exists", "file_non_empty"):
        if not rule.path:
            return f"{rule.kind} 缺少 path"
        try:
            target = _resolve_in_root(root, rule.path)
        except ValueError as exc:
            return f"{rule.kind} 路径非法：{exc}"
        if not target.exists():
            return f"{rule.kind} 失败：文件不存在 {rule.path}"
        if rule.kind == "file_non_empty":
            if not target.is_file() or target.stat().st_size == 0:
                return f"{rule.kind} 失败：文件为空 {rule.path}"
        return None

    if rule.kind == "text_contains":
        if not rule.text:
            return "text_contains 缺少 text"
        if not rule.path:
            return (
                "text_contains 必须指定 path（产物文件）；"
                "不允许检查模型输出措辞"
            )
        try:
            target = _resolve_in_root(root, rule.path)
        except ValueError as exc:
            return f"text_contains 路径非法：{exc}"
        if not target.is_file():
            return f"text_contains 失败：文件不存在 {rule.path}"
        try:
            content = target.read_text(encoding="utf-8")
        except OSError as exc:
            return f"text_contains 读取失败：{rule.path}（{exc}）"
        if rule.text not in content:
            return f"text_contains 失败：未找到 {rule.text!r}"
        return None

    if rule.kind == "exit_code":
        observed = _extract_exit_code(result)
        expected = 0 if rule.value is None else rule.value
        if observed is None:
            return "exit_code 失败：缺少退出码证据"
        try:
            matched = float(observed) == float(expected)
        except (TypeError, ValueError):
            return f"exit_code 失败：退出码不是数值（{observed!r}）"
        return None if matched else f"exit_code 失败：期望 {expected}，实际 {observed}"

    if rule.kind in ("json_field", "numeric_match"):
        if not rule.field:
            return f"{rule.kind} 缺少 field"
        data, error = _load_json_value(rule, result, root)
        if error is not None:
            return f"{rule.kind} 失败：{error}"
        observed, found = _dig(data, rule.field)
        if not found:
            return f"{rule.kind} 失败：字段不存在 {rule.field}"
        if rule.kind == "json_field":
            if rule.equals is not None and observed != rule.equals:
                return (
                    f"json_field 失败：字段 {rule.field} 期望 "
                    f"{rule.equals!r}，实际 {observed!r}"
                )
            return None
        if rule.value is None:
            return "numeric_match 缺少 value"
        try:
            numeric = float(observed)
        except (TypeError, ValueError):
            return f"numeric_match 失败：字段 {rule.field} 不是数值（{observed!r}）"
        if abs(numeric - float(rule.value)) > float(rule.tolerance):
            return (
                f"numeric_match 失败：字段 {rule.field} 期望 {rule.value}，"
                f"实际 {observed}"
            )
        return None

    return f"未知规则类型：{rule.kind}"


def evaluate_verification_rules(
    spec: SubtaskSpec, result: ExecutionResult, root: Path
) -> list[str]:
    """运行子任务声明的全部规则，返回失败原因列表（空表示全部通过）。"""
    failures: list[str] = []
    for rule in spec.verification:
        failure = evaluate_rule(rule, result, root)
        if failure is not None:
            failures.append(failure)
    return failures


__all__ = ["evaluate_rule", "evaluate_verification_rules"]

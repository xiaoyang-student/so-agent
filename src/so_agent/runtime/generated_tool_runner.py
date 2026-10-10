"""生成工具沙箱运行器。

职责：把动态生成的工具放到**沙箱子进程**执行（而非主进程 importlib 加载），
并强制其返回值为 JSON 可序列化。供 MCP handler（调用工具）与工作流的
沙箱运行验收（注册前试跑）共用。

工具运行器源码在沙箱中执行：importlib 加载 ``entry_file`` → 解析入口函数
（``run`` / 与工具同名 / ``main`` / ``execute``）→ 以 ``**arguments`` 调用入口
→ 把返回值 JSON 序列化后写到 stdout（带哨兵前缀）。返回值非 JSON 可序列化
或执行抛异常时输出 ``{"ok": false, "error": ...}`` 形态，由本模块解析为异常。

设计要点：
- 工具代码在子进程执行，主进程只收回 JSON，杜绝主进程任意代码执行；
- 参数经 ``json.dumps`` 后以 ``repr`` 字符串字面量内嵌，运行器内 ``json.loads``
  还原，避免直接拼接导致的代码注入；
- 返回值强制 JSON：运行器 ``json.dumps`` 不带 ``default``，非可序列化即报错。
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 仅类型标注，避免运行期循环依赖
    from so_agent.runtime.sandbox import SandboxRunner

# 结果哨兵：工具运行器把结果 JSON 以此前缀写入 stdout
_RESULT_SENTINEL: str = "__SO_AGENT_TOOL_RESULT__"

# 生成工具入口函数候选（按优先级：与工具同名优先，其次 run/main/execute）
_ENTRY_CANDIDATES: tuple[str, ...] = ("run", "main", "execute")


class GeneratedToolExecutionError(Exception):
    """生成工具在沙箱中执行失败（加载失败 / 无入口 / 返回非 JSON / 运行异常）。"""


def build_runner_code(
    entry_file: str | Path, tool_name: str, arguments: dict[str, Any]
) -> str:
    """构造在沙箱子进程执行的工具运行器源码。

    Args:
        entry_file: 生成工具入口源码文件（绝对路径）。
        tool_name: 工具名（用于优先解析同名入口函数）。
        arguments: 传给入口函数 ``run(**arguments)`` 的关键字参数。

    Returns:
        可直接交给 ``SandboxRunner.execute`` 执行的 Python 源码字符串。
    """
    entry_lit = repr(json.dumps(str(entry_file)))
    name_lit = repr(json.dumps(str(tool_name)))
    args_lit = repr(json.dumps(arguments, ensure_ascii=False, default=str))
    candidates_lit = repr(list(_ENTRY_CANDIDATES))
    sentinel_lit = repr(_RESULT_SENTINEL)
    return (
        "import importlib.util, json\n"
        f"ENTRY = json.loads({entry_lit})\n"
        f"TOOL_NAME = json.loads({name_lit})\n"
        f"ARGS = json.loads({args_lit})\n"
        f"CANDIDATES = {candidates_lit}\n"
        f"SENTINEL = {sentinel_lit}\n"
        "def _emit(payload):\n"
        "    print(SENTINEL + json.dumps(payload, ensure_ascii=False))\n"
        "try:\n"
        "    spec = importlib.util.spec_from_file_location"
        "('so_agent_generated_tool', ENTRY)\n"
        "    if spec is None or spec.loader is None:\n"
        "        raise RuntimeError('无法构造模块加载器: ' + ENTRY)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "    func = None\n"
        "    for candidate in [TOOL_NAME] + CANDIDATES:\n"
        "        fn = getattr(module, candidate, None)\n"
        "        if callable(fn):\n"
        "            func = fn\n"
        "            break\n"
        "    if func is None:\n"
        "        raise RuntimeError('生成工具缺少可调用入口: ' + ENTRY)\n"
        "    result = func(**ARGS)\n"
        "    _emit({'ok': True, 'result': result})\n"
        "except Exception as exc:\n"
        "    _emit({'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)})\n"
    )


def parse_runner_output(output: str) -> Any:
    """从工具运行器 stdout 解析结果（校验为 JSON 可序列化）。

    Args:
        output: 沙箱执行的 stdout 文本（含结果哨兵行）。

    Returns:
        工具入口函数的返回值（已确认为 JSON 可序列化）。

    Raises:
        GeneratedToolExecutionError: 无结果哨兵、结果非合法 JSON，
            或工具执行报错（``ok`` 为假）时。
    """
    text = output or ""
    idx = text.rfind(_RESULT_SENTINEL)
    if idx == -1:
        detail = text.strip() or "无输出"
        raise GeneratedToolExecutionError(f"生成工具运行器未产出结果哨兵：{detail!r}")
    payload_text = text[idx + len(_RESULT_SENTINEL):].splitlines()[0].strip()
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise GeneratedToolExecutionError(
            f"生成工具结果不是合法 JSON：{payload_text!r}"
        ) from exc
    if not isinstance(payload, dict) or not payload.get("ok"):
        error = payload.get("error") if isinstance(payload, dict) else payload
        raise GeneratedToolExecutionError(f"生成工具执行失败：{error}")
    return payload.get("result")


async def run_generated_tool_in_sandbox(
    sandbox: "SandboxRunner",
    entry_file: str | Path,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    task_id: str | None = None,
    run_id: str | None = None,
) -> Any:
    """在沙箱子进程执行生成工具并返回其结果（已校验为 JSON 可序列化）。

    Args:
        sandbox: 沙箱执行器（``SandboxRunner``）。
        entry_file: 生成工具入口源码文件（绝对路径）。
        tool_name: 工具名。
        arguments: 入口函数 ``run(**arguments)`` 的关键字参数。
        task_id: 沙箱目录隔离用的任务编号（默认 ``generated-tool``）。
        run_id: 本次运行编号（默认随机）。

    Returns:
        工具入口函数的返回值（JSON 可序列化）。

    Raises:
        GeneratedToolExecutionError: 工具加载/执行失败或返回非 JSON 时。
    """
    code = build_runner_code(entry_file, tool_name, arguments)
    exec_result = await sandbox.execute(
        code,
        task_id or "generated-tool",
        run_id or uuid.uuid4().hex[:12],
    )
    return parse_runner_output(exec_result.output or "")

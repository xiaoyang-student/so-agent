"""生成工具静态安全检查（准入第一道门）。

目标不是证明绝对安全，而是把已知高风险行为（网络、进程、动态执行、
破坏性文件操作、内省逃逸）拦截在评审与注册之前。检查失败即拒绝准入；
文件写入等中风险行为记录为告警，由评审与沙箱边界继续约束。静态检查
属于纵深防御的一层，不是隔离边界，真正隔离仍由沙箱后端负责。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

_MAX_SOURCE_BYTES = 200_000

# 禁止导入的模块根名：网络、进程、动态加载、序列化逃逸
_DENIED_IMPORTS: frozenset[str] = frozenset(
    {
        "subprocess",
        "socket",
        "ssl",
        "ctypes",
        "multiprocessing",
        "pty",
        "resource",
        "signal",
        "importlib",
        "pickle",
        "marshal",
        "shelve",
        "dbm",
        "http",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
        "ftplib",
        "smtplib",
        "telnetlib",
        "paramiko",
        "webbrowser",
        "xmlrpc",
        "asyncio.subprocess",
        "builtins",
        "posix",
        "nt",
    }
)

# 禁止从模块导入的具体名字（绕过整模块禁令的入口）
_DENIED_IMPORT_NAMES: dict[str, frozenset[str]] = {
    "sys": frozenset({"modules", "settrace", "setprofile", "_getframe"}),
}

# 禁止直接调用的内建函数
_DENIED_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "breakpoint",
        "getattr",
        "setattr",
        "delattr",
        "vars",
        "globals",
        "locals",
    }
)

# 禁止访问的危险属性（沙箱逃逸/内省链）
_DENIED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__globals__",
        "__subclasses__",
        "__builtins__",
        "__loader__",
        "__code__",
        "__closure__",
        "__mro__",
        "__base__",
        "__bases__",
        "__dict__",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__class__",
        "__reduce__",
        "__reduce_ex__",
    }
)

# 禁止的 os / shutil / pathlib / sys 高危属性名（调用或引用均拦截）
_DENIED_MODULE_ATTRS: dict[str, frozenset[str]] = {
    "sys": frozenset(
        {"modules", "settrace", "setprofile", "_getframe", "addaudithook"}
    ),
    "os": frozenset(
        {
            "system",
            "popen",
            "execv",
            "execve",
            "execvp",
            "execvpe",
            "execl",
            "execlp",
            "spawnv",
            "spawnve",
            "spawnl",
            "fork",
            "forkpty",
            "kill",
            "killpg",
            "putenv",
            "unsetenv",
            "remove",
            "unlink",
            "rmdir",
            "removedirs",
            "rename",
            "replace",
            "chmod",
            "chown",
            "symlink",
            "environ",
        }
    ),
    "shutil": frozenset({"rmtree", "move", "copyfile", "copytree"}),
    "pathlib": frozenset({"unlink", "rmdir", "rename", "replace", "symlink_to"}),
}


@dataclass(frozen=True)
class ToolSafetyAssessment:
    """静态检查结论；``passed`` 为 False 时不得进入评审。"""

    passed: bool
    issues: list[str]
    warnings: list[str]


def _attr_chain(node: ast.AST) -> str:
    """把 ``os.path.join`` 之类的属性链还原为点分字符串。"""
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _root_name(node: ast.AST) -> str:
    current = node
    while isinstance(current, (ast.Attribute, ast.Subscript)):
        current = current.value
    return current.id if isinstance(current, ast.Name) else ""


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return _attr_chain(func)
    return ""


def assess_tool_source(
    source: str, *, tool_name: str = "tool"
) -> ToolSafetyAssessment:
    """对工具源码做静态安全检查。

    Args:
        source: 待检查的完整 Python 源码。
        tool_name: 工具名（仅用于问题描述）。

    Returns:
        ``ToolSafetyAssessment``：``passed=False`` 时 ``issues`` 非空。
    """
    issues: list[str] = []
    warnings: list[str] = []

    if len(source.encode("utf-8")) > _MAX_SOURCE_BYTES:
        return ToolSafetyAssessment(
            passed=False,
            issues=[f"源码超过大小上限 {_MAX_SOURCE_BYTES} 字节"],
            warnings=[],
        )
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ToolSafetyAssessment(
            passed=False,
            issues=[f"源码无法解析：{exc.msg}（第 {exc.lineno} 行）"],
            warnings=[],
        )

    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                aliases[alias.asname or root] = root

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in _DENIED_IMPORTS or alias.name in _DENIED_IMPORTS:
                    issues.append(f"禁止导入高风险模块：{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".", 1)[0]
            if module in _DENIED_IMPORTS or (node.module or "") in _DENIED_IMPORTS:
                issues.append(f"禁止导入高风险模块：{node.module}")
            denied_names = _DENIED_IMPORT_NAMES.get(module) or frozenset()
            denied_names = denied_names | _DENIED_MODULE_ATTRS.get(
                module, frozenset()
            )
            for alias in node.names:
                if alias.name in denied_names:
                    issues.append(
                        f"禁止导入高风险名称：{node.module}.{alias.name}"
                    )
                if alias.name == "*" and denied_names:
                    issues.append(f"禁止通配导入高风险模块：{node.module}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            short = name.rsplit(".", 1)[-1]
            if short in _DENIED_CALLS:
                issues.append(f"禁止调用动态执行函数：{name}")
            if short == "open":
                mode = node.args[1] if len(node.args) > 1 else None
                if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
                    if any(flag in mode.value for flag in ("w", "a", "x", "+")):
                        warnings.append("工具包含文件写入操作（受沙箱与写作用域约束）")
                else:
                    warnings.append("工具调用 open() 且模式非字面量，无法静态判定写入范围")
            if short in {"write_text", "write_bytes"}:
                warnings.append(f"工具包含文件写入操作：{name}")
            if short in {"makedirs", "mkdir", "touch"}:
                warnings.append(f"工具包含目录/文件创建操作：{name}")
        elif isinstance(node, ast.Attribute):
            chain = _attr_chain(node)
            if node.attr in _DENIED_ATTRIBUTES:
                issues.append(f"禁止访问危险属性：{chain}")
            root = _root_name(node)
            denied_attrs = _DENIED_MODULE_ATTRS.get(aliases.get(root, root))
            if denied_attrs and node.attr in denied_attrs:
                issues.append(f"禁止访问高风险属性：{chain}")
        elif isinstance(node, ast.Name):
            if node.id == "__builtins__":
                issues.append("禁止访问 __builtins__")

    return ToolSafetyAssessment(
        passed=not issues,
        issues=sorted(set(issues)),
        warnings=sorted(set(warnings)),
    )


__all__ = ["ToolSafetyAssessment", "assess_tool_source"]

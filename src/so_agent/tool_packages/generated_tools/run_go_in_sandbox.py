"""run_go_in_sandbox: 在沙箱中运行/分析 Go 源代码的动态工具。

入口函数 run(**kwargs)，返回值保证 JSON 可序列化。
- 校验参数（source 非空字符串、timeout 为 1..30 整数、dry_run 为 bool）；
- dry_run=True 时仅静态解析（package 子句、行数、字符数），不执行；
- 否则尝试调用本机 go 工具链执行（go run），捕获 stdout/stderr/exit_code；
  沙箱内无 go 时返回 go_not_found 结构化错误。
"""

import re


def run(**kwargs):
    source = kwargs.get("source")
    timeout = kwargs.get("timeout", 10)
    dry_run = kwargs.get("dry_run", False)

    # ---- 参数校验 ----
    if not isinstance(source, str) or not source.strip():
        return {"success": False, "mode": "validate", "error_type": "invalid_input",
                "error": "source 必须为非空字符串"}
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not (1 <= timeout <= 30):
        return {"success": False, "mode": "validate", "error_type": "invalid_input",
                "error": "timeout 必须为 1 到 30 之间的整数"}
    if not isinstance(dry_run, bool):
        return {"success": False, "mode": "validate", "error_type": "invalid_input",
                "error": "dry_run 必须为布尔值"}

    # ---- 静态解析 ----
    m = re.search(r"^\s*package\s+([A-Za-z_][A-Za-z0-9_]*)", source, re.MULTILINE)
    package = m.group(1) if m else None

    if dry_run:
        return {
            "success": True,
            "mode": "dry_run",
            "package": package,
            "has_package_main": package == "main",
            "lines": len(source.splitlines()),
            "chars": len(source),
        }

    # ---- 真实执行（依赖本机 go 工具链） ----
    import os
    import shutil
    import subprocess
    import tempfile

    go = shutil.which("go")
    if not go:
        return {"success": False, "mode": "run", "go_available": False,
                "error_type": "go_not_found",
                "error": "未找到 go 可执行文件，无法在沙箱中运行 Go 代码",
                "stdout": "", "stderr": "", "exit_code": None}

    with tempfile.TemporaryDirectory(dir=".") as td:
        path = os.path.join(td, "main.go")
        with open(path, "w", encoding="utf-8") as f:
            f.write(source)
        try:
            p = subprocess.run([go, "run", path], capture_output=True, text=True,
                               timeout=timeout, cwd=td)
            if p.returncode == 0:
                return {"success": True, "mode": "run", "go_available": True,
                        "error_type": None, "error": None,
                        "stdout": p.stdout, "stderr": p.stderr,
                        "exit_code": p.returncode}
            return {"success": False, "mode": "run", "go_available": True,
                    "error_type": "go_run_failed", "error": "go run 退出码非 0",
                    "stdout": p.stdout, "stderr": p.stderr,
                    "exit_code": p.returncode}
        except subprocess.TimeoutExpired:
            return {"success": False, "mode": "run", "go_available": True,
                    "error_type": "timeout",
                    "error": "执行超时（%d 秒）" % timeout,
                    "stdout": "", "stderr": "", "exit_code": None}

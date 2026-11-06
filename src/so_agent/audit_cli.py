"""审计链校验 CLI：``so-agent-audit verify [path]``。"""

from __future__ import annotations

import argparse
import json
import os
import sys

from so_agent.runtime.audit_trail import (
    AUDIT_TRAIL_ENV,
    verify_audit_trail,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="so-agent-audit",
        description="校验 SO-Agent 跨任务防篡改审计链",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="逐条复核哈希链")
    verify.add_argument(
        "path",
        nargs="?",
        default="",
        help=f"审计库路径；省略时读取 {AUDIT_TRAIL_ENV}",
    )
    verify.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    args = parser.parse_args(argv)

    path = args.path.strip() or os.environ.get(AUDIT_TRAIL_ENV, "").strip()
    if not path:
        message = f"未提供审计库路径，且环境变量 {AUDIT_TRAIL_ENV} 为空"
        if args.json:
            print(json.dumps({"ok": False, "reason": message}, ensure_ascii=False))
        else:
            print(message, file=sys.stderr)
        return 2

    result = verify_audit_trail(path)
    if args.json:
        print(
            json.dumps(
                {
                    "ok": result.ok,
                    "entries": result.entries,
                    "first_bad_seq": result.first_bad_seq,
                    "reason": result.reason,
                    "path": path,
                },
                ensure_ascii=False,
            )
        )
    elif result.ok:
        print(f"审计链校验通过：{result.entries} 条记录，路径 {path}")
    else:
        print(
            f"审计链校验失败：{result.reason}"
            f"（记录数 {result.entries}，首个异常序号 {result.first_bad_seq}）",
            file=sys.stderr,
        )
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

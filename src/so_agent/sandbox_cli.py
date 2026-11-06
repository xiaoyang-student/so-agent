"""沙箱运维命令：doctor / selftest / setup。

- ``doctor``：只读体检——当前平台、配置后端、解析结果、各后端可用性；
- ``selftest``：真实执行探针代码，验证文件系统、网络、环境变量边界；
- ``setup``：检测/引导容器运行时，校验镜像，并可写入最小 ``.env`` 配置。

设计约定：本命令不会静默安装系统级软件，也不会覆盖已有配置；安装动作
只输出与当前平台匹配的命令，由使用者确认后执行。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from so_agent.config import get_settings
from so_agent.runtime.sandbox import available_sandbox_backends
from so_agent.runtime.sandbox_container import detect_container_runtime
from so_agent.runtime.sandbox_health import doctor_report, run_selftest


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _is_windows() -> bool:
    return os.name == "nt"


def _install_hint() -> list[str]:
    system = platform.system()
    if system == "Windows":
        release = platform.release()
        return [
            f"当前平台：Windows {release}（需要 Win10 2004+ / Win11 与 WSL2）",
            "1) winget install --id Docker.DockerDesktop -e",
            "   或 winget install --id RedHat.Podman-Desktop -e",
            "2) 启用 WSL2：wsl --install（如尚未安装），重启后启动 Docker Desktop",
            "3) 重新运行：so-agent-sandbox setup",
        ]
    if system == "Darwin":
        return [
            "当前平台：macOS",
            "1) brew install --cask docker",
            "   或 brew install podman && podman machine init && podman machine start",
            "2) 启动 Docker Desktop 后重新运行：so-agent-sandbox setup",
        ]
    return [
        f"当前平台：{system}",
        "1) curl -fsSL https://get.docker.com | sh（或使用发行版 docker/podman 包）",
        "2) sudo usermod -aG docker \"$USER\" 后重新登录",
        "3) 重新运行：so-agent-sandbox setup",
    ]


def _cmd_doctor(args: argparse.Namespace) -> int:
    settings = get_settings()
    report = doctor_report(settings)
    resolved = report["resolved_backend"]
    if args.json:
        _emit(report)
    else:
        os_info = report["os"]
        print(f"平台：{os_info['system']} {os_info['release']} ({os_info['machine']})")
        print(f"配置后端：{report['configured_backend']}")
        print(f"解析后端：{resolved}")
        print(
            "强制隔离：{}；允许本地回退：{}".format(
                "是" if report["require_isolation"] else "否",
                "是" if report["allow_local_fallback"] else "否",
            )
        )
        print(f"容器镜像：{report['container_image']}（网络：{report['container_network']}）")
        print(f"WSL 可用：{'是' if report['wsl'] else '否'}")
        print("后端探测：")
        for probe in report["backends"]:
            mark = "可用" if probe["available"] else "不可用"
            print(
                f"  - {probe['name']}: {mark}（隔离级别 {probe['isolation_level']}）"
                f" {probe['detail']}"
            )
        print(f"已注册后端：{', '.join(available_sandbox_backends())}")
        if resolved == "local":
            print("警告：当前将使用本地验证沙箱，它不是安全边界，仅限可信代码。")
    if args.require_isolation and resolved not in {"docker", "podman"}:
        print(
            "错误：--require-isolation 要求容器后端，但当前解析为 "
            f"{resolved!r}。",
            file=sys.stderr,
        )
        return 2
    return 0


def _cmd_selftest(args: argparse.Namespace) -> int:
    settings = get_settings()
    result = run_selftest(
        settings,
        backend=args.backend,
        sandbox_dir=args.sandbox_dir,
    )
    payload = result.as_dict()
    if args.json:
        _emit(payload)
    else:
        print(f"后端：{result.backend}（隔离级别 {result.isolation_level}）")
        if result.checks:
            labels = {
                "read_escape": "读取工作目录外文件",
                "write_escape": "写入工作目录外文件",
                "network_escape": "访问外网",
                "env_escape": "读取宿主环境密钥",
            }
            for key, value in result.checks.items():
                print(f"  - {labels.get(key, key)}：{'逃逸成功（不安全）' if value else '已阻断'}")
        if result.error:
            print(f"错误：{result.error}")
        print("结论：通过" if result.passed else "结论：未通过（未形成安全隔离）")
    return 0 if result.passed else 1


def _image_present(executable: str, image: str) -> bool:
    try:
        result = subprocess.run(
            [executable, "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _pull_image(executable: str, image: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [executable, "pull", image],
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    detail = (result.stderr or result.stdout or "").strip()[:1000]
    return result.returncode == 0, detail


def _update_env_file(path: Path, values: dict[str, str]) -> list[str]:
    """仅追加缺失键，不覆盖已有值；返回实际写入的键。"""
    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    present = {
        line.split("=", 1)[0].strip()
        for line in existing.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    }
    additions = [f"{key}={value}" for key, value in values.items() if key not in present]
    if not additions:
        return []
    prefix = "" if not existing or existing.endswith("\n") else "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(prefix + "\n".join(additions) + "\n")
    return [line.split("=", 1)[0] for line in additions]


def _cmd_setup(args: argparse.Namespace) -> int:
    settings = get_settings()
    runtime = detect_container_runtime("auto", refresh=True)
    image = (args.image or settings.sandbox_container_image).strip()
    payload: dict[str, object] = {
        "platform": platform.platform(),
        "runtime": None,
        "image": image,
        "image_present": False,
        "env_written": [],
        "hints": [],
    }

    if runtime is None:
        payload["hints"] = _install_hint()
        if args.json:
            _emit(payload)
        else:
            print("未检测到可用的 Docker/Podman 运行时。")
            for line in _install_hint():
                print(line)
            print(
                "在安装容器运行时之前，项目会按配置使用本地验证沙箱"
                "（非安全边界）。"
            )
        return 2

    payload["runtime"] = {
        "name": runtime.name,
        "version": runtime.version,
        "executable": runtime.executable,
    }
    present = _image_present(runtime.executable, image)
    payload["image_present"] = present
    if not present and args.pull_image:
        ok, detail = _pull_image(runtime.executable, image)
        payload["image_present"] = ok
        payload["pull_detail"] = detail
        if not ok and not args.json:
            print(f"镜像拉取失败：{detail}")
    if not payload["image_present"]:
        payload["hints"].append(
            f"镜像尚未就绪：{runtime.executable} pull {image}"
            "（或重新运行 setup --pull-image）"
        )

    if args.write_env:
        values = {
            "SO_AGENT_SANDBOX_BACKEND": "auto",
            "SO_AGENT_SANDBOX_CONTAINER_IMAGE": image,
        }
        if args.require_isolation:
            values["SO_AGENT_SANDBOX_REQUIRE_ISOLATION"] = "true"
        written = _update_env_file(Path(args.write_env), values)
        payload["env_written"] = written

    if args.json:
        _emit(payload)
    else:
        print(f"容器运行时：{runtime.name} {runtime.version}（{runtime.executable}）")
        print(f"沙箱镜像：{image} -> {'已就绪' if payload['image_present'] else '未就绪'}")
        for hint in payload["hints"]:
            print(f"提示：{hint}")
        if args.write_env:
            if payload["env_written"]:
                print(f"已写入 {args.write_env}：{', '.join(payload['env_written'])}")
            else:
                print(f"{args.write_env} 已包含所需配置，未做修改")
        print("下一步：so-agent-sandbox selftest")
    return 0 if payload["image_present"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="so-agent-sandbox",
        description="SO-Agent 沙箱体检、自检与运行时配置",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="只读体检当前沙箱能力")
    doctor.add_argument("--json", action="store_true", help="输出 JSON")
    doctor.add_argument(
        "--require-isolation",
        action="store_true",
        help="要求容器后端；未满足时以状态码 2 退出",
    )
    doctor.set_defaults(func=_cmd_doctor)

    selftest = sub.add_parser("selftest", help="执行隔离探针自检")
    selftest.add_argument("--backend", default=None, help="指定后端（默认取配置）")
    selftest.add_argument("--sandbox-dir", default=None, help="自检工作目录")
    selftest.add_argument("--json", action="store_true", help="输出 JSON")
    selftest.set_defaults(func=_cmd_selftest)

    setup = sub.add_parser("setup", help="检测容器运行时并准备沙箱镜像")
    setup.add_argument("--image", default=None, help="覆盖沙箱镜像")
    setup.add_argument(
        "--pull-image",
        action="store_true",
        help="镜像不存在时自动拉取",
    )
    setup.add_argument(
        "--write-env",
        default=None,
        metavar="PATH",
        help="把缺失配置追加到指定 .env（不覆盖已有键）",
    )
    setup.add_argument(
        "--require-isolation",
        action="store_true",
        help="写入 SO_AGENT_SANDBOX_REQUIRE_ISOLATION=true",
    )
    setup.add_argument("--json", action="store_true", help="输出 JSON")
    setup.set_defaults(func=_cmd_setup)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

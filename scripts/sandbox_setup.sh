#!/usr/bin/env sh
# 在 Linux / macOS 上准备 SO-Agent 基础安全沙箱。
# 用法：sh scripts/sandbox_setup.sh [--install] [--write-env] [--skip-pull] [--image IMG]
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON=${PYTHON:-python3}
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "未找到 Python，请先安装 Python 3.10+ 并加入 PATH" >&2
    exit 1
fi

INSTALL=0
WRITE_ENV=0
SKIP_PULL=0
IMAGE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --install) INSTALL=1 ;;
        --write-env) WRITE_ENV=1 ;;
        --skip-pull) SKIP_PULL=1 ;;
        --image) shift; IMAGE=${1:-} ;;
        *) echo "未知参数：$1" >&2; exit 1 ;;
    esac
    shift
done

runtime=""
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    runtime=docker
elif command -v podman >/dev/null 2>&1 && podman info >/dev/null 2>&1; then
    runtime=podman
fi

if [ -z "$runtime" ]; then
    echo "未检测到可用的 Docker/Podman。"
    if [ "$INSTALL" -eq 1 ] && [ "$(uname -s)" = "Linux" ]; then
        echo "将通过 get.docker.com 安装 Docker（需要 sudo）。"
        printf "继续安装？[y/N] "
        read -r answer
        if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
            curl -fsSL https://get.docker.com | sh
            echo "请把当前用户加入 docker 组后重新登录：sudo usermod -aG docker \"$USER\""
        fi
    else
        echo "Linux: curl -fsSL https://get.docker.com | sh  或安装发行版 docker/podman 包"
        echo "macOS: brew install --cask docker  或 brew install podman && podman machine init"
    fi
    echo "在此之前项目按配置回退到本地验证沙箱（不是安全边界）。"
    exit 2
fi

echo "检测到容器运行时：$runtime"
set -- -m so_agent.sandbox_cli setup
[ -n "$IMAGE" ] && set -- "$@" --image "$IMAGE"
[ "$SKIP_PULL" -eq 0 ] && set -- "$@" --pull-image
[ "$WRITE_ENV" -eq 1 ] && set -- "$@" --write-env "$REPO_ROOT/.env"

"$PYTHON" "$@"
"$PYTHON" -m so_agent.sandbox_cli selftest

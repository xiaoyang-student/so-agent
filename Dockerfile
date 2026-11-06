FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml constraints.txt ./
COPY src ./src
RUN python -m pip install -c constraints.txt ".[web]"

RUN useradd --create-home --uid 10001 soagent \
    && mkdir -p /app/sandbox /app/results \
    && chown -R soagent:soagent /app

USER soagent

# 容器内没有嵌套容器运行时，代码沙箱固定为 local（仅路径隔离）。
# 需要容器级代码隔离时请在宿主机运行，并使用 sandbox_backend=docker/podman。
ENV SO_AGENT_SANDBOX_BACKEND=local \
    SO_AGENT_SANDBOX_REQUIRE_ISOLATION=false \
    SO_AGENT_WEB_SANDBOX_DIR=/app/sandbox/web \
    SO_AGENT_WEB_RESULTS_DIR=/app/results

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3)"

CMD ["so-agent-web", "--host", "0.0.0.0", "--port", "8765"]

"""单元测试：四象限路径与长任务调度客户端的 HTTP 超时统一（任务 59）。

覆盖点（router.py 的同类验证见 test_router.py）：

- ``paths/persistent_task.py`` / ``paths/direct_answer.py`` /
  ``paths/deep_reasoning.py`` 的 ``AsyncOpenAI`` 客户端统一经
  ``runtime.config_loader.resolve_model_http_timeout`` 解析 timeout
  （与 orchestrator / 工具包客户端一致）；调度动作已改为经
  ``TaskRouter`` 执行，不再单独构造客户端（见
  ``test_scheduler_tool_path.py``）；
- 分维度（connect/read/write/pool）均未配置：客户端与 request 均收到
  单一 float，与旧 ``api_cfg.timeout`` 行为完全等价（向后兼容）；
- 任一维度配置：客户端与 request 均收到同一 ``openai.Timeout`` 对象
  （openai SDK 中 request 级 timeout 优先于客户端默认值，必须同步传
  解析结果，否则分维度永不生效）；
- 非法配置（非正数维度）：构造客户端前 fail fast 抛 ``ConfigLoadError``。

范围约束：仅验证超时注入链路，不引入 ContextManager / ModelGateway
等更大重构。``resolve_model_http_timeout`` 保持真实执行以验证完整
解析链。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from openai import Timeout

import so_agent.paths.deep_reasoning as deep_reasoning_mod
import so_agent.paths.direct_answer as direct_answer_mod
import so_agent.paths.persistent_task as persistent_task_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ClassificationResult, TaskStatus, TaskType
from so_agent.paths.deep_reasoning import DeepReasoningPath
from so_agent.paths.direct_answer import DirectAnswerPath
from so_agent.paths.persistent_task import PersistentTaskPath
from so_agent.runtime.config_loader import AgentAPIConfig, ConfigLoadError

# 可被解析为 ScheduleSpec 的合法调度 JSON（persistent_task 路径用）
VALID_SCHEDULE_JSON = (
    '{"trigger": "once", "action": "检查服务", '
    '"summary": "一次性检查", "is_recurring": false}'
)


def _make_config(**dimensions: float) -> AgentAPIConfig:
    """构造带直接密钥的测试配置（不依赖环境变量）。"""
    return AgentAPIConfig(
        provider="openai",
        model="fake-model",
        base_url="http://localhost/v1",
        timeout=12.0,
        retry=0,
        api_key="sk-test",
        **dimensions,
    )


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, target_module: object, content: str
) -> dict:
    """在目标模块上安装捕获型 AsyncOpenAI 桩，返回捕获字典。

    捕获 ``client_kwargs``（构造参数）与 ``request_kwargs``
    （chat.completions.create 参数）两个列表。
    """
    captured: dict = {}

    class _Completions:
        async def create(self, **kwargs):
            captured.setdefault("request_kwargs", []).append(kwargs)
            message = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class _Client:
        def __init__(self, **kwargs):
            captured.setdefault("client_kwargs", []).append(kwargs)
            self.chat = SimpleNamespace(completions=_Completions())

        async def close(self):
            pass

    monkeypatch.setattr(
        target_module, "AsyncOpenAI", lambda **kwargs: _Client(**kwargs)
    )
    return captured


def _assert_legacy_float(captured: dict) -> None:
    """分维度未配置：客户端与 request 均收到同一 float（等价旧行为）。"""
    client_kwargs = captured["client_kwargs"][0]
    request_kwargs = captured["request_kwargs"][0]
    assert isinstance(client_kwargs["timeout"], float)
    assert client_kwargs["timeout"] == 12.0
    assert request_kwargs["timeout"] == 12.0


def _assert_openai_timeout(captured: dict) -> None:
    """配置分维度：客户端与 request 均收到同一 openai.Timeout。"""
    client_timeout = captured["client_kwargs"][0]["timeout"]
    request_timeout = captured["request_kwargs"][0]["timeout"]
    assert isinstance(client_timeout, Timeout)
    assert client_timeout is request_timeout  # 同一解析结果
    assert client_timeout.connect == pytest.approx(3.0)
    assert client_timeout.read == pytest.approx(8.0)
    assert client_timeout.write == pytest.approx(12.0)  # 未配置维度回退总预算
    assert client_timeout.pool == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# persistent_task 路径（模块级导入，桩装在路径模块上）
# ---------------------------------------------------------------------------


@pytest.fixture
def persistent_path(tmp_path: Path) -> PersistentTaskPath:
    context = ProjectContext(sandbox_dir=tmp_path, project_name="t59-persistent")
    return PersistentTaskPath(context, Settings(_env_file=None))


class TestPersistentTaskHttpTimeout:
    async def test_no_dimensions_equivalent_legacy_float(
        self, persistent_path, monkeypatch
    ):
        captured = _install_fake_client(
            monkeypatch, persistent_task_mod, VALID_SCHEDULE_JSON
        )
        monkeypatch.setattr(
            persistent_task_mod, "load_agent_api_config", lambda *a, **kw: _make_config()
        )
        spec = await persistent_path._parse_schedule("每小时检查服务")
        assert spec.trigger == "once"  # 调用链走通
        _assert_legacy_float(captured)

    async def test_dimensions_produce_openai_timeout(
        self, persistent_path, monkeypatch
    ):
        captured = _install_fake_client(
            monkeypatch, persistent_task_mod, VALID_SCHEDULE_JSON
        )
        config = _make_config(connect_timeout=3.0, read_timeout=8.0)
        monkeypatch.setattr(
            persistent_task_mod, "load_agent_api_config", lambda *a, **kw: config
        )
        spec = await persistent_path._parse_schedule("每小时检查服务")
        assert spec.trigger == "once"
        _assert_openai_timeout(captured)

    async def test_invalid_dimension_fails_fast(self, persistent_path, monkeypatch):
        captured = _install_fake_client(
            monkeypatch, persistent_task_mod, VALID_SCHEDULE_JSON
        )
        config = _make_config(connect_timeout=0.0)
        monkeypatch.setattr(
            persistent_task_mod, "load_agent_api_config", lambda *a, **kw: config
        )
        with pytest.raises(ConfigLoadError, match="正数"):
            await persistent_path._parse_schedule("每小时检查服务")
        assert "client_kwargs" not in captured  # 客户端未被构造（fail fast）


# ---------------------------------------------------------------------------
# direct_answer / deep_reasoning 路径（模块级导入）
# ---------------------------------------------------------------------------


@pytest.fixture
def direct_path(tmp_path: Path) -> DirectAnswerPath:
    context = ProjectContext(sandbox_dir=tmp_path, project_name="t59-direct")
    return DirectAnswerPath(context, Settings(_env_file=None))


@pytest.fixture
def reasoning_path(tmp_path: Path) -> DeepReasoningPath:
    context = ProjectContext(sandbox_dir=tmp_path, project_name="t59-reasoning")
    return DeepReasoningPath(context, Settings(_env_file=None))


@pytest.fixture
def classification() -> ClassificationResult:
    return ClassificationResult(task_type=TaskType.DIRECT_ANSWER, reasoning="简单问答")


class TestDirectAnswerHttpTimeout:
    async def test_no_dimensions_equivalent_legacy_float(
        self, direct_path, classification, monkeypatch
    ):
        captured = _install_fake_client(monkeypatch, direct_answer_mod, "42")
        monkeypatch.setattr(
            direct_answer_mod, "load_agent_api_config", lambda *a, **kw: _make_config()
        )
        output = await direct_path.run("3+7", "", "", classification)
        assert output.status is TaskStatus.COMPLETED
        assert output.final_result == "42"
        _assert_legacy_float(captured)

    async def test_dimensions_produce_openai_timeout(
        self, direct_path, classification, monkeypatch
    ):
        captured = _install_fake_client(monkeypatch, direct_answer_mod, "42")
        config = _make_config(connect_timeout=3.0, read_timeout=8.0)
        monkeypatch.setattr(
            direct_answer_mod, "load_agent_api_config", lambda *a, **kw: config
        )
        output = await direct_path.run("3+7", "", "", classification)
        assert output.status is TaskStatus.COMPLETED
        _assert_openai_timeout(captured)

    async def test_invalid_dimension_fails_fast(
        self, direct_path, classification, monkeypatch
    ):
        captured = _install_fake_client(monkeypatch, direct_answer_mod, "42")
        config = _make_config(write_timeout=0.0)
        monkeypatch.setattr(
            direct_answer_mod, "load_agent_api_config", lambda *a, **kw: config
        )
        with pytest.raises(ConfigLoadError, match="正数"):
            await direct_path.run("3+7", "", "", classification)
        assert "client_kwargs" not in captured

    async def test_output_cap_is_applied(
        self, direct_path, classification, monkeypatch
    ):
        captured = _install_fake_client(monkeypatch, direct_answer_mod, "42")
        monkeypatch.setattr(
            direct_answer_mod, "load_agent_api_config", lambda *a, **kw: _make_config()
        )
        await direct_path.run("3+7", "", "", classification)
        assert captured["request_kwargs"][0]["max_tokens"] == 1024

    async def test_output_cap_zero_is_omitted(
        self, tmp_path, classification, monkeypatch
    ):
        path = DirectAnswerPath(
            ProjectContext(sandbox_dir=tmp_path, project_name="cap-zero"),
            Settings(_env_file=None, direct_answer_max_tokens=0),
        )
        captured = _install_fake_client(monkeypatch, direct_answer_mod, "42")
        monkeypatch.setattr(
            direct_answer_mod, "load_agent_api_config", lambda *a, **kw: _make_config()
        )
        await path.run("3+7", "", "", classification)
        assert "max_tokens" not in captured["request_kwargs"][0]


class TestDeepReasoningHttpTimeout:
    async def test_no_dimensions_equivalent_legacy_float(
        self, reasoning_path, classification, monkeypatch
    ):
        captured = _install_fake_client(
            monkeypatch, deep_reasoning_mod, "推理结论：X"
        )
        monkeypatch.setattr(
            deep_reasoning_mod, "load_agent_api_config", lambda *a, **kw: _make_config()
        )
        output = await reasoning_path.run("证明勾股定理", "", "", classification)
        assert output.status is TaskStatus.COMPLETED
        assert output.final_result == "推理结论：X"
        _assert_legacy_float(captured)

    async def test_dimensions_produce_openai_timeout(
        self, reasoning_path, classification, monkeypatch
    ):
        captured = _install_fake_client(
            monkeypatch, deep_reasoning_mod, "推理结论：X"
        )
        config = _make_config(connect_timeout=3.0, read_timeout=8.0)
        monkeypatch.setattr(
            deep_reasoning_mod, "load_agent_api_config", lambda *a, **kw: config
        )
        output = await reasoning_path.run("证明勾股定理", "", "", classification)
        assert output.status is TaskStatus.COMPLETED
        _assert_openai_timeout(captured)

    async def test_invalid_dimension_fails_fast(
        self, reasoning_path, classification, monkeypatch
    ):
        captured = _install_fake_client(
            monkeypatch, deep_reasoning_mod, "推理结论：X"
        )
        config = _make_config(pool_timeout=-0.5)
        monkeypatch.setattr(
            deep_reasoning_mod, "load_agent_api_config", lambda *a, **kw: config
        )
        with pytest.raises(ConfigLoadError, match="正数"):
            await reasoning_path.run("证明勾股定理", "", "", classification)
        assert "client_kwargs" not in captured

    async def test_output_cap_is_applied(
        self, reasoning_path, classification, monkeypatch
    ):
        captured = _install_fake_client(
            monkeypatch, deep_reasoning_mod, "推理结论：X"
        )
        monkeypatch.setattr(
            deep_reasoning_mod, "load_agent_api_config", lambda *a, **kw: _make_config()
        )
        await reasoning_path.run("证明勾股定理", "", "", classification)
        assert captured["request_kwargs"][0]["max_tokens"] == 4096

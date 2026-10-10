"""单元测试：四象限路由器 _llm_classify 的分类输出解析容错。

背景：运行日志出现“分类输出解析失败，降级为完整工作流”——模型可能
返回空字符串、Markdown 围栏包裹的 JSON 或前后带解释文字的输出。
回归验证解析链路的每一环：
- 空字符串 / 纯空白输出 → 直接降级为完整工作流（reasoning 明确指出空输出）；
- 围栏包裹、前后带解释文字的 JSON → 经 extract_json_object 宽容解析
  成功还原分类结果；
- 非 JSON 文本 / 非法枚举值 → 保守降级为完整工作流（最安全兜底）。

另含：分类客户端 HTTP 超时统一接入 ``resolve_model_http_timeout``
（任务 59）——分维度未配置时等价旧 float timeout，配置了分维度时
客户端与 request 均收到 openai.Timeout。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import so_agent.router as router_mod
from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import TaskType
from so_agent.router import TaskRouter


class _FakeApiConfig:
    """_llm_classify 所需的最小 API 配置桩（不读真实密钥）。

    分维度字段（connect/read/write/pool_timeout）补齐为 None：
    路由器客户端构造经 ``resolve_model_http_timeout`` 解析超时
    （任务 59），会读取这四个字段。
    """

    model = "fake-model"
    provider = "openai"
    base_url = "http://localhost/v1"
    timeout = 1.0
    retry = 0
    connect_timeout = None
    read_timeout = None
    write_timeout = None
    pool_timeout = None

    def resolve_api_key(self) -> str:
        return "sk-test"


def install_fake_llm(monkeypatch: pytest.MonkeyPatch, content: str) -> None:
    """把路由器的 OpenAI 客户端替换为返回固定文本的桩。"""

    class _Completions:
        async def create(self, **kwargs):
            message = SimpleNamespace(content=content)
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class _Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=_Completions())

        async def close(self):
            pass

    monkeypatch.setattr(
        router_mod, "load_agent_api_config", lambda *args, **kwargs: _FakeApiConfig()
    )
    monkeypatch.setattr(router_mod, "AsyncOpenAI", lambda **kwargs: _Client())


@pytest.fixture
def router(tmp_path):
    context = ProjectContext(sandbox_dir=tmp_path, project_name="router-test")
    settings = Settings(_env_file=None)
    return TaskRouter(context, settings)


class TestLlmClassifyParsing:
    """分类输出宽容解析：围栏与夹带文字均可还原分类结果。"""

    async def test_plain_json_parsed(self, router, monkeypatch):
        install_fake_llm(
            monkeypatch,
            '{"task_type": "direct_answer", "reasoning": "简单问答"}',
        )
        result = await router._llm_classify("解释什么是光合作用", "", "")
        assert result.task_type is TaskType.DIRECT_ANSWER
        assert result.reasoning == "简单问答"

    async def test_fenced_json_parsed(self, router, monkeypatch):
        # 千问兼容接口回归：JSON 被 ```json 围栏包裹
        install_fake_llm(
            monkeypatch,
            '```json\n{"task_type": "deep_reasoning", "reasoning": "需要推理"}\n```',
        )
        result = await router._llm_classify("证明勾股定理", "", "")
        assert result.task_type is TaskType.DEEP_REASONING

    async def test_json_with_prose_parsed(self, router, monkeypatch):
        # 前后带解释文字（非围栏起始）：走贪心 { ... } 提取
        install_fake_llm(
            monkeypatch,
            '好的，分类如下：\n{"task_type": "persistent_task", "reasoning": "定时执行"}\n请查收。',
        )
        result = await router._llm_classify("每小时检查一次日志", "", "")
        assert result.task_type is TaskType.PERSISTENT_TASK


class TestLlmClassifyDegradation:
    """解析失败一律保守降级为完整工作流（最安全兜底）。"""

    async def test_empty_output_degrades(self, router, monkeypatch):
        install_fake_llm(monkeypatch, "")
        result = await router._llm_classify("任意目标", "", "")
        assert result.task_type is TaskType.FULL_WORKFLOW
        assert "空" in result.reasoning

    async def test_whitespace_output_degrades(self, router, monkeypatch):
        install_fake_llm(monkeypatch, "   \n\t")
        result = await router._llm_classify("任意目标", "", "")
        assert result.task_type is TaskType.FULL_WORKFLOW
        assert "空" in result.reasoning

    async def test_non_json_output_degrades(self, router, monkeypatch):
        install_fake_llm(monkeypatch, "这不是 JSON，只是自由文本")
        result = await router._llm_classify("任意目标", "", "")
        assert result.task_type is TaskType.FULL_WORKFLOW
        assert "解析失败" in result.reasoning

    async def test_invalid_task_type_degrades(self, router, monkeypatch):
        install_fake_llm(
            monkeypatch,
            '{"task_type": "bogus quadrant", "reasoning": "枚举值非法"}',
        )
        result = await router._llm_classify("任意目标", "", "")
        assert result.task_type is TaskType.FULL_WORKFLOW
        assert "非法" in result.reasoning


class TestClassifierPromptFewShots:
    """分类提示词 few-shot 加固：8 个示例必须覆盖全部四象限。

    背景：示例过少时 LLM 倾向把长文本一律判为 full_workflow、把
    "监控/定时"类任务误判为普通问答；扩充到每象限 2 个示例提升
    分类边界清晰度。
    """

    def test_prompt_contains_eight_examples(self):
        example_lines = [
            line
            for line in router_mod.CLASSIFIER_PROMPT.splitlines()
            if line.strip().startswith("- 输入：")
        ]
        assert len(example_lines) == 8

    def test_prompt_covers_all_four_quadrants(self):
        for task_type in TaskType:
            assert f'"task_type": "{task_type.value}"' in router_mod.CLASSIFIER_PROMPT

    def test_prompt_covers_expected_example_inputs(self):
        # 每象限的新增示例输入都必须出现在提示词中
        for fragment in (
            "Python 的 GIL 是什么",
            "对比 Python asyncio 和 Go goroutine 并发模型，"
            "编写测试代码并给出选型建议",
            "分析 CAP 定理在微服务架构中的实际权衡",
            "每天早上 9 点检查服务器健康状态并发送报告",
            "监控 GitHub 仓库的新 issue 并自动分类",
        ):
            assert fragment in router_mod.CLASSIFIER_PROMPT


class TestClassifierHttpTimeoutUnification:
    """分类客户端 HTTP 超时统一接入 ``resolve_model_http_timeout``（任务 59）。

    验收点：
    - 分维度均未配置：客户端与 request 均收到单一 float（等价旧行为）；
    - 配置分维度：客户端与 request 均收到同一 ``openai.Timeout`` 对象
      （request 级参数优先于客户端默认值，必须同步传解析结果，否则
      分维度永不生效）；
    - 非法配置（非正数维度）：构造客户端前 fail fast 抛 ``ConfigLoadError``。
    """

    async def _classify_with_capture(
        self, monkeypatch: pytest.MonkeyPatch, config: object
    ) -> dict:
        """跑一次 _llm_classify，捕获 AsyncOpenAI 构造参数与 request 参数。"""
        captured: dict = {}

        class _Completions:
            async def create(self, **kwargs):
                captured["request_kwargs"] = kwargs
                message = SimpleNamespace(
                    content='{"task_type": "direct_answer", "reasoning": "x"}'
                )
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        class _Client:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs
                self.chat = SimpleNamespace(completions=_Completions())

            async def close(self):
                pass

        monkeypatch.setattr(
            router_mod, "load_agent_api_config", lambda *a, **kw: config
        )
        monkeypatch.setattr(
            router_mod, "AsyncOpenAI", lambda **kwargs: _Client(**kwargs)
        )

        context = ProjectContext(sandbox_dir=Path("."), project_name="timeout-test")
        router = TaskRouter(context, Settings(_env_file=None))
        await router._llm_classify("任意目标", "", "")
        return captured

    async def test_no_dimensions_yields_legacy_float(self, router, monkeypatch):
        """四个分维度均未配置：客户端与 request 收到同一 float（等价旧
        ``api_cfg.timeout`` 行为，完全向后兼容）。"""
        captured = await self._classify_with_capture(
            monkeypatch, _FakeApiConfig()  # timeout=1.0，分维度全 None
        )
        assert captured["client_kwargs"]["timeout"] == 1.0
        assert isinstance(captured["client_kwargs"]["timeout"], float)
        assert captured["request_kwargs"]["timeout"] == 1.0

    async def test_dimensions_produce_openai_timeout_for_client_and_request(
        self, router, monkeypatch
    ):
        """配置分维度：客户端与 request 均收到 openai.Timeout（同一解析结果）。"""
        from openai import Timeout

        config = _FakeApiConfig()
        config.connect_timeout = 3.0
        config.read_timeout = 8.0
        captured = await self._classify_with_capture(monkeypatch, config)

        client_timeout = captured["client_kwargs"]["timeout"]
        request_timeout = captured["request_kwargs"]["timeout"]
        assert isinstance(client_timeout, Timeout)
        assert isinstance(request_timeout, Timeout)
        assert client_timeout is request_timeout  # 同一解析结果
        assert client_timeout.connect == pytest.approx(3.0)
        assert client_timeout.read == pytest.approx(8.0)
        assert client_timeout.write == pytest.approx(1.0)  # 未配置维度回退总预算

    async def test_invalid_dimension_fails_fast(self, router, monkeypatch):
        """非正数分维度：构造客户端前抛 ConfigLoadError（fail fast）。"""
        from so_agent.runtime.config_loader import ConfigLoadError

        config = _FakeApiConfig()
        config.connect_timeout = 0.0
        context = ProjectContext(sandbox_dir=Path("."), project_name="timeout-test")
        router = TaskRouter(context, Settings(_env_file=None))
        monkeypatch.setattr(
            router_mod, "load_agent_api_config", lambda *a, **kw: config
        )
        with pytest.raises(ConfigLoadError, match="正数"):
            await router._llm_classify("任意目标", "", "")

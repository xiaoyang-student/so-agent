"""单元测试：PipelineGate 停止门、工具超时 profile 与模型 HTTP 超时 adapter（任务 45）。

覆盖任务 45 的验收点：

1. ``PipelineGate.remaining()`` 永不为负（deadline 过期返回 0.0）；
2. ``request_stop`` 幂等（首次原因保留、generation 只递增一次）；
3. ``check`` 在停止后抛 ``PipelineStoppedError``，且该异常继承
   ``BaseException``——穿透各协调器的 ``except Exception`` 宽捕获；
4. ``reset`` 清停止位但 generation 继续递增（同一引擎复用安全）；
5. ``timeout_profile_for_tool`` 与 ``ToolRegistry.get_timeout_profile``
   的映射（code_agent / subagent / sandbox / standard，生成工具回退
   SANDBOX）；
6. ``make_tool_timeout_resolver`` 按 profile 返回配置窗口；
7. ``Settings`` 派生：``code_agent_tool_execution_timeout`` 默认
   = agent_total + 清理宽限，显式设置被尊重；
8. ``resolve_model_http_timeout`` 向后兼容（无分维度时返回单一
   float）与分维度 ``openai.Timeout`` 构造、非正数拒绝。
"""

from __future__ import annotations

import asyncio
import time

import pytest

from so_agent.config import Settings
from so_agent.runtime.config_loader import (
    AgentAPIConfig,
    ConfigLoadError,
    resolve_model_http_timeout,
)
from so_agent.runtime.tool_registry import (
    ToolRegistry,
    ToolTimeoutProfile,
    timeout_profile_for_tool,
)
from so_agent.state_machine import PipelineStoppedError
from so_agent.workflow_support import PipelineGate, make_tool_timeout_resolver


# ---------------------------------------------------------------------------
# PipelineGate 停止门
# ---------------------------------------------------------------------------


class TestPipelineGateRemaining:
    """剩余时间计算：统一入口、永不为负。"""

    def test_remaining_none_without_deadline(self):
        gate = PipelineGate()
        assert gate.remaining() is None

    def test_remaining_never_negative_after_deadline(self):
        gate = PipelineGate()
        gate.set_deadline(time.monotonic() - 10.0)  # deadline 早已过期
        assert gate.remaining() == 0.0

    def test_remaining_positive_before_deadline(self):
        gate = PipelineGate()
        gate.set_deadline(time.monotonic() + 5.0)
        remaining = gate.remaining()
        assert 0.0 < remaining <= 5.0


class TestPipelineGateStop:
    """停止门：原子性、幂等与穿透语义。"""

    def test_initially_open(self):
        gate = PipelineGate()
        assert gate.stopping is False
        assert gate.stop_reason == ""
        gate.check("dispatch")  # 未停止：放行不抛

    def test_request_stop_is_idempotent_and_first_reason_wins(self):
        gate = PipelineGate()
        gate.request_stop("pipeline_timeout")
        generation = gate.generation
        gate.request_stop("cancelled")  # 二次请求不得覆盖
        assert gate.stopping is True
        assert gate.stop_reason == "pipeline_timeout"
        assert gate.generation == generation

    def test_check_raises_after_stop(self):
        gate = PipelineGate()
        gate.request_stop("cancelled")
        with pytest.raises(PipelineStoppedError, match="dispatch"):
            gate.check("dispatch")

    def test_stopped_error_pierces_except_exception(self):
        """PipelineStoppedError 继承 BaseException：穿透 except Exception 宽捕获。

        各协调器/调度器用 ``except Exception`` 把模型失败降级为可重试
        结果；停止信号若继承 Exception 会被吞掉并错误触发重试/重规划。
        """
        gate = PipelineGate()
        gate.request_stop("cancelled")
        with pytest.raises(PipelineStoppedError):
            try:
                gate.check("retry")
            except Exception as exc:  # 模拟宽捕获
                raise AssertionError("停止信号被 except Exception 吞掉") from exc

    def test_not_a_cancellation(self):
        """与 asyncio 取消语义解耦：不是 CancelledError，但可穿透宽捕获。"""
        assert issubclass(PipelineStoppedError, BaseException)
        assert not issubclass(PipelineStoppedError, asyncio.CancelledError)
        assert not issubclass(PipelineStoppedError, Exception)

    def test_generation_detects_stop_during_wait(self):
        """调用前记录 generation，等待期间发生停止 → 代数变化（迟到结果检测依据）。"""
        gate = PipelineGate()
        before = gate.generation
        gate.request_stop("external_stop")
        assert gate.stopping is True
        assert gate.generation != before

    def test_reset_clears_stop_but_generation_keeps_increasing(self):
        """同一引擎实例复用：reset 清停止位与 deadline，generation 继续递增。"""
        gate = PipelineGate()
        gate.set_deadline(time.monotonic() + 5.0)
        gate.request_stop("cancelled")
        before = gate.generation
        gate.reset()
        assert gate.stopping is False
        assert gate.stop_reason == ""
        assert gate.remaining() is None
        assert gate.generation > before


# ---------------------------------------------------------------------------
# 工具超时 profile
# ---------------------------------------------------------------------------


class TestTimeoutProfiles:
    """工具 → 超时 profile 映射。"""

    def test_module_default_mapping(self):
        assert timeout_profile_for_tool("code_agent") is ToolTimeoutProfile.CODE_AGENT
        assert (
            timeout_profile_for_tool("subagent_creator")
            is ToolTimeoutProfile.SUBAGENT
        )
        assert (
            timeout_profile_for_tool("execute_subagent") is ToolTimeoutProfile.SUBAGENT
        )
        assert timeout_profile_for_tool("run_in_sandbox") is ToolTimeoutProfile.SANDBOX
        assert timeout_profile_for_tool("read_file") is ToolTimeoutProfile.STANDARD
        assert timeout_profile_for_tool(None) is ToolTimeoutProfile.STANDARD
        assert timeout_profile_for_tool("") is ToolTimeoutProfile.STANDARD

    def test_registry_explicit_mapping_beats_generated_fallback(self, tmp_path):
        registry = ToolRegistry(generated_tools_dir=str(tmp_path))
        assert (
            registry.get_timeout_profile("code_agent") is ToolTimeoutProfile.CODE_AGENT
        )
        assert registry.get_timeout_profile("read_file") is ToolTimeoutProfile.STANDARD
        assert registry.get_timeout_profile(None) is ToolTimeoutProfile.STANDARD

    def test_registry_generated_tool_falls_back_to_sandbox(self, tmp_path):
        """注册来源为生成工具（沙箱内执行）→ SANDBOX profile。"""
        registry = ToolRegistry(generated_tools_dir=str(tmp_path))
        registry._tools["gen_tool"] = {"source": "generated", "assignable": True}
        assert registry.get_timeout_profile("gen_tool") is ToolTimeoutProfile.SANDBOX


class TestToolTimeoutResolver:
    """resolver 按 profile 返回配置窗口；profile 绝不延长总预算（由总预算截断保证）。"""

    def test_windows_by_profile(self):
        settings = Settings(_env_file=None)
        resolver = make_tool_timeout_resolver(settings)
        assert resolver("code_agent") == pytest.approx(
            settings.code_agent_tool_execution_timeout
        )
        assert resolver("subagent_creator") == pytest.approx(
            settings.subagent_tool_execution_timeout
        )
        assert resolver("execute_subagent") == pytest.approx(
            settings.subagent_tool_execution_timeout
        )
        assert resolver("run_in_sandbox") == pytest.approx(
            max(settings.tool_execution_timeout, settings.sandbox_timeout)
        )
        assert resolver("read_file") == pytest.approx(settings.tool_execution_timeout)
        assert resolver(None) == pytest.approx(settings.tool_execution_timeout)

    def test_code_agent_no_longer_ordinary_window(self):
        """code_agent 不再固定按普通 180s 工具窗口处理（任务 45 核心诉求）。"""
        settings = Settings(_env_file=None)
        assert settings.code_agent_tool_execution_timeout > (
            settings.tool_execution_timeout
        )
        resolver = make_tool_timeout_resolver(settings)
        assert resolver("code_agent") > resolver("read_file")

    def test_derived_code_agent_window_follows_agent_total(self):
        """未显式设置时派生：agent_total + 清理宽限（600 + 10 = 610 默认；
        缩短 agent_total 时派生窗口同步缩短——总预算语义不被放大）。"""
        settings = Settings(_env_file=None, agent_total_timeout=300.0)
        assert settings.code_agent_tool_execution_timeout == pytest.approx(310.0)

    def test_explicit_code_agent_window_respected(self):
        settings = Settings(
            _env_file=None,
            agent_total_timeout=300.0,
            code_agent_tool_execution_timeout=1200.0,
        )
        assert settings.code_agent_tool_execution_timeout == pytest.approx(1200.0)

    def test_registry_resolver_uses_generated_profile(self, tmp_path):
        settings = Settings(_env_file=None)
        registry = ToolRegistry(generated_tools_dir=str(tmp_path))
        registry._tools["gen_tool"] = {"source": "generated", "assignable": True}
        resolver = make_tool_timeout_resolver(settings, registry)
        assert resolver("gen_tool") == pytest.approx(
            max(settings.tool_execution_timeout, settings.sandbox_timeout)
        )


# ---------------------------------------------------------------------------
# 模型 HTTP 客户端超时 adapter（配置向后兼容）
# ---------------------------------------------------------------------------


class TestModelHttpTimeoutAdapter:
    def test_backward_compat_single_timeout(self):
        """四个分维度均未配置：返回单一 float，与旧行为完全一致。"""
        config = AgentAPIConfig(provider="openai", model="gpt-x", timeout=12.0)
        result = resolve_model_http_timeout(config)
        assert isinstance(result, float)
        assert result == 12.0

    def test_configured_dimensions_produce_openai_timeout(self):
        """任一分维度配置：构造 openai.Timeout（总预算仍取 timeout）。"""
        from openai import Timeout

        config = AgentAPIConfig(
            provider="openai",
            model="gpt-x",
            timeout=12.0,
            connect_timeout=3.0,
            read_timeout=8.0,
        )
        result = resolve_model_http_timeout(config)
        assert isinstance(result, Timeout)
        # 已配置维度生效；未配置维度回退到总预算 12.0（httpx.Timeout 不
        # 暴露总值属性，用回退值验证总预算传递）
        assert result.connect == pytest.approx(3.0)
        assert result.read == pytest.approx(8.0)
        assert result.write == pytest.approx(12.0)
        assert result.pool == pytest.approx(12.0)

    def test_non_positive_dimension_rejected(self):
        config = AgentAPIConfig(
            provider="openai", model="gpt-x", timeout=12.0, connect_timeout=0.0
        )
        with pytest.raises(ConfigLoadError, match="正数"):
            resolve_model_http_timeout(config)

    def test_negative_dimension_rejected(self):
        config = AgentAPIConfig(
            provider="openai", model="gpt-x", timeout=12.0, write_timeout=-1.0
        )
        with pytest.raises(ConfigLoadError):
            resolve_model_http_timeout(config)

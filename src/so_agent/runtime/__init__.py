"""运行时支撑模块。

为编排器（orchestrator）提供与具体业务无关的基础设施：

- registry：动态 Agent 的登记与生命周期管理（AgentRegistry）；
- scheduler：子任务依赖求解与并发调度（TaskScheduler）；
- sandbox：受限的代码/命令执行环境（SandboxRunner）；
- tool_registry：生成工具的注册与查询（ToolRegistry）；
- permissions：工具授权（ToolGrant）的签发与校验（PermissionGateway）；
- supervisor_router：Agent 间纵向通信的边界校验与留痕（SupervisorRouter）；
- config_loader：api_config.yaml 的加载与密钥环境变量解析；
- events：编排事件记录（写入 ProjectContext.event_log）。

全局约束：Agent 之间的一切通信必须严格使用 JSON（Pydantic 模型序列化），
由 supervisor_router（SupervisorRouter）强制校验，非 JSON 消息一律拒绝。
"""

from __future__ import annotations

from so_agent.runtime.config_loader import (
    AgentAPIConfig,
    ConfigLoadError,
    discover_tool_package_configs,
    load_agent_api_config,
)
from so_agent.runtime.events import ALL_EVENT_TYPES, EventLogger
from so_agent.runtime.permissions import PermissionGateway, PermissionGatewayError
from so_agent.runtime.registry import (
    DEFAULT_SUPERVISOR_ID,
    AgentRegistry,
    AgentRegistryError,
)
from so_agent.runtime.sandbox import SandboxError, SandboxRunner
from so_agent.runtime.scheduler import SchedulerError, TaskExecutor, TaskScheduler
from so_agent.runtime.supervisor_router import CommunicationError, SupervisorRouter
from so_agent.runtime.tool_registry import (
    DEFAULT_BUILTIN_AGENTS,
    ToolRegistry,
    ToolRegistryError,
)

__all__ = [
    # config_loader
    "AgentAPIConfig",
    "ConfigLoadError",
    "discover_tool_package_configs",
    "load_agent_api_config",
    # events
    "ALL_EVENT_TYPES",
    "EventLogger",
    # registry
    "DEFAULT_SUPERVISOR_ID",
    "AgentRegistry",
    "AgentRegistryError",
    # scheduler
    "SchedulerError",
    "TaskExecutor",
    "TaskScheduler",
    # sandbox
    "SandboxError",
    "SandboxRunner",
    # permissions
    "PermissionGateway",
    "PermissionGatewayError",
    # supervisor_router
    "CommunicationError",
    "SupervisorRouter",
    # tool_registry
    "DEFAULT_BUILTIN_AGENTS",
    "ToolRegistry",
    "ToolRegistryError",
]

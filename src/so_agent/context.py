"""项目级运行时上下文（ProjectContext）。

承载单个任务生命周期内的全部共享状态：沙箱目录、已创建 Agent 记录、
工具授权、事件日志与全局配置。orchestrator 及各 runtime 模块统一通过
该对象读写状态，禁止使用模块级全局变量，以保证任务间隔离与可测试性。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from so_agent.config import Settings, get_settings
from so_agent.models import AgentRecord, ToolGrant


@dataclass
class ProjectContext:
    """贯穿任务生命周期的共享上下文。

    Attributes:
        sandbox_dir: 沙箱根目录；所有文件操作必须限制在该目录内。
        project_name: 项目名称（用于标识当前任务空间）。
        created_agents: 已创建 Agent 的登记表，键为 agent_id。
        tool_grants: 工具授权登记表，键为 grant_id。
        event_log: 编排过程事件日志（追加写入，供审计与人工升级材料组装）。
        config: 全局运行配置（默认取进程内唯一实例）。
    """

    sandbox_dir: Path
    project_name: str
    created_agents: dict[str, AgentRecord] = field(default_factory=dict)
    tool_grants: dict[str, ToolGrant] = field(default_factory=dict)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    config: Settings = field(default_factory=get_settings)

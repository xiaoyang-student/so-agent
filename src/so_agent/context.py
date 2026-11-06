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
        task_store_dir: 持久化任务库的根目录；为 None 时回退到
            ``sandbox_dir``（CLI 默认），Web 运行时指向跨任务共享目录。
        created_agents: 已创建 Agent 的登记表，键为 agent_id。
        tool_grants: 工具授权登记表，键为 grant_id。
        event_log: 编排过程事件日志（追加写入，供审计与人工升级材料组装）。
        event_logger: 与控制层共享的事件记录器（含日志文件落盘）；工具
            包在内部运行嵌套 Agent 时经它上报 token 用量，保持全链路
            归因到同一事件流。未装配时为 None。
        token_budget: 任务级 token 预算追踪器（默认启用双阈值控制）；
            控制层在任务创建后挂载，嵌套 Agent 的用量回调据此累计。
        activity_sink: 可选活动回调（``(label: str) -> None``）；控制层
            挂载 pipeline 看门狗续时入口，嵌套 Agent 的流事件经它上报。
        config: 全局运行配置（默认取进程内唯一实例）。
    """

    sandbox_dir: Path
    project_name: str
    task_store_dir: Path | None = None
    created_agents: dict[str, AgentRecord] = field(default_factory=dict)
    tool_grants: dict[str, ToolGrant] = field(default_factory=dict)
    event_log: list[dict[str, Any]] = field(default_factory=list)
    event_logger: Any = None
    token_budget: Any = None
    activity_sink: Any = None
    config: Settings = field(default_factory=get_settings)

    def report_activity(self, label: str = "activity") -> None:
        """上报一次嵌套 Agent 活动（未挂载 sink 或回调失败时静默）。"""
        sink = self.activity_sink
        if sink is None:
            return
        try:
            sink(label)
        except Exception:  # 活动上报失败不得影响执行
            return

"""动态 Agent 注册表（AgentRegistry）。

职责与行为约定：
- 登记主管 Agent 动态创建的子 Agent（AgentRecord）；三个固定工具入口
  （code_agent / subagent_creator / review_agent）不是注册表条目；
- 强制 ``max_dynamic_agents`` 上限：活跃（未释放）Agent 数达到上限后，
  ``register`` 一律拒绝并抛出 ``AgentRegistryError``；
- ``release`` 仅把 Agent 移出活跃列表以释放配额，登记记录保留用于审计与
  替换溯源（replacement_of）；
- 提供 ``purge_expired`` 主动清理到期的 Agent（expires_at）。

访问边界：本对象仅供主管 Agent 与控制层（orchestrator）持有和使用，
不向子 Agent 开放任何查询接口；子 Agent 既不能查询、也不能注册/释放
Agent，其能力边界完全由 PermissionGateway 的工具授权决定。
"""

from __future__ import annotations

from datetime import datetime, timezone

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.models import AgentRecord, AgentStatus

# 主管 Agent 的默认标识（授权签发者与本表的控制者）
DEFAULT_SUPERVISOR_ID = "supervisor"


class AgentRegistryError(Exception):
    """注册表领域错误：重复登记、超过动态 Agent 上限、记录不存在等。"""


class AgentRegistry:
    """动态 Agent 注册表（内存实现）。"""

    def __init__(
        self,
        *,
        context: ProjectContext | None = None,
        settings: Settings | None = None,
        supervisor_id: str = DEFAULT_SUPERVISOR_ID,
    ) -> None:
        """初始化注册表。

        Args:
            context: 可选的共享上下文；提供时登记记录会同步镜像到
                ``context.created_agents``（同一对象引用，状态变化自动同步）。
            settings: 可选配置；优先级为 显式参数 > context.config > 全局配置。
            supervisor_id: 主管 Agent 标识，控制层身份校验使用。
        """
        self._context = context
        self._settings = settings or (
            context.config if context is not None else get_settings()
        )
        self._supervisor_id = supervisor_id
        self._agents: dict[str, AgentRecord] = {}   # 全部登记记录（含已释放，供审计）
        self._active: set[str] = set()              # 当前活跃的 agent_id

    # ------------------------------------------------------------------
    # 登记与释放
    # ------------------------------------------------------------------
    def register(self, agent_record: AgentRecord) -> None:
        """登记一个新 Agent。

        Raises:
            AgentRegistryError: 记录类型非法、agent_id 为空或重复、
                或活跃 Agent 数已达到 ``max_dynamic_agents`` 上限时。
        """
        if not isinstance(agent_record, AgentRecord):
            raise AgentRegistryError(
                f"agent_record 必须是 AgentRecord，实际为 {type(agent_record).__name__}"
            )
        agent_id = agent_record.agent_id
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise AgentRegistryError("agent_record.agent_id 不能为空")
        if agent_id in self._agents:
            raise AgentRegistryError(f"Agent 已登记，禁止重复注册：{agent_id}")
        limit = int(self._settings.max_dynamic_agents)
        if self.active_count() >= limit:
            raise AgentRegistryError(
                f"动态 Agent 数量已达上限（max_dynamic_agents={limit}），"
                f"拒绝注册 {agent_id}"
            )
        self._agents[agent_id] = agent_record
        self._active.add(agent_id)
        if self._context is not None:
            self._context.created_agents[agent_id] = agent_record

    def release(self, agent_id: str) -> None:
        """释放 Agent：从活跃列表移除以归还配额（重复释放为幂等操作）。

        登记记录仍保留在注册表与 ``context.created_agents`` 中，供审计与
        替换溯源使用。

        Raises:
            AgentRegistryError: agent_id 从未登记过时。
        """
        if agent_id not in self._agents:
            raise AgentRegistryError(f"Agent 不存在，无法释放：{agent_id}")
        self._active.discard(agent_id)

    def update_status(self, agent_id: str, status: AgentStatus) -> None:
        """更新 Agent 生命周期状态。

        Raises:
            AgentRegistryError: agent_id 不存在时。
        """
        record = self._agents.get(agent_id)
        if record is None:
            raise AgentRegistryError(f"Agent 不存在，无法更新状态：{agent_id}")
        record.status = status

    def purge_expired(self, now: datetime | None = None) -> list[str]:
        """清理已过期（``expires_at`` 到期）的活跃 Agent。

        Args:
            now: 参照时间；默认当前 UTC 时间（naive 时间按 UTC 处理）。

        Returns:
            被清理（状态置为 ``revoked`` 并移出活跃列表）的 agent_id 列表。
        """
        moment = now if now is not None else datetime.now(timezone.utc)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)

        purged: list[str] = []
        for agent_id in sorted(self._active):
            record = self._agents[agent_id]
            expires_at = record.expires_at
            if expires_at is None:
                continue
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= moment:
                record.status = "revoked"
                self._active.discard(agent_id)
                purged.append(agent_id)
        return purged

    # ------------------------------------------------------------------
    # 查询（仅主管 Agent / 控制层使用）
    # ------------------------------------------------------------------
    def get(self, agent_id: str) -> AgentRecord | None:
        """按 agent_id 查询登记记录（含已释放记录，供审计与替换溯源）。"""
        return self._agents.get(agent_id)

    def list_by_task(self, task_id: str, *, active_only: bool = False) -> list[AgentRecord]:
        """列出归属于指定任务的 Agent 记录（按创建时间升序）。

        Args:
            task_id: 任务编号（匹配 AgentRecord.parent_task_id）。
            active_only: 为 True 时仅返回活跃（未释放）记录。
        """
        records = [
            record
            for record in self._agents.values()
            if record.parent_task_id == task_id
            and (not active_only or record.agent_id in self._active)
        ]
        records.sort(key=lambda item: item.created_at)
        return records

    def active_count(self) -> int:
        """返回当前活跃（未释放）的 Agent 数量。"""
        return len(self._active)

    @property
    def supervisor_id(self) -> str:
        """返回主管 Agent 标识。"""
        return self._supervisor_id

    def assert_controller(self, caller_id: str) -> None:
        """校验调用者是否为主管/控制层。

        供控制层自检使用，防止误将本表暴露给子 Agent。

        Raises:
            PermissionError: caller_id 不是主管标识时。
        """
        if caller_id != self._supervisor_id:
            raise PermissionError(
                f"AgentRegistry 仅供主管 Agent（{self._supervisor_id}）与控制层使用，"
                f"拒绝 {caller_id!r} 的访问"
            )

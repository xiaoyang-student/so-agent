"""工作流状态机与共享状态（拆分自 workflow.py 的控制层基础设施）。

本模块是协调器架构的"状态底座"，承载四类职责：

- **领域异常**：``WorkflowError``（非法输入 / Agent 输出无法解析）、
  ``WorkflowCancelledError``（协作式取消）、``SupervisorNetworkError``
  （主管 Agent 网络连续失败）。置于叶子模块以供各协调器与 workflow
  模块共用（避免循环导入；workflow.py 对其 re-export）；
- **TaskStateMachine**：基于 python-statemachine 的声明式有限状态机，
  以 13 个高层阶段追踪任务流水线：idle → classifying → planning →
  reviewing_plan → granting_permissions → executing →（monitoring /
  replacing_agent / creating_tool）→ aggregating → reviewing_result →
  （reworking →）completed。它是权威 ``TaskStatus`` 的影子视图：
  ``WorkflowState.transition`` 在每次状态迁移时按映射表派发事件，
  全部合法迁移在图中显式声明；failed / cancelled / human_required
  三类终态没有对应阶段，影子图冻结在当前阶段（供审计观察最后阶段）；
- **WorkflowState**：任务级全部可变状态的数据类容器（自原
  WorkflowEngine.__init__ 的状态字段提取）。规划 / 执行 / 汇总三个
  协调器与薄编排器共享同一实例并借此交换状态（协调器之间不互相
  直接引用）；状态迁移（含事件留痕与进度提示）也统一在此发生；
- **结构化输出宽容解析**：``_coerce_structured_output``（纯文本返回 +
  围栏剥离 + Pydantic 校验）作为公共解析设施置于本模块，由
  workflow.py re-export，保持 ``workflow_mod._coerce_structured_output``
  的历史导入入口不变。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from statemachine import State, StateChart
from statemachine.exceptions import AttrNotFound, TransitionNotAllowed

from so_agent.json_utils import extract_json_object
from so_agent.models import (
    ExecutionResult,
    HumanEscalationRequest,
    ReviewDecision,
    SubtaskSpec,
    TaskRequest,
    TaskStatus,
    ToolGrant,
)
from so_agent.runtime.events import EVENT_STATUS_TRANSITION, EventLogger
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID

# ---------------------------------------------------------------------------
# 状态迁移的呈现层常量（进度提示 / 终态判定）
# ---------------------------------------------------------------------------

# 终态集合：到达后不允许继续迁移
_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.HUMAN_REQUIRED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    }
)

# 各执行阶段的进度提示（打印到 stderr，供用户实时观察工作流进展）
_STATUS_PROGRESS_MESSAGES: dict[TaskStatus, str] = {
    TaskStatus.PLANNING: "正在拆解任务...",
    TaskStatus.REPLANNING: "正在重新拆解任务...",
    TaskStatus.PLAN_REVIEW: "正在评审计划...",
    TaskStatus.EXECUTING: "正在执行子任务...",
    TaskStatus.REPLACEMENT_REQUESTED: "正在申请替代 Agent...",
    TaskStatus.REPLACEMENT_EXECUTING: "正在执行替代 Agent...",
    TaskStatus.AGGREGATING: "正在汇总结果...",
    TaskStatus.RESULT_REVIEW: "正在评审结果...",
}

# 失败/中断类终态：迁移原因作为错误信息一并打印到 stderr
_ERROR_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.HUMAN_REQUIRED,
    }
)


def _log_progress(status: TaskStatus, reason: str = "") -> None:
    """把状态迁移的进度提示或错误信息打印到标准错误（stderr）。"""
    message = _STATUS_PROGRESS_MESSAGES.get(status)
    if message:
        print(f"[so-agent] {message}", file=sys.stderr, flush=True)
    if status in _ERROR_STATUSES:
        detail = (reason or "").strip() or "未提供原因"
        print(
            f"[so-agent] 任务进入 {status.value} 状态：{detail}",
            file=sys.stderr,
            flush=True,
        )


# ---------------------------------------------------------------------------
# 工作流领域异常
# ---------------------------------------------------------------------------
class WorkflowError(Exception):
    """工作流领域错误：非法输入、主管/评审输出无法解析等。"""


class ReviewContractError(WorkflowError):
    """评审输出契约无效：提取出的 JSON/对象缺失显式可解析 passed 字段。

    任务 32 确定化契约：只有输入对象或提取出的 JSON 中存在显式可解析
    ``passed`` 字段时才可构造合法评审决定，绝不从评审文本语气或
    "通过/未通过"等关键词推断结论。捕获方（``_call_review_agent``）
    对该错误 fail closed（保守判定未通过并记录 ``review_contract_error``）。
    """


class WorkflowCancelledError(WorkflowError):
    """协作式取消：任务在检查点处被请求取消。"""


class SupervisorNetworkError(WorkflowError):
    """主管 Agent 网络连续失败（超时 / 连接错误）耗尽网络重试预算。

    此类错误由 ``PlanningCoordinator._plan_and_review`` 在独立网络重试
    预算耗尽后抛出，由编排流水线捕获并转入人工兜底（原因"网络连续
    失败"），不消耗重规划额度。
    """


class PipelineStoppedError(BaseException):
    """Pipeline 已进入停止状态（deadline 到期 / 取消 / 人工终止）后，
    停止门（``PipelineGate``）在 dispatch / retry / replacement /
    replan / 采纳计划 / 签发授权等新动作入口拒绝放行时抛出。

    刻意继承 ``BaseException``（与 ``asyncio.CancelledError`` 同策略）：
    各协调器与调度器中存在大量 ``except Exception`` 宽捕获（把模型
    调用失败降级为可重试结果），若继承 ``Exception``，停止信号会被
    这些宽捕获吞掉并错误触发重试 / 重规划；``BaseException`` 使停止
    信号穿透全部宽捕获，由 ``WorkflowEngine.run`` 显式捕获并映射为
    ``cancelled`` 终态。捕获方（调度器清理、迟到结果隔离）必须显式
    ``except PipelineStoppedError`` 才能吞并，且不得转为重试。
    """


# ---------------------------------------------------------------------------
# 声明式有限状态机（TaskStatus 的影子视图）
# ---------------------------------------------------------------------------
class TaskStateMachine(StateChart):
    """任务流水线阶段的声明式有限状态机。

    13 个状态覆盖从接收到完成的全部高层阶段；``classifying`` /
    ``granting_permissions`` / ``creating_tool`` 等阶段在当前控制层中
    由规划/执行协调器的动作映射（签发授权、创建缺口工具），事件命名
    与 :data:`_STATUS_TO_PHASE` 的 TaskStatus → (阶段, 事件) 映射一一
    对应。``completed`` 为唯一终态（failed / cancelled / human_required
    不推进影子图）。
    """

    idle = State(initial=True)
    classifying = State()
    planning = State()
    reviewing_plan = State()
    granting_permissions = State()
    executing = State()
    monitoring = State()
    replacing_agent = State()
    creating_tool = State()
    reviewing_result = State()
    reworking = State()
    aggregating = State()
    completed = State(final=True)

    classify = idle.to(classifying)

    plan = (
        idle.to(planning)
        | classifying.to(planning)
        | reviewing_plan.to(planning)
        | granting_permissions.to(planning)
        | monitoring.to(planning)
        | reworking.to(planning)
    )
    review_plan = planning.to(reviewing_plan) | granting_permissions.to(
        reviewing_plan
    )
    grant = (
        planning.to(granting_permissions)
        | reviewing_plan.to(granting_permissions)
        | executing.to(granting_permissions)
        | monitoring.to(granting_permissions)
    )
    execute = (
        reviewing_plan.to(executing)
        | granting_permissions.to(executing)
        | planning.to(executing)
        | monitoring.to(executing)
        | replacing_agent.to(executing)
        | creating_tool.to(executing)
        | reworking.to(executing)
    )
    monitor = executing.to(monitoring) | replacing_agent.to(monitoring)
    replace = executing.to(replacing_agent) | monitoring.to(replacing_agent)
    create_tool = executing.to(creating_tool) | monitoring.to(creating_tool)
    aggregate = (
        executing.to(aggregating)
        | monitoring.to(aggregating)
        | replacing_agent.to(aggregating)
        | creating_tool.to(aggregating)
    )
    review_result = aggregating.to(reviewing_result) | executing.to(reviewing_result)
    rework = reviewing_result.to(reworking)
    complete = (
        classifying.to(completed)
        | planning.to(completed)
        | granting_permissions.to(completed)
        | executing.to(completed)
        | monitoring.to(completed)
        | replacing_agent.to(completed)
        | creating_tool.to(completed)
        | aggregating.to(completed)
        | reviewing_result.to(completed)
        | reworking.to(completed)
    )


# 权威 TaskStatus → (影子图阶段, 派发事件)。failed / cancelled /
# human_required 不映射：影子图冻结在最后阶段。RECEIVED 为引擎初态
# （不发生迁移），影子图初始即为 idle，无需派发事件。
_STATUS_TO_PHASE: dict[TaskStatus, tuple[str, str]] = {
    TaskStatus.PLANNING: ("planning", "plan"),
    TaskStatus.REPLANNING: ("planning", "plan"),
    TaskStatus.PLAN_REVIEW: ("reviewing_plan", "review_plan"),
    TaskStatus.EXECUTING: ("executing", "execute"),
    TaskStatus.REPLACEMENT_REQUESTED: ("replacing_agent", "replace"),
    TaskStatus.REPLACEMENT_EXECUTING: ("executing", "execute"),
    TaskStatus.DECOMPOSITION_ISSUE: ("monitoring", "monitor"),
    TaskStatus.AGGREGATING: ("aggregating", "aggregate"),
    TaskStatus.RESULT_REVIEW: ("reviewing_result", "review_result"),
    TaskStatus.COMPLETED: ("completed", "complete"),
}


# ---------------------------------------------------------------------------
# 共享可变状态（协调器之间的唯一状态交换通道）
# ---------------------------------------------------------------------------
@dataclass
class WorkflowState:
    """任务级全部可变状态的数据类容器。

    自原 ``WorkflowEngine.__init__`` 的状态字段提取而来；规划 / 执行 /
    汇总协调器与薄编排器持有同一实例，通过它读写任务状态（协调器
    之间不互相引用）。字段名去掉了原私有下划线前缀。

    Attributes:
        events: 事件记录器（状态迁移留痕；测试可独立构造时为 None）。
        machine: 影子状态机（TaskStateMachine 实例）。
        current_status: 权威任务状态（TaskStatus，初始 RECEIVED）。
        plan_version: 当前计划版本号（从 1 开始；尚无计划时为 0）。
        replan_count: 已消耗的任务拆分修改次数。
        result_rework_count: 已消耗的结果评审返工次数。
        task_request: 当前任务请求；未接收任务时为 None。
        subtasks: 当前计划版本的子任务列表。
        pending_specs: 待执行的子任务批次（全量或返工子集）。
        results: ``{subtask_id: ExecutionResult}`` 执行结果。
        plan_history: 历代计划快照。
        agent_failure_history: Agent 失败历史（单行文本条目）。
        grants: 本任务累计签发的全部 ToolGrant。
        plan_grants: 本版本计划新签发的 ToolGrant。
        plan_grant_map: ``{subtask_id: ToolGrant}`` 本版本授权映射。
        plan_grant_proposals: 本版本计划评审用的授权候选（纯数据，
            未注册授权网关；两路评审均通过后才由 _issue_grants 真实签发）。
        last_review: 最近一次评审结论。
        human_escalation: 人工兜底材料包（进入 human_required 时设置）。
        cancel_requested: 协作式取消标记。
        seen_replacement_requests: 已处理的替换申请 request_id 去重集。
        seen_issue_fingerprints: 已写入失败历史的分解问题指纹。
        archive_results: 跨计划版本结果归档，键为
            ``(plan_version, subtask_id)``、值为当时的 ExecutionResult。
            重规划替换 ``results`` 前先把当前结果归档到此，供主管复用
            候选载荷与显式复用校验器查证历史来源（任务 35）；
            独立于 ``results`` 存活，避免清空后失去来源。
    """

    events: EventLogger | None = None
    machine: TaskStateMachine = field(default_factory=TaskStateMachine)
    current_status: TaskStatus = TaskStatus.RECEIVED
    plan_version: int = 0
    replan_count: int = 0
    result_rework_count: int = 0
    task_request: TaskRequest | None = None
    subtasks: list[SubtaskSpec] = field(default_factory=list)
    pending_specs: list[SubtaskSpec] = field(default_factory=list)
    results: dict[str, ExecutionResult] = field(default_factory=dict)
    plan_history: list[list[SubtaskSpec]] = field(default_factory=list)
    agent_failure_history: list[str] = field(default_factory=list)
    grants: list[ToolGrant] = field(default_factory=list)
    plan_grants: list[ToolGrant] = field(default_factory=list)
    plan_grant_map: dict[str, ToolGrant] = field(default_factory=dict)
    plan_grant_proposals: list[ToolGrant] = field(default_factory=list)
    last_review: ReviewDecision | None = None
    human_escalation: HumanEscalationRequest | None = None
    cancel_requested: bool = False
    seen_replacement_requests: set[str] = field(default_factory=set)
    seen_issue_fingerprints: set[tuple[str, int, tuple[str, ...]]] = field(
        default_factory=set
    )
    archive_results: dict[tuple[int, str], ExecutionResult] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # 影子状态机视图
    # ------------------------------------------------------------------
    @property
    def phase(self) -> str:
        """返回影子状态机当前阶段名（如 ``"planning"``）。"""
        state = next(iter(self.machine.configuration), None)
        return state.id if state is not None else ""

    def _advance_phase(self, new_status: TaskStatus) -> None:
        """按映射表派发影子状态机事件；无法映射的终态冻结影子图。"""
        mapping = _STATUS_TO_PHASE.get(new_status)
        if mapping is None:
            return
        target_phase, event = mapping
        if self.phase == target_phase:
            return
        try:
            self.machine.send(event)
        except (TransitionNotAllowed, AttrNotFound, AttributeError):
            # 影子图与权威状态偶发不同步时不阻断主流程（影子图仅供观察）
            return

    # ------------------------------------------------------------------
    # 状态迁移（一律通过 EventLogger 记录 status_transition 事件）
    # ------------------------------------------------------------------
    def transition(
        self, new_status: TaskStatus, *, reason: str = "", **extra: Any
    ) -> None:
        """状态迁移：更新当前状态并记录 ``status_transition`` 事件。

        状态实际发生变化时，同步推进影子状态机、把阶段进度提示（或
        失败/取消类状态的错误原因）打印到标准错误，便于用户实时观察
        工作流进展。

        Args:
            new_status: 目标状态。
            reason: 迁移原因（中文说明，写入事件 payload）。
            **extra: 附加到事件 payload 的 JSON 兼容字段（如 subtask_id）。
        """
        previous = self.current_status
        self.current_status = new_status
        if new_status == previous:
            return
        self._advance_phase(new_status)
        _log_progress(new_status, reason)
        payload: dict[str, Any] = {
            "task_id": self.task_request.task_id if self.task_request else None,
            "agent_id": DEFAULT_SUPERVISOR_ID,
            "from_status": previous.value,
            "to_status": new_status.value,
            "plan_version": self.plan_version,
            "replan_count": self.replan_count,
        }
        if reason:
            payload["reason"] = reason
        payload.update(extra)
        if self.events is not None:
            self.events.log_event(EVENT_STATUS_TRANSITION, payload)

    # ------------------------------------------------------------------
    # 协作式取消
    # ------------------------------------------------------------------
    def cancel(self) -> None:
        """请求取消当前任务（协作式：在下一个检查点生效）。"""
        self.cancel_requested = True

    def check_cancel(self) -> None:
        """协作式取消检查点：收到取消请求时抛出 WorkflowCancelledError。

        Raises:
            WorkflowCancelledError: 已通过 :meth:`cancel` 请求取消时。
        """
        if self.cancel_requested:
            raise WorkflowCancelledError("任务已收到取消请求")

    # ------------------------------------------------------------------
    # Agent 引用解析
    # ------------------------------------------------------------------
    def parse_agent_ref(self, agent_id: str) -> str:
        """从 Agent 引用解析所属子任务编号。

        调度器生成的格式为 ``{task_id}:{subtask_id}:{kind}-{hex}``；
        无法解析时返回空字符串。
        """
        raw = agent_id or ""
        if self.task_request is None:
            return ""
        prefix = f"{self.task_request.task_id}:"
        if not raw.startswith(prefix):
            return ""
        return raw[len(prefix):].split(":", 1)[0]

    def subtask_id_of(self, record: Any) -> str:
        """从 agent_id 解析所属子任务编号（委托 :meth:`parse_agent_ref`）。"""
        return self.parse_agent_ref(getattr(record, "agent_id", "") or "")


# ---------------------------------------------------------------------------
# 模块级工具：结构化输出宽容解析（纯文本返回 + 围栏剥离 + Pydantic 校验）
# ---------------------------------------------------------------------------
_StructuredOutputT = TypeVar("_StructuredOutputT", bound=BaseModel)

# 布尔宽容解释的肯定/否定词表（candidate.passed 为字符串形态时使用）。
# 任务 32：仅用于显式 passed 字段值的宽容解释（如 "true" / "yes"），
# 不再用于从自由文本关键词推导结论（文本语气不能决定放行与否）。
_BOOL_TRUE_TOKENS: frozenset[str] = frozenset(
    {"true", "yes", "on", "1", "pass", "passed", "通过"}
)
_BOOL_FALSE_TOKENS: frozenset[str] = frozenset(
    {"false", "no", "off", "0", "fail", "failed", "未通过", "不通过"}
)


def _coerce_bool_like(value: Any) -> bool | None:
    """尽力把任意值解释为布尔；无法解释时返回 None（不猜测）。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _BOOL_TRUE_TOKENS:
            return True
        if lowered in _BOOL_FALSE_TOKENS:
            return False
    return None


def _detect_passage_from_text(text: str) -> bool | None:  # pragma: no cover - 已废弃
    """已废弃（任务 32）：文本关键词不再用于推导评审结论。

    保留函数壳仅为兼容历史导入入口；任何调用都返回 None（无法判定）。
    评审放行只依据合法 ReviewDecision 的显式 ``passed`` 布尔字段，
    绝不从"通过/未通过"等文本关键词推断通过与否。
    """
    del text
    return None


def _rescue_or_contract_error(
    data: Any, label: str, exc: ValidationError
) -> ReviewDecision:
    """ReviewDecision 校验失败后的字段抢救；无显式 passed 时抛契约错误。

    先尝试 ``_rescue_review_decision`` 宽容抢救（字段类型不符但显式
    passed 存在）；抢救失败说明提取出的对象缺失显式可解析 passed
    字段——无法构造合法评审决定，抛 :class:`ReviewContractError`
    （调用方 fail closed，不猜测评审结论）。

    Raises:
        ReviewContractError: 对象中无显式可解析 passed 字段时。
    """
    rescued = _rescue_review_decision(data if isinstance(data, dict) else None)
    if rescued is not None:
        return rescued
    raise ReviewContractError(
        f"{label} 输出提取出的 JSON/对象缺失显式可解析 passed 字段，"
        f"无法构造合法评审决定：{exc}"
    ) from exc


def _coerce_str_list(value: Any) -> list[str]:
    """把任意值尽力转换为字符串列表（单条字符串包装为单元素列表）。"""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return []


def _review_contract_contradiction(decision: ReviewDecision) -> str | None:
    """校验 ReviewDecision 的字段契约；返回矛盾说明或 None（契约自洽）。

    任务 32 确定化契约（评审放行只依据合法 ReviewDecision 的 passed
    布尔字段，字段组合必须自洽）：

    - ``passed=true`` 时 ``required_fixes`` 必须为空（存在必须修复项
      就不得判定通过）；
    - ``passed=false`` 时 ``required_fixes`` 必须非空（拒绝结论必须
      写明至少一条必须修复项）；
    - ``suggestions`` 为非阻断建议，非空不构成阻断（任何组合均合法）。

    Returns:
        契约自洽时返回 None；矛盾时返回中文矛盾说明（供契约修复调用
        与审计事件引用）。
    """
    if decision.passed and decision.required_fixes:
        return (
            "passed=true 但 required_fixes 非空"
            f"（{len(decision.required_fixes)} 条）：通过结论不得携带必须修复项，"
            "非阻断建议应移入 suggestions"
        )
    if not decision.passed and not decision.required_fixes:
        return (
            "passed=false 但 required_fixes 为空："
            "拒绝结论必须至少给出一条必须修复项"
        )
    return None


def _rescue_review_decision(
    candidate: dict[str, Any] | None, raw_text: str = ""
) -> ReviewDecision | None:
    """从契约不符的评审输出中抢救关键字段，构造降级 ReviewDecision。

    字段类型宽容抢救（issues 为字符串而非列表、summary 缺失等），
    全部尽力而为、绝不抛异常。``passed`` 的唯一来源是 candidate 的
    显式 passed 字段（宽容布尔解释，如字符串 "true"）；任务 32 起
    不再从 raw_text 检测"通过/未通过"等结论关键词——文本语气不能
    推导放行结论，缺失显式可解析 passed 字段时返回 None，由调用方
    抛 ReviewContractError 并 fail closed（保守拦截）。

    抢救规则：

    - ``passed``：仅取 candidate 的显式 passed 字段（宽容布尔解释）；
      缺失或无法解释时返回 None（不猜测、不文本推断）；
    - ``issues`` / ``suggestions`` / ``required_fixes``：宽容转字符串
      列表，默认空列表；
    - ``summary``：仅接受字符串，否则回退空串；
    - ``retry_target``：仅接受字符串，否则丢弃（None）；
    - ``stage``：仅接受非空字符串，否则回退 plan_review（调用方
      ``_call_review_agent`` 最终会按评审输入统一覆盖）。

    Args:
        candidate: 提取出的候选对象（JSON 解析结果或输入字典）。
        raw_text: 原始文本（历史参数，仅为兼容保留，不再参与结论推断）。
    """
    data: dict[str, Any] = dict(candidate) if isinstance(candidate, dict) else {}
    passed = _coerce_bool_like(data.get("passed"))
    if passed is None:
        return None
    stage = data.get("stage")
    return ReviewDecision(
        stage=stage if isinstance(stage, str) and stage.strip() else "plan_review",
        passed=passed,
        issues=_coerce_str_list(data.get("issues")),
        suggestions=_coerce_str_list(data.get("suggestions")),
        required_fixes=_coerce_str_list(data.get("required_fixes")),
        retry_target=(
            data["retry_target"]
            if isinstance(data.get("retry_target"), str)
            else None
        ),
        summary=data["summary"] if isinstance(data.get("summary"), str) else "",
    )


def _coerce_structured_output(
    output: Any, model_cls: type[_StructuredOutputT], *, label: str
) -> _StructuredOutputT:
    """把 Agent 原始输出宽容解析为目标契约模型（Pydantic）。

    所有 Agent 以纯文本返回（output_type=None，兼容忽略
    response_format=json_schema 的千问等模型服务）；依次尝试：

    1. 目标模型实例（测试桩 / 已校验结果直接透传）；
    2. 其他 Pydantic 模型实例：序列化后按目标契约重校验（字段兼容即可
       通过，如旧全量形态经宽容归一化映射到精简契约）；
    3. 字典：直接按目标契约校验；
    4. JSON 文本：剥离 Markdown 围栏后整体/贪心提取再校验。

    ReviewDecision 专属降级（任务 32 确定化契约）：字典/JSON 校验
    失败时尝试 ``_rescue_review_decision`` 抢救关键字段（仅接受显式
    passed 字段，宽容布尔解释）；提取出对象但缺失显式可解析 passed
    字段时抛 ``ReviewContractError``（调用方 fail closed）；纯文本
    无 JSON 时不再做任何文本关键词结论推断，直接抛 WorkflowError
    （保守拦截）。评审放行只依据显式 ``passed`` 字段。

    Args:
        output: Runner 返回的 final_output（任意形态）。
        model_cls: 目标契约模型（PlanningOutput / ExecutionOutput /
            AggregationOutput / ReviewDecision 等）。
        label: 错误消息中的主体名称（如"主管 Agent"）。

    Raises:
        WorkflowError: 全部形态均无法解析为目标契约时。
        ReviewContractError: ReviewDecision 提取成功但缺失显式可解析
            passed 字段时（仅 ReviewDecision 分支）。
    """
    if isinstance(output, model_cls):
        return output
    if isinstance(output, BaseModel):
        try:
            return model_cls.model_validate(output.model_dump(mode="json"))
        except ValidationError as exc:
            if model_cls is ReviewDecision:
                return _rescue_or_contract_error(
                    output.model_dump(mode="json"), label, exc
                )
            raise WorkflowError(
                f"{label} 输出（{type(output).__name__}）不符合 "
                f"{model_cls.__name__} 契约：{exc}"
            ) from exc
    if isinstance(output, dict):
        try:
            return model_cls.model_validate(output)
        except ValidationError as exc:
            if model_cls is ReviewDecision:
                return _rescue_or_contract_error(output, label, exc)
            raise WorkflowError(
                f"{label} 输出字典不符合 {model_cls.__name__} 契约：{exc}"
            ) from exc
    if isinstance(output, str):
        obj = extract_json_object(output)
        if obj is not None:
            try:
                return model_cls.model_validate(obj)
            except ValidationError as exc:
                if model_cls is ReviewDecision:
                    return _rescue_or_contract_error(obj, label, exc)
                raise WorkflowError(
                    f"{label} 输出 JSON 不符合 {model_cls.__name__} 契约：{exc}"
                ) from exc
        # 纯文本无 JSON：任务 32 起不再按结论关键词降级（文本语气
        # 不能推导放行结论），直接保守拦截
    raise WorkflowError(
        f"{label} 输出无法解析为 {model_cls.__name__}（{type(output).__name__}）"
    )

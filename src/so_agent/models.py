"""领域数据模型（Pydantic v2）。

按架构计划第五节定义：任务请求、子任务规格、工具授权、Agent 记录、
替换请求、任务分解问题、执行结果、评审结论、生成工具清单、人工升级请求
以及任务状态枚举。

命名约定：所有标识符字段（*_id）使用字符串，时间字段统一为 UTC 时间；
所有可选集合字段默认空列表/字典，避免 None 判空分支。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utc_now() -> datetime:
    """返回当前 UTC 时间（统一时区来源）。"""
    return datetime.now(timezone.utc)


class TaskStatus(str, Enum):
    """任务生命周期状态机所允许的全部状态。"""

    RECEIVED = "received"                      # 已接收原始任务
    PLANNING = "planning"                      # 主 Agent 正在规划/拆分任务
    PLAN_REVIEW = "plan_review"                # 计划评审中（评审 Agent）
    EXECUTING = "executing"                    # 子任务执行中
    REPLACEMENT_REQUESTED = "replacement_requested"  # 子 Agent 已请求替换（待批准）
    REPLACEMENT_EXECUTING = "replacement_executing"  # 替换 Agent 执行中
    DECOMPOSITION_ISSUE = "decomposition_issue"      # 任务分解被判定存在问题
    REPLANNING = "replanning"                  # 主 Agent 正在修订任务分解
    AGGREGATING = "aggregating"                # 汇总各子任务结果
    RESULT_REVIEW = "result_review"            # 最终结果校验中（评审 Agent）
    COMPLETED = "completed"                    # 任务成功完成
    HUMAN_REQUIRED = "human_required"          # 需人工介入
    FAILED = "failed"                          # 任务失败终止
    CANCELLED = "cancelled"                    # 任务被取消


# 单个 Agent 记录的生命周期状态（与 TaskStatus 区分：描述 Agent 而非任务）
AgentStatus = Literal[
    "created",
    "running",
    "waiting",
    "completed",
    "failed",
    "replaced",
    "revoked",
]

# 评审阶段：计划评审 / 结果评审
ReviewStage = Literal["plan_review", "result_review"]

# 生成工具的评审状态
ToolReviewStatus = Literal["pending", "approved", "rejected"]


class TaskRequest(BaseModel):
    """进入系统的原始任务请求。"""

    task_id: str
    objective: str                                   # 任务目标（期望达成什么）
    constraints: list[str] = Field(default_factory=list)          # 约束条件
    acceptance_criteria: list[str] = Field(default_factory=list)  # 验收标准
    original_input: str = ""                         # 用户原始输入（原样保留）


class SubtaskSpec(BaseModel):
    """任务分解后的单个子任务规格（由主 Agent 产出，供子 Agent 执行）。

    ``reuse_from``（任务 35）是新计划子任务对旧计划子任务结果的
    **显式复用声明**：值为旧计划的 subtask_id，由主管在重规划时仅在
    确信语义仍适用时填写。该字段是声明元数据而非性语义的一部分，
    **不参与 ``spec_fingerprint``**（``fingerprint.spec_fingerprint``
    显式列举指纹字段，天然不含本字段）——否则声明本身会改变语义
    指纹，使复用判定自相矛盾。是否放行由规划协调器的安全校验器
    决定，代码绝不凭同名或指纹自动复用。
    """

    subtask_id: str
    plan_version: int = 1                            # 所属计划版本（重规划时递增）
    title: str                                       # 子任务标题
    instructions: str                                # 具体执行指令
    role: str                                        # 期望的执行角色（如 code / review / aggregate）
    dependencies: list[str] = Field(default_factory=list)   # 依赖的前置 subtask_id
    allowed_tools: list[str] = Field(default_factory=list)  # 授权的工具白名单
    expected_output: str = ""                        # 期望产出描述
    max_attempts: int = 3                            # 单子任务最大尝试次数
    reuse_from: str | None = None                    # 显式复用声明：旧计划 subtask_id（None=不复用）


class ToolGrant(BaseModel):
    """主 Agent 向某个 Agent 签发的工具授权凭证。"""

    grant_id: str
    task_id: str
    agent_id: str
    allowed_tools: list[str] = Field(default_factory=list)  # 本次授权的工具白名单
    issued_by: str                                   # 签发者（主 Agent 标识）
    issued_at: datetime = Field(default_factory=_utc_now)
    expires_at: datetime | None = None               # 过期时间；None 表示不过期
    revoked: bool = False                            # 是否已被撤销


class AgentRecord(BaseModel):
    """运行时 Agent 登记记录（含动态创建的子 Agent）。"""

    agent_id: str
    agent_type: str                                  # code_agent / subagent_creator / review_agent / dynamic
    parent_task_id: str
    plan_version: int = 1
    allowed_tools: list[str] = Field(default_factory=list)
    attempt_count: int = 0                           # 已尝试次数
    can_request_replacement: bool = True             # 是否允许请求替换（动态子 Agent 不具备该权限）
    replacement_requested: bool = False              # 是否已发起替换请求（每 Agent 至多一次）
    replacement_of: str | None = None                # 若本记录是替换产物，指向被替换的 agent_id
    status: AgentStatus = "created"
    created_at: datetime = Field(default_factory=_utc_now)
    expires_at: datetime | None = None               # 过期时间，供注册表清理


class ReplacementRequest(BaseModel):
    """子 Agent 向上请求替换自身的申请。"""

    request_id: str
    task_id: str
    subtask_id: str
    requester_agent_id: str                          # 发起请求的 Agent
    failure_summary: str                             # 失败摘要
    attempt_evidence: list[str] = Field(default_factory=list)  # 尝试过程证据
    requested_tools: list[str] = Field(default_factory=list)   # 希望新 Agent 获得的工具
    request_count: int = 1                           # 请求次数（全局上限见配置项）


class TaskDecompositionIssue(BaseModel):
    """任务分解存在问题时的诊断记录（用于触发重规划）。"""

    task_id: str
    subtask_id: str                                  # 出问题的子任务
    plan_version: int                                # 出问题时的计划版本
    failed_agent_ids: list[str] = Field(default_factory=list)  # 已失败的 Agent
    failure_reasons: list[str] = Field(default_factory=list)   # 失败原因
    attempted_approaches: list[str] = Field(default_factory=list)  # 已尝试过的方案
    recommendation: str = ""                         # 诊断建议（供重规划参考）


class ToolResponse(BaseModel):
    """统一的工具/Agent 响应信封（所有工具与 Agent 返回值的公共契约）。

    字段语义（全局 JSON 通信纪律的载体）：
    - success 必填：布尔真值，表示本次调用/执行是否成功；
    - output 统一为字符串：结构化产出须序列化为字符串后放入；
    - error：仅失败时填错误描述，成功时为 None；
    - artifacts：产出的文件路径列表（如位于沙箱目录内的产物）；
    - metadata：扩展元数据（如 exit_code、duration、token_usage 等），
      供调用方按需读取，不属于核心契约。
    """

    success: bool
    output: str = ""                                 # 主要产出（字符串）
    error: str | None = None                         # 失败时的错误描述
    artifacts: list[str] = Field(default_factory=list)   # 产出的文件路径等
    metadata: dict[str, Any] = Field(default_factory=dict)  # 扩展元数据


class ExecutionResult(ToolResponse):
    """单个子任务的执行结果（ToolResponse 信封的领域扩展）。

    继承 ToolResponse 统一信封：isinstance(ExecutionResult, ToolResponse)
    恒成立，使所有以 ToolResponse 语义消费结果的组件无需区分具体类型。
    本模型仅追加子任务归属与执行证据字段；与既有字段
    （subtask_id/success/output/evidence/artifacts/error/duration）保持
    完全向后兼容——旧代码的构造、校验、序列化方式均不受影响。
    """

    subtask_id: str
    evidence: list[str] = Field(default_factory=list)   # 证据（命令输出、日志片段等）
    duration: float = 0.0                            # 执行耗时（秒）


class ReviewDecision(BaseModel):
    """评审 Agent 的结论（计划评审或结果评审通用）。

    ``stage`` 有意保持 str 类型（不做 Literal 枚举校验）：评审 Agent 由
    LLM 驱动，可能返回 ``authorization_review`` / ``tool_review`` 等阶段
    值甚至中文阶段名；严格枚举会把内容正确的结论判为解析失败（进而被
    调用方保守判定为未通过）。调用方 ``workflow._call_review_agent``
    会把 stage 统一覆盖为 plan_review / result_review，此处宽松接受。

    字段契约（任务 32，由调用方 ``_call_review_agent`` 在解析后校验）：
    - ``passed=true`` 时 ``required_fixes`` 必须为空；
    - ``passed=false`` 时 ``required_fixes`` 必须非空；
    - ``suggestions`` 为非阻断建议，非空不单独阻断评审放行。
    """

    stage: str
    passed: bool
    issues: list[str] = Field(default_factory=list)          # 发现的问题
    suggestions: list[str] = Field(default_factory=list)     # 非阻断建议（不影响放行）
    required_fixes: list[str] = Field(default_factory=list)  # 必须修复项（阻断性）
    retry_target: str | None = None                  # 重试目标（subtask_id 或 agent_id）
    summary: str = ""                                # 结论摘要


class Execution(BaseModel):
    """单次子任务执行记录 — 每次 attempt 生成一条。

    ``failure_type`` / ``recovery_action`` 有意保持普通字符串（存
    ``FailureType.value`` / ``RecoveryAction.value``），不作枚举校验，
    以保持 SQLite 字符串列的向后兼容与宽松反序列化；取值域由
    :mod:`so_agent.runtime.failure` 的枚举定义，调度器在失败结果
    ``metadata`` 中同步回填同样的键值与值（运行时权威写入点）。
    """

    execution_id: str                                # uuid
    task_id: str
    subtask_id: str
    plan_version: int
    agent_id: str | None = None
    attempt: int = 1

    input_fingerprint: str = ""                      # SubtaskSpec 的 sha256
    output_fingerprint: str = ""                     # ExecutionResult.output 的 sha256

    status: Literal["pending", "running", "completed", "failed"] = "pending"
    result: ExecutionResult | None = None

    token_usage: int = 0
    duration: float = 0.0
    failure_type: str | None = None      # FailureType 值（network/rate_limit/timeout/permission/invalid_input/budget/quality/format/tool/sandbox/internal）
    recovery_action: str | None = None   # RecoveryAction 值（retry/repair_contract/replace/replan/stop）

    started_at: str | None = None
    completed_at: str | None = None


# 计划版本快照中执行记录的索引类型：subtask_id → 该版本的所有执行记录
PlanExecutions = dict[str, list[Execution]]


class PlanVersion(BaseModel):
    """一个计划版本的完整快照：规格 + 指纹索引 + 执行记录 + 评审结论。"""

    version: int
    specs: list[SubtaskSpec] = Field(default_factory=list)
    fingerprints: dict[str, str] = Field(default_factory=dict)  # subtask_id → spec_fingerprint
    executions: PlanExecutions = Field(default_factory=dict)   # subtask_id → 执行记录列表
    review_decision: ReviewDecision | None = None
    trigger_issue: TaskDecompositionIssue | None = None
    created_at: str = ""


class GeneratedToolManifest(BaseModel):
    """动态生成工具的清单条目（registry.json 的元素模型）。"""

    tool_id: str
    tool_name: str
    entry_file: str                                  # 相对 generated_tools 目录的入口文件路径
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)   # JSON Schema
    output_schema: dict[str, Any] = Field(default_factory=dict)  # JSON Schema
    created_by_task: str                             # 由哪个任务创建
    review_status: ToolReviewStatus = "pending"      # 需评审通过后方可注册使用
    version: str = "0.1.0"
    mcp_schema: dict[str, Any] = Field(default_factory=dict)     # MCP 暴露契约（MCPToolSchema 的 JSON 描述）
    test_cases: list[dict[str, Any]] = Field(default_factory=list)   # 验收测试用例（每项为入口函数 run() 的 kwargs）
    expected_outputs: list[Any] = Field(default_factory=list)        # 与 test_cases 一一对应的期望返回值
    verified: bool = False                                           # 沙箱运行验收是否通过


class HumanEscalationRequest(BaseModel):
    """升级至人工处理的完整材料包。"""

    task_id: str
    original_request: TaskRequest                    # 原始任务请求
    plan_history: list[list[SubtaskSpec]] = Field(default_factory=list)  # 历代计划快照
    agent_failure_history: list[str] = Field(default_factory=list)  # Agent 失败历史
    tool_call_evidence: list[str] = Field(default_factory=list)     # 工具调用证据
    final_review: ReviewDecision | None = None       # 最后一次评审结论
    created_at: datetime = Field(default_factory=_utc_now)


# ---------------------------------------------------------------------------
# 四象限任务路由
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """四象限任务类型。"""
    DIRECT_ANSWER = "direct_answer"          # ① 简单+短
    DEEP_REASONING = "deep_reasoning"        # ② 复杂+短
    PERSISTENT_TASK = "persistent_task"      # ③ 简单+长
    FULL_WORKFLOW = "full_workflow"           # ④ 复杂+长


class ClassificationResult(BaseModel):
    """LLM 分类结果。"""
    task_type: TaskType
    reasoning: str = ""


class ScheduleSpec(BaseModel):
    """持久化任务的调度规格。"""
    trigger: Literal["once", "cron", "interval", "event"]
    cron: str | None = None                  # cron 表达式（trigger=cron 时）
    interval_seconds: int | None = None      # 间隔秒数（trigger=interval 时）
    event_source: str | None = None          # 事件源描述（trigger=event 时）
    action: str = ""                         # 要执行的动作描述
    check_condition: str | None = None       # 触发条件描述（event 类型时）
    summary: str = ""                        # 人类可读摘要
    is_recurring: bool = False               # 是否周期执行

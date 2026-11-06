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
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator


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
    COMPLETED_PROVISIONAL = "completed_provisional"  # 低风险结果已交付，后台评审中
    COMPLETED = "completed"                    # 任务成功完成
    SCHEDULED = "scheduled"                    # 持久化任务已登记，等待调度执行
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


class VerificationRule(BaseModel):
    """可机检验收规则（由规划器生成，执行器确定性运行）。

    规则只读取沙箱文件与本次执行结果，不调用模型；任一规则失败即把
    子任务自述成功翻转为失败。字段含义按 ``kind`` 取用：

    - ``file_exists``：``path`` 存在（文件或目录）；
    - ``file_non_empty``：``path`` 是文件且大小 > 0；
    - ``json_field``：``path``（缺省用 output）解析为 JSON 后，``field``
      （点分路径）存在；``equals`` 非空时还要求值相等；
    - ``numeric_match``：``field`` 的数值与 ``value`` 在 ``tolerance``
      内相等；
    - ``text_contains``：``text`` 出现在 ``path`` 文件内容中（``path``
      必填，只检查产物文件，不检查模型输出措辞）；
    - ``exit_code``：结果 ``metadata.exit_code`` 等于 ``value``（默认 0）。
    """

    kind: Literal[
        "file_exists",
        "file_non_empty",
        "json_field",
        "numeric_match",
        "text_contains",
        "exit_code",
    ]
    path: str | None = None                          # 沙箱内相对路径
    field: str | None = None                         # JSON 点分字段路径
    equals: Any = None                               # json_field 期望值
    value: float | None = None                       # numeric_match / exit_code 期望值
    tolerance: float = 1e-6                          # 数值比较容差
    text: str | None = None                          # text_contains 子串
    description: str = ""                            # 人类可读说明


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
    # 写入范围声明（沙箱内相对路径，可含通配符）：仅用于并行冲突判定，
    # 不参与 spec_fingerprint。空数组表示"未声明"——并行调度时按未知
    # 副作用处理（fail-closed），会与其他可写子任务判为冲突。
    write_scope: list[str] = Field(default_factory=list)
    expected_output: str = ""                        # 期望产出描述
    max_attempts: int = 3                            # 单子任务最大尝试次数
    reuse_from: str | None = None                    # 显式复用声明：旧计划 subtask_id（None=不复用）
    # 可机检验收规则（可选）：执行后确定性运行；规则覆盖充分的低风险
    # 任务才允许跳过语义评审或先行交付（见 result_review_mode）。
    verification: list[VerificationRule] = Field(default_factory=list)


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
    content_hash: str = ""                                           # 源码 SHA-256（注册/执行前校验防篡改）
    # ---- 生命周期与安全准入 ----
    created_at: str = ""                                             # ISO 时间；空值加载时自动补齐
    last_used_at: str = ""                                           # 最近一次成功/失败调用时间
    use_count: int = 0                                               # 累计调用次数
    failure_count: int = 0                                           # 累计失败次数
    consecutive_failures: int = 0                                    # 连续失败次数（成功后清零）
    pinned: bool = False                                             # 固定工具：永不自动清理
    deprecated_at: str = ""                                          # 进入弃用期的时间
    safety_status: Literal[
        "unchecked", "passed", "failed", "quarantined"
    ] = "unchecked"                                                  # 静态安全检查结论
    safety_issues: list[str] = Field(default_factory=list)           # 安全检查发现


class SubtaskProgress(BaseModel):
    """人工兜底材料中的单条子任务进展（不含完整产出，仅摘要）。"""

    subtask_id: str
    title: str = ""
    status: Literal["completed", "failed", "pending"] = "pending"
    output_preview: str = ""
    error: str = ""


class HumanGuidance(BaseModel):
    """人工提交的指导：回退到哪些子任务、如何处理、补充什么信息。

    - ``resume``：不清除既有结果，直接继续执行（用于补齐信息后重试）；
    - ``rollback``：清除 ``rollback_to`` 及其下游子任务结果后重跑；
      ``rollback_to`` 为空时回退到全部失败子任务；
    - ``abort``：不恢复执行，任务保持人工兜底终态。
    """

    action: Literal["resume", "rollback", "abort"] = "rollback"
    rollback_to: list[str] = Field(default_factory=list, max_length=50)
    instructions: str = Field(default="", max_length=8000)
    additional_info: str = Field(default="", max_length=8000)


class HumanEscalationRequest(BaseModel):
    """升级至人工处理的完整材料包。"""

    task_id: str
    original_request: TaskRequest                    # 原始任务请求
    plan_history: list[list[SubtaskSpec]] = Field(default_factory=list)  # 历代计划快照
    agent_failure_history: list[str] = Field(default_factory=list)  # Agent 失败历史
    tool_call_evidence: list[str] = Field(default_factory=list)     # 工具调用证据
    final_review: ReviewDecision | None = None       # 最后一次评审结论
    reason: str = ""                                 # 触发人工介入的原因
    progress: list[SubtaskProgress] = Field(default_factory=list)   # 子任务进展摘要
    rollback_targets: list[str] = Field(default_factory=list)       # 可回退子任务候选
    created_at: datetime = Field(default_factory=_utc_now)


class AuditFinding(BaseModel):
    """确定性审计发现（问题码 + 级别 + 证据引用）。"""

    code: str
    level: Literal["info", "warning", "error"] = "info"
    message: str
    refs: list[str] = Field(default_factory=list)


class AuditPlanSummary(BaseModel):
    """单个计划版本的审计摘要（仅记录子任务标识，不携带业务载荷）。"""

    version: int
    subtask_ids: list[str] = Field(default_factory=list)


class AuditSubtaskResult(BaseModel):
    """当前计划版本子任务结果的审计摘要（不含完整产出）。"""

    subtask_id: str
    success: bool
    duration: float = 0.0
    evidence_count: int = 0
    artifact_count: int = 0
    output_chars: int = 0
    error: str = ""


class AuditAgentSummary(BaseModel):
    """Agent 生命周期审计摘要。"""

    agent_id: str
    agent_type: str
    status: str
    plan_version: int = 1
    attempt_count: int = 0
    allowed_tools: list[str] = Field(default_factory=list)
    replacement_requested: bool = False
    replacement_of: str | None = None


class AuditGrantSummary(BaseModel):
    """工具授权审计摘要。"""

    grant_id: str
    agent_id: str
    allowed_tools: list[str] = Field(default_factory=list)
    revoked: bool = False


class AuditUsageEntry(BaseModel):
    """按 Agent 聚合的模型用量。"""

    agent: str
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0


class AuditReport(BaseModel):
    """确定性审计报告：同一事件流与结果集必然产出同一报告。

    报告只包含标识、计数、摘要与证据引用，不复制完整业务产出；
    最终结果仅保留 SHA-256 与字符数，避免审计载体二次泄露内容。
    """

    task_id: str = ""
    status: str = ""
    verdict: Literal["pass", "warn", "fail"] = "pass"
    plan_versions: list[AuditPlanSummary] = Field(default_factory=list)
    results: list[AuditSubtaskResult] = Field(default_factory=list)
    agents: list[AuditAgentSummary] = Field(default_factory=list)
    grants: list[AuditGrantSummary] = Field(default_factory=list)
    usage: list[AuditUsageEntry] = Field(default_factory=list)
    total_tokens: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    findings: list[AuditFinding] = Field(default_factory=list)
    review_summary: str = ""
    final_result_sha256: str = ""
    final_result_chars: int = 0
    event_count: int = 0
    event_counts: dict[str, int] = Field(default_factory=dict)
    last_event_at: str = ""


# ---------------------------------------------------------------------------
# 四象限任务路由
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """四象限任务类型。"""
    DIRECT_ANSWER = "direct_answer"          # ① 简单+短
    DEEP_REASONING = "deep_reasoning"        # ② 复杂+短
    PERSISTENT_TASK = "persistent_task"      # ③ 简单+长
    FULL_WORKFLOW = "full_workflow"           # ④ 复杂+长


class RouteStep(BaseModel):
    """路由判断的单步记录（供控制台展示判断依据）。

    stage 取值：router_disabled / rule_check / llm_classify /
    tool_intent_upgrade / output_guard；status 取值随阶段而定
    （hit / miss / skipped / passed / fallback / upgraded / applied）。
    """
    stage: str
    status: str
    detail: str = ""


class ClassificationResult(BaseModel):
    """LLM 分类结果（含可展示的判断轨迹）。"""
    task_type: TaskType
    reasoning: str = ""
    steps: list[RouteStep] = Field(default_factory=list)


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
    timezone: str | None = None              # cron 计算所用 IANA 时区（空=UTC）

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str | None) -> str | None:
        """拒绝未知 IANA 时区，避免任务落库后每次调度都失败。"""
        if value is None or not str(value).strip():
            return None
        name = str(value).strip()
        try:
            ZoneInfo(name)
        except Exception as exc:
            raise ValueError(f"未知时区：{name}") from exc
        return name

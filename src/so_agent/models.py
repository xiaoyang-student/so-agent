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
    """任务分解后的单个子任务规格（由主 Agent 产出，供子 Agent 执行）。"""

    subtask_id: str
    plan_version: int = 1                            # 所属计划版本（重规划时递增）
    title: str                                       # 子任务标题
    instructions: str                                # 具体执行指令
    role: str                                        # 期望的执行角色（如 code / review / aggregate）
    dependencies: list[str] = Field(default_factory=list)   # 依赖的前置 subtask_id
    allowed_tools: list[str] = Field(default_factory=list)  # 授权的工具白名单
    expected_output: str = ""                        # 期望产出描述
    max_attempts: int = 3                            # 单子任务最大尝试次数


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


class ExecutionResult(BaseModel):
    """单个子任务的执行结果。"""

    subtask_id: str
    success: bool
    output: str = ""                                 # 文本产出
    evidence: list[str] = Field(default_factory=list)   # 证据（命令输出、日志片段等）
    artifacts: list[str] = Field(default_factory=list)  # 产物路径（位于沙箱目录内）
    error: str | None = None                         # 失败时的错误信息
    duration: float = 0.0                            # 执行耗时（秒）


class ReviewDecision(BaseModel):
    """评审 Agent 的结论（计划评审或结果评审通用）。"""

    stage: ReviewStage
    passed: bool
    issues: list[str] = Field(default_factory=list)          # 发现的问题
    required_fixes: list[str] = Field(default_factory=list)  # 必须修复项
    retry_target: str | None = None                  # 重试目标（subtask_id 或 agent_id）
    summary: str = ""                                # 结论摘要


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


class HumanEscalationRequest(BaseModel):
    """升级至人工处理的完整材料包。"""

    task_id: str
    original_request: TaskRequest                    # 原始任务请求
    plan_history: list[list[SubtaskSpec]] = Field(default_factory=list)  # 历代计划快照
    agent_failure_history: list[str] = Field(default_factory=list)  # Agent 失败历史
    tool_call_evidence: list[str] = Field(default_factory=list)     # 工具调用证据
    final_review: ReviewDecision | None = None       # 最后一次评审结论
    created_at: datetime = Field(default_factory=_utc_now)

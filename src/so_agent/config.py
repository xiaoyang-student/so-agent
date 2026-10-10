"""全局运行配置（Pydantic Settings）。

所有可调参数集中于此，支持环境变量覆盖：
- 环境变量前缀为 ``SO_AGENT_``，例如 ``SO_AGENT_MAX_CONCURRENCY=5``；
- 也可通过项目根目录下的 ``.env`` 文件提供同名变量；
- 未提供的参数使用下方默认值。

治理类参数与架构计划的约束一一对应（动态 Agent 上限、替换次数上限、
重规划次数上限等），编排器运行时应始终通过 ``get_settings()`` 读取，
不得在业务代码中硬编码。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 嵌套子 Agent 工具窗口的清理宽限（秒，任务 33 嵌套修复）：
# ``subagent_tool_execution_timeout`` 未显式设置时派生为
# ``agent_total_timeout + 本宽限``，保证子 Agent 自身的总预算硬上限
# 先触发（含取消 / 审计收尾），父级工具阶段窗口不会提前误杀。
_SUBAGENT_TOOL_CLEANUP_GRACE: float = 10.0

# 长时代码 Agent 工具窗口的清理宽限（秒，任务 45）：
# ``code_agent_tool_execution_timeout`` 未显式设置时派生为
# ``agent_total_timeout + 本宽限``，与嵌套子 Agent 同理：code_agent
# 内部运行的子 Agent 总预算先于父级工具窗口触发。
_CODE_AGENT_TOOL_CLEANUP_GRACE: float = 10.0


class Settings(BaseSettings):
    """so-agent 全局配置。"""

    model_config = SettingsConfigDict(
        env_prefix="SO_AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # 允许 model_timeout 等以 model_ 开头的配置字段，
        # 禁用 pydantic 对 model_ 前缀的保护命名空间警告。
        protected_namespaces=(),
    )

    # ---- 动态 Agent 与替换治理 ----
    max_dynamic_agents: int = 8          # 动态创建的子 Agent 数量上限
    max_agent_attempts: int = 3          # 单个 Agent 的最大尝试次数上限（真实次数由策略决定）
    max_replacement_requests: int = 1    # 替换请求次数上限（含动态子 Agent 无替换权约束）

    # ---- 确定性失败恢复内核（任务 46）----
    # 判定逻辑见 runtime.failure（纯函数 / 纯数据矩阵，LLM 不可修改）；
    # 预算与上下文治理不在本任务范围内，本组参数仅约束重试恢复行为。
    recovery_backoff_base_seconds: float = 1.0   # network/rate_limit 退避基数（秒）
    recovery_backoff_cap_seconds: float = 30.0   # 单次退避上限（秒）
    recovery_backoff_jitter_ratio: float = 0.2   # 退避抖动比例（0~1）
    recovery_retry_suppress_threshold: int = 2   # 同指纹连续失败达该次数即抑制重试
    recovery_repair_contract_max_attempts: int = 1  # 输出契约修复重试上限

    # ---- 编排与并发 ----
    max_concurrency: int = 3             # 同时在执行的最大子任务数
    max_replan_attempts: int = 3         # 主 Agent 修订任务分解的最大次数
    max_model_turns: int = 8             # 单次 Agent 运行的最大模型回合数；常规任务 8 轮足够，复杂任务可通过环境变量覆盖

    # ---- 超时与沙箱 ----
    model_timeout: float = 45.0          # 单次模型调用超时（秒）；配合 flash 级模型，45 秒足够单次 API 响应
    sandbox_timeout: float = 30.0        # 沙箱单次命令执行超时（秒）
    sandbox_output_limit: int = 1_048_576  # 沙箱输出上限（字节，默认 1 MiB）
    # 流式调用「五层预算」（任务 33）：
    # - first_event_timeout：流式首事件（含 SDK 冷启动与模型首包等待）
    #   窗口——首个流事件前的最大等待秒数；
    # - model_idle_timeout：模型阶段空闲窗口——连续该秒数没有任何
    #   SDK 原生流事件才判定卡死（不接受 LLM 文本心跳作为续时依据）；
    # - tool_execution_timeout：工具执行阶段窗口——语义 tool_called
    #   事件到对应 tool_output 事件之间的最大等待秒数；
    # - agent_total_timeout：单次主管/子 Agent 调用的绝对总预算——
    #   任何阶段窗口都被它截断，即使事件持续产出也不超过该上限；
    # - max_pipeline_timeout：整条流水线超时上限（秒）：超时携带部分
    #   结果优雅降级，不直接崩溃（同时作为工具授权 expires_at 的第二道
    #   防线）。
    # 关系：first_event / model_idle / tool_execution < agent_total
    #       < subagent_tool_execution_timeout ≤ supervisor_total_timeout
    #       < max_pipeline_timeout。
    first_event_timeout: float = 180.0
    model_idle_timeout: float = 90.0
    tool_execution_timeout: float = 180.0
    agent_total_timeout: float = 600.0
    max_pipeline_timeout: float = 2400.0
    # 嵌套子 Agent 工具窗口（任务 33 嵌套修复）：主管工具阶段中调用
    # 嵌套子 Agent 执行工具（subagent_creator / execute_subagent，按
    # ToolCallItem.tool_name 识别）时的独立等待窗口——必须覆盖子 Agent
    # 的 agent_total_timeout 与清理宽限（未显式设置时派生 =
    # agent_total_timeout + 10 秒 = 610 秒），否则复杂子任务会在父级
    # 工具窗口（tool_execution_timeout）被提前误杀；普通沙箱 / MCP /
    # 其他工具仍使用 tool_execution_timeout。
    subagent_tool_execution_timeout: float = 610.0
    # 长时代码 Agent 工具窗口（任务 45 超时 profile）：主管工具阶段中
    # 调用 code_agent（创建/修改文件、沙箱运行代码、生成工具）时的
    # 独立等待窗口——不再按普通工具窗口（tool_execution_timeout，180s）
    # 处理。未显式设置时派生 = agent_total_timeout + 10 秒（默认 610），
    # 覆盖内部子 Agent 总预算与清理宽限；仅影响工具阶段等待窗口，
    # 绝不延长 Agent/Pipeline 总 deadline（总预算仍在运行时截断）。
    code_agent_tool_execution_timeout: float = 610.0
    # 主管单次调用的绝对总预算（秒）：配置校验器保证不短于
    # subagent_tool_execution_timeout（父级总预算不会先于嵌套子 Agent
    # 工具窗口触发），运行时再被 pipeline 剩余预算截断
    # （workflow._run_supervisor）；子 Agent 自身仍是 agent_total_timeout
    # 硬上限，不被父级预算无限延长。
    supervisor_total_timeout: float = 900.0
    # ---- 旧超时字段（deprecated 兼容，任务 33）----
    # 仅在被显式设置（环境变量 / .env / 构造参数）时映射到新字段，
    # 见 _apply_deprecated_timeout_aliases：
    # - max_idle_timeout → first_event_timeout 与 model_idle_timeout
    #   （旧单一空闲窗口语义；工具执行阶段不再沿用旧窗口，修复旧的
    #   60 秒误杀工具执行问题，保持新默认 tool_execution_timeout）；
    # - max_supervisor_call_timeout → agent_total_timeout。
    max_idle_timeout: float = 60.0
    max_supervisor_call_timeout: float = 600.0

    # ---- 路由与分类 ----
    classifier_max_tokens: int = 512       # 分类调用的最大输出 token（保证含推理说明的完整 JSON 输出）
    enable_router: bool = True             # 是否启用四象限路由（False 则走原有流水线）

    # ---- 结果评审策略 ----
    skip_result_review_for_simple: bool = True  # 简单任务快速通道：子任务 <= 2 且无依赖且全成功且产出充分时跳过汇总与结果评审；False 则所有任务一律走结果评审

    # ---- 持久化调度 ----
    task_store_path: str = "sandbox/task_store.db"  # SQLite 存储路径
    scheduler_poll_interval: int = 60      # 调度器轮询间隔（秒）

    @model_validator(mode="after")
    def _apply_deprecated_timeout_aliases(self) -> "Settings":
        """旧超时字段显式设置时映射到新的分阶段字段（deprecated 兼容）。

        仅当旧字段出现在显式设置集合（环境变量 / .env / 构造参数）且
        对应新字段未被显式设置时才映射，避免静默改变显式新配置：

        - ``max_idle_timeout`` → ``first_event_timeout`` 与
          ``model_idle_timeout``：旧语义为全阶段单一空闲窗口；新语义
          下仅作用于非工具阶段（工具执行阶段采用独立的
          ``tool_execution_timeout``，修复旧的 60 秒误杀工具执行）。
        - ``max_supervisor_call_timeout`` → ``agent_total_timeout``：
          旧字段即单次调用的兜底总超时，语义一一对应。

        另维护嵌套修复不变量（任务 33 修订）：

        - ``subagent_tool_execution_timeout`` 未显式设置时派生为
          ``agent_total_timeout + 10 秒``：对任何自定义 agent_total 都
          覆盖子 Agent 总预算与清理宽限；
        - ``supervisor_total_timeout`` 保底不短于
          ``subagent_tool_execution_timeout``：父级总预算不会先于
          嵌套子 Agent 工具窗口触发（pipeline 剩余预算仍在运行时
          截断，2400 秒流水线上限不变）。
        """
        fields_set = self.model_fields_set
        if "max_idle_timeout" in fields_set:
            if "first_event_timeout" not in fields_set:
                self.first_event_timeout = float(self.max_idle_timeout)
            if "model_idle_timeout" not in fields_set:
                self.model_idle_timeout = float(self.max_idle_timeout)
        if "max_supervisor_call_timeout" in fields_set:
            if "agent_total_timeout" not in fields_set:
                self.agent_total_timeout = float(
                    self.max_supervisor_call_timeout
                )
        # 嵌套子 Agent 工具窗口：未显式设置时按 agent_total 派生
        if "subagent_tool_execution_timeout" not in fields_set:
            self.subagent_tool_execution_timeout = (
                float(self.agent_total_timeout)
                + _SUBAGENT_TOOL_CLEANUP_GRACE
            )
        # 长时代码 Agent 工具窗口（任务 45）：未显式设置时按 agent_total
        # 派生，与嵌套子 Agent 同理覆盖内部总预算与清理宽限
        if "code_agent_tool_execution_timeout" not in fields_set:
            self.code_agent_tool_execution_timeout = (
                float(self.agent_total_timeout)
                + _CODE_AGENT_TOOL_CLEANUP_GRACE
            )
        # 父级总预算保底：不短于嵌套子 Agent 工具窗口
        if self.supervisor_total_timeout < self.subagent_tool_execution_timeout:
            self.supervisor_total_timeout = self.subagent_tool_execution_timeout
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内唯一的全局配置实例。"""
    return Settings()

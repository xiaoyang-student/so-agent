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
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 嵌套子 Agent 工具窗口的清理宽限（秒，任务 33 嵌套修复）：
# ``subagent_tool_execution_timeout`` 未显式设置时派生为
# ``agent_total_timeout + 本宽限``；agent_total<=0（不设总预算）时
# 派生为 0（父级不设硬截止，由子 Agent 自身存活检测负责）。
_SUBAGENT_TOOL_CLEANUP_GRACE: float = 10.0

# 长时代码 Agent 工具窗口的清理宽限（秒，任务 45）：
# ``code_agent_tool_execution_timeout`` 未显式设置时派生为
# ``agent_total_timeout + 本宽限``，与嵌套子 Agent 同理；agent_total
# <=0 时派生为 0。
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
    # 多 Agent 效率目标（任务 103）：time_first 时间优先，接受更多
    # token 换取可感知的壁钟加速；token_first 维持更严格的净收益门槛。
    # 两个阈值 <= 0 时按目标默认解析（time_first 1.10 / 1.10；
    # token_first 1.30 / 1.00）。无论目标如何，绝对 token 仍受下方
    # 任务级预算熔断约束。
    optimization_objective: Literal["time_first", "token_first"] = "time_first"
    parallelism_net_gain_threshold: float = 0.0   # <=0 使用目标默认净收益门槛
    parallelism_min_speedup: float = 0.0          # <=0 使用目标默认最小加速比

    # ---- 超时与沙箱 ----
    model_timeout: float = 45.0          # 单次模型调用超时（秒）；配合 flash 级模型，45 秒足够单次 API 响应
    # 沙箱后端：auto 优先选择 Docker/Podman 容器后端，不可用时按
    # sandbox_allow_local_fallback 决定是否回退到本地验证型子进程；
    # 生产环境应设置 sandbox_require_isolation=True 关闭回退。
    sandbox_backend: str = "auto"
    sandbox_timeout: float = 300.0       # 沙箱单次命令执行超时（秒，<=0 不限制）
    sandbox_output_limit: int = 1_048_576  # 沙箱输出上限（字节，默认 1 MiB）
    sandbox_allow_local_fallback: bool = True
    sandbox_require_isolation: bool = False
    # 容器后端参数（L2）。默认无网络、只读根、丢弃全部 capability，
    # 仅把单次运行目录挂载为可写工作区。
    sandbox_container_image: str = "python:3.12-slim"
    sandbox_container_python: str = "python"
    sandbox_container_user: str = ""      # 空值：POSIX 使用宿主 uid:gid，Windows 交给运行时
    sandbox_container_memory_mb: int = 512
    sandbox_container_cpus: float = 1.0
    sandbox_container_pids_limit: int = 128
    sandbox_container_tmpfs_mb: int = 64
    sandbox_container_network: Literal["none", "bridge"] = "none"
    sandbox_container_pull: bool = True
    # 云沙箱配置占位：当前不实现具体协议，构造 cloud 后端仍 fail-closed；
    # 其他项目接入时通过 register_sandbox_backend("cloud", factory) 注入。
    sandbox_cloud_provider: str = ""
    sandbox_cloud_endpoint: str = ""
    sandbox_cloud_api_key_env: str = ""
    sandbox_cloud_template: str = ""
    sandbox_cloud_timeout: float = 120.0
    # 超时语义（任务 80）：正常任务不设总时长上限，只保留“存活/卡死”
    # 检测与“循环边界”。
    # - first_event_timeout：流式首事件（含 SDK 冷启动与模型首包等待）
    #   窗口——首个流事件前的最大等待秒数；
    # - model_idle_timeout：模型阶段空闲窗口——连续该秒数没有任何
    #   SDK 原生流事件才判定断流/卡死（不接受 LLM 文本心跳作为续时依据）；
    # - tool_execution_timeout：普通工具执行阶段窗口——语义 tool_called
    #   事件到对应 tool_output 事件之间的最大等待秒数；
    # - agent_total_timeout / supervisor_total_timeout：单次调用绝对总
    #   预算；<=0 表示不设总预算（默认），合法长调用可以持续任意时长；
    # - max_pipeline_timeout：整条流水线绝对上限；<=0 表示不设上限
    #   （默认），仅由部署方按需开启；
    # - pipeline_no_progress_timeout：活动看门狗——任何 SDK 流事件、
    #   工具开始/结束、子任务完成、阶段推进或检查点都会续时，连续该
    #   秒数完全没有活动才判定卡死。
    # 0 语义：所有窗口/预算 <=0 均表示“关闭该限制”，而不是立即超时。
    # 关系（仅当相关上限显式开启时）：
    #       first_event / model_idle / tool_execution
    #       < agent_total ≤ supervisor_total ≤ max_pipeline_timeout。
    first_event_timeout: float = 180.0
    model_idle_timeout: float = 90.0
    tool_execution_timeout: float = 180.0
    agent_total_timeout: float = 0.0
    # 流水线活动看门狗：任何真实活动（SDK 事件、工具生命周期、子任务
    # 完成、阶段推进、检查点落盘）都会重置窗口；它不限制正常工作的
    # 总时长（长任务可以持续数小时），只识别完全停滞。<=0 表示关闭。
    pipeline_no_progress_timeout: float = 1200.0
    # 绝对兜底：整条流水线的硬上限。正常运行时不应由它来管制任务；
    # 仅在看门狗失效或部署方明确要求时启用。<=0 表示不设置。
    max_pipeline_timeout: float = 0.0
    # 嵌套子 Agent / code_agent 的父级工具等待窗口：0（默认）表示不设
    # 父级硬截止，由子 Agent 自身的首事件/空闲窗口、工具窗口与回合
    # 上限负责卡死检测；显式设置为正数时作为父级兜底。
    subagent_tool_execution_timeout: float = 0.0
    code_agent_tool_execution_timeout: float = 0.0
    # 主管单次调用的绝对总预算（秒）：<=0（默认）表示不设总预算。
    # 显式开启时由配置校验器保证不短于嵌套子 Agent 工具窗口。
    supervisor_total_timeout: float = 0.0
    # 工具授权默认有效期（秒）：与任务总时长解耦；<=0 表示过期时间
    # 为 None（仍会在任务退出时统一撤销）。
    grant_ttl_seconds: float = 86_400.0
    # ---- 旧超时字段（deprecated 兼容，任务 33）----
    # 仅在被显式设置（环境变量 / .env / 构造参数）时映射到新字段，
    # 见 _apply_deprecated_timeout_aliases：
    # - max_idle_timeout → first_event_timeout 与 model_idle_timeout
    #   （旧单一空闲窗口语义；工具执行阶段不再沿用旧窗口，修复旧的
    #   60 秒误杀工具执行问题，保持新默认 tool_execution_timeout）；
    # - max_supervisor_call_timeout → agent_total_timeout。
    max_idle_timeout: float = 60.0
    max_supervisor_call_timeout: float = 600.0

    # ---- Token 预算与熔断（默认启用）----
    # 按任务级总量（max_total）与单 Agent 累计（max_agent）双阈值检查；
    # <=0 表示该维度不限制。默认值按历史真实任务的最坏观测（复杂任务
    # 峰值约 57 万 tokens）留出约 40% 余量，异常失控任务会在下一检查点
    # 终止，而不是无限消耗。启用但两个阈值都为 0 视为配置错误
    # （fail-closed），避免“开了开关却没有约束”的假安全。
    token_budget_enabled: bool = True
    token_budget_max_total_tokens: int = 800_000
    token_budget_max_agent_tokens: int = 400_000

    # ---- 路由与分类 ----
    classifier_max_tokens: int = 512       # 分类调用的最大输出 token（保证含推理说明的完整 JSON 输出）
    direct_answer_max_tokens: int = 1024   # 直答输出上限（<=0 表示交给服务端默认，不显式限制）
    deep_reasoning_max_tokens: int = 4096  # 深度推理输出上限（<=0 表示不显式限制）
    enable_router: bool = True             # 是否启用四象限路由（False 则走原有流水线）

    # ---- P3 轻量快通道（单文件纯写入，默认关闭）----
    # 仅当目标明确要求写入唯一一个沙箱内文本文件、且不含运行/安装/网络/
    # 多文件等动作时启用：一次轻量模型调用生成正文 + 控制层原子写入，
    # 跳过规划与执行主管调用。任何不确定或失败都回退完整工作流。
    fast_path_single_file_enabled: bool = False
    fast_path_max_output_tokens: int = 2048  # 正文生成输出上限（<=0 不显式限制）
    fast_path_max_file_bytes: int = 262_144  # 单文件写入上限（字节）

    # ---- 主管上下文瘦身 ----
    # 规划阶段 available_tools 仅保留名称与截断描述（均已审批且可分配，
    # source/approved 对决策无增益）；关闭后回退完整元数据。
    supervisor_context_slimming: bool = True
    supervisor_tool_description_max_chars: int = 160

    # ---- 结果评审策略 ----
    skip_result_review_for_simple: bool = True  # 简单任务快速通道：子任务 <= 2 且无依赖且全成功且产出充分时跳过汇总与结果评审；False 则所有任务一律走结果评审
    # 简单计划跳过 LLM 计划评审：仅首次规划、<=2 子任务、无依赖、
    # 工具均已批准且可分配、写入子任务均有可机检验收规则时生效；
    # 结果评审与其他确定性验收不受影响。
    skip_plan_review_for_simple: bool = True
    # 授权评审分级：risk_based（默认）= 候选满足确定性最小化条件时
    # 用控制层校验替代模型授权评审（工具必要性由计划评审负责）；
    # always = 恒做独立模型授权评审（审计或合规要求时使用）。
    authorization_review_mode: Literal["always", "risk_based"] = "risk_based"
    review_parse_retry_max_attempts: int = 1    # 评审输出无法解析时的严格契约重试次数（0 表示直接 fail-closed）
    # 确定性检查前置（默认开启）：控制层先运行计划结构 / 授权必要性 /
    # 结果正确性检查，结果随 ReviewInput.deterministic_findings 下发，
    # 评审 Agent 不再绑定校验工具（单轮完成）；关闭时评审 Agent 自行
    # 调用校验工具复核，恢复旧的多轮行为。
    review_precomputed_findings: bool = True
    # 结果评审模式：always=所有非简单任务阻塞评审；risk_based=低风险且
    # 确定性检查充分时跳过并记录原因；off=仅确定性验收（不建议生产）。
    result_review_mode: Literal["always", "risk_based", "off"] = "always"
    # 评审输出预算：结论会进入计划版本、事件与后续载荷，条目过多或单条
    # 过长只增加生成时间与下游上下文。超限按条数/字符截断并标注；
    # <=0 表示该项不限制。
    review_issues_max_items: int = 8
    review_suggestions_max_items: int = 8
    review_required_fixes_max_items: int = 8
    review_item_max_chars: int = 240
    review_summary_max_chars: int = 400
    # 汇总与结果评审并行（两者都只依赖执行结果，输入互不依赖）：
    # 默认开启以缩短串行等待；关闭后回退为"先汇总、后评审"。
    result_review_parallel_with_aggregation: bool = True
    # 异步 provisional 交付（默认关闭）：仅低风险、无副作用且确定性检查
    # 充分的计划先交付结果并后台评审；评审失败自动返工最多 1 次并保留
    # 旧版本，耗尽后转人工。
    result_review_async_provisional: bool = False

    # ---- 确定性验收（不依赖模型自述）----
    # code_agent 后置校验：create/modify 声明的文件必须真实存在且非空，
    # execute 必须带成功的沙箱执行结果；不满足时在工具调用内部即翻转为
    # 失败，避免把伪造的成功结果带回主管上下文。
    code_agent_postcondition_enabled: bool = True
    # 子任务工具证据：声明产出文件/产物的成功子任务，必须能在事件日志中
    # 找到本次尝试完成的工具调用；否则确定性翻转为失败。
    subtask_tool_evidence_required: bool = True
    # 计划工具范围一致性：instructions 明确要求执行代码却未授权
    # run_in_sandbox 时，在采纳阶段确定性补齐并留痕，避免执行期被授权
    # 评审拒绝后重试；关闭后保持原样，由评审拦截。
    plan_tool_scope_lint: bool = True
    # 计划内步骤合并：把“产生产物 → 基于该产物写报告”这类单依赖两步
    # 合并为一个子任务，省去一次 Agent 调用与跨调用的重复读取。只合并
    # 下游显式引用上游产物、且两步由同一执行器族承担的组合；0 表示关闭。
    plan_merge_report_max_pairs: int = 1
    # 合并步骤的回合余量：合并后的子任务一次调用承担原两步工作量，
    # 在 ``max_model_turns`` 之上加该余量，避免正常完成前耗尽回合上限。
    # 只对合并产生的那一个子任务生效；0 表示不加余量。
    plan_merge_turn_bonus: int = 6
    # code_agent 回合续期：达到回合上限但本轮已有成功的工具调用时，
    # 在同一工具调用内续轮，而不是整体失败后由上层重新派发一整轮。
    # 续期有界（次数与每次追加回合数），无成功工具调用的空转不续期。
    code_agent_turn_extension: bool = True
    code_agent_turn_extension_budget: int = 6
    code_agent_turn_extension_max: int = 1

    # ---- 子任务硬预算与执行器直派 ----
    # 标准内置工具子任务由代码直派给 code_agent（省去主管中转的固定
    # 开销）；关闭后回退为主管路由。
    subtask_direct_code_agent: bool = True
    # 单个子任务（跨全部尝试与替换）的 token 预算；<=0 表示不限制。
    # 用量经 usage 作用域精确归因到子任务，超限时停止继续尝试并转重规划。
    subtask_max_tokens: int = 150_000
    # 单个子任务（跨全部尝试与替换）的总时长预算（秒）；<=0 表示不限制。
    subtask_max_wall_seconds: float = 900.0
    # 确定性验收失败后的动作级修复次数（每次业务尝试内）；只修复验收
    # 指出的缺口，不重跑完整业务任务，0 表示关闭。
    subtask_action_repair_max_attempts: int = 1

    # ---- 工具输出治理（控制上下文膨胀）----
    # 工具返回值直接进入模型上下文；无上限时一次 read_file / 沙箱
    # stdout 就会把后续每轮输入 token 放大数倍。<=0 表示不截断。
    tool_output_max_chars: int = 12000       # read_file / run_in_sandbox 单次回报上限
    code_agent_output_max_chars: int = 8000  # code_agent 回传给主管的最终 JSON 上限
    # 输出治理：禁止模型把同一份代码同时写进正文与 code_diff；diff 只保留
    # 路径/行数/字节数摘要。<=0 的 max_output_tokens 表示不显式限制。
    code_agent_suppress_diff_echo: bool = True
    code_agent_max_output_tokens: int = 0
    # 叙述性字段的确定性预算：模型自述正文会整体进入上级上下文，缺少
    # 上限时一次“完成情况说明”就吃掉数千字符。超限按字符硬截断并标注；
    # <=0 表示该项不限制。
    code_agent_summary_max_chars: int = 400
    code_agent_action_max_chars: int = 160
    code_agent_evidence_item_max_chars: int = 240
    code_agent_evidence_max_items: int = 10
    # 单次写入上限（UTF-8 字节）：超限时拒绝并提示分段写入（append /
    # write_and_run），避免单轮生成超大文件造成时延长尾；<=0 不限制。
    code_agent_write_max_bytes: int = 16384
    # 同族任务脚本复用：成功执行过的 .py 产物按任务目标哈希缓存到工具库，
    # 同一目标再次执行时校验哈希后复制进沙箱，提示 code_agent 优先复用。
    code_agent_script_reuse: bool = True
    # 工具回执化：read_file 默认只返回窗口内容，整文件留在沙箱按需重读。
    # 模型每轮都会重发历史工具结果，整文件回显是 code_agent 输入 token
    # 的主要来源；<=0 表示不限制。
    code_agent_read_max_lines: int = 300
    code_agent_read_max_chars: int = 12000
    code_agent_read_dedup: bool = True
    # 写/执行去重：同一运行内相同路径同内容写入、或相同代码且工作区未
    # 变化时重复执行，返回回执而不重复产生副作用。
    code_agent_action_dedup: bool = True
    # write_file / write_and_run 返回的 diff 行数上限；代码已落盘，
    # 完整 diff 对后续轮次没有增量信息。
    code_agent_tool_diff_max_lines: int = 40
    # 模型输入裁剪：发送前把旧工具调用中的代码参数与旧工具输出替换为
    # 回执，只保留最近 N 条工具输出原文；<=0 表示关闭字符预算。
    # 默认阈值取高水位：折叠会重写前缀、令供应商前缀缓存整体失效，实测
    # 30k 时每次运行折叠 4–5 次，缓存命中率降 7–10 个百分点且模型因丢失
    # 上文追加读写；正常任务应不触发折叠，超长会话才由它兜底。
    code_agent_context_prune: bool = True
    code_agent_context_max_chars: int = 60000
    code_agent_context_keep_tool_outputs: int = 2

    # ---- 上下文载荷治理 ----
    # 每次主管调用记录 phase / 字符数 / 估算 token，用于按阶段定预算。
    context_payload_audit: bool = True
    # 向 code_agent 与动态子 Agent 注入沙箱浅层文件索引（只含元数据）。
    artifact_index_in_prompt: bool = True
    # code_agent 的索引只列与本子任务相关的文件（写入范围 + 最近产物），
    # 避免每次调用都把整个沙箱目录塞进上下文；<=0 表示不注入。
    artifact_index_max_entries: int = 20

    # ---- 工具库（生成工具与外部 MCP 目录）----
    tool_store_dir: str = "tools"          # 工具存储根目录（相对路径以 cwd 为根）
    tool_store_migrate_legacy: bool = True # 任务开始时把 {sandbox}/generated_tools 一次性复制到工具库
    # 旧配置兼容：显式设置时继续以 sandbox_dir 为根（不迁移已有部署）
    generated_tools_dir: str = "generated_tools"
    mcp_servers_dir: str = ""              # 外部 MCP 目录；为空时取 {tool_store}/mcp_servers
    mcp_directory_enabled: bool = True     # 是否在任务开始时扫描外部 MCP 目录
    mcp_directory_watch_seconds: int = 0   # Web 控制台目录轮询间隔（秒）；0 表示关闭

    # ---- 插件（外部能力可选加载）----
    # 目录约定：{plugins_dir}/{kind}/{name}/plugin.yaml。仅加载显式启用
    # 的插件；为空表示只用内置能力。沙箱插件注册后可用
    # sandbox_backend=<name> 选择。详见 docs/插件化规范.md。
    plugins_dir: str = "plugins"
    plugins_enabled: str = ""              # 逗号分隔插件 id；"*" 表示加载目录内全部
    plugins_require_hash: bool = False     # 强制入口文件 SHA-256 钉住

    # ---- 编排运行时 ----
    # native（默认）直接使用 WorkflowEngine；langgraph 为可选外层适配器，
    # 缺少依赖时启动失败，不引入硬依赖。见 runtime/orchestration_runtime.py。
    orchestration_runtime: str = "native"

    # ---- 高风险动作审批门 ----
    # 默认关闭；启用后在 run_in_sandbox 命中风险模式时先落盘审批记录，
    # 经 Web 控制台或 Webhook 决定后才执行，超时按拒绝处理。
    high_risk_approval_enabled: bool = False
    high_risk_approval_timeout_seconds: float = 300.0
    high_risk_approval_poll_seconds: float = 1.0
    high_risk_approval_patterns: str = ""  # 追加正则，分号分隔
    approval_store_path: str = ""           # 空 -> {共享存储目录}/tool_approvals.db

    # ---- 工具生命周期 ----
    tool_require_test_cases: bool = True   # 生成工具准入必须提供可执行验收用例（False 仅限受控调试）
    tool_failure_quarantine_threshold: int = 3  # 连续失败达到该次数自动隔离；0 表示关闭
    tool_candidate_ttl_days: int = 7       # 未批准候选的保留天数
    tool_approved_ttl_days: int = 30       # 已批准工具无使用记录的保留天数
    tool_deprecation_grace_days: int = 7   # 弃用到归档之间的宽限期
    tool_archive_retention_days: int = 90  # 归档目录保留天数
    tool_sweep_interval_minutes: int = 1440  # 自动清理最小间隔（默认每天一次）

    # ---- 持久化调度 ----
    task_store_path: str = "task_store.db"  # SQLite 存储路径（相对路径以 sandbox_dir 为根）
    scheduler_poll_interval: int = 60      # 调度器轮询间隔（秒）
    scheduler_max_concurrent_tasks: int = 4  # 调度器同时执行的任务上限（防止无限并发）
    # 错过触发时间的处理策略：coalesce=补跑一次后按当前时间重排；
    # catch_up=按到期顺序补跑（受 max_catch_up_runs 限制）；skip=直接跳到下一次。
    scheduler_misfire_policy: Literal["coalesce", "catch_up", "skip"] = "coalesce"
    # 超过该宽限才视为“错过”（秒）；宽限内按正常执行，避免轮询抖动触发策略。
    scheduler_misfire_grace_seconds: int = 60
    scheduler_max_catch_up_runs: int = 3
    # 单任务失败重试：0 表示失败即暂停（与旧行为一致）；>0 时按指数退避重排。
    scheduler_task_max_retries: int = 0
    scheduler_task_retry_backoff_seconds: float = 60.0

    # ---- Web 控制台 ----
    web_max_concurrent_tasks: int = 2      # Web 同时执行的任务上限（超出排队）
    web_auth_token: str = ""               # 访问令牌（为空时仅依赖默认 127.0.0.1 绑定）
    web_sandbox_dir: str = "sandbox/web"   # Web 任务沙箱根目录（相对路径以 cwd 为根）
    web_task_retention_days: int = 7       # 终态任务与产物保留天数（<=0 关闭按时间清理）
    web_max_retained_tasks: int = 200      # 保留的终态任务记录上限（<=0 关闭按数量清理）
    web_scheduler_enabled: bool = True     # 是否在 Web 进程内运行持久化任务调度器
    # 重启恢复：未完成任务自动重新排队执行，每个任务最多自动恢复 1 次；
    # 关闭后沿用旧行为（直接标记失败）。
    web_resume_interrupted_tasks: bool = True
    # 结果沉淀：默认只保留最近一次任务结果（latest.md / latest.json），
    # 新任务提交时清理上一次终态任务与产物；需要历史时打开
    # web_keep_history，结果存档接口无需改动。
    web_keep_history: bool = False
    web_results_dir: str = "results"       # 结果存档目录（相对 Web 任务根目录）

    # ---- 跨任务防篡改审计链（默认关闭）----
    # 设置路径后，所有事件同步追加到独立 SQLite 哈希链；空值保持旧行为。
    # 审计库只追加不裁剪，部署方需自行负责容量与备份。
    audit_trail_path: str = ""

    # ---- OpenTelemetry 追踪（默认关闭）----
    # exporter=none 时使用宿主已配置的全局 TracerProvider；console/otlp
    # 时由本进程自建 provider（otlp 读取标准 OTEL_EXPORTER_OTLP_* 环境变量）。
    otel_enabled: bool = False
    otel_service_name: str = "so-agent"
    otel_exporter: Literal["none", "console", "otlp"] = "none"

    # ---- 人工升级外部通知（默认关闭）----
    # URL 必须为 HTTPS；默认拒绝内网地址，令牌只从环境变量读取，不写入配置。
    escalation_webhook_url: str = ""
    escalation_webhook_token_env: str = ""
    escalation_webhook_timeout_seconds: float = 5.0
    escalation_webhook_allow_private_network: bool = False
    escalation_webhook_max_attempts: int = 3          # 总尝试次数（含首次；1 表示不重试）
    escalation_webhook_retry_backoff_seconds: float = 0.5  # 指数退避基数（秒）
    escalation_webhook_signing_secret_env: str = ""   # 可选 HMAC 签名密钥环境变量名
    # 通知 outbox：先持久化再投递，进程崩溃或端点短暂不可用时由后台
    # 任务按退避重试；同 event_id 只入队一次，接收方按 Idempotency-Key 去重。
    notification_outbox_enabled: bool = False
    notification_outbox_path: str = "sandbox/notification_outbox.db"
    notification_outbox_max_attempts: int = 8         # 单条通知的最大投递次数
    notification_outbox_retry_backoff_seconds: float = 5.0  # 退避基数（秒，上限 300）
    notification_outbox_drain_interval_seconds: float = 30.0
    notification_outbox_batch_size: int = 20

    # ---- 确定性上下文压缩 ----
    context_compression_enabled: bool = True
    context_compression_max_chars: int = 24000  # 单次载荷压缩后的总字符上限
    context_compression_result_chars: int = 4000  # 单条执行结果保留字符上限
    # 编号/契约枚举/路径/验收规则等关键字段不参与压缩，避免截断导致
    # 下游解析、路由与文件校验失真。
    context_compression_protect_critical: bool = True

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

        - ``subagent_tool_execution_timeout`` / ``code_agent_tool_execution_timeout``
          未显式设置时派生为 ``agent_total_timeout + 清理宽限``；
          ``agent_total_timeout <= 0``（不设总预算）时同样派生为 0，
          表示父级不设硬截止；
        - ``supervisor_total_timeout`` 仅在“自身与嵌套工具窗口都显式
          开启”时保底不短于嵌套工具窗口；任一方为 0（不限制）时不做
          保底，避免把“无限”改成固定值。
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
        # 嵌套子 Agent / code_agent 工具窗口：未显式设置时按 agent_total
        # 派生；agent_total<=0 表示不设总预算，派生窗口也为 0
        if "subagent_tool_execution_timeout" not in fields_set:
            self.subagent_tool_execution_timeout = (
                float(self.agent_total_timeout)
                + _SUBAGENT_TOOL_CLEANUP_GRACE
                if self.agent_total_timeout > 0
                else 0.0
            )
        if "code_agent_tool_execution_timeout" not in fields_set:
            self.code_agent_tool_execution_timeout = (
                float(self.agent_total_timeout)
                + _CODE_AGENT_TOOL_CLEANUP_GRACE
                if self.agent_total_timeout > 0
                else 0.0
            )
        # 父级总预算保底：仅当双方都是正数上限时才比较；0=不限制
        if (
            self.supervisor_total_timeout > 0
            and self.subagent_tool_execution_timeout > 0
            and self.supervisor_total_timeout
            < self.subagent_tool_execution_timeout
        ):
            self.supervisor_total_timeout = self.subagent_tool_execution_timeout
        return self

    @model_validator(mode="after")
    def _validate_token_budget(self) -> "Settings":
        """Token 预算配置校验：负数非法；启用时必须至少一个正上限。"""
        if (
            self.token_budget_max_total_tokens < 0
            or self.token_budget_max_agent_tokens < 0
        ):
            raise ValueError("token 预算上限不能为负数")
        if (
            self.token_budget_enabled
            and self.token_budget_max_total_tokens <= 0
            and self.token_budget_max_agent_tokens <= 0
        ):
            raise ValueError(
                "token_budget_enabled=True 时需至少配置一个正上限"
                "（token_budget_max_total_tokens 或 token_budget_max_agent_tokens）"
            )
        return self

    @model_validator(mode="after")
    def _validate_governance_limits(self) -> "Settings":
        """通知、输出与上下文参数校验：超时/上限不得为负。"""
        if self.escalation_webhook_timeout_seconds <= 0:
            raise ValueError("escalation_webhook_timeout_seconds 必须大于 0")
        if self.notification_outbox_drain_interval_seconds <= 0:
            raise ValueError(
                "notification_outbox_drain_interval_seconds 必须大于 0"
            )
        if self.context_compression_max_chars < 0:
            raise ValueError("context_compression_max_chars 不能为负数")
        if self.context_compression_result_chars < 0:
            raise ValueError("context_compression_result_chars 不能为负数")
        if self.pipeline_no_progress_timeout < 0:
            raise ValueError("pipeline_no_progress_timeout 不能为负数")
        if self.grant_ttl_seconds < 0:
            raise ValueError("grant_ttl_seconds 不能为负数")
        if self.review_parse_retry_max_attempts < 0:
            raise ValueError("review_parse_retry_max_attempts 不能为负数")
        if self.tool_output_max_chars < 0:
            raise ValueError("tool_output_max_chars 不能为负数")
        if self.code_agent_output_max_chars < 0:
            raise ValueError("code_agent_output_max_chars 不能为负数")
        if self.code_agent_max_output_tokens < 0:
            raise ValueError("code_agent_max_output_tokens 不能为负数")
        if self.scheduler_misfire_grace_seconds < 0:
            raise ValueError("scheduler_misfire_grace_seconds 不能为负数")
        if self.scheduler_max_catch_up_runs < 1:
            raise ValueError("scheduler_max_catch_up_runs 必须至少为 1")
        if self.scheduler_task_max_retries < 0:
            raise ValueError("scheduler_task_max_retries 不能为负数")
        if self.scheduler_task_retry_backoff_seconds < 0:
            raise ValueError("scheduler_task_retry_backoff_seconds 不能为负数")
        if self.plan_merge_report_max_pairs < 0:
            raise ValueError("plan_merge_report_max_pairs 不能为负数")
        if self.plan_merge_turn_bonus < 0:
            raise ValueError("plan_merge_turn_bonus 不能为负数")
        if self.code_agent_turn_extension_budget < 0:
            raise ValueError("code_agent_turn_extension_budget 不能为负数")
        if self.code_agent_turn_extension_max < 0:
            raise ValueError("code_agent_turn_extension_max 不能为负数")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程内唯一的全局配置实例。"""
    return Settings()

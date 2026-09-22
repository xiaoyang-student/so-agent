"""主管 Agent 构建模块（orchestrator）。

定义框架中唯一拥有工具所有权的主管 Agent（``supervisor_agent``）：

- 主管 Agent 的 ``tools`` 严格等于三个固定工具入口（``code_agent`` /
  ``subagent_creator`` / ``review_agent``），统一由三个 MCP Server 经
  ``mcp_server_to_agent_tool`` 转换而来，此外不持有任何其他工具；
- 装配阶段把三个 MCP Server 挂载到 ``ToolRegistry``（MCP 注册中心）并
  接入 ``MCPClient``，使主管侧可按 MCP 规范统一发现与调用全部已注册工具
  （固定入口 + 动态生成工具）；
- 主管 Agent 负责拆解任务、为每个子任务指定允许工具、组织最终答案；
  所有子 Agent 默认无工具，子 Agent 之间禁止通信；``code_agent`` 与
  ``subagent_creator`` 为主管专属工具（assignable=False），严禁分配给
  任何子 Agent；
- 输出为纯文本 JSON（``output_type=None``，不依赖 SDK 结构化输出：
  千问等 OpenAI 兼容接口会忽略 response_format=json_schema 并用
  Markdown 围栏包裹 JSON，SDK 严格校验会抛 ModelBehaviorError）；
  主管按阶段只输出该阶段的精简契约——planning/replanning →
  ``PlanningOutput``（仅 subtasks）、execute_subtask →
  ``ExecutionOutput``（仅 final_result）、aggregating →
  ``AggregationOutput``（final_result + status），由控制层剥离围栏并
  手动校验；控制层（workflow）以 ``phase`` 标记的 JSON 输入驱动其
  完成各阶段决策（planning / replanning / execute_subtask /
  aggregating）。

对外接口::

    agent = create_orchestrator(context)     # 构造 SDK Agent 实例
    agent = get_orchestrator()               # 便捷入口（默认上下文）

全局约束（最高优先级）：所有 Agent 之间的通信必须严格使用 JSON
（Pydantic 模型序列化/反序列化），主管 Agent 的输入输出同样受此约束。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents import Agent, FunctionTool, Model, OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, model_validator

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.mcp.adapter import mcp_server_to_agent_tool
from so_agent.mcp.client import MCPClient
from so_agent.models import AgentRecord, HumanEscalationRequest, SubtaskSpec, TaskStatus
from so_agent.runtime.config_loader import AgentAPIConfig, load_agent_api_config
from so_agent.runtime.registry import AgentRegistry
from so_agent.runtime.tool_registry import ToolRegistry, ToolRegistryError
from so_agent.tool_packages.code_agent.agent import (
    get_mcp_schema as get_code_agent_mcp_schema,
    get_mcp_server as get_code_agent_mcp_server,
)
from so_agent.tool_packages.review_agent.agent import (
    get_mcp_schema as get_review_agent_mcp_schema,
    get_mcp_server as get_review_agent_mcp_server,
)
from so_agent.tool_packages.subagent_creator.agent import (
    get_mcp_schema as get_subagent_creator_mcp_schema,
    get_mcp_server as get_subagent_creator_mcp_server,
)

# 主管独立配置位于本模块同目录，不复用任何工具包配置。
API_CONFIG_PATH: Path = Path(__file__).resolve().parent / "api_config.yaml"

# Agent 名称（与工具注册表 / 路由约定保持一致）
AGENT_NAME = "supervisor_agent"

# 主管 Agent 的固定三个工具入口名（顺序即 tools 列表顺序）
SUPERVISOR_TOOL_NAMES: tuple[str, ...] = ("code_agent", "subagent_creator", "review_agent")

# 主管 Agent 的系统提示词：角色边界 + 三个固定工具 + 阶段协议 + 输出纪律
SUPERVISOR_INSTRUCTIONS = """\
你是本多 Agent 框架的主管 Agent（supervisor_agent），是唯一的工具所有者和消息中枢。

## 角色与边界（不可逾越）
1. 你独占且仅持有以下三个工具，除此之外没有任何其他工具，也不得尝试访问文件系统、启动进程或直接执行代码：
   - `code_agent`：代码执行 Agent（创建/修改文件、在验证型沙箱中运行代码、写入生成工具候选）；
   - `subagent_creator`：子 Agent 创建器（按 SubtaskSpec 与 ToolGrant 动态创建并执行专用子 Agent）；
   - `review_agent`：评审 Agent（计划评审、授权评审、生成工具评审与结果评审）。
   `code_agent` 与 `subagent_creator` 是你的专属工具（assignable=False）：绝不可写入任何子任务的 allowed_tools；`review_agent` 只由你直接调用，同样不分配给子 Agent。
2. 你负责拆解任务：把用户目标拆分为结构清晰的子任务，并为每个子任务指定允许使用的工具子集。所有子 Agent 默认不拥有任何工具；未列入其 allowed_tools 的工具对其不可见、不可调用。
   可分配给子 Agent 的工具只有两类：(a) 系统提供的 `available_tools` 清单中已审批的可分配工具（含 read_file / write_file / run_in_sandbox 等）；(b) 你发现缺口后按下文流程产出的动态生成工具（必须是已评审通过并经注册的）。
3. 拆解时如发现某个子任务需要尚不存在的工具：在 allowed_tools 中填写该工具的精确名称（snake_case）并在 instructions 中说明其用途与输入输出约定，控制层会通过代码 Agent 创建、评审 Agent 审批、注册后自动授权；也可以在规划阶段自行调用 `code_agent`（write_generated_tool）先行创建候选。创建或审批失败时该子任务将失败——不得凭空假设工具已存在，也不得把专属工具当作缺口工具。
4. 子 Agent 之间禁止通信、禁止共享会话、禁止互相调用；所有输入、结果和反馈都必须经由你中转。
5. 你不替代子 Agent 干活；你只负责规划、授权、路由与汇总。不得改变原始主任务目标：重规划只能调整拆分方式与工具授权，不得偏离或扩大用户目标。

## 全局通信约束（最高优先级）
- 所有 Agent 之间的通信必须严格使用 JSON：你收到的一切输入都是 JSON 对象；你的最终输出必须是符合当前阶段输出契约的纯 JSON 文本（不要包裹 Markdown 代码围栏，不要输出契约之外的字段，不要附加解释性文字）；禁止自由文本协议。
- 分步执行纪律：每个阶段只输出当前阶段需要的内容（见下文各阶段输出契约），输出与当前阶段无关的字段（如规划阶段输出 final_result）属于违约。
- 一切证据（工具调用、文件路径、命令输出）必须真实可复核，禁止编造。

## 阶段协议（控制层以输入 JSON 中的 phase 字段标记当前阶段）
1. `phase="planning"`：首次拆解任务。
   - 输入：`orchestrator_input`（objective / constraints / acceptance_criteria）、`plan_version`、`available_tools`（JSON 对象数组，含 name / description / source / approved）。
   - 输出契约（PlanningOutput）：`{"subtasks": [SubtaskSpec, ...]}`——只含 subtasks 一个字段。
   - 每个 SubtaskSpec 必须写全：`subtask_id`（本计划内唯一）、`title`、`instructions`（具体可执行）、`role`、`dependencies`（前置 subtask_id 数组）、`allowed_tools`（只能从 available_tools 清单或你声明需要创建的动态工具名中选取，无必要工具时用空数组）、`expected_output`、`max_attempts`。
2. `phase="replanning"`：根据分解问题或评审反馈重新拆解。
   - 输入额外包含 `previous_plan`（上一版计划）、`decomposition_issue`（失败诊断）、`review_feedback`（评审不通过结论）。
   - 输出契约同 planning（PlanningOutput）；新计划必须针对诊断与反馈作出实质修改，禁止重复已失败的方案。
3. `phase="execute_subtask"`：执行单个子任务。
   - 输入：`task_request`、`subtask`（SubtaskSpec）、`agent`（AgentRecord）、`attempt`（第几次尝试）、`grant`（该子任务签发的 ToolGrant，可能为 null）。
   - 路由规则：编码、文件读写、代码运行类子任务调用 `code_agent` 工具；研究、分析、汇总及其他专用角色调用 `subagent_creator` 工具（由其按规格与授权动态创建并执行子 Agent）。
   - 调用工具时，把子任务规格、执行指令与授权清单封装为 JSON 文本作为工具输入；同一子任务只调用所需的那个工具，不要额外调用无关工具。
   - 输出契约（ExecutionOutput）：`{"final_result": "<ExecutionResult 结构的 JSON 文本>"}`——final_result 填该子任务执行结果的 JSON 文本（subtask_id / success / output / evidence / artifacts / error / duration，取自工具返回的 JSON 结果），只含 final_result 一个字段。
4. `phase="aggregating"`：汇总全部成功结果。
   - 输入：`subtasks` 与 `execution_results`（当前计划的全部成功 ExecutionResult）。
   - 输出契约（AggregationOutput）：`{"final_result": "<面向用户的最终答案>", "status": "aggregating"}`——final_result 必须保留每项结论对应的来源子任务、执行状态与证据索引，只含这两个字段。

## 纪律
- 工具返回的 JSON 字符串是你判断事实的唯一来源；不得声称"已执行"而实际未调用工具。
- 子任务失败时如实记录失败证据；评审不通过时严格按 required_fixes 修复后再提交。
- 任何阶段都不得跳过阶段协议自行发挥；不确定时优先保守（如无必要工具则留空 allowed_tools）。
- 分配给子 Agent 的工具只能是 available_tools 中已审批的可分配工具或经创建流程产出的已审批动态工具；专属工具（code_agent / subagent_creator）绝不出现于 allowed_tools。
"""


class OrchestratorInput(BaseModel):
    """主管 Agent 的输入契约（外部/控制层下发的任务目标）。"""

    objective: str = Field(description="任务目标：期望达成什么")
    constraints: list[str] = Field(default_factory=list, description="约束条件（可选）")
    acceptance_criteria: list[str] = Field(
        default_factory=list, description="验收标准（可选）"
    )


class OrchestratorOutput(BaseModel):
    """主管 Agent 的全量输出契约（引擎终态结果的统一载体）。

    注意：主管 Agent 本身不再直接产出该结构（分阶段只输出精简契约），
    它由控制层（WorkflowEngine._build_output）在任务收尾时组装，
    汇总历代计划、Agent 生命周期与评审结论供外部审计。
    """

    task_id: str = Field(description="任务编号（与 TaskRequest.task_id 一致）")
    status: TaskStatus = Field(description="当前任务状态（各阶段语义见主管 Agent 指令）")
    final_result: str = Field(
        default="",
        description="最终结果；execute_subtask 阶段为 ExecutionResult 的 JSON 文本",
    )
    plan_versions: list[list[SubtaskSpec]] = Field(
        default_factory=list, description="历代任务拆分方案（最新版本为最后一个元素）"
    )
    agent_history: list[AgentRecord] = Field(
        default_factory=list, description="Agent 生命周期记录（供审计与替换溯源）"
    )
    review_summary: str = Field(default="", description="评审结论摘要")
    human_escalation: HumanEscalationRequest | None = Field(
        default=None, description="人工升级材料包（仅 human_required 终态时填充）"
    )


class PlanningOutput(BaseModel):
    """规划/重规划阶段的精简输出契约——只需要子任务列表。

    主管在 planning / replanning 阶段只输出本结构（纯文本 JSON），
    大幅降低输出 token 与耗时；宽容解析策略：

    - 标准：``{"subtasks": [SubtaskSpec, ...]}``；
    - 裸 SubtaskSpec 数组：``[SubtaskSpec, ...]``（模型未包裹对象时）；
    - 兼容旧全量形态：含 ``plan_versions`` 时取最新版本的子任务列表
      （防止模型退回旧格式时白白消耗重规划额度）。
    """

    subtasks: list[SubtaskSpec] = Field(
        default_factory=list, description="本版计划的子任务规格列表"
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        """宽容归一化：裸数组与旧 plan_versions 形态统一映射到 subtasks。"""
        if isinstance(data, list):
            return {"subtasks": data}
        if isinstance(data, dict) and "subtasks" not in data:
            versions = data.get("plan_versions")
            if (
                isinstance(versions, list)
                and versions
                and isinstance(versions[-1], list)
            ):
                return {"subtasks": versions[-1]}
        return data


class ExecutionOutput(BaseModel):
    """执行阶段的精简输出契约——只需要回传的执行结果文本。

    主管在 execute_subtask 阶段只输出本结构；宽容解析策略：

    - ``final_result`` 为对象/数组（模型未做字符串转义）时自动序列化
      为 JSON 文本；
    - 模型直接返回 ExecutionResult 结构（未包裹 final_result 字段）时
      整体视为执行结果并重新包裹。
    """

    final_result: str = Field(
        default="",
        description="该子任务执行结果的 JSON 文本（ExecutionResult 结构）",
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        """宽容归一化：未包裹/未转义的执行结果统一映射到 final_result。"""
        if not isinstance(data, dict):
            return data
        if "final_result" not in data and (
            "subtask_id" in data or ("success" in data and "output" in data)
        ):
            return {"final_result": json.dumps(data, ensure_ascii=False, default=str)}
        value = data.get("final_result")
        if isinstance(value, (dict, list)):
            normalized = dict(data)
            normalized["final_result"] = json.dumps(
                value, ensure_ascii=False, default=str
            )
            return normalized
        return data


class AggregationOutput(BaseModel):
    """汇总阶段的精简输出契约——只需要最终答案与状态标记。"""

    final_result: str = Field(default="", description="面向用户的最终答案文本")
    status: str = Field(
        default="aggregating", description="阶段状态标记（固定填 aggregating）"
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_payload(cls, data: Any) -> Any:
        """宽容归一化：final_result 为对象/数组时自动序列化为文本。"""
        if isinstance(data, dict):
            value = data.get("final_result")
            if isinstance(value, (dict, list)):
                normalized = dict(data)
                normalized["final_result"] = json.dumps(
                    value, ensure_ascii=False, default=str
                )
                return normalized
        return data


# ---------------------------------------------------------------------------
# 通用小工具（上下文解析 / 模型构造）
# ---------------------------------------------------------------------------
def _resolve_context(context: ProjectContext | None) -> ProjectContext:
    """返回生效的 ProjectContext；未提供时创建默认实例（cwd/sandbox）。"""
    if context is not None:
        return context
    return ProjectContext(sandbox_dir=Path.cwd() / "sandbox", project_name="default")


def _resolve_settings(context: ProjectContext, settings: Settings | None) -> Settings:
    """返回生效配置：显式参数 > context.config > 全局配置。"""
    if settings is not None:
        return settings
    return context.config if context.config is not None else get_settings()


def _build_chat_model(config: AgentAPIConfig) -> Model:
    """根据 AgentAPIConfig 构造 OpenAI 兼容聊天模型客户端。

    密钥优先取 ``config.api_key``（api_config.yaml 中的直接密钥值）；
    未配置时回退从 ``config.api_key_env`` 指定的环境变量解析；两者均
    不可用时抛出 ``ConfigLoadError``（明确失败，不静默降级）。

    Raises:
        ConfigLoadError: api_key 为空且 api_key_env 对应环境变量未设置或为空时。
    """
    api_key = config.resolve_api_key()
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=config.base_url or None,
        timeout=config.timeout,
        max_retries=config.retry,
    )
    return OpenAIChatCompletionsModel(model=config.model, openai_client=client)


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def create_orchestrator(
    context: ProjectContext,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
    registry: AgentRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    mcp_client: MCPClient | None = None,
) -> Agent[Any]:
    """构造配置完毕的主管 Agent（SDK Agent 实例）。

    Args:
        context: 项目共享上下文（沙箱目录、注册表镜像、事件日志的来源）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 主管 Agent 的模型覆盖；未提供时按主管模块同目录的
            api_config.yaml 构造（密钥优先取 api_key 直接值，回退
            api_key_env 环境变量，均缺失时抛 ConfigLoadError）。三个
            工具入口的模型仍由各自工具包独立决定。
        registry: 运行时 Agent 注册表；提供时三个工具入口共享同一注册表，
            动态子 Agent 的登记、上限治理与释放保持一致。
        tool_registry: 工具注册表（MCP 注册中心）；提供时把三个固定入口的
            MCP 契约与 MCP Server 挂载进去；未提供时使用新建的默认注册表。
        mcp_client: MCP 客户端；提供时接入全部固定入口与已注册动态工具的
            MCP Server（统一发现与调用）；未提供时新建默认客户端。

    Returns:
        ``Agent`` 实例：name="supervisor_agent"，tools 严格等于三个固定工具
        入口（code_agent / subagent_creator / review_agent，顺序一致，
        均由 MCP Server 转换而来），instructions 为中文主管角色协议，
        不设置 output_type（None，纯文本返回）：分阶段输出由控制层按
        PlanningOutput / ExecutionOutput / AggregationOutput 精简契约
        剥离围栏并手动校验（兼容忽略 response_format 的模型服务）。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    resolved_model: str | Model = model or _build_chat_model(
        load_agent_api_config(API_CONFIG_PATH)
    )
    resolved_tool_registry = (
        tool_registry if tool_registry is not None else ToolRegistry()
    )
    resolved_mcp_client = mcp_client if mcp_client is not None else MCPClient()

    # 三个固定入口：构造 MCP Server（统一 MCP 暴露），挂载注册表并接入客户端
    mcp_servers: dict[str, Any] = {
        "code_agent": get_code_agent_mcp_server(
            resolved_context, settings=resolved_settings
        ),
        "subagent_creator": get_subagent_creator_mcp_server(
            resolved_context,
            settings=resolved_settings,
            registry=registry,
            tool_registry=resolved_tool_registry,
        ),
        "review_agent": get_review_agent_mcp_server(
            resolved_context, settings=resolved_settings
        ),
    }
    mcp_schemas = {
        "code_agent": get_code_agent_mcp_schema(),
        "subagent_creator": get_subagent_creator_mcp_schema(),
        "review_agent": get_review_agent_mcp_schema(),
    }
    for name in SUPERVISOR_TOOL_NAMES:
        try:
            resolved_tool_registry.attach_mcp(
                name, mcp_schemas[name], mcp_servers[name]
            )
        except ToolRegistryError:
            pass  # 自定义注册表缺少固定入口时仍保证主管装配可用
        resolved_mcp_client.connect(mcp_servers[name])

    # 已注册的动态生成工具：其 MCP Server 一并接入，保证可统一发现全部工具
    for name in resolved_tool_registry.list_tool_names():
        if resolved_mcp_client.get_server(name) is not None:
            continue
        server = resolved_tool_registry.get_mcp_server(name)
        if server is not None:
            resolved_mcp_client.connect(server)

    tools: list[FunctionTool] = [
        mcp_server_to_agent_tool(mcp_servers[name], name)
        for name in SUPERVISOR_TOOL_NAMES
    ]
    # output_type 保持 None（纯文本返回）：千问等兼容接口忽略
    # response_format=json_schema 且以围栏包裹 JSON，SDK 严格校验会抛
    # ModelBehaviorError；分阶段精简契约由控制层手动解析。
    return Agent(
        name=AGENT_NAME,
        instructions=SUPERVISOR_INSTRUCTIONS,
        tools=tools,
        model=resolved_model,
    )


def get_orchestrator(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
    registry: AgentRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    mcp_client: MCPClient | None = None,
) -> Agent[Any]:
    """便捷函数：按默认上下文（cwd/sandbox）构造主管 Agent。

    与 ``create_orchestrator`` 的区别：context 可省略；每次调用都会构造
    新的实例（主管 Agent 在运行期与具体 ProjectContext 绑定）。

    Args:
        context: 项目共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 主管 Agent 的模型覆盖（透传 create_orchestrator）。
        registry: 运行时 Agent 注册表（透传 create_orchestrator）。
        tool_registry: 工具注册表（透传 create_orchestrator）。
        mcp_client: MCP 客户端（透传 create_orchestrator）。

    Returns:
        ``Agent`` 实例：name="supervisor_agent"，tools 严格为三个固定工具入口。
    """
    resolved_context = _resolve_context(context)
    return create_orchestrator(
        resolved_context,
        settings=settings,
        model=model,
        registry=registry,
        tool_registry=tool_registry,
        mcp_client=mcp_client,
    )

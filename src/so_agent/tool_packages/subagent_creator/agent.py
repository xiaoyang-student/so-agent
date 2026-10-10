"""子 Agent 创建器构建模块（subagent_creator 工具包）。

职责：作为主 Agent 的固定工具入口之一，构造“子 Agent 创建器（Subagent
Creator）”——在不预先开发、不重启系统的前提下，按需动态构造专用子 Agent
并完成创建与执行：

- 读取本目录 ``prompt.md`` 作为系统提示词（instructions）；
- 通过 ``runtime.config_loader`` 加载本目录 ``api_config.yaml``，按其中
  ``api_key`` 直接值（或旧 ``api_key_env`` 环境变量回退）构造 OpenAI 兼容
  聊天模型（密钥值绝不写日志）；
- 绑定两个核心工具：``create_subagent``（按 SubtaskSpec + ToolGrant 动态
  构造并登记子 Agent）与 ``execute_subagent``（执行已登记子 Agent 并回收
  ExecutionResult）；两个工具返回值统一为 JSON 字符串；
- 工具权限双重防护：构造时（create_subagent）按 allowed_tools ∩ 工具池
  过滤注入（第一层）；提供 PermissionGateway 时每个注入工具再经
  ``_wrap_tool_with_permission_check`` 包裹，在每次工具调用的
  ``on_invoke_tool`` 回调内执行 ``verify_grant`` 运行时校验（第二层，
  授权撤销/过期后立即生效）；
- Agent 输出为纯文本 JSON（``output_type=None``，不依赖 SDK 结构化输出：
  千问等 OpenAI 兼容接口会忽略 ``response_format=json_schema`` 并把
  JSON 包裹在 Markdown 代码围栏中返回，故由调用方剥离围栏后按
  ``SubagentCreatorOutput`` 手动校验）；动态子 Agent 同样纯文本返回，
  由 ``execute_subagent`` 剥离围栏后按 ``ExecutionResult`` 手动校验；
- 对外同时提供 MCP 暴露契约与 MCP Server（``get_mcp_schema`` /
  ``get_mcp_server``），使本工具可按 MCP 规范被主管 Agent 与 MCPClient
  统一发现与调用。

对外接口::

    agent = create_subagent_creator(context)         # 构造 SDK Agent 实例
    tool = get_subagent_creator_as_tool(context)     # 转为主 Agent 可用的工具入口
    schema = get_mcp_schema()                        # MCP 暴露契约（MCPToolSchema）
    server = get_mcp_server(context)                 # 包装为 MCP Server（统一调用）

全局约束（最高优先级）：
- 所有 Agent 之间的通信必须严格使用 JSON：工具返回值与 Agent 最终输出
  一律为可解析的 JSON（Pydantic 序列化），不允许自由文本协议；
- 动态子 Agent 默认 tools 为空，只能注入 ToolGrant 中获批的工具；
- 不向子 Agent 暴露其他 Agent 的引用、handoff 或共享会话（禁止横向通信）；
- 替代 Agent（replacement_of 非空）固定不具备“请求替换”权限
  （can_request_replacement=False）。
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agents import (
    Agent,
    FunctionTool,
    Model,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    RunResult,
    RunResultStreaming,
    function_tool,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.json_utils import extract_json_object
from so_agent.mcp.adapter import agent_to_mcp_server, mcp_server_to_agent_tool
from so_agent.mcp.protocol import MCPToolSchema
from so_agent.mcp.server import MCPToolServer
from so_agent.models import AgentRecord, ExecutionResult, SubtaskSpec, ToolGrant
from so_agent.runtime.config_loader import (
    AgentAPIConfig,
    load_agent_api_config,
    resolve_model_http_timeout,
)
from so_agent.runtime.events import EventLogger
from so_agent.runtime.permissions import PermissionGateway
from so_agent.runtime.registry import AgentRegistry, AgentRegistryError
from so_agent.runtime.sandbox import SandboxRunner
from so_agent.runtime.tool_registry import ToolRegistry
from so_agent.tool_packages.code_agent.agent import build_code_tools
from so_agent.workflow_support import make_tool_timeout_resolver, run_agent_streamed

# ---------------------------------------------------------------------------
# 包内路径（用 pathlib 定位同目录下的提示词与 API 配置）
# ---------------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROMPT_PATH: Path = PACKAGE_DIR / "prompt.md"
API_CONFIG_PATH: Path = PACKAGE_DIR / "api_config.yaml"

# Agent 名称（与工具注册表 / 路由约定保持一致）
AGENT_NAME = "subagent_creator"

# 主管侧调用说明（同时用于 as_tool 入口与 MCP 暴露契约）
_TOOL_DESCRIPTION = (
    "子 Agent 创建器：按 SubtaskSpec + ToolGrant 动态创建专用子 Agent 并立即执行，"
    "返回已创建 AgentRecord 与 ExecutionResult 的结构化 JSON 汇总。"
)

# 动态子 Agent 可注入的工具池（复用 code_agent 的基础工具，不含任何 Agent 引用）
_DYNAMIC_TOOL_POOL_NAMES: tuple[str, ...] = ("read_file", "write_file", "run_in_sandbox")

# 追加在 prompt.md 之后的输入输出契约说明（不改动原始提示词文件）
_EXTRA_INSTRUCTIONS = """
## 结构化输入（SubagentCreatorInput）

主 Agent 会以如下字段（SubagentCreatorInput）下发创建需求：

- `subtasks`：需要动态创建执行者的子任务规格列表（SubtaskSpec 数组）；
- `tool_grants`：主 Agent 为这些子任务签发的工具授权列表（ToolGrant 数组）。

## 工具调用流程（严格按序）

1. 对每个需要动态执行者的子任务调用 `create_subagent(subtask_spec_json, tool_grant_json)`：

   - `subtask_spec_json` 为 SubtaskSpec 的 JSON 文本；
   - `tool_grant_json` 为 ToolGrant 的 JSON 文本；仅其 `allowed_tools` 中
     且在动态工具池内的工具会被注入；启用运行时校验时，注入工具
     在每次调用时还会向授权网关复核（授权撤销/过期后调用即被拦截
     并返回错误 JSON）；
   - 返回 AgentRecord 的 JSON（附注入/被忽略的工具清单）。

2. 随后对每个已创建 Agent 调用 `execute_subagent(agent_id, input_data)`：

   - `input_data` 为下发给该子 Agent 的 JSON 文本（任务上下文与执行指令）；
   - 返回 ExecutionResult 的 JSON（附 attempt_count）。

3. 不要执行未由本次 create_subagent 成功返回的 agent_id。

## 最终输出（结构化强约束，覆盖上文输出格式段落）

最终回复必须是 SubagentCreatorOutput 结构的纯 JSON 文本——不包裹
Markdown 代码围栏，不附加任何解释性文字：

- `created_agents`：成功创建的 AgentRecord 列表（原样 JSON 对象数组）；
- `results`：全部 execute_subagent 的 ExecutionResult 列表；
- `failed_subtasks`：创建或执行失败的 subtask_id 列表；
- `summary`：汇总说明（创建数量、注入工具、失败原因）。

硬性纪律：

1. 严格“先创建后执行”；一次缺口只创建一个精准匹配的子 Agent，禁止批量碰运气；
2. 不得为子 Agent 注入授权清单之外的工具；工具池外的请求须在 summary 中说明；
3. 不得向子 Agent 提供其他 Agent 的引用、通信方式或 handoff；
4. 替代 Agent 不具备替换申请权；其失败只能如实上报，由主 Agent 决策；
5. 所有证据（工具调用、返回结构）必须真实，禁止编造。
"""


class SubagentCreatorInput(BaseModel):
    """子 Agent 创建器的输入契约（主 Agent 下发）。"""

    subtasks: list[SubtaskSpec] = Field(
        default_factory=list, description="需要动态创建执行者的子任务规格列表"
    )
    tool_grants: list[ToolGrant] = Field(
        default_factory=list, description="主 Agent 为这些子任务签发的工具授权列表"
    )


class SubagentCreatorOutput(BaseModel):
    """子 Agent 创建器的输出契约（结构化 JSON，禁止自由文本）。"""

    created_agents: list[dict[str, Any]] = Field(
        default_factory=list, description="成功创建的 AgentRecord 列表"
    )
    results: list[ExecutionResult] = Field(
        default_factory=list, description="全部子 Agent 的执行结果列表"
    )
    failed_subtasks: list[str] = Field(
        default_factory=list, description="创建或执行失败的 subtask_id 列表"
    )
    summary: str = Field(default="", description="汇总说明")


@dataclass
class _CreatorState:
    """创建器运行期内部状态：agent_id 到 SDK Agent 实例的映射。

    仅存活于同一次“创建/执行”配对调用内；注册表中的 AgentRecord 才是
    跨组件传播的权威登记信息。
    """

    instances: dict[str, Agent[Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 通用小工具（JSON 序列化 / 上下文解析 / 输入解析）
# ---------------------------------------------------------------------------
def _json(payload: Any) -> str:
    """将载荷序列化为 JSON 字符串（工具返回值的统一格式）。"""
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(message: str, **fields: Any) -> str:
    """构造失败形态的 JSON 工具返回值。"""
    return _json({"status": "error", "error": message, **fields})


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


def _load_prompt() -> str:
    """读取本目录 prompt.md 作为系统提示词（含输入输出契约补充说明）。"""
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return f"{text}\n\n{_EXTRA_INSTRUCTIONS}"


def _build_chat_model(config: AgentAPIConfig) -> Model:
    """根据 api_config.yaml 构造 OpenAI 兼容聊天模型客户端。

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
        timeout=resolve_model_http_timeout(config),
        max_retries=config.retry,
    )
    return OpenAIChatCompletionsModel(model=config.model, openai_client=client)


async def _extract_output_json(result: RunResult | RunResultStreaming) -> str:
    """``as_tool`` 输出提取器：保证工具返回值始终为 JSON 字符串。

    Agent 以纯文本返回（``output_type=None``），字符串输出先剥离
    Markdown 代码围栏提取 JSON 对象；无法结构化时回退为
    ``{"output": ...}`` 文本包装。
    """
    output = result.final_output
    if isinstance(output, BaseModel):
        return output.model_dump_json()
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False, default=str)
    if isinstance(output, str):
        obj = extract_json_object(output)
        if obj is not None:
            return json.dumps(obj, ensure_ascii=False)
    return json.dumps({"output": "" if output is None else str(output)}, ensure_ascii=False)


def _parse_json_object(value: Any, *, label: str) -> dict[str, Any]:
    """解析 JSON 对象参数（接受 JSON 文本或已解析对象）。

    Raises:
        ValueError: 参数为空、非法 JSON 或不是对象时。
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"{label} 不能为空（应为 JSON 文本）")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} 不是合法 JSON：{exc}") from exc
    else:
        data = value
    if not isinstance(data, dict):
        raise ValueError(f"{label} 必须是 JSON 对象")
    return data


def _normalize_subagent_input(input_data: Any) -> str:
    """将下发给子 Agent 的输入规范化为 JSON 文本。"""
    if isinstance(input_data, str):
        text = input_data.strip()
        if text.startswith(("{", "[")):
            try:
                return json.dumps(json.loads(text), ensure_ascii=False)
            except json.JSONDecodeError:
                return input_data
        return input_data
    return json.dumps(input_data, ensure_ascii=False, default=str)


def _subtask_id_from_agent_id(agent_id: str) -> str:
    """从 agent_id（``{task_id}:{subtask_id}:{kind}-{hex}``）中解析 subtask_id。"""
    parts = agent_id.split(":")
    return parts[1] if len(parts) >= 2 and parts[1] else agent_id


def _detect_replacement_of(
    spec_payload: dict[str, Any],
    registry: AgentRegistry,
    task_id: str,
    subtask_id: str,
) -> str | None:
    """判定本次创建是否属于“替代 Agent”，并返回被替代的 agent_id。

    优先级：
    1. 显式标记：SubtaskSpec 的 JSON 载荷中带 ``replacement_of`` 字段；
    2. 注册表推断：同一任务下存在同 subtask_id 且状态为 ``failed`` 的历史记录，
       取最新一条视为被替代者。

    Returns:
        被替代的 agent_id；非替代场景返回 None。
    """
    explicit = spec_payload.get("replacement_of")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    marker = f":{subtask_id}:"
    for record in reversed(registry.list_by_task(task_id)):
        if record.status == "failed" and marker in record.agent_id:
            return record.agent_id
    return None


def _compose_dynamic_instructions(spec: SubtaskSpec, injected_tools: list[str]) -> str:
    """为动态子 Agent 组装专用系统提示词（含权限与输出纪律）。

    输出契约与 ToolResponse 统一信封对齐：success / output / error /
    artifacts / metadata 语义不变，另加 ExecutionResult 扩展字段
    subtask_id / evidence（duration 可选——由调用方计时时无需模型填写）。
    """
    tool_text = "、".join(injected_tools) if injected_tools else "（无工具授权）"
    return (
        "你是由子 Agent 创建器按需动态构造的专用子 Agent，只对下述子任务负责。\n\n"
        f"【子任务 ID】{spec.subtask_id}\n"
        f"【子任务标题】{spec.title}\n"
        f"【执行指令】{spec.instructions}\n"
        f"【期望产出】{spec.expected_output or '（未指定）'}\n"
        f"【授权工具】{tool_text}\n\n"
        "硬性约束：\n"
        "1. 只能使用上述已注入的工具；需要额外工具时必须停止执行并说明诉求，"
        "不得自行获取、发现或伪造工具；\n"
        "2. 不得创建新的子 Agent，不得与其他子 Agent 通信或共享会话；\n"
        "3. 不得访问项目沙箱目录之外的路径；所有证据必须真实可复核；\n"
        f"4. 最终输出必须是 ExecutionResult 结构的纯 JSON 文本（subtask_id 固定填 "
        f"\"{spec.subtask_id}\"）——不包裹 Markdown 代码围栏，不附加解释性文字，"
        "禁止自由文本。\n"
        "\n"
        "你的最终输出必须严格为以下 JSON 格式（统一 ToolResponse 信封字段，"
        "不要包裹 Markdown 代码围栏）：\n"
        "{\n"
        "  \"subtask_id\": \"<当前子任务ID，即上面指定的 subtask_id>\",\n"
        "  \"success\": true,\n"
        "  \"output\": \"<你的完整分析/执行结果，作为字符串>\",\n"
        "  \"evidence\": [\"相关证据（可选）\"],\n"
        "  \"artifacts\": [\"产出文件路径（可选）\"],\n"
        "  \"error\": null,\n"
        "  \"metadata\": {}\n"
        "}\n"
        "\n"
        "字段语义（与 ToolResponse 信封完全一致）：\n"
        "- success：执行是否成功；失败时将 success 设为 false 并在 error 字段说明原因；\n"
        "- output：主要产出，必须是字符串类型；如果你的结果是 JSON 对象，"
        "请将其序列化为字符串放入 output；\n"
        "- error：失败时的错误描述，成功时为 null；\n"
        "- artifacts：产出的文件路径列表（通常为沙箱内的产物）；\n"
        "- evidence：支撑结论的证据列表（命令输出、日志片段等）；\n"
        "- metadata：扩展元数据对象（如耗时、统计信息等），无内容时填空对象 {}。"
    )


def _coerce_execution_result(output: Any, fallback_subtask_id: str) -> ExecutionResult:
    """将子 Agent 的最终输出强制转换为 ExecutionResult（防御性兜底）。

    子 Agent 以纯文本返回（``output_type=None``），字符串输出先剥离
    Markdown 代码围栏提取 JSON 对象再校验；无法结构化时按失败结果
    兜底（保留原始文本便于诊断）。
    """
    if isinstance(output, ExecutionResult):
        result = output
    elif isinstance(output, dict):
        try:
            result = ExecutionResult.model_validate(output)
        except ValidationError:
            # 降级：子 Agent 产出了有效 JSON 但不是 ExecutionResult 格式，
            # 将其整体作为 output 字段包裹进 ExecutionResult（避免无效重试）
            result = ExecutionResult(
                subtask_id=fallback_subtask_id,
                success=True,
                output=json.dumps(output, ensure_ascii=False, default=str),
                evidence=[],
                artifacts=[],
            )
    elif isinstance(output, str):
        # 纯文本返回：剥离 ```json 围栏后提取 JSON 对象再校验
        obj = extract_json_object(output)
        try:
            result = ExecutionResult.model_validate(obj if obj is not None else output)
        except ValidationError:
            if obj is not None:
                # 降级：剥离围栏后得到了有效 JSON 但不是 ExecutionResult 格式，
                # 将其整体作为 output 字段包裹进 ExecutionResult（避免无效重试）
                result = ExecutionResult(
                    subtask_id=fallback_subtask_id,
                    success=True,
                    output=json.dumps(obj, ensure_ascii=False, default=str),
                    evidence=[],
                    artifacts=[],
                )
            else:
                # 纯自由文本（无可提取 JSON）：保留原文并判失败
                result = ExecutionResult(
                    subtask_id=fallback_subtask_id,
                    success=False,
                    output=output,
                    error="子 Agent 未返回 ExecutionResult 结构化输出",
                )
    else:
        result = ExecutionResult(
            subtask_id=fallback_subtask_id,
            success=False,
            output="" if output is None else str(output),
            error="子 Agent 未返回 ExecutionResult 结构化输出",
        )
    if not result.subtask_id:
        result.subtask_id = fallback_subtask_id
    return result


def _wrap_tool_with_permission_check(
    tool: FunctionTool,
    gateway: PermissionGateway,
    subtask_id: str,
    *,
    agent_id: str,
    task_id: str,
    grant_id: str | None = None,
) -> FunctionTool:
    """包裹 FunctionTool，在每次调用前执行运行时权限校验。

    双重防护的第二层：构造时（create_subagent）已按 allowed_tools 过滤注入；
    本包裹在工具真正被调用的时刻再向 PermissionGateway 校验一次，授权
    被撤销或过期后立即失效（不依赖重新构造 Agent）。

    拦截机制：OpenAI Agents SDK 的 ``FunctionTool`` 为 dataclass，
    ``on_invoke_tool`` 是每次工具调用的唯一入口回调；因此用
    ``dataclasses.replace`` 生成保留全部元数据（name / description /
    params_json_schema / strict_json_schema 等）的副本，仅替换
    ``on_invoke_tool`` 为校验前置的守卫回调。

    校验身份说明：``verify_grant(agent_id, tool_name, task_id)`` 以授权
    凭证（ToolGrant）登记的身份为准——即规划阶段按子任务槽位签发的
    ``grant.agent_id``（形如 ``{task_id}:{subtask_id}:grant-slot``）与
    ``grant.task_id``；动态子 Agent 自身的 agent_id
    （``...dynamic-{hex}``）不是网关登记的授权主体，不参与校验。

    grant_id 绑定校验（防伪造第三道）：提供 ``grant_id`` 时，在三元组
    校验通过后额外把网关登记记录（``get_grant(grant_id)``）与本次调用
    逐一比对——grant 存在、未撤销、agent_id/task_id 与凭证一致、
    tool_name 在其白名单内。传入的 grant JSON 在到达本工具前可能被
    主管模型转写（字段被篓改/拼接他单授权），绑定校验确保“凭证内容”
    与“网关登记事实”一致，凭证字段被篓改后即使三元组侥幸匹配也会被
    拦截；``grant_id`` 为 None 时保持旧行为（仅三元组校验，向后兼容）。

    Args:
        tool: 待包裹的原始 FunctionTool（不会被修改，返回新副本）。
        gateway: 工具授权网关（运行时校验的权威判定者）。
        subtask_id: 所属子任务编号（用于错误消息诊断）。
        agent_id: 授权凭证登记的调用者身份（ToolGrant.agent_id）。
        task_id: 授权凭证绑定的任务编号（ToolGrant.task_id）。
        grant_id: 授权凭证编号；提供时追加网关登记记录绑定比对，None 时
            跳过绑定校验（向后兼容）。

    Returns:
        新的 FunctionTool：校验通过则透传原始工具结果；校验失败返回
        明确的错误 JSON（不执行原始工具），由子 Agent 模型可见并停止重试。
    """
    original_invoke = tool.on_invoke_tool
    tool_name = tool.name

    async def _guarded_invoke(
        ctx: RunContextWrapper[Any], args_json: str
    ) -> Any:
        if not gateway.verify_grant(agent_id, tool_name, task_id):
            return _error(
                "工具调用被运行时权限校验拦截："
                f"tool={tool_name}, subtask_id={subtask_id}, "
                f"agent_id={agent_id}, task_id={task_id} —— 该授权不存在、"
                "已撤销或已过期，本次调用未执行。请勿重复尝试该工具；"
                "如确有必要请在最终输出中说明权限缺口。"
            )
        if grant_id is not None:
            grant = gateway.get_grant(grant_id)
            if (
                grant is None
                or grant.revoked
                or grant.agent_id != agent_id
                or grant.task_id != task_id
                or tool_name not in grant.allowed_tools
            ):
                return _error(
                    "工具调用被 grant_id 绑定校验拦截："
                    f"grant_id={grant_id}, tool={tool_name}, "
                    f"subtask_id={subtask_id} —— 授权凭证与网关登记记录"
                    "不一致（凭证不存在、已撤销、主体不符或工具越权），"
                    "本次调用未执行。请勿重复尝试该工具。"
                )
        return await original_invoke(ctx, args_json)

    return replace(tool, on_invoke_tool=_guarded_invoke)


# ---------------------------------------------------------------------------
# 内部工具构建
# ---------------------------------------------------------------------------
def build_creator_tools(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    registry: AgentRegistry | None = None,
    sandbox_runner: SandboxRunner | None = None,
    subagent_model: str | Model | None = None,
    tool_registry: ToolRegistry | None = None,
    gateway: PermissionGateway | None = None,
) -> dict[str, FunctionTool]:
    """构造子 Agent 创建器的全部内部工具（以工具名索引的字典）。

    注意：``create_subagent`` 与 ``execute_subagent`` 通过同一个内部状态
    （agent_id → Agent 实例）配对，因此**必须使用同一次调用返回的工具集合**。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        registry: Agent 注册表；未提供时按 context/settings 新建。
        sandbox_runner: 沙箱执行器（供动态工具池使用）；未提供时按
            context/settings 新建。
        subagent_model: 动态子 Agent 使用的模型；未提供时按本包
            api_config.yaml 构造（密钥优先取 api_key 直接值，回退
            api_key_env 环境变量，均缺失时抛 ConfigLoadError）。
        tool_registry: 工具注册表；提供时把其中已审批的动态生成工具
            （经 MCP Server 反向包装为 FunctionTool）并入动态工具池，
            使子 Agent 可按授权使用新生成的专用工具。
        gateway: 工具授权网关；提供时对注入子 Agent 的每个工具追加
            运行时校验包裹（verify_grant），形成“构造时注入 + 运行时
            校验”双重防护；未提供时仅保留构造时过滤（向后兼容）。

    Returns:
        ``{工具名: FunctionTool}``，包含 create_subagent / execute_subagent。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    resolved_registry = registry or AgentRegistry(
        context=resolved_context, settings=resolved_settings
    )
    resolved_model = subagent_model or _build_chat_model(load_agent_api_config(API_CONFIG_PATH))

    # 动态工具池：复用 code_agent 的基础工具（不含任何 Agent 引用）
    all_tools = build_code_tools(
        resolved_context, sandbox_runner=sandbox_runner, settings=resolved_settings
    )
    tool_pool: dict[str, FunctionTool] = {
        name: all_tools[name] for name in _DYNAMIC_TOOL_POOL_NAMES if name in all_tools
    }

    # 动态工具池扩充：已审批的生成工具（经 MCP Server 反向包装为 FunctionTool）
    if tool_registry is not None:
        for item in tool_registry.list_assignable_tools(approved_only=True):
            name = str(item.get("name") or "")
            if not name or name in tool_pool or item.get("source") != "generated":
                continue
            server = tool_registry.get_mcp_server(name)
            if server is None:
                continue
            tool_pool[name] = mcp_server_to_agent_tool(server, name)

    state = _CreatorState()
    # 超时审计事件通道（任务 33）：流式超时发生时写入 timeout_audit，
    # 事件同步镜像到共享上下文的 event_log
    audit_events = EventLogger(context=resolved_context)
    # 运行时权限校验网关（None 表示关闭第二层防护，仅保留构造时过滤）
    resolved_gateway = gateway

    @function_tool
    def create_subagent(
        ctx: RunContextWrapper[Any],
        subtask_spec_json: str,
        tool_grant_json: str,
    ) -> str:
        """按子任务规格与工具授权，动态创建一个专用子 Agent 并登记到运行时注册表。

        动态子 Agent 的工具集合默认空，仅注入 ToolGrant.allowed_tools 中且在
        工具池内的工具；不注入任何其他 Agent 的引用或通信能力。若该子任务在
        注册表中已有失败记录（或 SubtaskSpec 载荷显式带 ``replacement_of``），
        则本次创建视为替代 Agent：can_request_replacement 固定为 false。

        启用运行时校验（装配时提供了 PermissionGateway）时，注入的工具会在
        每次被调用的时刻向授权网关复核：授权被撤销或过期后调用立即被拦截
        并返回错误 JSON（不执行原工具），与构造时过滤形成双重防护。

        Args:
            subtask_spec_json: SubtaskSpec 的 JSON 文本（可附带 replacement_of 字段）。
            tool_grant_json: ToolGrant 的 JSON 文本（本子任务下发的授权凭证）。

        Returns:
            JSON 字符串：成功为 AgentRecord 的完整序列化（附 injected_tools /
            ignored_tools）；失败为 ``{"status": "error", "error"}``。
        """
        # 1) 解析规格与授权
        try:
            spec_payload = _parse_json_object(subtask_spec_json, label="subtask_spec_json")
            spec = SubtaskSpec.model_validate(spec_payload)
        except (ValueError, ValidationError) as exc:
            return _error(f"subtask_spec_json 非法：{exc}")
        try:
            grant_payload = _parse_json_object(tool_grant_json, label="tool_grant_json")
            grant = ToolGrant.model_validate(grant_payload)
        except (ValueError, ValidationError) as exc:
            return _error(f"tool_grant_json 非法：{exc}")

        # 2) 工具注入：默认空，仅注入授权清单内且在工具池中的工具
        injected_names = [name for name in grant.allowed_tools if name in tool_pool]
        ignored_names = [name for name in grant.allowed_tools if name not in tool_pool]
        injected_tools = [tool_pool[name] for name in injected_names]

        # 2.5) 运行时权限包裹（双重防护第二层 + grant_id 绑定校验）：注入的
        #      每个工具都经 PermissionGateway.verify_grant 前置校验，授权
        #      撤销/过期后立即生效；同时把本凭证的 grant_id 传入包裹，
        #      运行时比对网关登记记录，防止凭证字段在传递链路中被篓改。
        #      以授权凭证登记的身份（grant.agent_id / grant.task_id）
        #      为校验主体，与规划阶段槽位授权保持同一登记链。
        if resolved_gateway is not None and injected_tools:
            injected_tools = [
                _wrap_tool_with_permission_check(
                    tool,
                    resolved_gateway,
                    spec.subtask_id,
                    agent_id=grant.agent_id,
                    task_id=grant.task_id,
                    grant_id=grant.grant_id,
                )
                for tool in injected_tools
            ]

        # 3) 替代判定与替换权限
        replacement_of = _detect_replacement_of(
            spec_payload, resolved_registry, grant.task_id, spec.subtask_id
        )
        record = AgentRecord(
            agent_id=f"{grant.task_id}:{spec.subtask_id}:dynamic-{uuid.uuid4().hex[:8]}",
            agent_type="dynamic",
            parent_task_id=grant.task_id,
            plan_version=spec.plan_version,
            allowed_tools=injected_names,
            attempt_count=0,
            can_request_replacement=replacement_of is None,
            replacement_of=replacement_of,
            status="created",
        )

        # 4) 构造动态子 Agent（纯文本返回，输出由 execute_subagent
        #    剥离围栏后按 ExecutionResult 手动校验）
        sub_agent: Agent[Any] = Agent(
            name=f"dynamic-{spec.subtask_id}",
            instructions=_compose_dynamic_instructions(spec, injected_names),
            tools=injected_tools,
            model=resolved_model,
            # output_type 保持 None（纯文本返回）：兼容千问等忽略
            # response_format=json_schema 的接口，避免 SDK 严格校验报错。
        )

        # 5) 登记注册表（受 max_dynamic_agents 上限治理）
        try:
            resolved_registry.register(record)
        except AgentRegistryError as exc:
            return _error(f"子 Agent 注册失败：{exc}", agent_id=record.agent_id)
        state.instances[record.agent_id] = sub_agent

        payload = record.model_dump(mode="json")
        payload["injected_tools"] = injected_names
        payload["ignored_tools"] = ignored_names
        return _json(payload)

    @function_tool
    async def execute_subagent(
        ctx: RunContextWrapper[Any],
        agent_id: str,
        input_data: str,
    ) -> str:
        """执行一个已登记的动态子 Agent，返回其 ExecutionResult 结果。

        执行前从注册表校验 agent_id 存在；执行后跟踪 attempt_count：
        成功置为 completed；失败且达到 max_agent_attempts 置为 failed，
        否则置为 waiting（等待上层决定是否重试）。

        Args:
            agent_id: create_subagent 返回的 AgentRecord.agent_id。
            input_data: 下发给子 Agent 的 JSON 文本（任务上下文与执行指令）。

        Returns:
            JSON 字符串：ExecutionResult 的完整序列化（附 agent_id 与
            attempt_count）；失败为 ``{"status": "error", "error"}``。
        """
        record = resolved_registry.get(agent_id)
        if record is None:
            return _error(f"Agent 不存在或未登记：{agent_id}")
        agent_instance = state.instances.get(agent_id)
        if agent_instance is None:
            return _error(
                "该 Agent 没有可执行的运行时实例（必须由本次会话的 create_subagent "
                "创建后立即执行）",
                agent_id=agent_id,
            )

        record.attempt_count += 1
        try:
            resolved_registry.update_status(agent_id, "running")
        except AgentRegistryError:
            pass  # 状态更新失败不影响执行本身

        fallback_subtask_id = _subtask_id_from_agent_id(agent_id)
        prompt = _normalize_subagent_input(input_data)
        # 分阶段流式超时（任务 33）：首事件 / 模型空闲 / 工具执行三类
        # 阶段窗口 + agent_total_timeout 绝对总预算；流式执行下只要
        # SDK 原生事件持续产出就不会被空闲超时误杀，阶段窗口内无事件
        # 或总预算耗尽才终止
        timeout = float(resolved_settings.agent_total_timeout)

        try:
            run_result = await run_agent_streamed(
                agent_instance,
                input=prompt,
                max_turns=int(resolved_settings.max_model_turns),
                first_event_timeout=float(
                    resolved_settings.first_event_timeout
                ),
                model_idle_timeout=float(
                    resolved_settings.model_idle_timeout
                ),
                tool_execution_timeout=float(
                    resolved_settings.tool_execution_timeout
                ),
                agent_total_timeout=timeout,
                # 工具窗口按 profile 解析（任务 45）：code_agent / 嵌套
                # 子 Agent / sandbox 使用长窗口（不受普通 180s 限制），
                # 仍受上方 agent_total_timeout 绝对总预算封顶
                tool_timeout_resolver=make_tool_timeout_resolver(
                    resolved_settings, tool_registry
                ),
                runner=Runner,
                events=audit_events,
                audit_context={
                    "agent": "subagent",
                    "task_id": record.parent_task_id,
                    "agent_id": agent_id,
                    "subtask_id": fallback_subtask_id,
                },
            )
        except asyncio.TimeoutError:
            execution = ExecutionResult(
                subtask_id=fallback_subtask_id,
                success=False,
                error=f"执行超时（>{timeout:g}s），已终止",
                duration=timeout,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 模型/网络异常按失败处理，不中断上层
            execution = ExecutionResult(
                subtask_id=fallback_subtask_id,
                success=False,
                error=f"执行异常：{type(exc).__name__}: {exc}",
            )
        else:
            execution = _coerce_execution_result(run_result.final_output, fallback_subtask_id)

        final_status = (
            "completed"
            if execution.success
            else (
                "failed"
                if record.attempt_count >= int(resolved_settings.max_agent_attempts)
                else "waiting"
            )
        )
        try:
            resolved_registry.update_status(agent_id, final_status)  # type: ignore[arg-type]
        except AgentRegistryError:
            pass

        payload = execution.model_dump(mode="json")
        payload["agent_id"] = agent_id
        payload["attempt_count"] = record.attempt_count
        return _json(payload)

    return {
        "create_subagent": create_subagent,
        "execute_subagent": execute_subagent,
    }


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def create_subagent_creator(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
    registry: AgentRegistry | None = None,
    sandbox_runner: SandboxRunner | None = None,
    tool_registry: ToolRegistry | None = None,
    gateway: PermissionGateway | None = None,
) -> Agent[Any]:
    """构造配置完毕的子 Agent 创建器（SDK Agent 实例）。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 模型实例或名称覆盖；未提供时按本包 api_config.yaml 构造
            （密钥优先取 api_key 直接值，回退 api_key_env 环境变量，
            均缺失时抛 ConfigLoadError）。动态子 Agent
            与创建器共享同一模型配置。
        registry: Agent 注册表；未提供时按 context/settings 新建。
        sandbox_runner: 沙箱执行器（供动态工具池使用）；未提供时按
            context/settings 新建。
        tool_registry: 工具注册表；提供时把已审批的生成工具并入
            动态子 Agent 可注入工具池。
        gateway: 工具授权网关；提供时对注入子 Agent 的工具追加运行时
            校验（verify_grant），与工作流共享同一网关实例以复用
            规划阶段签发的槽位授权。

    Returns:
        ``Agent`` 实例：name="subagent_creator"，instructions 取自 prompt.md，
        不设置 output_type（None，纯文本返回）：输出由调用方剥离
        Markdown 围栏后按 ``SubagentCreatorOutput`` 手动校验（MCP
        适配层内置该容错路径）。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    resolved_model = model or _build_chat_model(load_agent_api_config(API_CONFIG_PATH))
    tools = build_creator_tools(
        resolved_context,
        settings=resolved_settings,
        registry=registry,
        sandbox_runner=sandbox_runner,
        subagent_model=resolved_model,
        tool_registry=tool_registry,
        gateway=gateway,
    )
    return Agent(
        name=AGENT_NAME,
        instructions=_load_prompt(),
        tools=list(tools.values()),
        model=resolved_model,
        # output_type 保持 None（纯文本返回）：千问等兼容接口忽略
        # response_format=json_schema 时 SDK 严格校验会抛 ModelBehaviorError，
        # 改由调用方手动剥离围栏 + SubagentCreatorOutput 校验。
    )


def get_subagent_creator_as_tool(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
    registry: AgentRegistry | None = None,
    sandbox_runner: SandboxRunner | None = None,
    tool_registry: ToolRegistry | None = None,
    gateway: PermissionGateway | None = None,
) -> FunctionTool:
    """将子 Agent 创建器转为主 Agent 可直接调用的工具入口（Agent.as_tool）。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 模型实例或名称覆盖；未提供时按本包 api_config.yaml 构造。
        registry: Agent 注册表；未提供时按 context/settings 新建。
        sandbox_runner: 沙箱执行器（供动态工具池使用）。
        tool_registry: 工具注册表；提供时把已审批的生成工具并入
            动态子 Agent 可注入工具池。
        gateway: 工具授权网关；透传给 build_creator_tools 启用运行时校验。

    Returns:
        ``FunctionTool``：工具名 "subagent_creator"，返回值经输出提取器
        规范化为结构化 JSON 字符串。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    agent = create_subagent_creator(
        resolved_context,
        settings=resolved_settings,
        model=model,
        registry=registry,
        sandbox_runner=sandbox_runner,
        tool_registry=tool_registry,
        gateway=gateway,
    )
    return agent.as_tool(
        tool_name=AGENT_NAME,
        tool_description=_TOOL_DESCRIPTION,
        custom_output_extractor=_extract_output_json,
        max_turns=resolved_settings.max_model_turns,
    )


def get_mcp_schema() -> MCPToolSchema:
    """返回子 Agent 创建器的 MCP 暴露契约（MCPToolSchema）。

    inputSchema 取自 ``SubagentCreatorInput.model_json_schema()``。
    """
    return MCPToolSchema(
        name=AGENT_NAME,
        description=_TOOL_DESCRIPTION,
        inputSchema=SubagentCreatorInput.model_json_schema(),
    )


def get_mcp_server(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
    registry: AgentRegistry | None = None,
    sandbox_runner: SandboxRunner | None = None,
    tool_registry: ToolRegistry | None = None,
    gateway: PermissionGateway | None = None,
) -> MCPToolServer:
    """将子 Agent 创建器包装为 MCP Server（供主管与 MCPClient 统一调用）。

    与 ``get_subagent_creator_as_tool`` 等价的 MCP 暴露形态：调用时按
    ``SubagentCreatorInput`` 校验参数、运行 Agent，并按
    ``SubagentCreatorOutput`` 提取结构化 JSON 文本。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 模型实例或名称覆盖；未提供时按本包 api_config.yaml 构造。
        registry: Agent 注册表；未提供时按 context/settings 新建。
        sandbox_runner: 沙箱执行器（供动态工具池使用）。
        tool_registry: 工具注册表；提供时把已审批的生成工具并入
            动态子 Agent 可注入工具池。
        gateway: 工具授权网关；透传给 build_creator_tools 启用运行时校验
            （主管编排器应传入与规划阶段共享的同一网关实例）。

    Returns:
        ``MCPToolServer``：name="subagent_creator"，含单个工具的 MCP Server。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    agent = create_subagent_creator(
        resolved_context,
        settings=resolved_settings,
        model=model,
        registry=registry,
        sandbox_runner=sandbox_runner,
        tool_registry=tool_registry,
        gateway=gateway,
    )
    return agent_to_mcp_server(
        agent,
        AGENT_NAME,
        _TOOL_DESCRIPTION,
        SubagentCreatorInput,
        SubagentCreatorOutput,
        max_turns=resolved_settings.max_model_turns,
    )

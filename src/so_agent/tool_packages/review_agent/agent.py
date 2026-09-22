"""评审 Agent 构建模块（review_agent 工具包）。

职责：作为主 Agent 的固定工具入口之一，构造“评审 Agent（Review Agent）”：

- 读取本目录 ``prompt.md`` 作为系统提示词（instructions）；
- 通过 ``runtime.config_loader`` 加载本目录 ``api_config.yaml``，按其中
  ``api_key`` 直接值（或旧 ``api_key_env`` 环境变量回退）构造 OpenAI 兼容
  聊天模型（密钥值绝不写日志）；
- 绑定三个自动校验工具：``check_plan_completeness`` /
  ``check_tool_necessity`` / ``check_result_correctness``，所有工具返回值
  统一为 JSON 字符串；
- Agent 输出为纯文本 JSON（``output_type=None``，不依赖 SDK 结构化输出：
  千问等 OpenAI 兼容接口会忽略 ``response_format=json_schema`` 并把
  JSON 包裹在 Markdown 代码围栏中返回，故由调用方剥离围栏后按
  ``models.ReviewDecision`` 手动校验），覆盖计划评审（含授权最小化评审）、
  生成工具评审与结果评审各阶段的全部结论；
- 对外同时提供 MCP 暴露契约与 MCP Server（``get_mcp_schema`` /
  ``get_mcp_server``），使本工具可按 MCP 规范被主管 Agent 与 MCPClient
  统一发现与调用。

对外接口::

    agent = create_review_agent(context)         # 构造 SDK Agent 实例
    tool = get_review_agent_as_tool(context)     # 转为主 Agent 可用的工具入口
    schema = get_mcp_schema()                    # MCP 暴露契约（MCPToolSchema）
    server = get_mcp_server(context)             # 包装为 MCP Server（统一调用）

全局约束（最高优先级）：
- 所有 Agent 之间的通信必须严格使用 JSON：工具返回值与 Agent 最终输出
  一律为可解析的 JSON（Pydantic 序列化），不允许自由文本协议；
- 评审 Agent 只读核验，不修改任何产物、不代替执行 Agent 完成任务；
- 一切评审结论必须有证据支撑，无法核验的断言按未通过处理。
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from agents import (
    Agent,
    FunctionTool,
    Model,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    RunResult,
    RunResultStreaming,
    function_tool,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from so_agent.config import Settings, get_settings
from so_agent.context import ProjectContext
from so_agent.json_utils import extract_json_object
from so_agent.mcp.adapter import agent_to_mcp_server
from so_agent.mcp.protocol import MCPToolSchema
from so_agent.mcp.server import MCPToolServer
from so_agent.models import (
    ExecutionResult,
    GeneratedToolManifest,
    ReviewDecision,
    SubtaskSpec,
    TaskRequest,
    ToolGrant,
)
from so_agent.runtime.config_loader import AgentAPIConfig, load_agent_api_config

# ---------------------------------------------------------------------------
# 包内路径（用 pathlib 定位同目录下的提示词与 API 配置）
# ---------------------------------------------------------------------------
PACKAGE_DIR: Path = Path(__file__).resolve().parent
PROMPT_PATH: Path = PACKAGE_DIR / "prompt.md"
API_CONFIG_PATH: Path = PACKAGE_DIR / "api_config.yaml"

# Agent 名称（与工具注册表 / 路由约定保持一致）
AGENT_NAME = "review_agent"

# 主管侧调用说明（同时用于 as_tool 入口与 MCP 暴露契约）
_TOOL_DESCRIPTION = (
    "评审 Agent：对任务拆解计划、工具授权必要性、生成工具与执行结果做独立质检，"
    "返回 ReviewDecision 结构的 JSON 结论（passed/issues/required_fixes/retry_target）。"
)

# 授权评审中视为“过于宽泛/高风险”的工具名（提示模型重点核验）
_SUSPICIOUS_TOOLS: frozenset[str] = frozenset(
    {"*", "all", "any", "any_tool", "shell", "exec", "sudo"}
)

# 追加在 prompt.md 之后的输入输出契约说明（不改动原始提示词文件）
_EXTRA_INSTRUCTIONS = """
## 结构化输入（ReviewInput）

主 Agent 会以如下字段（ReviewInput）下发评审材料：

- `stage`：评审阶段，取值 `plan_review` / `authorization_review` / `tool_review` / `result_review`；
- `task_request`：原始任务请求（TaskRequest，含 objective / constraints / acceptance_criteria）；
- `subtasks`：待评审的子任务规格列表（计划评审、授权评审时提供，可为 null）；
- `tool_grants`：本次计划的工具授权列表（授权评审时提供，可为 null）；
- `generated_tool`：待评审的生成工具清单（GeneratedToolManifest；工具评审时提供，可为 null）；
- `execution_results`：全部子任务的执行结果列表（结果评审时提供，可为 null）；
- `plan_version`：当前计划版本号。

## 内部校验工具（自动检查，供结论参考）

1. 计划评审：调用 `check_plan_completeness` 检查子任务完整性
   （重复 ID、空字段、依赖缺失、自依赖、循环依赖）；
2. 授权评审：调用 `check_tool_necessity` 检查授权必要性与最小化
   （超额授权、未授权缺口、空授权、可疑宽泛工具）；
3. 结果评审：调用 `check_result_correctness` 检查执行结果的失败项、
   证据完整性、产物路径是否越界；
4. 生成工具评审（`tool_review`）：核验 `generated_tool` 候选的用途必要性、
   名称规范（snake_case、不与既有工具重名）、MCP 契约完整性
   （name/description/inputSchema 与实现入口一致）与安全边界
   （不得引用其他 Agent、不得越出沙箱）；必要时结合 `check_tool_necessity`
   核对声明与授权的对应关系。

自动检查结论必须纳入考量，但不得替代你的独立判断；
自动检查未覆盖的语义问题（如验收标准覆盖度）仍需你逐条核对。

## 输出硬约束（覆盖上文输出格式段落）

最终输出必须是 ReviewDecision 结构的纯 JSON 文本——不包裹 Markdown
代码围栏（不要 ```json 标记），不附加任何解释性文字：

```json
{
  "stage": "plan_review 或 result_review",
  "passed": true,
  "issues": ["发现的问题（无则空数组）"],
  "required_fixes": ["必须修复项（无则空数组）"],
  "retry_target": "需要重试的 subtask_id 或 agent_id；无需重试则为 null",
  "summary": "结论摘要（含逐条验收标准核对结论）"
}
```

- ReviewDecision 的 `stage` 字段只接受 `plan_review` / `result_review` 两个值；
  当本次输入 `stage` 为 `authorization_review` 或 `tool_review` 时，输出 `stage`
  必须填 `plan_review`，并在 `summary` 中注明“本次为授权最小化评审”或
  “本次为生成工具评审”；
- `passed` 为 true 仅当不存在 `required_fixes`；
- 每个 `required_fixes` 条目应指向具体问题并给出可操作建议；
- 检查工具的 JSON 返回值仅作证据，结论必须独立作出且与证据一致。
"""


class ReviewStage(str, Enum):
    """评审 Agent 可处理的阶段类型。"""

    PLAN_REVIEW = "plan_review"                    # 计划评审
    AUTHORIZATION_REVIEW = "authorization_review"  # 授权最小化评审（输出时映射为 plan_review）
    TOOL_REVIEW = "tool_review"                    # 生成工具评审（输出时映射为 plan_review）
    RESULT_REVIEW = "result_review"                # 结果评审


class ReviewInput(BaseModel):
    """评审 Agent 的输入契约（主 Agent 下发）。"""

    stage: ReviewStage = Field(
        description="评审阶段：plan_review/authorization_review/tool_review/result_review"
    )
    task_request: TaskRequest = Field(description="原始任务请求")
    subtasks: list[SubtaskSpec] | None = Field(
        default=None, description="待评审的子任务规格列表（计划/授权评审时提供）"
    )
    tool_grants: list[ToolGrant] | None = Field(
        default=None, description="本次计划的工具授权列表（授权评审时提供）"
    )
    generated_tool: GeneratedToolManifest | None = Field(
        default=None, description="待评审的生成工具清单（工具评审时提供）"
    )
    execution_results: list[ExecutionResult] | None = Field(
        default=None, description="全部子任务的执行结果列表（结果评审时提供）"
    )
    plan_version: int = Field(default=1, description="当前计划版本号")


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


def _sandbox_root(context: ProjectContext) -> Path:
    """返回项目沙箱根目录（绝对路径、已规范化）。"""
    return Path(context.sandbox_dir).expanduser().resolve()


def _is_within(root: Path, target: Path) -> bool:
    """判断 target 是否位于 root 之内（Windows 大小写不敏感比较）。"""
    root_text = os.path.normcase(str(root))
    target_text = os.path.normcase(str(target))
    return target_text == root_text or target_text.startswith(root_text + os.sep)


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
        timeout=config.timeout,
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


# ---------------------------------------------------------------------------
# 输入解析（工具参数一律为 JSON 文本；解析失败返回错误结构）
# ---------------------------------------------------------------------------
def _loads(value: str, *, label: str) -> Any:
    """解析 JSON 文本参数。

    Raises:
        ValueError: 参数为空或不是合法 JSON 时。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} 不能为空（应为 JSON 文本）")
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} 不是合法 JSON：{exc}") from exc


def _parse_subtask_list(value: str) -> list[SubtaskSpec]:
    """解析子任务列表 JSON（接受数组或 ``{"subtasks": [...]}`` 形态）。"""
    data = _loads(value, label="subtasks_json")
    if isinstance(data, dict):
        data = data.get("subtasks", [])
    if not isinstance(data, list):
        raise ValueError("subtasks_json 必须是 JSON 数组或含 subtasks 数组的对象")
    return [SubtaskSpec.model_validate(item) for item in data]


def _parse_grant_list(value: str) -> list[ToolGrant]:
    """解析工具授权列表 JSON（接受数组或 ``{"tool_grants": [...]}`` 形态）。"""
    data = _loads(value, label="grants_json")
    if isinstance(data, dict):
        data = data.get("tool_grants", data.get("grants", []))
    if not isinstance(data, list):
        raise ValueError("grants_json 必须是 JSON 数组或含 tool_grants 数组的对象")
    return [ToolGrant.model_validate(item) for item in data]


def _parse_result_list(value: str) -> list[ExecutionResult]:
    """解析执行结果列表 JSON（接受数组或 ``{"results": [...]}`` 形态）。"""
    data = _loads(value, label="results_json")
    if isinstance(data, dict):
        data = data.get("execution_results", data.get("results", []))
    if not isinstance(data, list):
        raise ValueError("results_json 必须是 JSON 数组或含 results 数组的对象")
    return [ExecutionResult.model_validate(item) for item in data]


def _parse_criteria(value: str) -> list[str]:
    """解析验收标准（接受字符串 / 字符串数组 / 键值对象，统一为列表）。"""
    if not isinstance(value, str) or not value.strip():
        return []
    data = json.loads(value)
    if isinstance(data, str):
        return [data]
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        return [f"{key}: {item}" for key, item in data.items()]
    return [str(data)]


def _detect_cycles(specs: list[SubtaskSpec]) -> list[str]:
    """用 Kahn 算法检测依赖环，返回无法完成拓扑排序的 subtask_id（含环成员）。"""
    ids = [spec.subtask_id for spec in specs]
    id_set = set(ids)
    indegree: dict[str, int] = {sid: 0 for sid in ids}
    dependents: dict[str, list[str]] = {sid: [] for sid in ids}
    for spec in specs:
        for dep in spec.dependencies:
            if dep in id_set and dep != spec.subtask_id:
                indegree[spec.subtask_id] += 1
                dependents[dep].append(spec.subtask_id)
    queue = [sid for sid in ids if indegree[sid] == 0]
    resolved: set[str] = set()
    while queue:
        current = queue.pop(0)
        resolved.add(current)
        for nxt in dependents[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return sorted(sid for sid in ids if sid not in resolved)


def _flat_tools(specs: Iterable[SubtaskSpec]) -> set[str]:
    """汇总全部子任务声明的工具白名单（去重）。"""
    return {tool for spec in specs for tool in spec.allowed_tools if tool.strip()}


# ---------------------------------------------------------------------------
# 内部工具构建
# ---------------------------------------------------------------------------
def build_review_tools(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
) -> dict[str, FunctionTool]:
    """构造评审 Agent 的全部内部校验工具（以工具名索引的字典）。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；校验工具为纯逻辑实现，该参数为接口一致性保留。

    Returns:
        ``{工具名: FunctionTool}``，包含 check_plan_completeness /
        check_tool_necessity / check_result_correctness 三个工具。
    """
    resolved_context = _resolve_context(context)
    sandbox_root = _sandbox_root(resolved_context)

    @function_tool
    def check_plan_completeness(ctx: RunContextWrapper[Any], subtasks_json: str) -> str:
        """检查任务拆解的完整性：重复 ID、空字段、依赖缺失、自依赖与循环依赖。

        Args:
            subtasks_json: 子任务规格列表的 JSON 文本（SubtaskSpec 数组，或
                含 subtasks 数组的对象）。

        Returns:
            JSON 字符串：包含 ``duplicate_ids`` / ``empty_fields`` /
            ``self_dependencies`` / ``missing_dependencies`` / ``cyclic_ids``
            / ``passed``（无任何问题才为 true）与 ``notes``；输入非法时返回
            ``{"status": "error", "error"}``。
        """
        try:
            specs = _parse_subtask_list(subtasks_json)
        except (ValueError, ValidationError) as exc:
            return _error(f"subtasks_json 校验失败：{exc}")

        ids = [spec.subtask_id for spec in specs]
        id_set = set(ids)
        duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
        empty_fields = [
            {
                "subtask_id": spec.subtask_id,
                "missing": [
                    name
                    for name, value in (
                        ("title", spec.title),
                        ("instructions", spec.instructions),
                        ("role", spec.role),
                    )
                    if not str(value).strip()
                ],
            }
            for spec in specs
            if not (spec.title.strip() and spec.instructions.strip() and spec.role.strip())
        ]
        self_dependencies = sorted(
            spec.subtask_id for spec in specs if spec.subtask_id in spec.dependencies
        )
        missing_dependencies = [
            {
                "subtask_id": spec.subtask_id,
                "missing": sorted(dep for dep in spec.dependencies if dep not in id_set),
            }
            for spec in specs
            if any(dep not in id_set for dep in spec.dependencies)
        ]
        cyclic_ids = _detect_cycles(specs)

        notes: list[str] = []
        if not specs:
            notes.append("子任务清单为空：计划评审不可能通过，请要求主 Agent 重新拆解。")
        if missing_dependencies:
            notes.append("存在依赖不存在的子任务：请核对 subtask_id 拼写或补充缺失子任务。")
        if cyclic_ids:
            notes.append("依赖图存在环：必须调整依赖关系后才能执行。")
        passed = not (
            duplicates or empty_fields or self_dependencies or missing_dependencies or cyclic_ids
        )
        if not specs:
            passed = False

        return _json(
            {
                "status": "ok",
                "subtask_count": len(specs),
                "duplicate_ids": duplicates,
                "empty_fields": empty_fields,
                "self_dependencies": self_dependencies,
                "missing_dependencies": missing_dependencies,
                "cyclic_ids": cyclic_ids,
                "passed": passed,
                "notes": notes,
            }
        )

    @function_tool
    def check_tool_necessity(ctx: RunContextWrapper[Any], subtasks_json: str, grants_json: str) -> str:
        """检查工具授权的必要性与最小化：超额授权、空授权、可疑宽泛工具与授权缺口。

        Args:
            subtasks_json: 子任务规格列表的 JSON 文本（可为空数组）。
            grants_json: 工具授权列表的 JSON 文本（ToolGrant 数组）。

        Returns:
            JSON 字符串：包含 ``over_granted``（授权超出子任务声明的工具）、
            ``under_granted``（声明但未授权的工具）、``empty_grants``（空授权
            grant_id）、``suspicious_tools``（宽泛高风险工具名）、``passed``
            与 ``notes``；输入非法时返回 ``{"status": "error", "error"}``。
        """
        try:
            specs = _parse_subtask_list(subtasks_json)
        except (ValueError, ValidationError) as exc:
            return _error(f"subtasks_json 校验失败：{exc}")
        try:
            grants = _parse_grant_list(grants_json)
        except (ValueError, ValidationError) as exc:
            return _error(f"grants_json 校验失败：{exc}")

        declared = _flat_tools(specs)
        granted = {tool for grant in grants for tool in grant.allowed_tools if tool.strip()}
        over_granted = sorted(granted - declared)
        under_granted = sorted(declared - granted)
        empty_grants = [grant.grant_id for grant in grants if not grant.allowed_tools]
        suspicious = sorted(tool for tool in granted if tool.lower() in _SUSPICIOUS_TOOLS)

        notes: list[str] = []
        if over_granted:
            notes.append("存在超出子任务声明的授权工具：须核验是否为必要最小化，否则应收紧授权。")
        if under_granted:
            notes.append("存在声明但未授权的工具：相关子任务可能因缺少工具而失败，请补充授权或调整声明。")
        if empty_grants:
            notes.append("存在空授权条目：空授权不产生任何放行能力，应删除或补全。")
        if suspicious:
            notes.append("存在宽泛/高风险工具名：必须重点核验必要性，必要时要求替换为细粒度工具。")

        passed = not (over_granted or empty_grants or suspicious)
        return _json(
            {
                "status": "ok",
                "declared_tools": sorted(declared),
                "granted_tools": sorted(granted),
                "over_granted": over_granted,
                "under_granted": under_granted,
                "empty_grants": empty_grants,
                "suspicious_tools": suspicious,
                "passed": passed,
                "notes": notes,
            }
        )

    @function_tool
    def check_result_correctness(ctx: RunContextWrapper[Any], results_json: str, criteria_json: str) -> str:
        """检查执行结果的正确性：失败项、证据完整性、产物路径越界与验收标准清单。

        Args:
            results_json: 执行结果列表的 JSON 文本（ExecutionResult 数组）。
            criteria_json: 验收标准的 JSON 文本（字符串 / 字符串数组 / 键值对象）。

        Returns:
            JSON 字符串：包含 ``failed_subtasks`` / ``empty_output`` /
            ``missing_evidence`` / ``artifact_violations``（产物路径越界）/
            ``criteria`` 与 ``criteria_total`` / ``passed`` 与 ``notes``；
            输入非法时返回 ``{"status": "error", "error"}``。
        """
        try:
            results = _parse_result_list(results_json)
        except (ValueError, ValidationError) as exc:
            return _error(f"results_json 校验失败：{exc}")
        try:
            criteria = _parse_criteria(criteria_json)
        except (ValueError, json.JSONDecodeError) as exc:
            return _error(f"criteria_json 校验失败：{exc}")

        failed = [
            {"subtask_id": result.subtask_id, "error": result.error}
            for result in results
            if not result.success
        ]
        empty_output = [
            result.subtask_id for result in results if result.success and not result.output.strip()
        ]
        missing_evidence = [
            result.subtask_id for result in results if result.success and not result.evidence
        ]
        artifact_violations: list[dict[str, str]] = []
        for result in results:
            for artifact in result.artifacts:
                try:
                    path = Path(artifact)
                    target = path.resolve() if path.is_absolute() else (sandbox_root / path).resolve()
                    if not _is_within(sandbox_root, target):
                        artifact_violations.append(
                            {"subtask_id": result.subtask_id, "artifact": artifact}
                        )
                except OSError:
                    artifact_violations.append(
                        {"subtask_id": result.subtask_id, "artifact": artifact}
                    )

        notes: list[str] = []
        if failed:
            notes.append("存在失败子任务：结论不应为通过，应给出重试目标与修复要求。")
        if empty_output:
            notes.append("存在成功但无文本产出的子任务：须核验其证据链是否闭环。")
        if missing_evidence:
            notes.append("存在无证据的成功结果：证据缺失的断言按未通过处理，不得放行。")
        if artifact_violations:
            notes.append("存在沙箱目录之外的产物路径：属于越界痕迹，必须重点核验并可能判定不通过。")
        if not criteria:
            notes.append("未提供验收标准：无法逐条核对，请要求主 Agent 补充 TaskRequest.acceptance_criteria。")

        passed = not (failed or artifact_violations or missing_evidence or empty_output)
        return _json(
            {
                "status": "ok",
                "result_count": len(results),
                "failed_subtasks": failed,
                "empty_output": empty_output,
                "missing_evidence": missing_evidence,
                "artifact_violations": artifact_violations,
                "criteria": criteria,
                "criteria_total": len(criteria),
                "passed": passed,
                "notes": notes,
            }
        )

    return {
        "check_plan_completeness": check_plan_completeness,
        "check_tool_necessity": check_tool_necessity,
        "check_result_correctness": check_result_correctness,
    }


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------
def create_review_agent(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
) -> Agent[Any]:
    """构造配置完毕的评审 Agent（SDK Agent 实例）。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 模型实例或名称覆盖；未提供时按本包 api_config.yaml 构造
            （密钥优先取 api_key 直接值，回退 api_key_env 环境变量，
            均缺失时抛 ConfigLoadError）。

    Returns:
        ``Agent`` 实例：name="review_agent"，instructions 取自 prompt.md，
        不设置 output_type（None，纯文本返回）：评审结论由调用方剥离
        Markdown 围栏后按 ``ReviewDecision`` 手动校验（MCP 适配层与
        workflow 控制层均内置该容错路径）。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    resolved_model = model or _build_chat_model(load_agent_api_config(API_CONFIG_PATH))
    tools = build_review_tools(resolved_context, settings=resolved_settings)
    return Agent(
        name=AGENT_NAME,
        instructions=_load_prompt(),
        tools=list(tools.values()),
        model=resolved_model,
        # output_type 保持 None（纯文本返回）：千问等兼容接口忽略
        # response_format=json_schema 时 SDK 严格校验会抛 ModelBehaviorError，
        # 改由调用方手动剥离围栏 + ReviewDecision 校验。
    )


def get_review_agent_as_tool(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
) -> FunctionTool:
    """将评审 Agent 转为主 Agent 可直接调用的工具入口（Agent.as_tool）。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 模型实例或名称覆盖；未提供时按本包 api_config.yaml 构造。

    Returns:
        ``FunctionTool``：工具名 "review_agent"，返回值经输出提取器规范化为
        结构化 JSON 字符串。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    agent = create_review_agent(
        resolved_context, settings=resolved_settings, model=model
    )
    return agent.as_tool(
        tool_name=AGENT_NAME,
        tool_description=_TOOL_DESCRIPTION,
        custom_output_extractor=_extract_output_json,
        max_turns=resolved_settings.max_model_turns,
    )


def get_mcp_schema() -> MCPToolSchema:
    """返回评审 Agent 的 MCP 暴露契约（MCPToolSchema）。

    inputSchema 取自 ``ReviewInput.model_json_schema()``。
    """
    return MCPToolSchema(
        name=AGENT_NAME,
        description=_TOOL_DESCRIPTION,
        inputSchema=ReviewInput.model_json_schema(),
    )


def get_mcp_server(
    context: ProjectContext | None = None,
    *,
    settings: Settings | None = None,
    model: str | Model | None = None,
) -> MCPToolServer:
    """将评审 Agent 包装为 MCP Server（供主管与 MCPClient 统一调用）。

    与 ``get_review_agent_as_tool`` 等价的 MCP 暴露形态：调用时按
    ``ReviewInput`` 校验参数、运行 Agent，并按 ``ReviewDecision``
    提取结构化 JSON 文本。

    Args:
        context: 共享上下文；未提供时使用默认沙箱目录（cwd/sandbox）。
        settings: 生效配置；未提供时取 context.config（再回退全局配置）。
        model: 模型实例或名称覆盖；未提供时按本包 api_config.yaml 构造。

    Returns:
        ``MCPToolServer``：name="review_agent"，含单个工具的 MCP Server。
    """
    resolved_context = _resolve_context(context)
    resolved_settings = _resolve_settings(resolved_context, settings)
    agent = create_review_agent(
        resolved_context, settings=resolved_settings, model=model
    )
    return agent_to_mcp_server(
        agent,
        AGENT_NAME,
        _TOOL_DESCRIPTION,
        ReviewInput,
        ReviewDecision,
        max_turns=resolved_settings.max_model_turns,
    )

"""四象限任务路由器（TaskRouter）。

分类策略（两级，先便宜后精确）：
1. 规则预判（零成本、不调用 LLM）：只捕获毫无歧义的简单场景——
   纯数学表达式与明确问候/闲聊。注意：短文本不一定简单、长文本不
   一定复杂，严禁以文本长度作为复杂度判断依据；
2. LLM 分类（一次轻量调用）：其余场景一律交由 LLM 按四象限判定
   （复杂度 × 时长），不猜测、不启发式。分类调用失败、输出解析
   失败或结果非法时，保守降级为完整工作流（最安全的兜底）。

路由分发（分类结果 → 执行路径）：
- direct_answer   → DirectAnswerPath（① 简单+短：一次 LLM 直接回答）
- deep_reasoning  → DeepReasoningPath（② 复杂+短：单次深度推理）
- persistent_task → PersistentTaskPath（③ 简单+长：持久化 + 调度）
- full_workflow   → WorkflowEngine（④ 复杂+长：完整多 Agent 协作）

延迟导入说明：PersistentTaskPath 与 WorkflowEngine 均在函数内按需
导入，避免模块级循环导入（workflow 依赖 orchestrator，router 是位于
其上的分发入口），同时保证对应模块暂时缺失时不影响其他路径可用。
"""

from __future__ import annotations

import re
from pathlib import Path

from openai import AsyncOpenAI

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.json_utils import extract_json_object
from so_agent.models import ClassificationResult, RouteStep, TaskStatus, TaskType
from so_agent.orchestrator import OrchestratorOutput
from so_agent.paths.deep_reasoning import DeepReasoningPath
from so_agent.paths.direct_answer import DirectAnswerPath
from so_agent.paths.single_file import maybe_run_single_file_fast_path
from so_agent.runtime.config_loader import (
    load_agent_api_config,
    resolve_api_config_path,
    resolve_model_http_timeout,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.model_gateway import build_openai_client
from so_agent.runtime.usage import record_completion_usage
from so_agent.runtime.telemetry import async_span_scope, set_span_attributes

logger = get_logger("router")

# 复用主管 Agent 的 API 配置（分类调用与主管共用同一模型入口）
API_CONFIG_PATH: Path = Path(__file__).resolve().parent / "api_config.yaml"

# 规则预判模式一：纯数学表达式（如 "3+7"、"12*5"）
MATH_EXPR_RE = re.compile(r"^\s*[\d\s\+\-\*\/\.\(\)\^%]+[=?？]?\s*$")
# 规则预判模式二：明确问候/闲聊（忽略大小写与尾部标点）
GREETING_RE = re.compile(
    r"^(你好|hello|hi|嗨|谢谢|thanks|再见|bye|hey)\s*[!！。.？?]*$",
    re.IGNORECASE,
)
# 目标中出现“创建/写入/运行”等动作且指向文件、脚本、命令时，必须走工具路径。
# 动作必须带明确对象：单独的“执行/写入”也可能出现在知识问答里（如“解释
# goroutine 如何执行”），不能仅凭动词升级为工具任务。
_TOOL_ACTION_HINT_RE = re.compile(
    r"(创建|新建|写(?:入)?|保存|读取|修改|删除|重命名|复制|移动|压缩|解压|"
    r"安装|部署|下载|上传|运行|执行|生成|"
    r"\b(?:create|write|save|read|modify|delete|rename|copy|move|run|execute|install|deploy|download|upload|generate)\b)",
    re.IGNORECASE,
)
_TOOL_OBJECT_RE = (
    r"(?:文件|文档|目录|文件夹|脚本|代码|报告|表格|图片|附件|配置|数据库|"
    r"仓库|命令|程序|测试|构建|依赖|包|插件|工具|环境|服务|接口|"
    r"python|bash|shell|node|npm|pip|git|docker|curl|"
    r"[\w./\\-]+\.(?:md|txt|py|js|ts|json|csv|log|html|ya?ml|xlsx?|pdf|docx?))"
)
_TOOL_INTENT_RE = re.compile(
    rf"{_TOOL_ACTION_HINT_RE.pattern}[^，,。；;!?！？\n]{{0,12}}{_TOOL_OBJECT_RE}"
    r"|(\b(?:create|write|save|read|modify|delete|rename|copy|move|run|execute|install|deploy|download|upload|generate)\b"
    r".{0,40}\b(?:file|script|code|report|directory|folder|command|config|database|repo|repository|test|build|dependency|package|plugin|tool|environment)\b)",
    re.IGNORECASE,
)
# 否定性工具指令（“不要写入任何文件”“无需运行命令”）表达的是禁止，
# 不能当成工具需求；按分句剔除，避免升级误判。
_TOOL_NEGATION_RE = re.compile(
    r"(不要|无需|不需要|禁止|严禁|不得|不用|不必|不能|不可|避免|"
    r"\b(?:do not|don't|without|no need to|avoid)\b)",
    re.IGNORECASE,
)
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[，,。；;!?！？\n])")
# 验收标准明确要求产物存在/包含内容时，直接回答无法满足
_ARTIFACT_ACCEPTANCE_RE = re.compile(
    r"((文件|目录|脚本|产物|附件)|[\w./\\-]+\.(md|txt|py|js|ts|json|csv|log|html|ya?ml|xlsx?|pdf|docx?))"
    r".{0,16}(存在|包含|生成|创建|写入|下载)",
    re.IGNORECASE,
)
# 无工具路径出现工具调用标记，说明模型伪造了工具调用，不能当作最终答复
_TOOL_MARKUP_RE = re.compile(r"<\s*(tool_call|function|parameter)\b", re.IGNORECASE)

# 模型偶发输出非标准枚举（如 "direct"），归一化为四象限标准值；
# 只映射语义明确的同义词，未知取值仍走保守降级。
_TASK_TYPE_ALIASES = {
    "direct": TaskType.DIRECT_ANSWER,
    "direct-answer": TaskType.DIRECT_ANSWER,
    "direct answer": TaskType.DIRECT_ANSWER,
    "directanswer": TaskType.DIRECT_ANSWER,
    "deep": TaskType.DEEP_REASONING,
    "deep-reasoning": TaskType.DEEP_REASONING,
    "deep reasoning": TaskType.DEEP_REASONING,
    "reasoning": TaskType.DEEP_REASONING,
    "persistent": TaskType.PERSISTENT_TASK,
    "persistent-task": TaskType.PERSISTENT_TASK,
    "persistent task": TaskType.PERSISTENT_TASK,
    "schedule": TaskType.PERSISTENT_TASK,
    "scheduled": TaskType.PERSISTENT_TASK,
    "workflow": TaskType.FULL_WORKFLOW,
    "full": TaskType.FULL_WORKFLOW,
    "full-workflow": TaskType.FULL_WORKFLOW,
    "full workflow": TaskType.FULL_WORKFLOW,
}


def _normalize_task_type(value: object) -> object:
    """把模型输出的任务类型别名归一化为标准枚举值。"""
    if isinstance(value, str):
        return _TASK_TYPE_ALIASES.get(value.strip().lower(), value)
    return value


def _strip_negated_tool_clauses(text: str) -> str:
    """剔除含否定词的工具指令分句，保留正面需求。"""
    kept: list[str] = []
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        if _TOOL_NEGATION_RE.search(clause) and _TOOL_ACTION_HINT_RE.search(
            clause
        ):
            continue
        kept.append(clause)
    return "".join(kept)


def _needs_tool_execution(objective: str, acceptance_criteria: str) -> bool:
    objective = _strip_negated_tool_clauses(objective)
    acceptance_criteria = _strip_negated_tool_clauses(acceptance_criteria)
    return bool(
        _TOOL_INTENT_RE.search(objective or "")
        or _ARTIFACT_ACCEPTANCE_RE.search(acceptance_criteria or "")
    )


def _tool_intent_evidence(objective: str, acceptance_criteria: str) -> str:
    """返回触发工具意图升级的确定性证据（供判断轨迹展示）。"""
    matched = _TOOL_INTENT_RE.search(
        _strip_negated_tool_clauses(objective)
    )
    if matched is not None:
        return f"目标命中工具动作“{matched.group(0)}”"
    matched = _ARTIFACT_ACCEPTANCE_RE.search(
        _strip_negated_tool_clauses(acceptance_criteria)
    )
    if matched is not None:
        return f"验收标准命中产物要求“{matched.group(0)}”"
    return ""

CLASSIFIER_PROMPT = """你是一个任务分类器。根据用户目标，判断任务类型并返回 JSON。

分类维度：
- 复杂度：任务是否需要多步骤协作、工具调用、代码执行？（注意：短文本不一定简单，长文本不一定复杂）
- 时长：任务是否需要持久化状态、定时执行、长期监控？

四种类型：
1. direct_answer — 简单+短事务：可以直接回答的问题（算术、事实查询、简单翻译、常识问答等）
2. deep_reasoning — 复杂+短事务：需要深度推理但不需要工具（数学证明、复杂分析、逻辑推导、创意写作等）
3. persistent_task — 简单+长事务：需要持久化和调度（定时任务、文件监控、持续集成、周期性检查等）
4. full_workflow — 复杂+长事务：需要完整多 Agent 协作（代码开发、项目分析、多步骤工程任务等）

硬性规则：只要目标或验收标准要求创建/读取/修改文件、运行命令或脚本、生成可下载产物，
即使描述很短，也必须归为 full_workflow（direct_answer / deep_reasoning 没有工具，无法完成）。

分类示例（覆盖全部四象限）：
- 输入："3+7" → 输出：{"task_type": "direct_answer", "reasoning": "纯算术，直接计算即可回答"}
- 输入："Python 的 GIL 是什么" → 输出：{"task_type": "direct_answer", "reasoning": "概念性事实查询，直接回答即可"}
- 输入："帮我分析市面上最好的蓝牙音箱" → 输出：{"task_type": "full_workflow", "reasoning": "需要多步骤调研与工具协作"}
- 输入："对比 Python asyncio 和 Go goroutine 并发模型，编写测试代码并给出选型建议" → 输出：{"task_type": "full_workflow", "reasoning": "需要调研、编写并运行基准代码、多步骤工程协作"}
- 输入："什么是量子计算" → 输出：{"task_type": "deep_reasoning", "reasoning": "概念解释，需要深度推理但无需工具"}
- 输入："分析 CAP 定理在微服务架构中的实际权衡" → 输出：{"task_type": "deep_reasoning", "reasoning": "复杂架构分析，需要深度推理但无需工具"}
- 输入："每天早上 9 点检查服务器健康状态并发送报告" → 输出：{"task_type": "persistent_task", "reasoning": "定时执行的周期性任务，需要持久化与调度"}
- 输入："监控 GitHub 仓库的新 issue 并自动分类" → 输出：{"task_type": "persistent_task", "reasoning": "长期持续监控，需要持久化状态与周期性调度"}
- 输入："创建 web-smoke.txt，内容写 WEB_OK" → 输出：{"task_type": "full_workflow", "reasoning": "需要文件写入工具，直接回答无法产出文件"}

严格按以下 JSON 格式输出，不要输出其他内容：
{"task_type": "...", "reasoning": "一句话说明分类理由"}"""


class TaskRouter:
    """四象限任务路由器：规则预判 + LLM 分类 + 路由分发。"""

    def __init__(
        self,
        context: ProjectContext,
        settings: Settings,
        *,
        log_file: str | Path | None = None,
    ) -> None:
        self._context = context
        self._settings = settings
        self._log_file = log_file

    def _rule_based_classify(self, objective: str) -> TaskType | None:
        """零成本规则预判：只捕获毫无歧义的简单场景。

        返回 None 表示需要走 LLM 分类（绝不以文本长度判断复杂度）。
        """
        text = objective.strip()
        if MATH_EXPR_RE.match(text):
            return TaskType.DIRECT_ANSWER
        if GREETING_RE.match(text):
            return TaskType.DIRECT_ANSWER
        return None

    @staticmethod
    def _rule_label(objective: str) -> str:
        """规则命中项的可读名称（仅用于判断轨迹展示）。"""
        text = objective.strip()
        if MATH_EXPR_RE.match(text):
            return "纯数学表达式"
        if GREETING_RE.match(text):
            return "问候/闲聊"
        return ""

    async def _llm_classify(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> ClassificationResult:
        """LLM 分类：一次轻量调用。

        API 配置加载失败时抛 ``ConfigLoadError``（明确失败，不静默降级）；
        HTTP 超时经 ``resolve_model_http_timeout`` 解析（任务 59，与
        orchestrator / 工具包客户端一致：分维度未配置时等价旧单一
        float timeout，非法配置 fail fast）；调用异常、输出解析失败或
        结果非法时降级为完整工作流。
        """
        api_cfg = load_agent_api_config(
            resolve_api_config_path(API_CONFIG_PATH), resolve_secrets=True
        )
        http_timeout = resolve_model_http_timeout(api_cfg)
        client = build_openai_client(
            api_cfg,
            config_path=resolve_api_config_path(API_CONFIG_PATH),
            client_factory=AsyncOpenAI,
            http_timeout=http_timeout,
        )

        user_msg = f"任务目标：{objective}"
        if constraints:
            user_msg += f"\n约束条件：{constraints}"
        if acceptance_criteria:
            user_msg += f"\n验收标准：{acceptance_criteria}"

        try:
            async with async_span_scope(
                "so_agent.model.call",
                {"task.type": "classify", "model.name": api_cfg.model},
            ) as call_span:
                response = await client.chat.completions.create(
                    model=api_cfg.model,
                    messages=[
                        {"role": "system", "content": CLASSIFIER_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    max_tokens=self._settings.classifier_max_tokens,
                    timeout=http_timeout,
                )
                raw = response.choices[0].message.content or ""
                record_completion_usage(self._context, response, "router_classifier")
                set_span_attributes(call_span, {"answer.chars": len(raw)})
        except Exception as exc:  # 网络/服务异常：降级为完整工作流（最安全）
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning=(
                    f"分类调用失败（{type(exc).__name__}: {exc}），"
                    "降级为完整工作流"
                ),
                steps=[
                    RouteStep(
                        stage="llm_classify",
                        status="fallback",
                        detail=(
                            f"分类调用异常（{type(exc).__name__}），"
                            "降级为完整工作流"
                        ),
                    )
                ],
            )
        finally:
            await client.close()

        # 空输出直接降级（与“非 JSON 输出”区分开，便于从运行日志定位原因）
        if not raw.strip():
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning="分类输出为空字符串，降级为完整工作流",
                steps=[
                    RouteStep(
                        stage="llm_classify",
                        status="fallback",
                        detail="分类输出为空字符串，降级为完整工作流",
                    )
                ],
            )
        # 宽容解析（剥离 Markdown 围栏、容错前后解释文字，提取首个 JSON
        # 对象），与项目 json_utils 的“纯文本返回 + 手动校验”解析路径保持一致
        data = extract_json_object(raw)
        if data is None:
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning=(
                    "分类输出解析失败，降级为完整工作流："
                    f"{raw.strip()[:100]}"
                ),
                steps=[
                    RouteStep(
                        stage="llm_classify",
                        status="fallback",
                        detail="分类输出不是合法 JSON，降级为完整工作流",
                    )
                ],
            )
        try:
            result = ClassificationResult(
                task_type=_normalize_task_type(data.get("task_type")),
                reasoning=str(data.get("reasoning") or ""),
            )
        except Exception as exc:  # 字段缺失/枚举值非法等
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning=(
                    f"分类结果非法（{type(exc).__name__}），"
                    f"降级为完整工作流：{raw.strip()[:100]}"
                ),
                steps=[
                    RouteStep(
                        stage="llm_classify",
                        status="fallback",
                        detail=(
                            f"分类结果字段非法（{type(exc).__name__}），"
                            "降级为完整工作流"
                        ),
                    )
                ],
            )
        result.steps = [
            RouteStep(
                stage="llm_classify",
                status="passed",
                detail=(
                    f"模型判定 {result.task_type.value}"
                    + (f"：{result.reasoning}" if result.reasoning else "")
                ),
            )
        ]
        return result

    async def classify(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> ClassificationResult:
        """分类任务：先规则预判，不确定则走 LLM 分类。"""
        if not self._settings.enable_router:
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning="路由器已禁用，走完整工作流",
                steps=[
                    RouteStep(
                        stage="router_disabled",
                        status="fallback",
                        detail="enable_router=false，跳过规则与模型分类",
                    )
                ],
            )

        steps: list[RouteStep] = []
        # 1) 规则预判（仅在无约束无验收标准时尝试）
        if not constraints and not acceptance_criteria:
            quick = self._rule_based_classify(objective)
            if quick is not None:
                label = self._rule_label(objective)
                steps.append(
                    RouteStep(
                        stage="rule_check",
                        status="hit",
                        detail=f"命中{label}规则，直接判定 direct_answer",
                    )
                )
                return ClassificationResult(
                    task_type=quick,
                    reasoning="规则预判命中",
                    steps=steps,
                )
            steps.append(
                RouteStep(
                    stage="rule_check",
                    status="miss",
                    detail="未命中纯数学/问候规则，交由模型分类",
                )
            )
        else:
            steps.append(
                RouteStep(
                    stage="rule_check",
                    status="skipped",
                    detail="存在约束或验收标准，规则预判不适用",
                )
            )

        # 2) LLM 分类
        result = await self._llm_classify(
            objective, constraints, acceptance_criteria
        )
        steps.extend(result.steps)
        if result.task_type in {
            TaskType.DIRECT_ANSWER,
            TaskType.DEEP_REASONING,
        } and _needs_tool_execution(objective, acceptance_criteria):
            evidence = _tool_intent_evidence(objective, acceptance_criteria)
            steps.append(
                RouteStep(
                    stage="tool_intent_upgrade",
                    status="upgraded",
                    detail=(
                        f"原分类 {result.task_type.value}；{evidence}；"
                        "升级为完整工作流"
                    ),
                )
            )
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning=(
                    f"目标或验收标准要求文件/命令工具（原分类 "
                    f"{result.task_type.value}），升级为完整工作流"
                ),
                steps=steps,
            )
        return result.model_copy(update={"steps": steps})

    async def _guard_no_tool_output(
        self,
        output: OrchestratorOutput,
        objective: str,
        constraints: str,
        acceptance_criteria: str,
        classification: ClassificationResult,
    ) -> OrchestratorOutput:
        """无工具路径若输出工具调用标记，说明模型伪造了工具调用，降级重跑。"""
        if _TOOL_MARKUP_RE.search(output.final_result or ""):
            logger.warning("直答/推理路径出现工具调用标记，降级为完整工作流")
            classification.steps.append(
                RouteStep(
                    stage="output_guard",
                    status="applied",
                    detail=(
                        "直答/推理输出包含 <tool_call>/<function> 等伪造工具调用"
                        "标记，已降级为完整工作流"
                    ),
                )
            )
            return await self._run_full_workflow(
                objective, constraints, acceptance_criteria
            )
        classification.steps.append(
            RouteStep(
                stage="output_guard",
                status="passed",
                detail="输出未包含伪造工具调用标记，维持当前分支",
            )
        )
        return output

    async def _run_full_workflow(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> OrchestratorOutput:
        """完整工作流兜底：交给 WorkflowEngine，按配置选择编排运行时。"""
        from so_agent.runtime.orchestration_runtime import (
            create_orchestration_runtime,
        )
        from so_agent.workflow import WorkflowEngine

        engine = WorkflowEngine(
            self._context, settings=self._settings, log_file=self._log_file
        )
        runtime = create_orchestration_runtime(engine, self._settings)
        output = await runtime.run(objective, constraints, acceptance_criteria)
        if output.status is TaskStatus.COMPLETED_PROVISIONAL:
            # CLI / 一次性调用没有独立的快照消费方：等待后台评审收口，
            # 返回确定性的最终结果（Web 控制台直接持有引擎，可先行展示）
            final = await runtime.wait_for_provisional()
            return final or output
        return output

    async def route(
        self,
        objective: str,
        constraints: str,
        acceptance_criteria: str,
        *,
        classification: ClassificationResult | None = None,
    ) -> OrchestratorOutput:
        """分类任务并路由到对应执行路径。

        ``classification`` 可由调用方预先提供（Web 控制台需要先展示分类
        结果再选路），避免重复调用分类模型。
        """
        if classification is None:
            async with async_span_scope(
                "so_agent.classify",
                {"task.objective_chars": len(objective)},
            ) as span:
                classification = await self.classify(
                    objective, constraints, acceptance_criteria
                )
                set_span_attributes(
                    span,
                    {
                        "task.type": classification.task_type.value,
                        "task.reasoning_chars": len(
                            classification.reasoning or ""
                        ),
                    },
                )
        logger.info(
            "任务分类：%s（%s）",
            classification.task_type.value,
            classification.reasoning,
        )

        match classification.task_type:
            case TaskType.DIRECT_ANSWER:
                output = await DirectAnswerPath(
                    self._context, self._settings
                ).run(objective, constraints, acceptance_criteria, classification)
                return await self._guard_no_tool_output(
                    output,
                    objective,
                    constraints,
                    acceptance_criteria,
                    classification,
                )
            case TaskType.DEEP_REASONING:
                output = await DeepReasoningPath(
                    self._context, self._settings
                ).run(objective, constraints, acceptance_criteria, classification)
                return await self._guard_no_tool_output(
                    output,
                    objective,
                    constraints,
                    acceptance_criteria,
                    classification,
                )
            case TaskType.PERSISTENT_TASK:
                # 延迟导入：该模块按需可用，缺失时不影响其他路径
                from so_agent.paths.persistent_task import PersistentTaskPath

                return await PersistentTaskPath(
                    self._context, self._settings
                ).run(
                    objective, constraints, acceptance_criteria, classification
                )
            case _:
                # FULL_WORKFLOW：先尝试严格判定的单文件快通道；未启用、
                # 不匹配或失败均回退完整工作流（不改变原有终态语义）。
                if bool(self._settings.enable_router):
                    outcome = await maybe_run_single_file_fast_path(
                        self._context,
                        self._settings,
                        objective,
                        constraints,
                        acceptance_criteria,
                    )
                    if outcome.applied and outcome.output is not None:
                        classification.steps.append(
                            RouteStep(
                                stage="fast_path",
                                status="hit",
                                detail=(
                                    f"单文件纯写入快通道：{outcome.target}"
                                ),
                            )
                        )
                        return outcome.output
                    if outcome.reason not in {"disabled", "not_applicable"}:
                        classification.steps.append(
                            RouteStep(
                                stage="fast_path",
                                status="fallback",
                                detail=(
                                    f"快通道未完成（{outcome.reason}），"
                                    "回退完整工作流"
                                ),
                            )
                        )
                # FULL_WORKFLOW 与未知类型统一降级为完整工作流
                return await self._run_full_workflow(
                    objective, constraints, acceptance_criteria
                )


# 供外部按需引用的统一导出（TaskStatus 从 models.py 导入，便于调用方
# 直接判断路由结果的终态）
__all__ = [
    "API_CONFIG_PATH",
    "CLASSIFIER_PROMPT",
    "GREETING_RE",
    "MATH_EXPR_RE",
    "TaskRouter",
    "TaskStatus",
    "TaskType",
]

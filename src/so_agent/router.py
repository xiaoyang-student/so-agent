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
import sys
from pathlib import Path

from openai import AsyncOpenAI

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.json_utils import extract_json_object
from so_agent.models import ClassificationResult, TaskStatus, TaskType
from so_agent.orchestrator import OrchestratorOutput
from so_agent.paths.deep_reasoning import DeepReasoningPath
from so_agent.paths.direct_answer import DirectAnswerPath
from so_agent.runtime.config_loader import (
    load_agent_api_config,
    resolve_model_http_timeout,
)

# 复用主管 Agent 的 API 配置（分类调用与主管共用同一模型入口）
API_CONFIG_PATH: Path = Path(__file__).resolve().parent / "api_config.yaml"

# 规则预判模式一：纯数学表达式（如 "3+7"、"12*5"）
MATH_EXPR_RE = re.compile(r"^\s*[\d\s\+\-\*\/\.\(\)\^%]+[=?？]?\s*$")
# 规则预判模式二：明确问候/闲聊（忽略大小写与尾部标点）
GREETING_RE = re.compile(
    r"^(你好|hello|hi|嗨|谢谢|thanks|再见|bye|hey)\s*[!！。.？?]*$",
    re.IGNORECASE,
)

CLASSIFIER_PROMPT = """你是一个任务分类器。根据用户目标，判断任务类型并返回 JSON。

分类维度：
- 复杂度：任务是否需要多步骤协作、工具调用、代码执行？（注意：短文本不一定简单，长文本不一定复杂）
- 时长：任务是否需要持久化状态、定时执行、长期监控？

四种类型：
1. direct_answer — 简单+短事务：可以直接回答的问题（算术、事实查询、简单翻译、常识问答等）
2. deep_reasoning — 复杂+短事务：需要深度推理但不需要工具（数学证明、复杂分析、逻辑推导、创意写作等）
3. persistent_task — 简单+长事务：需要持久化和调度（定时任务、文件监控、持续集成、周期性检查等）
4. full_workflow — 复杂+长事务：需要完整多 Agent 协作（代码开发、项目分析、多步骤工程任务等）

分类示例（覆盖全部四象限）：
- 输入："3+7" → 输出：{"task_type": "direct_answer", "reasoning": "纯算术，直接计算即可回答"}
- 输入："Python 的 GIL 是什么" → 输出：{"task_type": "direct_answer", "reasoning": "概念性事实查询，直接回答即可"}
- 输入："帮我分析市面上最好的蓝牙音箱" → 输出：{"task_type": "full_workflow", "reasoning": "需要多步骤调研与工具协作"}
- 输入："对比 Python asyncio 和 Go goroutine 并发模型，编写测试代码并给出选型建议" → 输出：{"task_type": "full_workflow", "reasoning": "需要调研、编写并运行基准代码、多步骤工程协作"}
- 输入："什么是量子计算" → 输出：{"task_type": "deep_reasoning", "reasoning": "概念解释，需要深度推理但无需工具"}
- 输入："分析 CAP 定理在微服务架构中的实际权衡" → 输出：{"task_type": "deep_reasoning", "reasoning": "复杂架构分析，需要深度推理但无需工具"}
- 输入："每天早上 9 点检查服务器健康状态并发送报告" → 输出：{"task_type": "persistent_task", "reasoning": "定时执行的周期性任务，需要持久化与调度"}
- 输入："监控 GitHub 仓库的新 issue 并自动分类" → 输出：{"task_type": "persistent_task", "reasoning": "长期持续监控，需要持久化状态与周期性调度"}

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
        api_cfg = load_agent_api_config(API_CONFIG_PATH, resolve_secrets=True)
        http_timeout = resolve_model_http_timeout(api_cfg)
        client = AsyncOpenAI(
            api_key=api_cfg.resolve_api_key(),
            base_url=api_cfg.base_url or None,
            timeout=http_timeout,
            max_retries=api_cfg.retry,
        )

        user_msg = f"任务目标：{objective}"
        if constraints:
            user_msg += f"\n约束条件：{constraints}"
        if acceptance_criteria:
            user_msg += f"\n验收标准：{acceptance_criteria}"

        try:
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
        except Exception as exc:  # 网络/服务异常：降级为完整工作流（最安全）
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning=(
                    f"分类调用失败（{type(exc).__name__}: {exc}），"
                    "降级为完整工作流"
                ),
            )
        finally:
            await client.close()

        # 空输出直接降级（与“非 JSON 输出”区分开，便于从运行日志定位原因）
        if not raw.strip():
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning="分类输出为空字符串，降级为完整工作流",
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
            )
        try:
            return ClassificationResult(**data)
        except Exception as exc:  # 字段缺失/枚举值非法等
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning=(
                    f"分类结果非法（{type(exc).__name__}），"
                    f"降级为完整工作流：{raw.strip()[:100]}"
                ),
            )

    async def classify(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> ClassificationResult:
        """分类任务：先规则预判，不确定则走 LLM 分类。"""
        if not self._settings.enable_router:
            return ClassificationResult(
                task_type=TaskType.FULL_WORKFLOW,
                reasoning="路由器已禁用，走完整工作流",
            )

        # 1) 规则预判（仅在无约束无验收标准时尝试）
        if not constraints and not acceptance_criteria:
            quick = self._rule_based_classify(objective)
            if quick is not None:
                return ClassificationResult(
                    task_type=quick,
                    reasoning="规则预判命中",
                )

        # 2) LLM 分类
        return await self._llm_classify(objective, constraints, acceptance_criteria)

    async def _run_full_workflow(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> OrchestratorOutput:
        """完整工作流兜底：交给现有 WorkflowEngine（延迟导入）。"""
        from so_agent.workflow import WorkflowEngine

        engine = WorkflowEngine(
            self._context, settings=self._settings, log_file=self._log_file
        )
        return await engine.run(objective, constraints, acceptance_criteria)

    async def route(
        self, objective: str, constraints: str, acceptance_criteria: str
    ) -> OrchestratorOutput:
        """分类任务并路由到对应执行路径。"""
        classification = await self.classify(
            objective, constraints, acceptance_criteria
        )
        print(
            f"[so-agent] 任务分类：{classification.task_type.value}"
            f"（{classification.reasoning}）",
            file=sys.stderr,
            flush=True,
        )

        match classification.task_type:
            case TaskType.DIRECT_ANSWER:
                return await DirectAnswerPath(self._context, self._settings).run(
                    objective, constraints, acceptance_criteria, classification
                )
            case TaskType.DEEP_REASONING:
                return await DeepReasoningPath(self._context, self._settings).run(
                    objective, constraints, acceptance_criteria, classification
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

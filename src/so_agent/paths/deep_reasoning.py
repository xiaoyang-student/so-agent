"""深度推理路径（DeepReasoningPath）。

四象限中的 ② 复杂+短：单次 LLM 调用完成深度推理，不经过规划 / 评审 /
工具调用。与 DirectAnswerPath 的区别：
- system prompt 不同：引导模型进行严谨、分步的深度推理；
- 输出上限由 ``deep_reasoning_max_tokens`` 控制（<=0 时不显式限制）；
- 用户消息中显式包含约束条件与验收标准。

实现约定与 DirectAnswerPath 一致：复用主管 Agent 的
``src/so_agent/api_config.yaml``；输出包装为 ``OrchestratorOutput``
（定义于 ``so_agent.orchestrator``）；调用失败返回 ``failed`` 终态
而非向上抛异常。
"""

from __future__ import annotations

import uuid
from pathlib import Path

from openai import AsyncOpenAI

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ClassificationResult, TaskStatus
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.config_loader import (
    load_agent_api_config,
    resolve_api_config_path,
    resolve_model_http_timeout,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.model_gateway import build_openai_client
from so_agent.runtime.telemetry import async_span_scope, set_span_attributes
from so_agent.runtime.usage import record_completion_usage

# 复用主管 Agent 的 API 配置（src/so_agent/api_config.yaml）
API_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "api_config.yaml"

logger = get_logger("path.deep_reasoning")


def _build_prompt(objective: str, constraints: str, acceptance_criteria: str) -> str:
    """构造推理 prompt：任务目标 + 约束条件 + 验收标准（后两者可选）。"""
    parts = [f"任务目标：{objective.strip()}"]
    if constraints.strip():
        parts.append(f"约束条件：{constraints.strip()}")
    if acceptance_criteria.strip():
        parts.append(f"验收标准：{acceptance_criteria.strip()}")
    return "\n\n".join(parts)


class DeepReasoningPath:
    """复杂+短任务：单次深度推理。"""

    SYSTEM_PROMPT = (
        "你是一个深度推理助手。请对以下问题进行严谨的分析和推理。\n"
        "要求：\n"
        "1. 先分析问题的关键要素\n"
        "2. 逐步推导并给出明确的结论\n"
        "3. 用户约束优先级最高：若用户限制句数、字数、语言、格式或要求"
        "只输出某类内容，必须严格满足，不得附加分析过程或总结段\n"
        "如有约束条件，必须严格遵守。\n"
        "你没有工具，不能创建文件、读写磁盘或执行命令；"
        "不得输出 <tool_call>、<function=...> 等工具调用标记来假装执行操作。"
    )

    def __init__(self, context: ProjectContext, settings: Settings) -> None:
        self._context = context
        self._settings = settings

    async def run(
        self,
        objective: str,
        constraints: str,
        acceptance_criteria: str,
        classification: ClassificationResult,
    ) -> OrchestratorOutput:
        """一次 LLM 调用完成深度推理，并包装为 OrchestratorOutput。

        Args:
            objective: 任务目标（用户问题原文）。
            constraints: 约束条件文本（可为空）。
            acceptance_criteria: 验收标准文本（可为空）。
            classification: 路由器给出的分类结果（记录进输出摘要）。

        Returns:
            ``OrchestratorOutput``：成功时为 ``completed`` 终态，完整推理
            结论置于 ``final_result``；调用失败时为 ``failed`` 终态。
        """
        task_id = uuid.uuid4().hex
        prompt = _build_prompt(objective, constraints, acceptance_criteria)
        summary = "执行路径：deep_reasoning"
        if classification.reasoning:
            summary += f"；分类理由：{classification.reasoning}"

        logger.info("深度推理中...")

        api_cfg = load_agent_api_config(
            resolve_api_config_path(API_CONFIG_PATH), resolve_secrets=True
        )
        # HTTP 超时经 resolve_model_http_timeout 解析（任务 59，与
        # orchestrator / router 客户端一致）：分维度未配置时等价旧单一
        # float timeout；request 级 timeout 同步传同一解析结果，避免
        # float 覆盖客户端级分维度 Timeout（openai SDK 中 request 级
        # 参数优先于客户端默认值）。
        http_timeout = resolve_model_http_timeout(api_cfg)
        client = build_openai_client(
            api_cfg,
            config_path=resolve_api_config_path(API_CONFIG_PATH),
            client_factory=AsyncOpenAI,
            http_timeout=http_timeout,
        )
        request: dict[str, object] = {
            "model": api_cfg.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "timeout": http_timeout,
        }
        max_tokens = int(self._settings.deep_reasoning_max_tokens)
        if max_tokens > 0:
            request["max_tokens"] = max_tokens
        try:
            async with async_span_scope(
                "so_agent.model.call",
                {"task.type": "deep_reasoning", "model.name": api_cfg.model},
            ) as span:
                response = await client.chat.completions.create(**request)
                answer = (response.choices[0].message.content or "").strip()
                record_completion_usage(self._context, response, "deep_reasoning")
                set_span_attributes(span, {"answer.chars": len(answer)})
        except Exception as exc:  # 网络/服务异常统一转为失败终态
            message = f"深度推理调用失败：{type(exc).__name__}: {exc}"
            logger.warning(message)
            return OrchestratorOutput(
                task_id=task_id,
                status=TaskStatus.FAILED,
                final_result=message,
                review_summary=summary,
            )
        finally:
            await client.close()

        return OrchestratorOutput(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            final_result=answer or "（模型未返回内容）",
            review_summary=summary,
        )

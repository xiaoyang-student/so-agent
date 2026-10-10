"""直接回答路径（DirectAnswerPath）。

四象限中的 ① 简单+短：一次 LLM 调用直接回答，不经过规划 / 评审 /
工具调用，是四象限中成本最低的执行路径。

实现约定：
- API 配置复用主管 Agent 的 ``src/so_agent/api_config.yaml``（与
  orchestrator.py 相同的 ``load_agent_api_config`` 加载机制，密钥
  解析失败时抛 ``ConfigLoadError``，明确失败不静默降级）；
- 输出统一包装为 ``OrchestratorOutput``（定义于 ``so_agent.orchestrator``，
  models.py 中不存在该模型），与完整工作流引擎的终态输出格式兼容；
- 调用失败时返回 ``TaskStatus.FAILED`` 的终态输出（不向上抛异常），
  与 WorkflowEngine"始终返回终态输出"的行为保持一致。
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from openai import AsyncOpenAI

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import ClassificationResult, TaskStatus
from so_agent.orchestrator import OrchestratorOutput
from so_agent.runtime.config_loader import (
    load_agent_api_config,
    resolve_model_http_timeout,
)

# 复用主管 Agent 的 API 配置（src/so_agent/api_config.yaml）
API_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "api_config.yaml"


def _build_prompt(objective: str, constraints: str, acceptance_criteria: str) -> str:
    """构造简单 prompt：目标原文 +（可选）约束条件与验收标准。"""
    parts = [objective.strip()]
    if constraints.strip():
        parts.append(f"约束条件：{constraints.strip()}")
    if acceptance_criteria.strip():
        parts.append(f"验收标准：{acceptance_criteria.strip()}")
    return "\n\n".join(parts)


class DirectAnswerPath:
    """简单+短任务：一次 LLM 直接回答。"""

    SYSTEM_PROMPT = (
        "你是一个简洁高效的助手。请直接、准确地回答用户的问题，"
        "不要输出与问题无关的冗余内容；如有约束条件，必须严格遵守。"
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
        """一次 LLM 调用完成直接回答，并包装为 OrchestratorOutput。

        Args:
            objective: 任务目标（用户问题原文）。
            constraints: 约束条件文本（可为空）。
            acceptance_criteria: 验收标准文本（可为空）。
            classification: 路由器给出的分类结果（记录进输出摘要）。

        Returns:
            ``OrchestratorOutput``：成功时为 ``completed`` 终态，模型内容
            置于 ``final_result``；调用失败时为 ``failed`` 终态。
        """
        task_id = uuid.uuid4().hex
        prompt = _build_prompt(objective, constraints, acceptance_criteria)
        summary = "执行路径：direct_answer"
        if classification.reasoning:
            summary += f"；分类理由：{classification.reasoning}"

        print("[so-agent] 直接回答中...", file=sys.stderr, flush=True)

        api_cfg = load_agent_api_config(API_CONFIG_PATH, resolve_secrets=True)
        # HTTP 超时经 resolve_model_http_timeout 解析（任务 59，与
        # orchestrator / router 客户端一致）：分维度未配置时等价旧单一
        # float timeout；request 级 timeout 同步传同一解析结果，避免
        # float 覆盖客户端级分维度 Timeout（openai SDK 中 request 级
        # 参数优先于客户端默认值）。
        http_timeout = resolve_model_http_timeout(api_cfg)
        client = AsyncOpenAI(
            api_key=api_cfg.resolve_api_key(),
            base_url=api_cfg.base_url or None,
            timeout=http_timeout,
            max_retries=api_cfg.retry,
        )
        try:
            response = await client.chat.completions.create(
                model=api_cfg.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                timeout=http_timeout,
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as exc:  # 网络/服务异常统一转为失败终态
            message = f"直接回答调用失败：{type(exc).__name__}: {exc}"
            print(f"[so-agent] {message}", file=sys.stderr, flush=True)
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

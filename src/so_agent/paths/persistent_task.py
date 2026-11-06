"""路径 ③：简单+长任务 — 持久化状态 + 调度执行。

该路径处理"目标简单但持续时间长"的任务（如定时检查、周期监控）：
1. 用一次 LLM 调用把任务目标解析为 ``ScheduleSpec`` 调度规格；
2. 将任务持久化到 ``TaskStore``（SQLite），跨进程可恢复；
3. 通过 ``PeriodicTaskPoller.register`` 计算并写入首次执行时间；
4. 立即返回确认信息（任务本身由调度守护进程异步执行）。
"""

from __future__ import annotations

import json
from pathlib import Path

from openai import AsyncOpenAI

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.cron import CronExpressionError, parse_cron
from so_agent.models import (
    ClassificationResult,
    ScheduleSpec,
    TaskStatus,
    TaskType,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.persistence import TaskStore, resolve_task_store_path
from so_agent.runtime.config_loader import (
    load_agent_api_config,
    resolve_api_config_path,
    resolve_model_http_timeout,
)
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.model_gateway import build_openai_client
from so_agent.runtime.telemetry import async_span_scope, set_span_attributes
from so_agent.runtime.usage import record_completion_usage
from so_agent.task_scheduler import PeriodicTaskPoller

API_CONFIG_PATH = Path(__file__).resolve().parent.parent / "api_config.yaml"

logger = get_logger("path.persistent_task")

def _normalize_schedule(spec: ScheduleSpec, objective: str) -> ScheduleSpec:
    """校验并纠正 LLM 给出的调度规格。

    - ``cron`` 表达式非法 / ``interval_seconds`` 非正数时降级；
    - ``cron`` / ``interval`` 强制为周期任务，``once`` 强制为非周期；
    - ``event`` 不在此降级，由调用方明确拒绝（避免创建永不触发的任务）。
    """
    if spec.trigger == "interval" and (
        not spec.interval_seconds or spec.interval_seconds <= 0
    ):
        logger.warning("interval_seconds 非法，已降级为一次性任务。")
        return _fallback_schedule(objective)
    if spec.trigger == "cron":
        try:
            parse_cron(spec.cron or "")
        except CronExpressionError as exc:
            logger.warning("cron 表达式非法（%s），已降级为一次性任务。", exc)
            return _fallback_schedule(objective)
        return spec.model_copy(update={"is_recurring": True})
    if spec.trigger == "interval":
        return spec.model_copy(update={"is_recurring": True})
    return spec.model_copy(update={"is_recurring": False})


def _fallback_schedule(objective: str) -> ScheduleSpec:
    """构造降级用的一次性调度规格（trigger=once，不周期执行）。

    LLM 调用失败或输出解析失败时统一回退到此默认规格，
    保证任务仍可被持久化并调度执行，CLI 不崩溃。
    """
    return ScheduleSpec(
        trigger="once",
        action=objective,
        summary=f"一次性任务：{objective[:50]}",
        is_recurring=False,
    )


SCHEDULE_PARSER_PROMPT = """你是一个任务调度解析器。根据用户描述，解析出调度规格并返回 JSON。

输出格式（严格 JSON，不要输出其他内容）：
{
    "trigger": "once|cron|interval|event",
    "cron": "cron表达式（仅 trigger=cron 时）",
    "interval_seconds": 数字（仅 trigger=interval 时）,
    "event_source": "事件源描述（仅 trigger=event 时）",
    "action": "要执行的具体动作",
    "check_condition": "触发条件（可选）",
    "summary": "一句话描述调度计划",
    "is_recurring": true或false,
    "timezone": "IANA时区名（可选，仅当用户明确指定时，例如 Asia/Shanghai）"
}

示例：
- "每天检查目录有没有新CSV" → {"trigger":"cron","cron":"0 0 * * *","action":"检查目录并统计新CSV文件","summary":"每天零点检查目录中的新CSV文件","is_recurring":true}
- "每隔5分钟检测服务状态" → {"trigger":"interval","interval_seconds":300,"action":"检测服务健康状态","summary":"每5分钟检测一次服务状态","is_recurring":true}
- "每天北京时间上午9点汇总数据" → {"trigger":"cron","cron":"0 9 * * *","timezone":"Asia/Shanghai","action":"汇总数据","summary":"每天北京时间 9 点汇总","is_recurring":true}
"""


class PersistentTaskPath:
    """简单+长任务：持久化状态 + 调度执行。"""

    def __init__(self, context: ProjectContext, settings: Settings):
        self._context = context
        self._settings = settings

    async def run(
        self,
        objective: str,
        constraints: str,
        acceptance_criteria: str,
        classification: ClassificationResult,
    ) -> OrchestratorOutput:
        """解析调度规格 → 持久化任务 → 注册调度器 → 返回确认。"""

        # 1) LLM 解析调度规格（落库前校验，非法规格降级而非留死任务）
        parsed = await self._parse_schedule(objective)
        if parsed.trigger == "event":
            reason = (
                "未创建任务：event 触发依赖外部事件源，当前版本尚未实现。"
                "请改用 cron 或 interval 调度。"
            )
            logger.warning("event 触发未实现，拒绝创建持久化任务。")
            return self._build_output(
                "", reason, status=TaskStatus.FAILED
            )
        schedule_spec = _normalize_schedule(parsed, objective)

        # 2) 持久化任务
        store = TaskStore(resolve_task_store_path(self._context, self._settings))
        task_record = store.create_task(
            objective=objective,
            task_type=TaskType.PERSISTENT_TASK,
            constraints=constraints,
            acceptance_criteria=acceptance_criteria,
            schedule_spec=schedule_spec,
            status="scheduled",
        )

        # 3) 注册到调度器（计算首次执行时间）
        scheduler = PeriodicTaskPoller(store, self._context, self._settings)
        scheduler.register(task_record)

        store.add_log(
            task_record.task_id,
            "task_created",
            {"schedule": schedule_spec.model_dump()},
        )

        summary = (
            f"持久化任务已创建并进入调度队列。\n"
            f"- 任务ID：{task_record.task_id}\n"
            f"- 调度方式：{schedule_spec.summary or schedule_spec.trigger}\n"
            f"- 执行动作：{schedule_spec.action}\n"
            f"- 时区：{schedule_spec.timezone or 'UTC'}\n"
            f"- 是否周期：{'是' if schedule_spec.is_recurring else '否'}\n\n"
            f"提示：CLI 使用 `python -m so_agent.main --daemon` 启动调度守护"
            "进程；Web 控制台已内置调度器，无需额外启动。"
        )

        logger.info("持久化任务已创建：%s", task_record.task_id)
        return self._build_output(
            task_record.task_id, summary, status=TaskStatus.SCHEDULED
        )

    async def _parse_schedule(self, objective: str) -> ScheduleSpec:
        """用一次 LLM 调用解析任务目标为调度规格。

        失败语义（均为降级而非崩溃，与 direct_answer 路径一致）：
        - LLM 网络/服务异常：记录不含密钥的简短 stderr 信息，
          降级为一次性任务（once）；
        - 输出 JSON 解析失败：同样降级为一次性任务；
        - 客户端在 finally 中关闭，避免连接泄漏；
        - HTTP 超时经 ``resolve_model_http_timeout`` 解析（任务 59，
          与 orchestrator / router 客户端一致）：分维度未配置时等价旧
          单一 float timeout，非法配置在构造客户端前 fail fast。
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
        try:
            async with async_span_scope(
                "so_agent.schedule.parse",
                {"task.type": "persistent_task", "model.name": api_cfg.model},
            ) as span:
                response = await client.chat.completions.create(
                    model=api_cfg.model,
                    messages=[
                        {"role": "system", "content": SCHEDULE_PARSER_PROMPT},
                        {"role": "user", "content": objective},
                    ],
                    max_tokens=200,
                    timeout=http_timeout,
                )
                raw = (response.choices[0].message.content or "").strip()
                record_completion_usage(
                    self._context, response, "persistent_schedule_parser"
                )
                set_span_attributes(span, {"schedule.raw_chars": len(raw)})
        except Exception as exc:
            # 网络/服务异常：仅记录异常类型（不含任何请求细节与密钥），
            # 降级为一次性任务，不让 CLI 崩溃。
            logger.warning(
                "调度规格解析失败（%s），已降级为一次性任务。", type(exc).__name__
            )
            return _fallback_schedule(objective)
        finally:
            await client.close()

        # 去除 markdown 围栏
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )

        try:
            data = json.loads(raw)
            return ScheduleSpec(**data)
        except Exception:
            # JSON 解析失败时同样降级为一次性任务
            logger.warning("调度规格输出解析失败，已降级为一次性任务。")
            return _fallback_schedule(objective)

    def _build_output(
        self,
        task_id: str,
        summary: str,
        *,
        status: TaskStatus = TaskStatus.SCHEDULED,
    ) -> OrchestratorOutput:
        """构造 OrchestratorOutput（终态载体）。

        字段核对（见 so_agent.orchestrator.OrchestratorOutput）：
        - ``task_id`` / ``status`` 为必填；
        - ``final_result`` / ``plan_versions`` / ``agent_history`` /
          ``review_summary`` / ``human_escalation`` 均有默认值，
          本路径无子任务计划与 Agent 历史，直接使用默认空值。
        """
        return OrchestratorOutput(
            task_id=task_id,
            status=status,
            final_result=summary,
        )

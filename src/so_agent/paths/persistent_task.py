"""路径 ③：简单+长任务 — 持久化状态 + 调度执行。

该路径处理"目标简单但持续时间长"的任务（如定时检查、周期监控）：
1. 用一次 LLM 调用把任务目标解析为 ``ScheduleSpec`` 调度规格；
2. 将任务持久化到 ``TaskStore``（SQLite），跨进程可恢复；
3. 通过 ``PeriodicTaskPoller.register`` 计算并写入首次执行时间；
4. 立即返回确认信息（任务本身由调度守护进程异步执行）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

from so_agent.config import Settings
from so_agent.context import ProjectContext
from so_agent.models import (
    ClassificationResult,
    ScheduleSpec,
    TaskStatus,
    TaskType,
)
from so_agent.orchestrator import OrchestratorOutput
from so_agent.persistence import TaskStore
from so_agent.runtime.config_loader import (
    load_agent_api_config,
    resolve_model_http_timeout,
)
from so_agent.task_scheduler import PeriodicTaskPoller

API_CONFIG_PATH = Path(__file__).resolve().parent.parent / "api_config.yaml"

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
    "is_recurring": true或false
}

示例：
- "每天检查目录有没有新CSV" → {"trigger":"cron","cron":"0 0 * * *","action":"检查目录并统计新CSV文件","summary":"每天零点检查目录中的新CSV文件","is_recurring":true}
- "每隔5分钟检测服务状态" → {"trigger":"interval","interval_seconds":300,"action":"检测服务健康状态","summary":"每5分钟检测一次服务状态","is_recurring":true}
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

        # 1) LLM 解析调度规格
        schedule_spec = await self._parse_schedule(objective)

        # 2) 持久化任务
        store = TaskStore(
            Path(self._settings.task_store_path)
            if Path(self._settings.task_store_path).is_absolute()
            else self._context.sandbox_dir / self._settings.task_store_path
        )
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
            f"持久化任务已创建并调度。\n"
            f"- 任务ID：{task_record.task_id}\n"
            f"- 调度方式：{schedule_spec.summary or schedule_spec.trigger}\n"
            f"- 执行动作：{schedule_spec.action}\n"
            f"- 是否周期：{'是' if schedule_spec.is_recurring else '否'}\n\n"
            f"提示：使用 `python -m so_agent.main --daemon` 启动调度守护进程"
            "以执行已调度的任务。"
        )

        print(
            f"[so-agent] 持久化任务已创建：{task_record.task_id}",
            file=sys.stderr,
            flush=True,
        )
        return self._build_output(task_record.task_id, summary)

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
        api_cfg = load_agent_api_config(API_CONFIG_PATH, resolve_secrets=True)
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
                    {"role": "system", "content": SCHEDULE_PARSER_PROMPT},
                    {"role": "user", "content": objective},
                ],
                max_tokens=200,
                timeout=http_timeout,
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            # 网络/服务异常：仅记录异常类型（不含任何请求细节与密钥），
            # 降级为一次性任务，不让 CLI 崩溃。
            print(
                f"[so-agent] 调度规格解析失败（{type(exc).__name__}），"
                "已降级为一次性任务。",
                file=sys.stderr,
                flush=True,
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
            print(
                "[so-agent] 调度规格输出解析失败，已降级为一次性任务。",
                file=sys.stderr,
                flush=True,
            )
            return _fallback_schedule(objective)

    def _build_output(self, task_id: str, summary: str) -> OrchestratorOutput:
        """构造 OrchestratorOutput（终态载体）。

        字段核对（见 so_agent.orchestrator.OrchestratorOutput）：
        - ``task_id`` / ``status`` 为必填；
        - ``final_result`` / ``plan_versions`` / ``agent_history`` /
          ``review_summary`` / ``human_escalation`` 均有默认值，
          本路径无子任务计划与 Agent 历史，直接使用默认空值。
        """
        return OrchestratorOutput(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            final_result=summary,
        )

"""事件记录器（EventLogger）。

职责与行为约定：
- 以统一结构记录编排过程中的全部事件（任务、Agent、授权、替换、通信、
  人工升级等），首版仅追加写入内存列表，接口稳定后可平滑替换为持久化存储；
- 事件 payload 强制 JSON 兼容：Pydantic 模型、datetime、set/tuple 等对象
  会被自动转换为 JSON 可序列化形态后再写入，不可序列化的内容会被拒绝；
- 支持按 ``task_id`` / ``agent_id`` / ``event_type`` 过滤查询；
- 支持订阅回调（subscribe），供审计、监控与人工升级材料组装使用。

全局约束：所有 Agent 之间的通信必须严格使用 JSON；本模块是事件留痕入口，
同样遵循该约束，确保任何事件都可被序列化、审计与回放。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeAlias

from pydantic import BaseModel

from so_agent.context import ProjectContext
from so_agent.models import _utc_now
from so_agent.runtime.audit_trail import AuditTrail, get_default_audit_trail
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.redaction import redact_payload

logger = get_logger("runtime.events")

# ---------------------------------------------------------------------------
# 事件类型常量（全量清单；新增事件类型必须先在此登记）
# ---------------------------------------------------------------------------
EVENT_TASK_CREATED: str = "task_created"
EVENT_STATUS_TRANSITION: str = "status_transition"
EVENT_TASK_DISPATCHED: str = "task_dispatched"
EVENT_TOOL_GRANTED: str = "tool_granted"
EVENT_TOOL_REVOKED: str = "tool_revoked"
EVENT_TOOL_DENIED: str = "tool_denied"
EVENT_TOOL_CALLED: str = "tool_called"
EVENT_AGENT_CREATED: str = "agent_created"
EVENT_AGENT_RELEASED: str = "agent_released"
EVENT_REPLACEMENT_REQUESTED: str = "replacement_requested"
# 替代生命周期事件（任务 34）：由执行协调器在调度器真实时点通知的
# 受控回调中写入，与全局 FSM 迁移同锁串行化，保证时间线与实际执行
# 一致（requested → executing → completed / failed）。
EVENT_REPLACEMENT_EXECUTING: str = "replacement_executing"
EVENT_REPLACEMENT_COMPLETED: str = "replacement_completed"
EVENT_REPLACEMENT_FAILED: str = "replacement_failed"
EVENT_DECOMPOSITION_ISSUE: str = "decomposition_issue"
EVENT_REPLAN: str = "replan"
EVENT_HUMAN_ESCALATION: str = "human_escalation"
EVENT_MESSAGE_ROUTED: str = "message_routed"
EVENT_MESSAGE_REJECTED: str = "message_rejected"
EVENT_TOKEN_USAGE: str = "token_usage"
EVENT_REVIEW_PARSE_ERROR: str = "review_parse_error"
# 评审输出无法解析时的一次严格契约重试（attempt / max_attempts / error，
# 不携带完整原始输出）；耗尽后仍 fail-closed 并记 review_parse_error。
EVENT_REVIEW_PARSE_RETRY: str = "review_parse_retry"
EVENT_REVIEW_AGENT_OUTPUT: str = "review_agent_output"
EVENT_REVIEW_CONTRACT_REPAIR: str = "review_contract_repair"
EVENT_REVIEW_CONTRACT_ERROR: str = "review_contract_error"
# 流式超时审计（任务 33）：阶段空闲超时或 Agent 总预算超时发生时写入，
# 记录 phase / configured_timeout / elapsed / agent 或 subtask 上下文。
EVENT_TIMEOUT_AUDIT: str = "timeout_audit"
# 跨计划版本显式结果复用（任务 35）：规划协调器对主管的 reuse_from
# 声明做安全校验后写入。校验通过时 result_reused 记录新 subtask_id、
# 来源 subtask_id 与来源 plan_version；校验失败时 reuse_rejected 记录
# 机器可读的 reason 码（不携带完整输出），新任务回退正常 pending 队列。
EVENT_RESULT_REUSED: str = "result_reused"
EVENT_REUSE_REJECTED: str = "reuse_rejected"
# 确定性失败恢复内核（任务 46）：调度器在真实失败时点写入。
# - failure_classified：每次失败尝试的确定性分类（failure_type /
#   fingerprint / occurrences，不含业务产出）；
# - recovery_decided：策略判定结果（action / reason / backoff）；
# - retry_suppressed：同指纹连续失败达到抑制阈值、停止重试的审计点。
EVENT_FAILURE_CLASSIFIED: str = "failure_classified"
EVENT_RECOVERY_DECIDED: str = "recovery_decided"
EVENT_RETRY_SUPPRESSED: str = "retry_suppressed"
# 多 Agent 效率治理：规划阶段的确定性并行度决策与调度期熔断。
# - parallelism_decided：计划形状、闸门结论与并发上限（不含业务载荷）；
# - runtime_breaker_tripped：provider 压力或连续失败触发收敛为串行；
# - review_skipped：确定性条件满足而省略的评审调用（记录原因）。
EVENT_PARALLELISM_DECIDED: str = "parallelism_decided"
EVENT_RUNTIME_BREAKER_TRIPPED: str = "runtime_breaker_tripped"
EVENT_REVIEW_SKIPPED: str = "review_skipped"
# 计划结构确定性校验失败：重复/空 ID、依赖缺失/自依赖/成环、越界写入
# 路径等结构问题在进入模型评审前即被拦截，直接进入重新拆解。
EVENT_PLAN_VALIDATION_FAILED: str = "plan_validation_failed"
# 写入范围验收：子任务自述成功但声明的 write_scope 文件在沙箱中不存在，
# 由控制层确定性翻转为失败（记录缺失条目，不携带文件内容）。
EVENT_WRITE_SCOPE_UNMET: str = "write_scope_unmet"
# 代码 Agent 后置校验：模型自述成功但文件/执行结果未满足确定性条件，
# 在工具调用内部翻转为失败（记录动作与实际观测，不携带文件内容）。
EVENT_CODE_AGENT_POSTCONDITION_FAILED: str = "code_agent_postcondition_failed"
# 子任务工具证据缺失：声明产出文件/产物的子任务本次尝试没有任何完成
# 的工具调用，由控制层确定性翻转为失败。
EVENT_TOOL_EVIDENCE_MISSING: str = "tool_evidence_missing"
# 可机检验收失败：规划器声明的 verification 规则未全部通过。
EVENT_VERIFICATION_FAILED: str = "verification_failed"
# 验收规则 lint：采纳计划时丢弃无法客观判定或结构不完整的规则（如
# 无 path 的 text_contains），记录规则摘要与原因，不改变计划其余部分。
EVENT_VERIFICATION_RULE_REJECTED: str = "verification_rule_rejected"
# 计划工具范围一致性：instructions 要求执行代码但 allowed_tools 未包含
# run_in_sandbox 时，在采纳阶段补齐并留痕（只记录补齐的工具与命中标记）。
EVENT_PLAN_TOOL_SCOPE_REPAIRED: str = "plan_tool_scope_repaired"
# 计划内步骤合并：把“产生产物 → 基于产物写报告”的单依赖两步合并为
# 一个子任务（记录合并后的 id、被并入的 id 与上游产物清单）。
EVENT_PLAN_STEPS_MERGED: str = "plan_steps_merged"
# 同族任务脚本复用：action=recorded 记录成功产物；action=staged 表示
# 已把校验通过的缓存脚本复制进沙箱；action=rejected 表示哈希/路径校验失败。
EVENT_SCRIPT_REUSED: str = "script_reused"
# 动作级修复：子任务自述成功但被确定性验收翻转后，在同一尝试内发起
# 一次定向修复（记录首次失败原因与修复结果，不携带业务产出全文）。
EVENT_ACTION_REPAIR: str = "action_repair"
# 异步 provisional 交付与纠正闭环（默认关闭）：
# - provisional_delivered：低风险结果先行交付、后台评审开始；
# - provisional_review_passed：后台评审通过，任务转为 completed；
# - provisional_review_failed：后台评审未通过（随后自动返工或升级）；
# - provisional_revised：返工完成且结果已修订（旧版本保留在输出中）；
# - provisional_escalated：后台评审返工耗尽，转人工兜底。
EVENT_PROVISIONAL_DELIVERED: str = "provisional_delivered"
EVENT_PROVISIONAL_REVIEW_PASSED: str = "provisional_review_passed"
EVENT_PROVISIONAL_REVIEW_FAILED: str = "provisional_review_failed"
EVENT_PROVISIONAL_REVISED: str = "provisional_revised"
EVENT_PROVISIONAL_ESCALATED: str = "provisional_escalated"
# 主管调用载荷审计：记录阶段、字符数与估算 token（不记录载荷内容）。
EVENT_CONTEXT_PAYLOAD_AUDITED: str = "context_payload_audited"
# Token 预算熔断（默认关闭）：启用后任务级 / Agent 级用量超过配置上限
# 时写入，记录 scope / limit / observed（同一 scope 只记一次）。
EVENT_TOKEN_BUDGET_EXCEEDED: str = "token_budget_exceeded"
# 人工指导闭环：人工提交指导并恢复执行时写入（action / 回退目标 /
# 受影响子任务 / 是否补充信息；不记录指导全文）。
EVENT_HUMAN_GUIDANCE_APPLIED: str = "human_guidance_applied"
# 人工升级外部通知：配置 Webhook 后投递升级材料包的结果（成功/失败）。
# 失败只记录，不影响任务进入 human_required。
EVENT_HUMAN_ESCALATION_NOTIFIED: str = "human_escalation_notified"
EVENT_HUMAN_ESCALATION_NOTIFY_FAILED: str = "human_escalation_notify_failed"
# 确定性上下文压缩：超过预算的载荷被头尾保留并打截断标记。
EVENT_CONTEXT_COMPRESSED: str = "context_compressed"
# code_agent 模型输入裁剪：旧工具调用参数/输出被替换为回执（只记录
# 前后字符数与条数，不记录载荷内容）。
EVENT_CONTEXT_PRUNED: str = "context_pruned"
# 同文件重复读取折叠：同一运行内已读窗口再次被读取时只返回回执。
# 记录路径、窗口与被省略的正文长度，不记录文件内容。
EVENT_READ_COLLAPSED: str = "read_collapsed"
# 重复写/执行折叠：同一运行内相同路径同内容写入、或相同代码且工作区
# 未变化时重复执行，返回回执而不重复产生副作用（记录工具与目标，
# 不记录代码或文件内容）。
EVENT_ACTION_REPLAYED: str = "action_replayed"
# code_agent 回合续期：达到回合上限但本轮已有成功工具调用时，在同一
# 工具调用内放宽回合上限继续（记录前后上限、续期次数与成功工具数）。
EVENT_TURN_BUDGET_EXTENDED: str = "turn_budget_extended"
# 断点续跑：人工兜底时保存检查点，CLI 恢复时读取并继续。
EVENT_CHECKPOINT_SAVED: str = "checkpoint_saved"
EVENT_CHECKPOINT_RESUMED: str = "checkpoint_resumed"
# 单文件轻量快通道：记录判定与结果（hit / completed / fallback），
# 只含目标路径、原因与正文长度，不记录文件内容。
EVENT_FAST_PATH: str = "fast_path"

ALL_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_TASK_CREATED,
        EVENT_STATUS_TRANSITION,
        EVENT_TASK_DISPATCHED,
        EVENT_TOOL_GRANTED,
        EVENT_TOOL_REVOKED,
        EVENT_TOOL_DENIED,
        EVENT_TOOL_CALLED,
        EVENT_AGENT_CREATED,
        EVENT_AGENT_RELEASED,
        EVENT_REPLACEMENT_REQUESTED,
        EVENT_REPLACEMENT_EXECUTING,
        EVENT_REPLACEMENT_COMPLETED,
        EVENT_REPLACEMENT_FAILED,
        EVENT_DECOMPOSITION_ISSUE,
        EVENT_REPLAN,
        EVENT_HUMAN_ESCALATION,
        EVENT_MESSAGE_ROUTED,
        EVENT_MESSAGE_REJECTED,
        EVENT_TOKEN_USAGE,
        EVENT_REVIEW_PARSE_ERROR,
        EVENT_REVIEW_PARSE_RETRY,
        EVENT_REVIEW_AGENT_OUTPUT,
        EVENT_REVIEW_CONTRACT_REPAIR,
        EVENT_REVIEW_CONTRACT_ERROR,
        EVENT_TIMEOUT_AUDIT,
        EVENT_RESULT_REUSED,
        EVENT_REUSE_REJECTED,
        EVENT_FAILURE_CLASSIFIED,
        EVENT_RECOVERY_DECIDED,
        EVENT_RETRY_SUPPRESSED,
        EVENT_PARALLELISM_DECIDED,
        EVENT_RUNTIME_BREAKER_TRIPPED,
        EVENT_REVIEW_SKIPPED,
        EVENT_PLAN_VALIDATION_FAILED,
        EVENT_WRITE_SCOPE_UNMET,
        EVENT_CODE_AGENT_POSTCONDITION_FAILED,
        EVENT_TOOL_EVIDENCE_MISSING,
        EVENT_VERIFICATION_FAILED,
        EVENT_VERIFICATION_RULE_REJECTED,
        EVENT_PLAN_TOOL_SCOPE_REPAIRED,
        EVENT_PLAN_STEPS_MERGED,
        EVENT_SCRIPT_REUSED,
        EVENT_ACTION_REPAIR,
        EVENT_PROVISIONAL_DELIVERED,
        EVENT_PROVISIONAL_REVIEW_PASSED,
        EVENT_PROVISIONAL_REVIEW_FAILED,
        EVENT_PROVISIONAL_REVISED,
        EVENT_PROVISIONAL_ESCALATED,
        EVENT_CONTEXT_PAYLOAD_AUDITED,
        EVENT_TOKEN_BUDGET_EXCEEDED,
        EVENT_HUMAN_GUIDANCE_APPLIED,
        EVENT_HUMAN_ESCALATION_NOTIFIED,
        EVENT_HUMAN_ESCALATION_NOTIFY_FAILED,
        EVENT_CONTEXT_COMPRESSED,
        EVENT_CONTEXT_PRUNED,
        EVENT_READ_COLLAPSED,
        EVENT_ACTION_REPLAYED,
        EVENT_TURN_BUDGET_EXTENDED,
        EVENT_CHECKPOINT_SAVED,
        EVENT_CHECKPOINT_RESUMED,
        EVENT_FAST_PATH,
    }
)

# 事件类型名与事件回调签名
EventType: TypeAlias = str
EventCallback: TypeAlias = Callable[[dict[str, Any]], None]


def _json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    """将常见对象递归转换为 JSON 兼容形态。

    - Pydantic 模型 → ``model_dump(mode="json")`` 结果；
    - datetime → ISO 8601 字符串；
    - dict / list / tuple / set → 同构容器；
    - 其余值原样返回（是否合法由 ``json.dumps`` 最终校验）。

    对自引用容器做循环检测（``_seen`` 记录在途容器 id），命中时以
    ``"<circular-reference>"`` 占位，避免无限递归导致 ``RecursionError``。
    """
    if _seen is None:
        _seen = set()
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"), _seen)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        marker = id(value)
        if marker in _seen:
            return "<circular-reference>"
        _seen.add(marker)
        try:
            if isinstance(value, dict):
                return {
                    str(key): _json_safe(item, _seen)
                    for key, item in value.items()
                }
            return [_json_safe(item, _seen) for item in value]
        finally:
            _seen.discard(marker)
    return value


class EventLogger:
    """编排过程事件记录器（内存实现）。"""

    def __init__(
        self,
        *,
        context: ProjectContext | None = None,
        max_events: int | None = None,
        log_file: str | Path | None = None,
        audit_trail: AuditTrail | None = None,
    ) -> None:
        """初始化记录器。

        Args:
            context: 可选的共享上下文；提供时事件会同步追加到
                ``context.event_log``，便于编排器与人工升级材料统一取用。
            max_events: 事件保留上限（None 表示不限制）；内部列表与
                ``context.event_log`` 镜像同步裁剪，避免任一列表无界增长。
            log_file: 可选的日志文件路径；提供时每条事件实时追加写入该文件
                （JSON 行格式，含时间戳），父目录自动创建；None 表示不落文件。
            audit_trail: 可选的跨任务审计链；None 时按
                ``SO_AGENT_AUDIT_TRAIL_PATH`` / 配置惰性解析，未配置即禁用。
        """
        self._context = context
        self._max_events = max_events
        self._events: list[dict[str, Any]] = []
        self._subscribers: list[EventCallback] = []
        self._log_file: Path | None = Path(log_file) if log_file is not None else None
        if self._log_file is not None:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
        self._audit_trail = audit_trail
        self._audit_trail_resolved = audit_trail is not None

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------
    def log_event(
        self,
        event_type: str,
        payload: dict[str, Any] | BaseModel | None = None,
        timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """记录一条事件并返回事件记录（JSON 兼容 dict）。

        Args:
            event_type: 事件类型，必须是 ``ALL_EVENT_TYPES`` 中已登记的类型。
            payload: 事件内容；支持 dict / Pydantic 模型 / None。写入前统一
                转换为 JSON 兼容形态；顶层 ``task_id`` / ``agent_id`` 会被
                提取为索引字段，供过滤查询使用。
            timestamp: 事件时间；默认当前 UTC 时间；naive 时间按 UTC 处理。

        Raises:
            ValueError: 事件类型未登记，或 payload 无法序列化为 JSON 时。
            TypeError: payload 类型不受支持时。
        """
        if not isinstance(event_type, str) or event_type not in ALL_EVENT_TYPES:
            raise ValueError(
                f"未登记的事件类型：{event_type!r}；"
                f"合法事件类型见 ALL_EVENT_TYPES：{sorted(ALL_EVENT_TYPES)}"
            )

        if isinstance(payload, BaseModel):
            data = payload.model_dump(mode="json")
        elif payload is None:
            data = {}
        elif isinstance(payload, dict):
            data = payload
        else:
            raise TypeError(
                "payload 必须是 dict / Pydantic 模型 / None，"
                f"实际为 {type(payload).__name__}"
            )
        data = redact_payload(_json_safe(data))
        try:
            json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"事件 payload 无法序列化为 JSON：{exc}") from exc

        ts = timestamp if timestamp is not None else _utc_now()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        record: dict[str, Any] = {
            "event_id": uuid.uuid4().hex[:12],
            "event_type": event_type,
            "timestamp": ts.isoformat(),
            "task_id": data.get("task_id"),
            "agent_id": data.get("agent_id"),
            "payload": data,
        }

        self._events.append(record)
        if self._context is not None:
            self._context.event_log.append(record)
        if self._max_events is not None:
            # 内部列表与 context 镜像同步裁剪，避免任一列表无界增长
            if len(self._events) > self._max_events:
                del self._events[: len(self._events) - self._max_events]
            if (
                self._context is not None
                and len(self._context.event_log) > self._max_events
            ):
                del self._context.event_log[
                    : len(self._context.event_log) - self._max_events
                ]

        for callback in list(self._subscribers):
            try:
                callback(record)
            except Exception:  # 订阅者异常不得影响主流程
                continue
        self._append_file(record)
        self._append_audit(record)
        return record

    def _append_audit(self, record: dict[str, Any]) -> None:
        """把一条事件追加到跨任务审计链；失败不影响主流程。"""
        trail = self._audit_trail
        if not self._audit_trail_resolved:
            self._audit_trail_resolved = True
            trail = get_default_audit_trail()
            self._audit_trail = trail
        if trail is None:
            return
        try:
            trail.append(record)
        except Exception:
            logger.warning("审计链写入失败（不影响任务）", exc_info=True)

    def _append_file(self, record: dict[str, Any]) -> None:
        """把一条事件实时追加写入日志文件（JSON 行格式）；失败不影响主流程。"""
        if self._log_file is None:
            return
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            with open(self._log_file, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:  # 日志写入失败不得影响主流程
            return

    # ------------------------------------------------------------------
    # 查询与订阅
    # ------------------------------------------------------------------
    def get_events(
        self,
        task_id: str | None = None,
        agent_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """按条件过滤事件（未提供的条件不参与过滤），返回浅拷贝列表。"""
        matched: list[dict[str, Any]] = []
        for record in self._events:
            if task_id is not None and record.get("task_id") != task_id:
                continue
            if agent_id is not None and record.get("agent_id") != agent_id:
                continue
            if event_type is not None and record.get("event_type") != event_type:
                continue
            matched.append(dict(record))
        return matched

    def subscribe(self, callback: EventCallback) -> None:
        """注册事件回调（重复注册自动忽略）；回调异常不影响主流程。"""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: EventCallback) -> None:
        """移除已注册的事件回调（不存在时静默忽略）。"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def clear(self) -> None:
        """清空内部事件列表（不影响 ``context.event_log`` 中已同步记录）。"""
        self._events.clear()

    @property
    def events(self) -> list[dict[str, Any]]:
        """返回内部事件列表的浅拷贝。"""
        return list(self._events)

    def __len__(self) -> int:
        """返回当前已记录的事件数量。"""
        return len(self._events)


def shared_event_logger(context: ProjectContext | None) -> EventLogger:
    """返回与共享上下文绑定的同一个事件记录器（供工具内部复用）。

    MCP 工具内部的 Agent 若各自新建 ``EventLogger``，事件只会进入内存
    与 ``context.event_log``，不会写入编排器的事件文件，造成 Agent
    生命周期与嵌套 token 用量在审计日志中缺失。本函数优先复用
    ``context.event_logger``（编排器装配时带 log_file 的实例），缺失时
    新建并回填，保证全链路事件写入同一通道。
    """
    if context is not None:
        existing = getattr(context, "event_logger", None)
        if isinstance(existing, EventLogger):
            return existing
    logger = EventLogger(context=context)
    if context is not None:
        try:
            context.event_logger = logger
        except Exception:
            pass
    return logger

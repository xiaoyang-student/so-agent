"""主管 / 评审 Agent 调用与任务级预算（WorkflowEngine 混入）。

``Runner`` 在调用时经 :func:`_workflow_runner` 从 ``so_agent.workflow``
模块读取，保持测试对 ``workflow_mod.Runner`` 的模块级替换语义。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from so_agent.models import ReviewDecision
from so_agent.orchestrator import supervisor_phase_scope
from so_agent.runtime.budget import TokenBudgetLimits, TokenBudgetTracker
from so_agent.runtime.events import (
    EVENT_CONTEXT_PAYLOAD_AUDITED,
    EVENT_REVIEW_AGENT_OUTPUT,
    EVENT_REVIEW_CONTRACT_ERROR,
    EVENT_REVIEW_CONTRACT_REPAIR,
    EVENT_REVIEW_PARSE_ERROR,
    EVENT_REVIEW_PARSE_RETRY,
)
from so_agent.runtime.usage import log_model_usage
from so_agent.state_machine import (
    PipelineDeadlineExceeded,
    PipelineStoppedError,
    ReviewContractError,
    WorkflowError,
    _StructuredOutputT,
    _coerce_structured_output,
    _review_contract_contradiction,
)
from so_agent.tool_packages.review_agent.agent import (
    ReviewInput,
    ReviewStage,
    build_deterministic_findings,
)
from so_agent.workflow_support import (
    estimate_tokens,
    make_tool_timeout_resolver,
    prune_empty,
    run_agent_streamed,
)

# 契约修复调用的严格字段模板（任务 32）：随修复指令下发给评审 Agent，
# 要求只修复字段一致性（passed 与 required_fixes/suggestions 的组合），
# 禁止重新评审或改变评审事实
_REVIEW_CONTRACT_TEMPLATE: dict[str, Any] = {
    "stage": "plan_review 或 result_review（与本次评审阶段一致）",
    "passed": True,
    "issues": [],
    "suggestions": [],
    "required_fixes": [],
    "retry_target": None,
    "summary": "",
}

# 契约修复消息中携带的原始输出截断上限（避免把完整敏感载荷无限写日志
# /无限回传模型；正常评审结论远小于该阈值，仅防御异常超长输出）
_REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS: int = 4000


def _workflow_runner() -> Any:
    """读取 workflow 模块当前的 Runner（保持模块级 monkeypatch 语义）。"""
    from so_agent import workflow as workflow_module

    return workflow_module.Runner


def _clip_text(text: str, limit: int) -> str:
    """按字符预算截断单条文本（<=0 不限制）。"""
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}…（已截断）"


def _clip_items(items: list[str], max_items: int, max_chars: int) -> list[str]:
    """按条数与单条字符预算截断字符串列表（不改变是否为空）。"""
    if max_items <= 0:
        selected = list(items)
        omitted = 0
    else:
        selected = list(items[:max_items])
        omitted = max(len(items) - max_items, 0)
    clipped = [_clip_text(item, max_chars) for item in selected]
    if omitted:
        clipped.append(f"…（另有 {omitted} 条已省略）")
    return clipped


def _trim_review_decision(decision: Any, settings: Any) -> Any:
    """按配置裁剪评审结论的叙述性字段（只改文本长度，不改 passed 等事实）。

    评审输出会进入计划版本、事件与后续载荷；条目过多或单条过长只增加
    生成时间与下游上下文，对放行判断没有增益。
    """
    decision.issues = _clip_items(
        decision.issues,
        int(getattr(settings, "review_issues_max_items", 0) or 0),
        int(getattr(settings, "review_item_max_chars", 0) or 0),
    )
    decision.suggestions = _clip_items(
        decision.suggestions,
        int(getattr(settings, "review_suggestions_max_items", 0) or 0),
        int(getattr(settings, "review_item_max_chars", 0) or 0),
    )
    decision.required_fixes = _clip_items(
        decision.required_fixes,
        int(getattr(settings, "review_required_fixes_max_items", 0) or 0),
        int(getattr(settings, "review_item_max_chars", 0) or 0),
    )
    decision.summary = _clip_text(
        decision.summary,
        int(getattr(settings, "review_summary_max_chars", 0) or 0),
    )
    return decision


class SupervisorMixin:
    def _activate_token_budget(self) -> None:
        """创建任务级预算追踪器并挂到共享上下文（供嵌套 Agent 累计）。"""
        tracker = TokenBudgetTracker(
            TokenBudgetLimits.from_settings(self._settings),
            events=self._events,
            task_id=self._task_request.task_id if self._task_request else None,
        )
        self._token_budget = tracker
        self._context.token_budget = tracker

    def _record_model_usage(self, result: Any, agent_label: str) -> None:
        """记录一次模型调用的 token 用量，并在安全检查点执行熔断。

        默认 ``token_budget_enabled=True``：超限抛出 ``TokenBudgetExceeded``
        （BaseException），由 ``run()`` 映射为 failed 终态且不触发重试；
        显式关闭时只写用量事件与累计快照，行为与改造前一致。
        """
        log_model_usage(
            self._events, result, agent_label, budget=self._token_budget
        )
        self._enforce_token_budget()

    def _enforce_token_budget(self) -> None:
        """控制层统一预算检查点：任务总量与全部 Agent 累计一起检查。"""
        if self._token_budget is not None:
            self._token_budget.enforce()

    def _audit_supervisor_payload(
        self, payload: dict[str, Any], message: str
    ) -> None:
        """记录主管载荷体量与阶段归因（只记体积，不记载荷内容）。"""
        if not getattr(self._settings, "context_payload_audit", True):
            return
        key_chars: dict[str, int] = {}
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_chars[str(key)] = len(
                    json.dumps(value, ensure_ascii=False, default=str)
                )
        top_keys = sorted(key_chars, key=key_chars.get, reverse=True)[:10]
        self._events.log_event(
            EVENT_CONTEXT_PAYLOAD_AUDITED,
            {
                "task_id": (
                    self._task_request.task_id
                    if self._task_request is not None
                    else None
                ),
                "phase": (
                    payload.get("phase") if isinstance(payload, dict) else None
                ),
                "chars": len(message),
                "estimated_tokens": estimate_tokens(message),
                "key_chars": {key: key_chars[key] for key in top_keys},
            },
        )

    def _supervisor_budget(self) -> float:
        """主管单次调用的绝对总预算（秒）；0 表示不设总预算。

        - 基准取 ``supervisor_total_timeout``；<=0（默认）表示不设
          总预算，正常长调用可以持续任意时长；
        - pipeline 显式开启绝对 deadline 时（``max_pipeline_timeout>0``），
          才与剩余预算取较小值；未开启时剩余为 None，不做截断；
        - 直接调用 ``_run_supervisor``（未启动流水线）时使用完整基准，
          保持既有测试与降级路径语义。
        """
        budget = float(self._settings.supervisor_total_timeout)
        remaining = self._gate.remaining()
        if remaining is None:
            return budget
        if budget <= 0:
            return remaining
        return min(budget, remaining)

    async def _run_supervisor(
        self, payload: dict[str, Any], output_model: type[_StructuredOutputT]
    ) -> _StructuredOutputT:
        """调用主管 Agent：JSON 输入 → 按阶段契约手动解析的结构化输出。

        主管 Agent 以纯文本返回（output_type=None，兼容忽略
        response_format 的千问等模型服务）：此处剥离 Markdown 代码围栏
        并按 ``output_model`` 阶段性精简契约（PlanningOutput /
        ExecutionOutput / AggregationOutput）手动校验，每个阶段只
        要求当前需要的字段，大幅降低输出 token 与耗时；对模型实例 /
        字典 / JSON 文本形态做宽容解析。

        Args:
            payload: 含 ``phase`` 阶段标记的 JSON 输入载荷。
            output_model: 该阶段的输出契约模型。

        Raises:
            WorkflowError: 输出无法解析为 output_model 时。
        """
        if isinstance(payload, dict):
            payload = prune_empty(payload)
        message = json.dumps(payload, ensure_ascii=False, default=str)
        self._audit_supervisor_payload(payload, message)
        # 分阶段存活检测（任务 80）：首事件窗口（覆盖 SDK 冷启动与模型
        # 首包等待）/ 模型空闲窗口 / 工具执行窗口；默认不设单次调用
        # 总预算，合法长调用不会被固定时长截断。工具阶段中的嵌套子
        # Agent / code_agent 默认不设父级硬截止，由子 Agent 自身的
        # 存活检测与回合上限负责；任一空闲超时先取消底层流再抛
        # asyncio.TimeoutError（消息携带 phase），由调用方按各自策略
        # 优雅降级；超时前写入 timeout_audit 审计事件；runner 由
        # _workflow_runner() 读取，保持 workflow_mod.Runner 的模块级
        # monkeypatch 语义。
        supervisor_budget = self._supervisor_budget()
        # pipeline 显式开启绝对 deadline 时的剩余预算；未开启为 None
        base_budget = float(self._settings.supervisor_total_timeout)
        remaining = self._gate.remaining()
        deadline_clipped = remaining is not None and (
            base_budget <= 0 or remaining < base_budget
        )
        # 停止门预检：停止后拒绝新调用（穿透宽捕获）；显式 deadline
        # 已耗尽时不调用模型、不进入流式等待，直接抛出可审计的预算
        # 超时；未开启绝对 deadline 时 supervisor_budget=0 表示无限，
        # 不再是“预算不足”。
        self._gate.check("supervisor_call")
        if remaining is not None and remaining <= 0.0:
            raise PipelineDeadlineExceeded(
                f"pipeline 绝对 deadline 已耗尽（剩余 {remaining:g}s），"
                "跳过本次主管调用"
            )
        gate_generation_before = self._gate.generation
        task_request = self._task_request
        phase = payload.get("phase") if isinstance(payload, dict) else None
        # 阶段注入：工具桥据此在 planning / replanning 阶段确定性拒绝
        # 主管的固定工具入口（提示词约束之外的代码层兜底）
        with supervisor_phase_scope(phase if isinstance(phase, str) else None):
            try:
                result = await run_agent_streamed(
                    self._orchestrator,
                    input=message,
                    max_turns=int(self._settings.max_model_turns),
                    first_event_timeout=float(self._settings.first_event_timeout),
                    model_idle_timeout=float(self._settings.model_idle_timeout),
                    tool_execution_timeout=float(
                        self._settings.tool_execution_timeout
                    ),
                    subagent_tool_execution_timeout=float(
                        self._settings.subagent_tool_execution_timeout
                    ),
                    agent_total_timeout=supervisor_budget,
                    runner=_workflow_runner(),
                    events=self._events,
                    # 工具阶段窗口按实际 tool_name 的超时 profile 解析
                    #（任务 45：code_agent / subagent / sandbox / standard），
                    # 总预算仍被 agent_total 与 pipeline deadline 截断
                    tool_timeout_resolver=make_tool_timeout_resolver(
                        self._settings, self._tool_registry
                    ),
                    # 流事件即真实活动：持续产出时续时 pipeline 看门狗
                    activity_hook=self._gate.record_progress,
                    audit_context={
                        "agent": "supervisor",
                        "pipeline_phase": phase,
                        "subtask_id": (
                            (payload.get("subtask") or {}).get("subtask_id")
                            if isinstance(payload, dict)
                            and isinstance(payload.get("subtask"), dict)
                            else None
                        ),
                        "attempt": (
                            payload.get("attempt")
                            if isinstance(payload, dict)
                            else None
                        ),
                        "task_id": (
                            task_request.task_id
                            if task_request is not None
                            else None
                        ),
                    },
                )
            except asyncio.TimeoutError as exc:
                if deadline_clipped:
                    raise PipelineDeadlineExceeded(
                        f"pipeline 绝对 deadline 耗尽："
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                raise
        # 迟到结果隔离（任务 45）：调用返回时门已停止或代数变化（等待
        # 期间发生取消 / deadline 到期 / 人工终止），本结果视为迟到——
        # 不写 WorkflowState、不记用量、不触发重试或授权签发，抛出
        # 穿透宽捕获的停止信号由 run() 映射为 cancelled 终态
        if (
            self._gate.stopping
            or self._gate.generation != gate_generation_before
        ):
            raise PipelineStoppedError(
                f"主管调用返回时 pipeline 已停止"
                f"（{self._gate.stop_reason}），结果按迟到隔离"
            )
        self._record_model_usage(result, "supervisor")
        self._gate.record_progress(
            f"supervisor:{phase}" if isinstance(phase, str) else "supervisor"
        )
        return _coerce_structured_output(
            result.final_output, output_model, label="主管 Agent"
        )

    async def _call_review_agent(self, review_input: Any) -> ReviewDecision:
        """调用评审 Agent：JSON 输入 → 契约校验后的 ReviewDecision。

        任务 32 确定化契约：评审放行只依据解析出的合法 ReviewDecision
        的 ``passed`` 布尔字段，绝不从评审文本语气或"通过/未通过"等
        关键词推断通过与否；结构矛盾先修复契约，无法修复时 fail closed。

        处理链路（全部 fail closed，宁可拦截不可放行）：

        1. 调用 + 宽容解析（``_coerce_review_decision``）：提取出的
           JSON/对象缺失显式可解析 ``passed`` 字段时抛
           ``ReviewContractError`` → 保守未通过 +
           ``review_contract_error`` 事件；完全无法解析
           （``WorkflowError``）时按 ``review_parse_retry_max_attempts``
           做严格契约重试（``review_parse_retry`` 事件），耗尽后保守
           未通过 + ``review_parse_error`` 事件；调用超时不重试，直接
           fail closed；其余意外异常向上传播不吞掉；
        2. 解析成功后执行字段契约校验（``_review_contract_contradiction``）：
           ``passed=true`` 时 ``required_fixes`` 必须为空、``passed=false``
           时必须非空（``suggestions`` 非空不阻断）；
        3. 契约矛盾时仅进行一次契约修复调用（只修字段一致性，禁止重新
           评审/改变评审事实）；修复结果再次解析与校验；
        4. 修复后仍矛盾、缺失显式 passed 或解析失败 → fail closed 的
           ReviewDecision（``required_fixes`` 写明评审契约无效）+
           ``review_contract_error`` 事件。

        ``review_agent_output`` 事件保留 stage / passed / raw_output_preview
        并新增 ``contract_valid``；原始输出预览截断至 1000 字符。``stage``
        统一归一化为 plan_review / result_review（授权评审在结论中映射为
        plan_review）。
        """
        expected_stage: str = (
            "result_review"
            if review_input.stage == ReviewStage.RESULT_REVIEW
            else "plan_review"
        )
        # 确定性检查前置：控制层先算出结构 / 授权 / 结果检查结果，随输入
        # 下发；评审 Agent 因此无需调用校验工具，一轮模型调用即可结论。
        if (
            isinstance(review_input, ReviewInput)
            and bool(
                getattr(self._settings, "review_precomputed_findings", True)
            )
            and not review_input.deterministic_findings
        ):
            review_input.deterministic_findings = build_deterministic_findings(
                review_input, self._context.sandbox_dir
            )
        result: Any = None
        message = review_input.model_dump_json()
        retry_limit = max(
            int(getattr(self._settings, "review_parse_retry_max_attempts", 0) or 0),
            0,
        )
        retry_attempt = 0
        while True:
            try:
                result = await _workflow_runner().run(
                    self._review_agent,
                    input=message,
                    max_turns=int(self._settings.max_model_turns),
                )
                self._record_model_usage(result, "review_agent")
                decision = self._coerce_review_decision(result.final_output)
                break
            except ReviewContractError as exc:
                # 输出缺失显式可解析 passed 字段：无法构造合法评审决定，
                # fail closed（不猜测结论），契约错误事件留痕
                raw_output = str(result.final_output) if result else "N/A"
                self._events.log_event(
                    EVENT_REVIEW_CONTRACT_ERROR,
                    {
                        "stage": expected_stage,
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_output_preview": raw_output[:500],
                    },
                )
                decision = ReviewDecision(
                    stage=expected_stage,
                    passed=False,
                    issues=[f"评审契约无效：{exc}"],
                    required_fixes=[
                        "评审契约无效：输出缺失显式 passed 字段，"
                        "须修复评审输出契约后重新提交评审"
                    ],
                    summary="评审契约无效（缺失显式 passed 字段），保守判定为未通过",
                )
                self._log_review_agent_output(
                    expected_stage, decision, raw_output, contract_valid=False
                )
                return decision
            except (WorkflowError, asyncio.TimeoutError) as exc:
                # 仅对解析失败做一次严格契约重试；调用超时不重试，
                # 避免把单次评审耗时放大为多倍。
                if isinstance(exc, WorkflowError) and retry_attempt < retry_limit:
                    retry_attempt += 1
                    raw_output = (
                        str(result.final_output) if result is not None else ""
                    )
                    self._events.log_event(
                        EVENT_REVIEW_PARSE_RETRY,
                        {
                            "stage": expected_stage,
                            "attempt": retry_attempt,
                            "max_attempts": retry_limit,
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw_output_preview": raw_output[:500] or "N/A",
                        },
                    )
                    message = self._build_review_parse_retry_message(
                        review_input, raw_output, exc
                    )
                    continue
                self._events.log_event(
                    EVENT_REVIEW_PARSE_ERROR,
                    {
                        "stage": expected_stage,
                        "error": f"{type(exc).__name__}: {exc}",
                        "retry_attempts": retry_attempt,
                        "raw_output_preview": (
                            str(result.final_output)[:500] if result else "N/A"
                        ),
                    },
                )
                return ReviewDecision(
                    stage=expected_stage,
                    passed=False,
                    issues=[
                        f"评审 Agent 调用/解析失败：{type(exc).__name__}: {exc}"
                    ],
                    required_fixes=["修复评审链路后重新提交评审"],
                    summary="评审不可用，保守判定为未通过",
                )

        # 解析成功后执行字段契约校验；矛盾时仅一次契约修复调用
        contradiction = _review_contract_contradiction(decision)
        contract_valid = contradiction is None
        if contradiction is not None:
            decision, contract_valid = await self._repair_review_contract(
                expected_stage, review_input, result.final_output, contradiction
            )
        decision.stage = expected_stage
        # 契约校验完成后再裁剪叙述字段：只缩短文本，不改变 passed 与
        # required_fixes 的非空语义
        decision = _trim_review_decision(decision, self._settings)
        # 成功解析后记录评审原始输出（截断避免日志膨胀）
        raw_output = str(result.final_output) if result.final_output is not None else ""
        self._log_review_agent_output(
            expected_stage, decision, raw_output, contract_valid=contract_valid
        )
        return decision

    def _log_review_agent_output(
        self,
        stage: str,
        decision: ReviewDecision,
        raw_output: str,
        *,
        contract_valid: bool,
    ) -> None:
        """记录 ``review_agent_output`` 审计事件（含契约有效性标记）。"""
        self._events.log_event(
            EVENT_REVIEW_AGENT_OUTPUT,
            {
                "stage": stage,
                "passed": decision.passed,
                "contract_valid": contract_valid,
                "raw_output_preview": raw_output[:1000],
            },
        )

    async def _repair_review_contract(
        self,
        expected_stage: str,
        review_input: Any,
        raw_output: Any,
        contradiction: str,
    ) -> tuple[ReviewDecision, bool]:
        """对契约矛盾的评审输出执行一次（且仅一次）契约修复调用。

        修复输入 = 原始评审输入 JSON + ``contract_repair`` 附加字段
        （原始输出、矛盾说明、严格字段模板与修复指令），要求评审
        Agent 只修复字段一致性：调整 passed 与 required_fixes /
        suggestions 的组合使契约自洽，禁止重新评审或改变评审事实。
        修复输出再次解析并做同样的契约校验；仍矛盾、缺失显式
        passed 或解析失败时 fail closed（保守未通过）。

        审计：``review_contract_repair`` 事件记录 attempt（固定 1）、
        原始矛盾与最终状态（repaired / failed），修复输出预览截断至
        500 字符；最终 fail closed 时另记 ``review_contract_error``。

        Args:
            expected_stage: 归一化后的评审阶段（plan_review / result_review）。
            review_input: 原始评审输入（ReviewInput，修复消息保留其全部
                字段供评审 Agent 定位上下文）。
            raw_output: 原始评审输出（final_output，任意形态）。
            contradiction: ``_review_contract_contradiction`` 给出的矛盾说明。

        Returns:
            ``(decision, contract_valid)``：修复成功时为修复后的合法
            结论（True）；失败时为 fail closed 结论（False）。
        """
        repair_message = self._build_contract_repair_message(
            review_input, raw_output, contradiction
        )
        repair_result: Any = None
        repair_error = ""
        decision: ReviewDecision | None = None
        try:
            repair_result = await _workflow_runner().run(
                self._review_agent,
                input=repair_message,
                max_turns=int(self._settings.max_model_turns),
            )
            self._record_model_usage(repair_result, "review_agent")
            repaired = self._coerce_review_decision(repair_result.final_output)
            residual = _review_contract_contradiction(repaired)
            if residual is not None:
                repair_error = f"修复输出仍违反契约：{residual}"
            else:
                decision = repaired
        except ReviewContractError as exc:
            repair_error = f"修复输出缺失显式 passed 字段：{exc}"
        except (WorkflowError, asyncio.TimeoutError) as exc:
            repair_error = f"修复调用/解析失败：{type(exc).__name__}: {exc}"

        final_status = "repaired" if decision is not None else "failed"
        self._events.log_event(
            EVENT_REVIEW_CONTRACT_REPAIR,
            {
                "stage": expected_stage,
                "attempt": 1,
                "contradiction": contradiction,
                "final_status": final_status,
                "repair_error": repair_error or None,
                "repair_output_preview": (
                    str(repair_result.final_output)[:500]
                    if repair_result is not None
                    else "N/A"
                ),
            },
        )
        if decision is not None:
            return decision, True

        self._events.log_event(
            EVENT_REVIEW_CONTRACT_ERROR,
            {
                "stage": expected_stage,
                "contradiction": contradiction,
                "repair_error": repair_error,
                "raw_output_preview": (
                    str(raw_output)[:500] if raw_output is not None else "N/A"
                ),
            },
        )
        return (
            ReviewDecision(
                stage=expected_stage,
                passed=False,
                issues=[f"评审契约无效：{contradiction}；{repair_error}"],
                required_fixes=[
                    "评审契约无效：须修复评审输出的字段契约后重新提交评审"
                ],
                summary="评审契约无效（契约修复未成功），保守判定为未通过",
            ),
            False,
        )

    @staticmethod
    def _build_contract_repair_message(
        review_input: Any, raw_output: Any, contradiction: str
    ) -> str:
        """构造契约修复消息：原始评审输入 JSON + contract_repair 附加字段。

        消息保留 ReviewInput 全部字段（评审 Agent 据此定位上下文；
        Pydantic 校验会忽略附加字段，既有调用链/测试桩兼容），附加的
        ``contract_repair`` 字段携带修复指令、矛盾说明、原始输出
        （超长截断至 ``_REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS``）
        与严格字段模板。
        """
        original_text = str(raw_output) if raw_output is not None else ""
        if len(original_text) > _REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS:
            original_text = (
                original_text[:_REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS]
                + "…（已截断）"
            )
        payload = json.loads(review_input.model_dump_json())
        payload["contract_repair"] = {
            "instruction": (
                "你上一次输出的评审结论 JSON 存在字段契约矛盾。"
                "请只修复字段一致性，使输出满足 strict_template 的字段约束："
                "passed=true 时 required_fixes 必须为空数组；"
                "passed=false 时 required_fixes 必须至少包含一条修复项；"
                "非阻断的改进建议一律写入 suggestions 数组。"
                "禁止重新评审、禁止改变评审事实：不得修改 issues 的内容、"
                "不得翻转 passed 的评审结论语义，只做上述字段组合的"
                "一致性调整。只输出修复后的 ReviewDecision JSON"
                "（不包裹代码围栏、不附加任何解释文字）。"
            ),
            "contradiction": contradiction,
            "original_output": original_text,
            "strict_template": _REVIEW_CONTRACT_TEMPLATE,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _build_review_parse_retry_message(
        review_input: Any, raw_output: Any, error: Exception
    ) -> str:
        """构造解析失败重试消息：原始评审输入 JSON + parse_retry 附加字段。

        重试消息要求评审 Agent 严格按模板重新输出合法 ReviewDecision
        JSON；保留原始输入与上次输出（截断至修复上限），便于模型定位
        解析失败原因而不丢失评审上下文。
        """
        original_text = str(raw_output) if raw_output is not None else ""
        if len(original_text) > _REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS:
            original_text = (
                original_text[:_REVIEW_CONTRACT_REPAIR_MAX_ORIGINAL_CHARS]
                + "…（已截断）"
            )
        payload = json.loads(review_input.model_dump_json())
        payload["parse_retry"] = {
            "instruction": (
                "你上一次输出无法被解析为合法的 ReviewDecision JSON。"
                "请基于原始评审输入重新给出评审结论，并严格满足 "
                "strict_template 的字段约束：必须包含显式 passed 布尔字段；"
                "passed=true 时 required_fixes 必须为空数组；"
                "passed=false 时 required_fixes 必须至少包含一条修复项；"
                "非阻断的改进建议写入 suggestions 数组。"
                "只输出完整的 ReviewDecision JSON"
                "（不包裹代码围栏、不附加任何解释文字）。"
            ),
            "parse_error": f"{type(error).__name__}: {error}",
            "previous_output": original_text,
            "strict_template": _REVIEW_CONTRACT_TEMPLATE,
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _coerce_review_decision(output: Any) -> ReviewDecision:
        """把评审 Agent 输出规范化为 ReviewDecision（宽容解析，含围栏剥离）。

        评审 Agent 同样以纯文本返回（output_type=None），复用
        ``_coerce_structured_output`` 的四形态宽容解析；仅接受显式
        可解析 ``passed`` 字段（缺失时抛 ``ReviewContractError``），
        不做任何文本关键词结论推断。

        Raises:
            WorkflowError: 输出无法解析为 ReviewDecision 时。
            ReviewContractError: 提取出 JSON/对象但缺失显式可解析
                passed 字段时。
        """
        return _coerce_structured_output(output, ReviewDecision, label="评审 Agent")

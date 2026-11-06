"""执行旅程：工具调用、动态 Agent、生成工具与路线节点。"""

from __future__ import annotations

from so_agent.models import TaskType
from so_agent.web.text import tr
from typing import Any

from so_agent.web.common import (
    TERMINAL_STATUSES,
    WebTask,
    _GENERATED_TOOL_ACTIONS,
    _GENERATED_TOOL_STEP_LABELS,
    _GENERATED_TOOL_STEP_LABELS_EN,
    _MAX_TOOL_CALLS,
    _PHASE_RANK,
    _preview,
    _status_label,
)


class JourneyMixin:

    @staticmethod
    def _agent_indexes(
        task: WebTask,
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
        """建立 Agent → 子任务索引，并汇总派发尝试次数。"""
        agent_subtasks: dict[str, str] = {}
        dispatches: dict[str, dict[str, Any]] = {}
        for record in task.context.event_log:
            event_type = record.get("event_type")
            payload = record.get("payload") or {}
            if event_type == "task_dispatched":
                subtask_id = payload.get("subtask_id")
                agent_id = payload.get("agent_id")
                if subtask_id and agent_id:
                    agent_subtasks[str(agent_id)] = str(subtask_id)
                if subtask_id:
                    dispatches[str(subtask_id)] = {
                        "attempt": int(payload.get("attempt") or 1),
                        "max_attempts": int(payload.get("max_attempts") or 3),
                        "agent_id": agent_id,
                    }
            elif event_type == "agent_created":
                nested = (
                    payload.get("agent")
                    if isinstance(payload.get("agent"), dict)
                    else {}
                )
                agent_id = payload.get("agent_id") or nested.get("agent_id")
                subtask_id = payload.get("subtask_id")
                if agent_id and subtask_id:
                    agent_subtasks[str(agent_id)] = str(subtask_id)
        return agent_subtasks, dispatches


    def _build_tool_calls(
        self, task: WebTask, agent_subtasks: dict[str, str], lang: str = "zh"
    ) -> list[dict[str, Any]]:
        """把 tool_called 审计事件配对为工具调用记录（开始 ↔ 结束）。"""
        entries: list[dict[str, Any]] = []
        open_index: dict[tuple[str, str, str, str], int] = {}
        for record in task.context.event_log:
            if record.get("event_type") != "tool_called":
                continue
            payload = record.get("payload") or {}
            action = str(payload.get("action") or "")
            if not action.startswith("invoke_"):
                continue
            agent_id = payload.get("agent_id")
            subtask_id = payload.get("subtask_id") or (
                agent_subtasks.get(str(agent_id)) if agent_id else None
            )
            tool_name = str(payload.get("tool_name") or "unknown")
            call_id = payload.get("call_id")
            key = (
                str(call_id or ""),
                str(agent_id or ""),
                str(subtask_id or ""),
                tool_name,
            )
            agent_label = payload.get("agent") or (
                "supervisor" if not agent_id else "subagent"
            )
            if action == "invoke_started":
                entries.append(
                    {
                        "tool": tool_name,
                        "call_id": call_id,
                        "agent": agent_label,
                        "agent_id": agent_id,
                        "subtask_id": subtask_id,
                        "pipeline_phase": payload.get("pipeline_phase"),
                        "arguments": _preview(
                            str(payload.get("arguments_preview") or ""), 240
                        ),
                        "output": "",
                        "duration": 0.0,
                        "status": "running",
                        "result_ok": None,
                        "started_at": record.get("timestamp"),
                        "finished_at": None,
                    }
                )
                open_index[key] = len(entries) - 1
                continue

            index = open_index.pop(key, None)
            if index is None and call_id:
                for cursor in range(len(entries) - 1, -1, -1):
                    candidate = entries[cursor]
                    if (
                        candidate["status"] == "running"
                        and candidate["call_id"] == call_id
                    ):
                        index = cursor
                        break
            if index is None:
                entries.append(
                    {
                        "tool": tool_name,
                        "call_id": call_id,
                        "agent": agent_label,
                        "agent_id": agent_id,
                        "subtask_id": subtask_id,
                        "pipeline_phase": payload.get("pipeline_phase"),
                        "arguments": "",
                        "output": "",
                        "duration": 0.0,
                        "status": "running",
                        "result_ok": None,
                        "started_at": None,
                        "finished_at": None,
                    }
                )
                index = len(entries) - 1
            entry = entries[index]
            entry.update(
                {
                    "output": _preview(
                        str(payload.get("output_preview") or ""), 240
                    ),
                    "duration": float(payload.get("duration") or 0.0),
                    "status": (
                        "failed" if action == "invoke_failed" else "completed"
                    ),
                    "result_ok": payload.get("result_ok"),
                    "finished_at": record.get("timestamp"),
                }
            )
            if not entry["arguments"]:
                entry["arguments"] = _preview(
                    str(payload.get("arguments_preview") or ""), 240
                )
            if entry["tool"] == "unknown" and tool_name != "unknown":
                entry["tool"] = tool_name
        if task.status in TERMINAL_STATUSES:
            # 终态任务不应再有进行中的调用：历史日志缺失结束事件时
            # 按失败收口，避免界面永久显示 running
            for entry in entries:
                if entry["status"] != "running":
                    continue
                entry["status"] = "failed"
                entry["result_ok"] = False
                if not entry["output"]:
                    entry["output"] = tr(
                        lang,
                        "未收到工具输出（任务已结束）",
                        "No tool output received (task finished)",
                    )
        return entries[-_MAX_TOOL_CALLS:]


    def _build_dynamic_agents(
        self, task: WebTask, agent_subtasks: dict[str, str]
    ) -> list[dict[str, Any]]:
        """合并 agent_created / agent_released 事件为 Agent 生命周期记录。"""
        agents: dict[str, dict[str, Any]] = {}
        for record in task.context.event_log:
            event_type = record.get("event_type")
            if event_type not in {"agent_created", "agent_released"}:
                continue
            payload = record.get("payload") or {}
            nested = (
                payload.get("agent")
                if isinstance(payload.get("agent"), dict)
                else {}
            )
            agent_id = str(payload.get("agent_id") or nested.get("agent_id") or "")
            if not agent_id:
                continue
            entry = agents.get(agent_id)
            if entry is None:
                entry = {
                    "agent_id": agent_id,
                    "agent_type": "",
                    "subtask_id": agent_subtasks.get(agent_id),
                    "plan_version": None,
                    "replacement_of": None,
                    "allowed_tools": [],
                    "injected_tools": [],
                    "ignored_tools": [],
                    "attempt_count": 0,
                    "status": "unknown",
                    "created_at": None,
                    "released_at": None,
                }
                agents[agent_id] = entry
            if event_type == "agent_created":
                entry["agent_type"] = str(
                    payload.get("agent_type")
                    or nested.get("agent_type")
                    or entry["agent_type"]
                )
                entry["subtask_id"] = (
                    payload.get("subtask_id")
                    or entry["subtask_id"]
                    or agent_subtasks.get(agent_id)
                )
                entry["plan_version"] = payload.get(
                    "plan_version", nested.get("plan_version", entry["plan_version"])
                )
                entry["replacement_of"] = payload.get(
                    "replacement_of",
                    nested.get("replacement_of", entry["replacement_of"]),
                )
                allowed = nested.get("allowed_tools")
                if payload.get("allowed_tools"):
                    entry["allowed_tools"] = list(payload["allowed_tools"])
                elif isinstance(allowed, list):
                    entry["allowed_tools"] = list(allowed)
                if payload.get("injected_tools"):
                    entry["injected_tools"] = list(payload["injected_tools"])
                if payload.get("ignored_tools"):
                    entry["ignored_tools"] = list(payload["ignored_tools"])
                entry["attempt_count"] = int(
                    nested.get("attempt_count")
                    or payload.get("attempt_count")
                    or entry["attempt_count"]
                    or 0
                )
                if entry["status"] == "unknown":
                    entry["status"] = str(nested.get("status") or "created")
                entry["created_at"] = record.get("timestamp")
            else:
                entry["status"] = str(payload.get("status") or entry["status"])
                entry["attempt_count"] = int(
                    payload.get("attempt_count") or entry["attempt_count"] or 0
                )
                entry["released_at"] = record.get("timestamp")
        return sorted(
            agents.values(),
            key=lambda item: (item.get("created_at") or "", item["agent_id"]),
        )


    def _build_generated_tools(
        self, task: WebTask, lang: str = "zh"
    ) -> list[dict[str, Any]]:
        """按工具名汇总缺口工具的动态生成生命周期。"""
        step_labels = (
            _GENERATED_TOOL_STEP_LABELS_EN
            if lang == "en"
            else _GENERATED_TOOL_STEP_LABELS
        )
        tools: dict[str, dict[str, Any]] = {}
        for record in task.context.event_log:
            if record.get("event_type") != "tool_called":
                continue
            payload = record.get("payload") or {}
            action = str(payload.get("action") or "")
            if action not in _GENERATED_TOOL_ACTIONS:
                continue
            name = str(payload.get("tool_name") or "")
            if not name:
                continue
            entry = tools.get(name)
            if entry is None:
                entry = {
                    "tool_name": name,
                    "status": "in_progress",
                    "reason": "",
                    "plan_version": None,
                    "steps": [],
                }
                tools[name] = entry
            if payload.get("plan_version") is not None:
                entry["plan_version"] = payload.get("plan_version")
            reason = str(payload.get("reason") or "")
            if action == "tool_verified":
                detail = tr(
                    lang,
                    f"沙箱验收通过（{payload.get('test_count') or 0} 条用例）",
                    f"Sandbox verification passed ({payload.get('test_count') or 0} cases)",
                )
            elif action == "tool_verify_skipped":
                detail = tr(
                    lang,
                    f"跳过沙箱验收：{reason or '未提供测试用例'}",
                    f"Sandbox verification skipped: {reason or 'no test cases provided'}",
                )
            elif action == "candidate_created":
                detail = tr(
                    lang,
                    "代码 Agent 已产出工具候选",
                    "Code agent produced a tool candidate",
                )
            elif action == "gap_detected":
                detail = tr(lang, "检测到工具缺口", "Tool gap detected")
            elif action == "registered":
                detail = tr(lang, "评审通过并已注册", "Approved and registered")
            elif action == "gap_unresolved":
                detail = tr(
                    lang,
                    f"未能补齐：{reason or '未知原因'}",
                    f"Could not fill the gap: {reason or 'unknown reason'}",
                )
                entry["reason"] = _preview(reason, 300)
            else:
                detail = reason
            entry["steps"].append(
                {
                    "action": action,
                    "label": step_labels.get(action, action),
                    "detail": _preview(detail, 240),
                    "timestamp": record.get("timestamp"),
                }
            )
            if action == "registered":
                entry["status"] = "registered"
            elif action == "gap_unresolved":
                entry["status"] = "unresolved"
        return sorted(tools.values(), key=lambda item: item["tool_name"])


    def _build_journey(
        self,
        task: WebTask,
        plan: dict[str, Any],
        route: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        dynamic_agents: list[dict[str, Any]],
        generated_tools: list[dict[str, Any]],
        agent_dispatches: dict[str, dict[str, Any]],
        lang: str = "zh",
    ) -> dict[str, Any]:
        """构建全链路流程图数据（路由 → 分支 → 工作流 → 工具 / Agent）。"""
        nodes: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        row_ids: dict[int, list[str]] = {}
        edges: list[dict[str, Any]] = []

        def add(node: dict[str, Any]) -> dict[str, Any]:
            node.setdefault("w", 240)
            node.setdefault("chips", [])
            node.setdefault("rows", [])
            node.setdefault("blocks", [])
            node.setdefault("tool_calls", [])
            node.setdefault("agents", [])
            node.setdefault("generated_tools", [])
            nodes.append(node)
            by_id[node["id"]] = node
            row_ids.setdefault(int(node["row"]), []).append(node["id"])
            return node

        def link(
            source: str | None,
            target: str,
            *,
            kind: str = "main",
            label: str = "",
        ) -> None:
            if source and source in by_id and target in by_id:
                edges.append(
                    {"from": source, "to": target, "kind": kind, "label": label}
                )

        terminal = task.status in TERMINAL_STATUSES
        failed = task.status in {"failed", "human_required"}
        cancelled = task.status == "cancelled"
        engine = task.engine
        phase = task.status
        if engine is not None:
            try:
                phase = engine.current_status.value
            except Exception:
                pass

        if task.classification is None:
            current_rank = 0
        elif not terminal:
            current_rank = _PHASE_RANK.get(phase, 1)
        elif failed:
            current_rank = _PHASE_RANK.get(phase, 3)
        elif cancelled:
            current_rank = _PHASE_RANK.get(phase, 2)
        else:
            current_rank = 6

        def rank_status(rank: int) -> str:
            if current_rank <= 0:
                return "pending"
            if terminal:
                if failed:
                    if rank < current_rank:
                        return "done"
                    return "failed" if rank == current_rank else "pending"
                if cancelled:
                    if rank < current_rank:
                        return "done"
                    return "warn" if rank == current_rank else "skipped"
                return "done"
            if rank < current_rank:
                return "done"
            return "active" if rank == current_rank else "pending"

        add(
            {
                "id": "start",
                "row": 0,
                "w": 320,
                "kind": "terminal",
                "title": tr(lang, "任务提交", "Task submitted"),
                "subtitle": _preview(task.objective, 60),
                "status": "done",
                "rows": [[tr(lang, "任务编号", "Task ID"), task.task_id]],
                "blocks": [
                    {
                        "label": tr(lang, "任务目标", "Task objective"),
                        "text": _preview(task.objective, 800),
                    }
                ],
            }
        )

        if task.classification is None:
            add(
                {
                    "id": "classify",
                    "row": 1,
                    "w": 320,
                    "kind": "decision",
                    "title": tr(lang, "任务分类", "Task classification"),
                    "subtitle": tr(
                        lang,
                        "等待规则预判与模型分类",
                        "Awaiting rule check and model classification",
                    ),
                    "status": (
                        "active" if task.status == "running" else "pending"
                    ),
                }
            )
            link("start", "classify")
            add(
                {
                    "id": "end",
                    "row": 2,
                    "w": 280,
                    "kind": "terminal",
                    "title": tr(lang, "最终答复", "Final answer"),
                    "subtitle": tr(
                        lang, "分类完成后继续", "Continues after classification"
                    ),
                    "status": "pending",
                }
            )
            link("classify", "end")
            return self._finish_journey(nodes, edges, row_ids, by_id)

        steps = route["steps"]
        rule_step = next(
            (item for item in steps if item["stage"] == "rule_check"), None
        )
        llm_step = next(
            (item for item in steps if item["stage"] == "llm_classify"), None
        )
        upgrade_step = next(
            (item for item in steps if item["stage"] == "tool_intent_upgrade"),
            None,
        )
        disabled_step = next(
            (item for item in steps if item["stage"] == "router_disabled"), None
        )
        guard_step = next(
            (item for item in steps if item["stage"] == "output_guard"), None
        )

        chain = "start"
        if rule_step is not None:
            add(
                {
                    "id": "rule",
                    "row": 1,
                    "w": 340,
                    "kind": "decision",
                    "title": tr(lang, "规则预判", "Rule pre-check"),
                    "subtitle": {
                        "hit": tr(
                            lang,
                            "命中确定性规则，直接判定",
                            "Matched a deterministic rule",
                        ),
                        "miss": tr(
                            lang,
                            "未命中规则，交由模型分类",
                            "No rule matched; classified by model",
                        ),
                        "skipped": tr(
                            lang,
                            "存在约束/验收标准，跳过规则",
                            "Constraints present; rule check skipped",
                        ),
                    }.get(rule_step["status"], rule_step["status"]),
                    "status": {
                        "hit": "done",
                        "miss": "done",
                        "skipped": "skipped",
                    }.get(rule_step["status"], "done"),
                    "rows": [
                        [
                            tr(lang, "判断方式", "Decision method"),
                            route["method_label"],
                        ]
                    ],
                    "blocks": [
                        {
                            "label": tr(lang, "判断依据", "Decision basis"),
                            "text": rule_step["detail"],
                        }
                    ],
                }
            )
            link(chain, "rule")
            chain = "rule"

        if llm_step is not None:
            fallback = llm_step["status"] == "fallback"
            add(
                {
                    "id": "llm",
                    "row": 2,
                    "w": 340,
                    "kind": "decision",
                    "title": tr(lang, "模型分类", "Model classification"),
                    "subtitle": (
                        tr(
                            lang,
                            "分类失败，安全降级为完整工作流",
                            "Classification failed; safely fell back to full workflow",
                        )
                        if fallback
                        else tr(
                            lang,
                            "模型判定任务象限",
                            "Model determined the task quadrant",
                        )
                    ),
                    "status": "warn" if fallback else "done",
                    "rows": [
                        [
                            tr(lang, "分类结果", "Classification"),
                            route["classified_task_type"] or "—",
                        ],
                        [
                            tr(lang, "分类方式", "Method"),
                            route["method_label"],
                        ],
                    ],
                    "blocks": [
                        {
                            "label": tr(lang, "模型判断", "Model decision"),
                            "text": llm_step["detail"],
                        }
                    ],
                }
            )
            link(chain, "llm")
            chain = "llm"
        elif disabled_step is not None:
            add(
                {
                    "id": "disabled",
                    "row": 2,
                    "w": 340,
                    "kind": "decision",
                    "title": tr(lang, "路由器已禁用", "Router disabled"),
                    "subtitle": tr(
                        lang,
                        "跳过分类，直接进入完整工作流",
                        "Skipped classification; going straight to full workflow",
                    ),
                    "status": "skipped",
                    "blocks": [
                        {
                            "label": tr(lang, "判断依据", "Decision basis"),
                            "text": disabled_step["detail"],
                        }
                    ],
                }
            )
            link(chain, "disabled")
            chain = "disabled"

        if upgrade_step is not None:
            add(
                {
                    "id": "upgrade",
                    "row": 3,
                    "w": 340,
                    "kind": "decision",
                    "title": tr(lang, "工具意图升级", "Tool intent upgrade"),
                    "subtitle": tr(
                        lang,
                        "目标或验收标准要求工具执行，升级为完整工作流",
                        "Goal or acceptance criteria require tools; upgraded to full workflow",
                    ),
                    "status": "warn",
                    "rows": [
                        [
                            tr(lang, "原分类", "Original"),
                            route["original_task_type"] or "—",
                        ],
                        [
                            tr(lang, "升级后", "Upgraded to"),
                            "full_workflow",
                        ],
                    ],
                    "blocks": [
                        {
                            "label": tr(lang, "升级依据", "Upgrade basis"),
                            "text": upgrade_step["detail"],
                        }
                    ],
                }
            )
            link(chain, "upgrade")
            chain = "upgrade"

        for branch in route["branches"]:
            chips: list[str] = (
                [tr(lang, "已选择", "Selected")] if branch["selected"] else []
            )
            if branch["selected"] and route["guard_fallback"]:
                chips.append(
                    tr(
                        lang,
                        "守卫降级 → 完整工作流",
                        "Guard fallback → full workflow",
                    )
                )
            add(
                {
                    "id": "branch-" + branch["type"],
                    "row": 4,
                    "w": 230,
                    "kind": "branch",
                    "title": branch["label"],
                    "subtitle": branch["path"],
                    "status": "chosen" if branch["selected"] else "unselected",
                    "chips": chips,
                    "rows": [
                        [
                            tr(lang, "适用条件", "Applies when"),
                            branch["condition"],
                        ]
                    ],
                }
            )
            link(chain, "branch-" + branch["type"], kind="branch")
        selected_branch = "branch-" + route["classified_task_type"]
        exec_chain: str = selected_branch if selected_branch in by_id else chain

        if route["guard_fallback"]:
            add(
                {
                    "id": "guard",
                    "row": 5,
                    "w": 340,
                    "kind": "guard",
                    "title": tr(
                        lang,
                        "输出守卫：降级重跑",
                        "Output guard: fallback rerun",
                    ),
                    "subtitle": tr(
                        lang,
                        "直答/推理输出含伪造工具调用标记，改走完整工作流",
                        "Direct/reasoning output contained fake tool-call markers; switched to full workflow",
                    ),
                    "status": "warn",
                    "rows": [
                        [
                            tr(lang, "初始分支", "Initial branch"),
                            route["classified_task_type"] or "—",
                        ],
                        [
                            tr(lang, "降级后", "After fallback"),
                            route["effective_task_type"] or "—",
                        ],
                    ],
                    "blocks": [
                        {
                            "label": tr(lang, "守卫判定", "Guard decision"),
                            "text": (
                                guard_step["detail"]
                                if guard_step
                                else tr(
                                    lang, "已触发降级", "Fallback triggered"
                                )
                            ),
                        }
                    ],
                }
            )
            link(
                exec_chain,
                "guard",
                kind="fallback",
                label=tr(lang, "守卫降级", "Guard fallback"),
            )
            exec_chain = "guard"

        effective_type = route["effective_task_type"]
        if effective_type != TaskType.FULL_WORKFLOW.value:
            path_label = route["effective_label"]
            path_node = add(
                {
                    "id": "path",
                    "row": 5,
                    "w": 340,
                    "kind": "phase",
                    "title": tr(
                        lang,
                        path_label + "执行",
                        path_label + " execution",
                    ),
                    "subtitle": tr(
                        lang,
                        "单次模型调用，不进入多 Agent 工作流",
                        "Single model call; no multi-agent workflow",
                    ),
                    "status": rank_status(1),
                    "rows": [[tr(lang, "执行路径", "Execution path"), path_label]],
                    "blocks": [],
                }
            )
            if task.output is not None and task.output.final_result:
                path_node["blocks"].append(
                    {
                        "label": tr(lang, "最终输出", "Final output"),
                        "text": _preview(task.output.final_result, 600),
                    }
                )
            link(exec_chain, "path")
            end_node = add(
                {
                    "id": "end",
                    "row": 6,
                    "w": 300,
                    "kind": "terminal",
                    "title": self._terminal_title(task.status, lang),
                    "subtitle": tr(lang, "任务结束", "Task finished"),
                    "status": self._terminal_status(task.status),
                    "rows": [
                        [tr(lang, "最终状态", "Final status"), task.status],
                        [tr(lang, "任务类型", "Task type"), route["effective_label"]],
                    ],
                }
            )
            if task.error:
                end_node["blocks"].append(
                    {
                        "label": tr(lang, "错误信息", "Error"),
                        "text": _preview(task.error, 400),
                    }
                )
            link("path", "end")
            return self._finish_journey(nodes, edges, row_ids, by_id)

        planning = add(
            {
                "id": "planning",
                "row": 6,
                "w": 360,
                "kind": "phase",
                "title": tr(lang, "规划：拆解子任务", "Planning: decompose subtasks"),
                "subtitle": tr(
                    lang,
                    "主管生成子任务规格与依赖关系",
                    "Supervisor generates subtask specs and dependencies",
                ),
                "status": rank_status(1),
                "rows": [
                    [
                        tr(lang, "计划版本", "Plan version"),
                        f"v{plan.get('current') or 0}",
                    ],
                    [
                        tr(lang, "重规划次数", "Replan count"),
                        str(plan.get("replan_count") or 0),
                    ],
                ],
            }
        )
        link(exec_chain, "planning")

        review = add(
            {
                "id": "plan_review",
                "row": 7,
                "w": 360,
                "kind": "phase",
                "title": tr(lang, "计划评审", "Plan review"),
                "subtitle": tr(
                    lang,
                    "两阶段评审通过后才放行执行",
                    "Execution starts only after two-stage review passes",
                ),
                "status": rank_status(2),
            }
        )
        link("planning", "plan_review")

        version = None
        versions = plan.get("versions") or []
        if versions:
            current = plan.get("current")
            version = next(
                (item for item in versions if item["version"] == current),
                versions[-1],
            )
        subtasks = list(version["nodes"]) if version else []
        by_subtask = {item["id"]: item for item in subtasks}

        levels: dict[str, int] = {}
        visiting: set[str] = set()

        def resolve_level(subtask_id: str) -> int:
            if subtask_id in levels:
                return levels[subtask_id]
            if subtask_id in visiting:
                return 0
            visiting.add(subtask_id)
            node = by_subtask.get(subtask_id)
            value = 0
            for dependency in (node or {}).get("dependencies") or []:
                if dependency in by_subtask:
                    value = max(value, resolve_level(dependency) + 1)
            visiting.discard(subtask_id)
            levels[subtask_id] = value
            return value

        for item in subtasks:
            resolve_level(item["id"])
        max_level = max(levels.values(), default=-1)

        for item in subtasks:
            subtask_id = item["id"]
            calls = [
                call for call in tool_calls if call["subtask_id"] == subtask_id
            ]
            agents = [
                agent
                for agent in dynamic_agents
                if agent["subtask_id"] == subtask_id
            ]
            tools = [
                tool
                for tool in generated_tools
                if tool["tool_name"] in (item.get("allowed_tools") or [])
            ]
            chips: list[str] = []
            allowed = item.get("allowed_tools") or []
            if allowed:
                chips.append(
                    tr(lang, "工具 ", "Tools ")
                    + " / ".join(allowed[:3])
                    + ("…" if len(allowed) > 3 else "")
                )
            if calls:
                chips.append(
                    tr(lang, f"调用 {len(calls)} 次", f"{len(calls)} calls")
                )
            if agents:
                chips.append(f"Agent {len(agents)}")
            if tools:
                chips.append(
                    tr(lang, "动态工具 ", "Dynamic tools ")
                    + " / ".join(t["tool_name"] for t in tools)
                )
            blocks = []
            if item.get("instructions"):
                blocks.append(
                    {
                        "label": tr(lang, "执行指令", "Instructions"),
                        "text": item["instructions"],
                    }
                )
            if item.get("expected_output"):
                blocks.append(
                    {
                        "label": tr(lang, "期望产出", "Expected output"),
                        "text": item["expected_output"],
                    }
                )
            if item.get("output_preview"):
                blocks.append(
                    {
                        "label": tr(lang, "实际产出", "Actual output"),
                        "text": item["output_preview"],
                    }
                )
            if item.get("error"):
                blocks.append(
                    {"label": tr(lang, "错误", "Error"), "text": item["error"]}
                )
            rows = [
                [tr(lang, "子任务编号", "Subtask ID"), subtask_id],
                [tr(lang, "角色", "Role"), item.get("role") or "—"],
                [
                    tr(lang, "状态", "Status"),
                    _status_label(lang, str(item.get("status") or "")) or "—",
                ],
                [
                    tr(lang, "尝试次数", "Attempts"),
                    f"{item.get('attempt') or 0} / {item.get('max_attempts') or '?'}",
                ],
                [
                    tr(lang, "前置依赖", "Dependencies"),
                    (
                        ", " if lang == "en" else "、"
                    ).join(item.get("dependencies") or [])
                    or tr(lang, "无", "None"),
                ],
                [
                    tr(lang, "耗时", "Duration"),
                    f"{item.get('duration') or 0:.2f}s",
                ],
            ]
            if item.get("reused_from"):
                rows.append(
                    [
                        tr(lang, "复用来源", "Reused from"),
                        item["reused_from"],
                    ]
                )
            dispatch = agent_dispatches.get(subtask_id)
            dispatch_agent = (
                str(dispatch["agent_id"])
                if dispatch and dispatch.get("agent_id")
                else None
            )
            # 派发 Agent 是调度器建的主 Agent；嵌套生成的动态 Agent 才
            # 是实际执行工具的角色，两者分开呈现避免误导
            nested_agents = [
                str(agent["agent_id"])
                for agent in agents
                if agent.get("agent_id")
                and str(agent["agent_id"]) != dispatch_agent
            ]
            if dispatch_agent:
                rows.append([tr(lang, "派发 Agent", "Dispatch agent"), dispatch_agent])
            if nested_agents:
                suffix = (
                    tr(
                        lang,
                        f"（共 {len(nested_agents)} 个动态 Agent）",
                        f" ({len(nested_agents)} dynamic agents)",
                    )
                    if len(nested_agents) > 1
                    else ""
                )
                rows.append(
                    [tr(lang, "执行 Agent", "Executing agent"), nested_agents[-1] + suffix]
                )
            elif dispatch_agent:
                rows.append(
                    [
                        tr(lang, "最近执行 Agent", "Latest executing agent"),
                        dispatch_agent,
                    ]
                )
            add(
                {
                    "id": subtask_id,
                    "row": 8 + levels.get(subtask_id, 0),
                    "w": 320,
                    "kind": "subtask",
                    "title": item.get("title") or subtask_id,
                    "subtitle": item.get("role") or "",
                    "status": item.get("status") or "pending",
                    "chips": chips,
                    "rows": rows,
                    "blocks": blocks,
                    "tool_calls": calls,
                    "agents": agents,
                    "generated_tools": tools,
                }
            )

        roots = [
            item["id"]
            for item in subtasks
            if not any(
                dependency in by_subtask
                for dependency in (item.get("dependencies") or [])
            )
        ]
        for root in roots:
            link("plan_review", root)
        for item in subtasks:
            for dependency in item.get("dependencies") or []:
                if dependency in by_subtask:
                    link(dependency, item["id"])
        leaves = [
            item["id"]
            for item in subtasks
            if not any(
                item["id"] in (other.get("dependencies") or [])
                for other in subtasks
            )
        ]

        aggregation_row = 8 + (max_level + 1 if max_level >= 0 else 0)
        aggregation = add(
            {
                "id": "aggregation",
                "row": aggregation_row,
                "w": 360,
                "kind": "phase",
                "title": tr(lang, "汇总子任务结果", "Aggregate subtask results"),
                "subtitle": tr(
                    lang,
                    "聚合各子任务产出并生成最终答复",
                    "Combine subtask outputs into the final answer",
                ),
                "status": rank_status(4),
            }
        )
        if leaves:
            for leaf in leaves:
                link(leaf, "aggregation")
        else:
            link("plan_review", "aggregation")
        result_review = add(
            {
                "id": "result_review",
                "row": aggregation_row + 1,
                "w": 360,
                "kind": "phase",
                "title": tr(lang, "结果评审", "Result review"),
                "subtitle": tr(
                    lang,
                    "最终答复通过评审后才发布",
                    "Final answer is published only after review passes",
                ),
                "status": rank_status(5),
            }
        )
        link("aggregation", "result_review")
        end_node = add(
            {
                "id": "end",
                "row": aggregation_row + 2,
                "w": 320,
                "kind": "terminal",
                "title": self._terminal_title(task.status, lang),
                "subtitle": tr(lang, "任务结束", "Task finished"),
                "status": self._terminal_status(task.status),
                "rows": [
                    [tr(lang, "最终状态", "Final status"), task.status],
                    [tr(lang, "子任务数", "Subtasks"), str(len(subtasks))],
                    [tr(lang, "工具调用", "Tool calls"), str(len(tool_calls))],
                    [tr(lang, "动态 Agent", "Dynamic agents"), str(len(dynamic_agents))],
                ],
            }
        )
        if task.output is not None and task.output.final_result:
            end_node["blocks"].append(
                {
                    "label": tr(lang, "最终答复", "Final answer"),
                    "text": _preview(task.output.final_result, 800),
                }
            )
        if task.error:
            end_node["blocks"].append(
                {
                    "label": tr(lang, "错误信息", "Error"),
                    "text": _preview(task.error, 400),
                }
            )
        link("result_review", "end")
        return self._finish_journey(nodes, edges, row_ids, by_id)


    @staticmethod
    def _terminal_title(status: str, lang: str = "zh") -> str:
        return {
            "completed": tr(lang, "最终答复", "Final answer"),
            "failed": tr(lang, "任务失败", "Task failed"),
            "cancelled": tr(lang, "任务已取消", "Task cancelled"),
            "human_required": tr(lang, "需要人工处理", "Human action required"),
        }.get(status, tr(lang, "最终答复", "Final answer"))


    @staticmethod
    def _terminal_status(status: str) -> str:
        if status == "completed":
            return "done"
        if status == "failed":
            return "failed"
        if status in {"cancelled", "human_required"}:
            return "warn"
        if status == "running":
            return "active"
        return "pending"


    @staticmethod
    def _finish_journey(
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        row_ids: dict[int, list[str]],
        by_id: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """按行居中排布节点，并推导连线状态。"""
        gap_x = 40
        gap_y = 64
        pad = 28
        heights = {
            "terminal": 64,
            "decision": 84,
            "branch": 96,
            "phase": 88,
            "guard": 88,
            "subtask": 110,
        }
        offsets: dict[int, float] = {}
        y = pad
        for row in sorted(row_ids):
            offsets[row] = y
            row_height = max(
                heights.get(by_id[node_id]["kind"], 92) for node_id in row_ids[row]
            )
            for node_id in row_ids[row]:
                node = by_id[node_id]
                node["h"] = heights.get(node["kind"], 92) + (
                    22 if node["chips"] else 0
                )
            y += row_height + gap_y

        width = 0
        for row in row_ids.values():
            total = sum(by_id[node_id]["w"] for node_id in row) + gap_x * (
                len(row) - 1
            )
            width = max(width, total)
        width += pad * 2

        for row, ids in row_ids.items():
            total = sum(by_id[node_id]["w"] for node_id in ids) + gap_x * (
                len(ids) - 1
            )
            x = (width - total) / 2
            for node_id in ids:
                node = by_id[node_id]
                node["x"] = x
                node["y"] = offsets[row]
                x += node["w"] + gap_x

        for edge in edges:
            source = by_id.get(edge["from"])
            target = by_id.get(edge["to"])
            if edge["kind"] == "branch" and target and target["status"] == "unselected":
                edge["status"] = "muted"
            elif (source and source["status"] == "failed") or (
                target and target["status"] == "failed"
            ):
                edge["status"] = "failed"
            elif target and target["status"] in {"active"}:
                edge["status"] = "active"
            elif target and target["status"] in {"done", "chosen"}:
                edge["status"] = "success"
            elif target and target["status"] == "warn":
                edge["status"] = "warn"
            else:
                edge["status"] = "muted"
            if edge["kind"] in {"branch", "fallback"}:
                edge["dashed"] = edge["status"] != "success"

        return {
            "width": round(width),
            "height": round(y - gap_y + pad),
            "nodes": nodes,
            "edges": edges,
        }


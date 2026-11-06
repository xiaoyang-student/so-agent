"""任务快照：阶段、计划、路线与调度/预算载荷。"""

from __future__ import annotations

from so_agent.models import ExecutionResult
from so_agent.models import SubtaskSpec
from so_agent.models import TaskType
from so_agent.web.text import tr
from typing import Any
import copy

from so_agent.web.common import (
    WebTask,
    _ROUTE_BRANCHES,
    _ROUTE_BRANCHES_EN,
    _ROUTE_METHOD_LABELS,
    _ROUTE_METHOD_LABELS_EN,
    _preview,
    _route_labels,
    _split_items,
    logger,
)


class SnapshotMixin:

    def snapshot(self, task_id: str, lang: str = "zh") -> dict[str, Any] | None:
        """返回任务详情快照（分类、阶段、计划 DAG、时间线、产物）。"""
        task = self.get_task(task_id)
        if task is None:
            return None
        if task.restored and task.restored_snapshot:
            restored = self._restored_snapshot(task, lang)
            if restored is not None:
                return restored
        engine = task.engine
        phase = task.status
        plan_version = 0
        replan_count = 0
        if engine is not None:
            try:
                phase = engine.current_status.value
                plan_version = engine.plan_version
                replan_count = engine.replan_count
            except Exception:  # 状态读取失败不得影响页面
                pass
        output = task.output
        plan = self._build_plan_snapshot(task, lang)
        route = self._build_route(task, lang)
        agent_subtasks, agent_dispatches = self._agent_indexes(task)
        tool_calls = self._build_tool_calls(task, agent_subtasks, lang)
        dynamic_agents = self._build_dynamic_agents(task, agent_subtasks)
        generated_tools = self._build_generated_tools(task, lang)
        return {
            "task_id": task.task_id,
            "objective": task.objective,
            "constraints": _split_items(task.constraints),
            "acceptance_criteria": _split_items(task.acceptance_criteria),
            "status": task.status,
            "phase": phase,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "cancel_requested": task.cancel_requested,
            "classification": (
                {
                    "task_type": task.classification.task_type.value,
                    "label": _route_labels(lang).get(
                        task.classification.task_type,
                        task.classification.task_type.value,
                    ),
                    "reasoning": task.classification.reasoning,
                }
                if task.classification is not None
                else None
            ),
            "route": route,
            "plan_version": plan_version,
            "replan_count": replan_count,
            "plan": plan,
            "journey": self._build_journey(
                task,
                plan,
                route,
                tool_calls,
                dynamic_agents,
                generated_tools,
                agent_dispatches,
                lang,
            ),
            "tool_calls": tool_calls,
            "dynamic_agents": dynamic_agents,
            "generated_tools": generated_tools,
            "final_result": output.final_result if output is not None else "",
            "provisional": bool(output.provisional) if output is not None else False,
            "previous_result": (
                output.previous_result if output is not None else ""
            ),
            "notice": output.notice if output is not None else "",
            "human_escalation": self._human_escalation_payload(task),
            "human_guidance": (
                task.last_guidance.model_dump(mode="json")
                if task.last_guidance is not None
                else None
            ),
            "token_budget": self._token_budget_payload(task),
            "review_summary": output.review_summary if output is not None else "",
            "audit": (
                output.audit.model_dump(mode="json")
                if output is not None and output.audit is not None
                else None
            ),
            "engine_task_id": output.task_id if output is not None else "",
            "agent_history": self._build_agent_history(task),
            "error": task.error,
            "timeline": self._build_timeline(task, lang),
            "artifacts": self._scan_artifacts(task),
        }


    @staticmethod
    def _human_escalation_payload(task: WebTask) -> dict[str, Any] | None:
        """返回人工兜底材料（优先实时引擎状态，其次已产出的输出）。"""
        engine = task.engine
        material = None
        if engine is not None:
            try:
                material = engine.human_escalation
            except Exception:
                material = None
        if material is None and task.output is not None:
            material = task.output.human_escalation
        return material.model_dump(mode="json") if material is not None else None


    @staticmethod
    def _token_budget_payload(task: WebTask) -> dict[str, Any]:
        """返回本任务 token 预算累计快照（默认仅统计）。"""
        engine = task.engine
        if engine is None:
            return {}
        try:
            return engine.token_budget_snapshot()
        except Exception:
            return {}


    def _build_plan_snapshot(
        self, task: WebTask, lang: str = "zh"
    ) -> dict[str, Any]:
        engine = task.engine
        if engine is None:
            return self._single_step_plan(task, lang)
        try:
            plan_history = engine.plan_history
            results = engine.results
            current_version = engine.plan_version
        except Exception as exc:
            logger.debug("计划快照读取失败：%s", exc)
            return {"versions": [], "current": 0, "replan_count": 0}

        dispatches = self._dispatch_index(task)
        reuse_targets = {
            spec.reuse_from
            for plan in plan_history
            for spec in plan
            if spec.reuse_from
        }
        versions: list[dict[str, Any]] = []
        for index, specs in enumerate(plan_history, start=1):
            nodes = [
                self._build_node(
                    spec,
                    version=index,
                    current_version=current_version,
                    results=results,
                    dispatches=dispatches,
                    reuse_targets=reuse_targets,
                    task=task,
                )
                for spec in specs
            ]
            ids = {spec.subtask_id for spec in specs}
            edges = [
                {"from": dependency, "to": spec.subtask_id}
                for spec in specs
                for dependency in spec.dependencies
                if dependency in ids
            ]
            versions.append(
                {"version": index, "nodes": nodes, "edges": edges}
            )
        return {
            "versions": versions,
            "current": current_version,
            "replan_count": engine.replan_count,
        }


    def _single_step_plan(self, task: WebTask, lang: str = "zh") -> dict[str, Any]:
        """非完整工作流路径的单节点“计划”（用于统一流程图渲染）。"""
        classification = task.classification
        if classification is None:
            label = tr(lang, "等待分类", "Awaiting classification")
            reason = ""
        else:
            label = _route_labels(lang).get(
                classification.task_type, classification.task_type.value
            )
            reason = classification.reasoning
        node = {
            "id": "route",
            "title": label,
            "role": classification.task_type.value if classification else "router",
            "status": "pending" if task.status == "queued" else task.status,
            "attempt": 0,
            "max_attempts": 0,
            "dependencies": [],
            "allowed_tools": [],
            "instructions": reason,
            "expected_output": "",
            "output_preview": _preview(
                task.output.final_result if task.output is not None else ""
            ),
            "error": task.error or "",
            "duration": 0.0,
            "reused_from": None,
            "archived": False,
        }
        return {
            "versions": [{"version": 1, "nodes": [node], "edges": []}],
            "current": 1,
            "replan_count": 0,
        }


    @staticmethod
    def _dispatch_index(task: WebTask) -> dict[str, dict[str, Any]]:
        index: dict[str, dict[str, Any]] = {}
        for record in task.context.event_log:
            if record.get("event_type") != "task_dispatched":
                continue
            payload = record.get("payload") or {}
            subtask_id = payload.get("subtask_id")
            if not subtask_id:
                continue
            index[str(subtask_id)] = {
                "attempt": int(payload.get("attempt") or 1),
                "max_attempts": int(payload.get("max_attempts") or 3),
                "agent_id": payload.get("agent_id"),
            }
        return index


    @staticmethod
    def _build_node(
        spec: SubtaskSpec,
        *,
        version: int,
        current_version: int,
        results: dict[str, ExecutionResult],
        dispatches: dict[str, dict[str, Any]],
        reuse_targets: set[str | None],
        task: WebTask,
    ) -> dict[str, Any]:
        result = results.get(spec.subtask_id) if version == current_version else None
        archived = version != current_version
        reused_from = spec.reuse_from
        if result is not None:
            reused_from = result.metadata.get("reused_from") or reused_from
        dispatch = dispatches.get(spec.subtask_id)

        if archived:
            status = "reused" if spec.subtask_id in reuse_targets else "archived"
        elif result is not None:
            if result.success:
                status = "reused" if reused_from else "completed"
            else:
                status = "failed"
        elif task.cancel_requested or task.status == "cancelled":
            status = "cancelled"
        elif dispatch is not None:
            status = "running"
        else:
            status = "pending"

        return {
            "id": spec.subtask_id,
            "title": spec.title,
            "role": spec.role,
            "status": status,
            "attempt": dispatch["attempt"] if dispatch else 0,
            "max_attempts": (
                dispatch["max_attempts"] if dispatch else spec.max_attempts
            ),
            "dependencies": list(spec.dependencies),
            "allowed_tools": list(spec.allowed_tools),
            "instructions": _preview(spec.instructions, 400),
            "expected_output": _preview(spec.expected_output, 240),
            "output_preview": _preview(result.output if result else ""),
            "error": _preview(result.error if result else "", 300),
            "duration": float(result.duration) if result else 0.0,
            "reused_from": reused_from,
            "archived": archived,
        }


    def _build_route(self, task: WebTask, lang: str = "zh") -> dict[str, Any]:
        branches = [
            dict(branch, selected=False, status="unselected")
            for branch in (_ROUTE_BRANCHES_EN if lang == "en" else _ROUTE_BRANCHES)
        ]
        classification = task.classification
        if classification is None:
            return {
                "available": False,
                "classified_task_type": None,
                "effective_task_type": None,
                "effective_label": None,
                "method": None,
                "method_label": "",
                "reasoning": "",
                "steps": [],
                "branches": branches,
                "guard_fallback": False,
                "original_task_type": None,
            }
        steps = [
            {
                "stage": str(getattr(step, "stage", "") or ""),
                "status": str(getattr(step, "status", "") or ""),
                "detail": _preview(str(getattr(step, "detail", "") or ""), 400),
            }
            for step in classification.steps
        ]

        method = "llm"
        if any(item["stage"] == "router_disabled" for item in steps):
            method = "disabled"
        elif any(item["stage"] == "tool_intent_upgrade" for item in steps):
            method = "upgrade"
        elif any(
            item["stage"] == "rule_check" and item["status"] == "hit"
            for item in steps
        ):
            method = "rule"
        elif any(
            item["stage"] == "llm_classify" and item["status"] == "fallback"
            for item in steps
        ):
            method = "fallback"

        guard_fallback = any(
            item["stage"] == "output_guard" and item["status"] == "applied"
            for item in steps
        )
        effective = (
            TaskType.FULL_WORKFLOW if guard_fallback else classification.task_type
        )

        original_type = None
        for item in steps:
            if item["stage"] != "tool_intent_upgrade":
                continue
            prefix = "原分类 "
            if item["detail"].startswith(prefix):
                value = item["detail"][len(prefix):].split("；")[0].strip()
                original_type = value or None
            break

        for branch in branches:
            if branch["type"] == classification.task_type.value:
                branch["selected"] = True
                branch["status"] = "chosen"

        return {
            "available": True,
            "classified_task_type": classification.task_type.value,
            "effective_task_type": effective.value,
            "effective_label": _route_labels(lang).get(effective, effective.value),
            "method": method,
            "method_label": (
                _ROUTE_METHOD_LABELS_EN if lang == "en" else _ROUTE_METHOD_LABELS
            ).get(method, method),
            "reasoning": _preview(classification.reasoning, 600),
            "steps": steps,
            "branches": branches,
            "guard_fallback": guard_fallback,
            "original_task_type": original_type,
        }


    def _restored_snapshot(
        self, task: WebTask, lang: str
    ) -> dict[str, Any] | None:
        key = "en" if lang == "en" else "zh"
        raw = task.restored_snapshot.get(key) or task.restored_snapshot.get("zh")
        if not raw:
            return None
        snapshot = copy.deepcopy(raw)
        snapshot["status"] = task.status
        snapshot["phase"] = task.status
        snapshot["finished_at"] = task.finished_at
        snapshot["error"] = task.error
        snapshot["cancel_requested"] = task.cancel_requested
        snapshot["restored"] = True
        snapshot["artifacts"] = self._scan_artifacts(task)
        snapshot["human_guidance"] = (
            task.last_guidance.model_dump(mode="json")
            if task.last_guidance is not None
            else snapshot.get("human_guidance")
        )
        return snapshot


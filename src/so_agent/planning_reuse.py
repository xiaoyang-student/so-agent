"""规划结果复用（PlanningCoordinator 的 reuse_from 校验能力）。

从 planning_coordinator 拆分（行为不变）：归档、候选载荷、显式复用
声明的安全校验、失败级联闭包与计划版本指纹入库。跨模块状态统一经
``self`` 访问协调器共享状态。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from so_agent.fingerprint import spec_fingerprint, specs_fingerprint_equivalent
from so_agent.models import (
    ExecutionResult,
    PlanVersion,
    SubtaskSpec,
    TaskDecompositionIssue,
)
from so_agent.runtime.events import EVENT_RESULT_REUSED, EVENT_REUSE_REJECTED
from so_agent.runtime.logging_setup import get_logger
from so_agent.runtime.registry import DEFAULT_SUPERVISOR_ID

logger = get_logger("planning")


class PlanningReuseMixin:
    """结果复用候选构造与显式 reuse_from 安全校验。"""

    def _archive_completed_results(self) -> None:
        """把当前版本的执行结果归档到 ``archive_results``。

        归档键为 ``(当前 plan_version, subtask_id)``，深拷贝固化成功与
        失败结果。``self._results`` 在重规划采纳时会被清空，历史来源
        只能经此归档查证——归档独立于 ``results`` 存活，InMemory 与
        SQLite 场景行为一致（不依赖 ResultStore 读回，旧库无需迁移）。
        幂等：同版本重复归档按 subtask_id 覆盖。

        调用时机：``_plan_and_review`` 入口（重规划进入、请求新计划
        之前）——候选载荷的构造发生在主管返回新计划之前，归档必须
        先行，否则首次重规划的候选载荷查无来源；采纳新计划的
        ``if not problems`` 块不再重复归档。
        """
        for subtask_id, result in self._results.items():
            self._archive_results[(self._plan_version, subtask_id)] = (
                result.model_copy(deep=True)
            )

    def _reuse_candidates_payload(
        self,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """构造重规划输入的可复用候选载荷（最小充分信息）。

        候选条目覆盖上一版计划中**已有结果**（成功或失败）的子任务，
        每条含：旧 plan version、subtask_id、spec 语义字段（title /
        instructions / role / dependencies / allowed_tools /
        expected_output）、success、输出摘要（截断）、evidence、
        artifacts 与失败诊断（error，仅失败项）。失败项列入载荷是为
        了向主管完整呈现失败语境；只有 ``success=true`` 的条目才可能
        通过复用校验（校验器强制）。

        调用时机：主管拆解请求前（``self._subtasks`` 仍是旧计划、
        ``self._plan_version`` 仍是旧版本号）。

        Returns:
            ``(候选条目列表, 失败级联范围)``；级联范围为上一版失败
            子任务及其全部下游的 subtask_id 列表（含失败项自身）；
            无历史版本时两者均为空。
        """
        if self._plan_version < 1 or len(self._plan_history) < self._plan_version:
            return [], []
        previous_version = self._plan_version
        previous_specs = self._plan_history[previous_version - 1]
        failed_ids = {
            subtask_id
            for (version, subtask_id), result in self._archive_results.items()
            if version == previous_version and not result.success
        }
        cascade_scope = self._downstream_closure(failed_ids, previous_specs)

        candidates: list[dict[str, Any]] = []
        for spec in previous_specs:
            result = self._archive_results.get((previous_version, spec.subtask_id))
            if result is None:
                continue  # 未执行到：无结果可复用，也不构成失败语境
            entry: dict[str, Any] = {
                "plan_version": previous_version,
                "subtask_id": spec.subtask_id,
                "title": spec.title,
                "instructions": spec.instructions,
                "role": spec.role,
                "dependencies": list(spec.dependencies),
                "allowed_tools": list(spec.allowed_tools),
                "expected_output": spec.expected_output,
                "success": result.success,
                "output_summary": (result.output or "")[:500],
                "evidence": list(result.evidence)[:10],
                "artifacts": list(result.artifacts),
            }
            if not result.success:
                entry["failure_diagnosis"] = (result.error or "")[:500]
            candidates.append(entry)
        return candidates, sorted(cascade_scope)

    def _validate_reuse_declarations(self) -> dict[str, ExecutionResult]:
        """对新计划的显式 ``reuse_from`` 声明逐项安全校验。

        核心原则：LLM（主管）负责判断旧结果在新计划语境下是否可
        复用；代码不凭同名或 fingerprint 自动作语义决定，只校验声明
        是否满足安全硬约束。无声明时绝不复用，即使 id、名称与指纹
        完全相同。

        校验矩阵（全部通过才放行）：

        1. 来源存在：``reuse_from`` 必须是上一被采纳计划版本中存在
           且有归档结果的 subtask_id（``source_not_found``）；
        2. 来源成功：归档结果 ``success=True``（``source_not_successful``）；
        3. 非污染：来源不在上一版失败子任务的下游级联闭包内
           （``source_poisoned_by_failure``）；
        4. 指纹成分兼容：instructions / role / expected_output 完全
           一致，依赖集合经本轮合法复用声明形成的 id 映射后集合等价
           （``fingerprint_incompatible``；禁止文本相似度放宽）；
        5. 依赖成立：新任务依赖的旧对应者若留有成功结果，则该依赖
           必须同时复用通过（否则旧结果建立在被丢弃的旧成功结果
           之上，依赖链断裂）；旧对应无成功结果时允许其重新执行
           （``dependency_not_satisfied``）；
        6. artifacts 可信：本地路径必须位于沙箱范围内且无路径穿越、
           文件仍然存在；URL 等非本地引用跳过本地校验，纯文本伪装
           的产物路径无法通过存在性校验（``artifact_missing`` /
           ``artifact_outside_sandbox`` / ``artifact_path_traversal``）；
        7. 工具范围不扩大：新 spec ``allowed_tools`` 的 manifest 稳定
           标识集合必须 ⊆ 旧 spec（``tool_scope_expanded``）；
        8. 单一消费：同一旧结果不可被两个新任务重复引用
           （``duplicate_reference``，先完全通过者先消费）。

        校验通过：复制 ``ExecutionResult`` 到新 subtask_id（metadata
        写入 ``reused_from`` / ``reused_from_plan_version`` 来源标记，
        随序列化进入汇总与最终评审输入，评审不因复用而跳过），记录
        ``result_reused`` 事件。校验失败：记录 ``reuse_rejected`` 事件
        （机器可读 reason 码，不携带完整输出），新任务保持无结果
        状态 → 执行协调器的 pending 过滤自动将其放回正常执行队列
        （不静默成功）。

        固定点迭代：依赖映射与依赖成立校验依赖“同时复用”的其它
        声明，按“每轮用当前通过集重校验全部未通过声明”迭代至收敛
        （依赖链逐层解析：无依赖改名的一轮内先通过）。

        Returns:
            ``{新 subtask_id: 复制后的 ExecutionResult}``；无声明或
            全部被拒时为空字典。
        """
        declarations = [
            spec
            for spec in self._subtasks
            if spec.reuse_from and str(spec.reuse_from).strip()
        ]
        if not declarations:
            # 无显式声明：绝不复用（即使 id、名称与指纹完全相同）
            return {}

        task_id = self._task_request.task_id
        previous_version = self._plan_version - 1
        if 1 <= previous_version <= len(self._plan_history):
            previous_specs = self._plan_history[previous_version - 1]
        else:
            previous_specs = []
        previous_spec_map = {spec.subtask_id: spec for spec in previous_specs}

        # 上一版失败子任务及其下游污染闭包（按上一版依赖图计算）
        failed_ids = {
            subtask_id
            for (version, subtask_id), result in self._archive_results.items()
            if version == previous_version and not result.success
        }
        poisoned = self._downstream_closure(failed_ids, previous_specs)

        def _log_rejected(spec: SubtaskSpec, reason: str) -> None:
            # 机器可读 reason 码；不携带完整输出，防泄露
            self._events.log_event(
                EVENT_REUSE_REJECTED,
                {
                    "task_id": task_id,
                    "agent_id": DEFAULT_SUPERVISOR_ID,
                    "plan_version": self._plan_version,
                    "subtask_id": spec.subtask_id,
                    "source_subtask_id": str(spec.reuse_from).strip(),
                    "reason": reason,
                },
            )
            logger.warning(
                "复用声明被拒：[%s] ← [%s]（原因 %s），回退重新执行",
                spec.subtask_id,
                str(spec.reuse_from).strip(),
                reason,
            )

        # 静态校验（不依赖其它声明）：来源/成功/污染/artifacts/工具范围
        pending: list[tuple[SubtaskSpec, SubtaskSpec, ExecutionResult]] = []
        for spec in declarations:
            source_id = str(spec.reuse_from).strip()
            old_spec = previous_spec_map.get(source_id)
            old_result = self._archive_results.get((previous_version, source_id))
            if old_spec is None or old_result is None:
                _log_rejected(spec, "source_not_found")
                continue
            if not old_result.success:
                _log_rejected(spec, "source_not_successful")
                continue
            if source_id in poisoned:
                _log_rejected(spec, "source_poisoned_by_failure")
                continue
            artifact_problem = self._validate_reuse_artifacts(old_result)
            if artifact_problem is not None:
                _log_rejected(spec, artifact_problem)
                continue
            if not self._reuse_tool_scope_compatible(spec, old_spec):
                _log_rejected(spec, "tool_scope_expanded")
                continue
            pending.append((spec, old_spec, old_result))

        # 固定点迭代：指纹兼容（依赖映射来自当前通过集）+ 依赖成立 +
        # 单一消费（先完全通过者先消费来源；重复引用检查后置，收敛后
        # 按"指纹 → 依赖 → 重复引用"优先级诊断根本原因）
        passed: dict[str, tuple[str, ExecutionResult]] = {}
        while True:
            id_map = {source: new_id for new_id, (source, _) in passed.items()}
            progressed = False
            for item in list(pending):
                spec, old_spec, old_result = item
                if not specs_fingerprint_equivalent(spec, old_spec, id_map):
                    continue  # 映射可能随其它声明通过而扩展，下一轮再试
                problem = self._reuse_dependencies_satisfied(
                    spec, id_map, previous_version, passed
                )
                if problem is not None:
                    continue  # 依赖的声明可能后续轮次通过，下一轮再试
                source_id = str(spec.reuse_from).strip()
                if source_id in {source for source, _ in passed.values()}:
                    continue  # 来源已被其它声明消费：收敛后按重复引用拒绝
                passed[spec.subtask_id] = (source_id, old_result)
                pending.remove(item)
                progressed = True
            if not progressed:
                break

        # 收敛后剩余声明：最终诊断并拒绝（机器可读 reason；先报根本
        # 原因——指纹不兼容 / 依赖不满足，均不成立时才报重复引用）
        final_id_map = {source: new_id for new_id, (source, _) in passed.items()}
        for spec, old_spec, old_result in pending:
            if not specs_fingerprint_equivalent(spec, old_spec, final_id_map):
                reason = "fingerprint_incompatible"
            else:
                reason = self._reuse_dependencies_satisfied(
                    spec, final_id_map, previous_version, passed
                ) or "duplicate_reference"
            _log_rejected(spec, reason)

        # 通过校验：复制结果到新 subtask_id（来源标记写入 metadata，
        # 随序列化进入汇总与最终评审输入）并记录 result_reused 事件
        reused: dict[str, ExecutionResult] = {}
        for new_id, (source_id, old_result) in passed.items():
            metadata = dict(old_result.metadata or {})
            metadata["reused_from"] = source_id
            metadata["reused_from_plan_version"] = previous_version
            reused[new_id] = old_result.model_copy(
                update={"subtask_id": new_id, "metadata": metadata}
            )
            self._events.log_event(
                EVENT_RESULT_REUSED,
                {
                    "task_id": task_id,
                    "agent_id": DEFAULT_SUPERVISOR_ID,
                    "plan_version": self._plan_version,
                    "subtask_id": new_id,
                    "source_subtask_id": source_id,
                    "source_plan_version": previous_version,
                },
            )
        return reused

    def _validate_reuse_artifacts(self, result: ExecutionResult) -> str | None:
        """校验复用结果的 artifacts（返回 None 表示通过，否则 reason 码）。

        规则（任务 35 安全硬约束）：

        - URL 等非本地引用（含 ``://``）不是本地产物路径，跳过本地
          路径校验（非本地证据不能伪装为产物路径也无需路径校验）；
        - 其余条目视为本地产物路径：相对路径按沙箱内相对路径解释、
          绝对路径直接解析，解析结果必须位于沙箱范围内（越界且含
          ``..`` 为 ``artifact_path_traversal``，越界无 ``..`` 为
          ``artifact_outside_sandbox``），且文件必须仍然存在
          （``artifact_missing``；纯文本等非路径内容伪装的产物路径
          无法通过存在性校验，同样落入此 reason）。
        """
        sandbox = Path(self._context.sandbox_dir).resolve()
        for artifact in result.artifacts:
            if not isinstance(artifact, str) or not artifact.strip():
                continue
            text = artifact.strip()
            if "://" in text:
                continue  # URL 等非本地引用：不作本地路径校验
            raw = Path(text)
            resolved = (raw if raw.is_absolute() else sandbox / raw).resolve()
            traversal = ".." in raw.parts
            try:
                inside = resolved.is_relative_to(sandbox)
            except (TypeError, ValueError):
                inside = False
            if not inside:
                return (
                    "artifact_path_traversal"
                    if traversal
                    else "artifact_outside_sandbox"
                )
            if not resolved.exists():
                return "artifact_missing"
        return None

    def _tool_stable_ids(self, names: list[str]) -> set[str]:
        """把工具名集合映射为 manifest 稳定标识集合。

        动态生成工具（已注册）用 ``generated:{tool_id}`` 标识——跨版本
        改名后仍可经 manifest 稳定标识对齐；其余工具（内置/预置/
        未注册缺口）用 ``tool:{name}`` 名字标识（名字本身即稳定
        标识，未注册名不产生稳定 id，只能按名字比对）。
        """
        stable: set[str] = set()
        for name in names:
            entry = self._tool_registry.get_tool(name)
            tool_id = entry.get("tool_id") if entry else None
            if tool_id:
                stable.add(f"generated:{tool_id}")
            else:
                stable.add(f"tool:{name}")
        return stable

    def _reuse_tool_scope_compatible(
        self, new_spec: SubtaskSpec, old_spec: SubtaskSpec
    ) -> bool:
        """新 spec 的工具范围不得扩大旧 spec（按 manifest 稳定标识）。"""
        return self._tool_stable_ids(new_spec.allowed_tools) <= self._tool_stable_ids(
            old_spec.allowed_tools
        )

    def _reuse_dependencies_satisfied(
        self,
        new_spec: SubtaskSpec,
        id_map: dict[str, str],
        previous_version: int,
        passed: dict[str, tuple[str, ExecutionResult]],
    ) -> str | None:
        """校验复用任务的新依赖是否成立（返回 None 表示满足）。

        指纹兼容已保证新依赖集合 == 旧依赖集合经 ``id_map`` 映射。
        对每个新依赖 ``dep``：找到其旧对应 ``old_dep``（同名直通或
        ``id_map`` 中的改名目标）；若上一版 ``old_dep`` 留有**成功**
        结果，则该结果只有经“同时复用”保留（``dep`` 在通过集中）才
        仍然成立——否则 ``dep`` 将重新执行，复用的旧结果建立在被
        丢弃的旧成功结果之上，依赖链断裂。旧对应无成功结果（失败
        /未执行）时 ``dep`` 本来就会（重新）执行，“仍会在执行前
        完成”，允许。
        """
        reverse_map = {new: old for old, new in id_map.items()}
        passed_new_ids = set(passed.keys())
        for dep in new_spec.dependencies:
            old_dep = reverse_map.get(dep, dep)
            old_dep_result = self._archive_results.get(
                (previous_version, old_dep)
            )
            if old_dep_result is not None and old_dep_result.success:
                if dep not in passed_new_ids:
                    return "dependency_not_satisfied"
        return None

    def _downstream_closure(
        self, seeds: set[str], specs: list[SubtaskSpec] | None = None
    ) -> set[str]:
        """计算给定计划中种子子任务的全部下游子任务（含种子自身）。

        依据 ``specs``（默认当前计划 ``self._subtasks``）的依赖边沿
        扇出方向做 BFS：失败子任务的下游结果受级联影响不可信，不得
        跨版本复用（任务 35 校验器显式传入上一版计划计算污染闭包）。
        """
        effective = self._subtasks if specs is None else specs
        dependents: dict[str, list[str]] = {}
        for spec in effective:
            for dep in spec.dependencies:
                dependents.setdefault(dep, []).append(spec.subtask_id)
        closure = set(seeds)
        stack = list(seeds)
        while stack:
            current = stack.pop()
            for child in dependents.get(current, []):
                if child not in closure:
                    closure.add(child)
                    stack.append(child)
        return closure

    def _record_plan_version(self, issue: TaskDecompositionIssue | None) -> None:
        """把当前采纳的计划版本快照（含 spec 指纹索引）写入结果存储。

        指纹索引供跨版本的结果复用审计与持久化实现（SQLite）复用；
        存储未配置（依赖容器未注入）时静默跳过。
        """
        result_store = self._result_store
        if result_store is None:
            return
        plan_version = PlanVersion(
            version=self._plan_version,
            specs=[spec.model_copy(deep=True) for spec in self._subtasks],
            fingerprints={
                spec.subtask_id: spec_fingerprint(spec) for spec in self._subtasks
            },
            trigger_issue=issue.model_copy(deep=True) if issue is not None else None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        result_store.save_plan_version(plan_version)

    def _completed_results_summary(self) -> list[dict[str, Any]]:
        """汇总当前计划下已成功子任务的结果摘要（重规划载荷参考用）。

        LLM（主管）参考摘要调整计划；结果复用由主管的显式
        ``reuse_from`` 声明驱动（候选信息见 ``_reuse_candidates_payload``），
        代码只做安全校验（``_validate_reuse_declarations``）；失败结果
        不进入摘要（避免误导，失败语境由候选载荷与级联范围提供）。
        """
        summary: list[dict[str, Any]] = []
        for spec in self._subtasks:
            result = self._results.get(spec.subtask_id)
            if result is None or not result.success:
                continue
            summary.append(
                {
                    "subtask_id": spec.subtask_id,
                    "title": spec.title,
                    "fingerprint": spec_fingerprint(spec),
                    "output": result.output,
                    "artifacts": list(result.artifacts),
                }
            )
        return summary

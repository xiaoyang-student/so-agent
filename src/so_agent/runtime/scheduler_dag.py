"""子任务依赖图校验与拓扑排序（SubtaskDAGScheduler 的 DAG 能力）。

从 runtime.scheduler 拆分（行为不变）：SchedulerError 与 DAGMixin 在此
定义，scheduler 模块继续 re-export，外部导入路径保持不变。
"""

from __future__ import annotations

from so_agent.models import SubtaskSpec


class SchedulerError(Exception):
    """调度器领域错误：子任务清单为空、依赖非法、存在循环依赖等。"""


class DAGMixin:
    """子任务依赖图的拓扑排序与完整性校验。"""

    def topological_order(
        self,
        subtasks: list[SubtaskSpec],
        *,
        external_deps: set[str] | None = None,
    ) -> list[str]:
        """计算子任务的拓扑序（Kahn 算法，保持输入相对顺序）。

        Args:
            subtasks: 待调度子任务（可以是不含已完成上游的局部批次）。
            external_deps: 批次外已满足的依赖 ID（续跑/返工场景）。

        Raises:
            SchedulerError: 清单为空、ID 重复、自依赖、依赖缺失或存在环时。
        """
        if not subtasks:
            raise SchedulerError("子任务列表为空，无法调度")

        ids = [spec.subtask_id for spec in subtasks]
        if len(ids) != len(set(ids)):
            duplicated = sorted({sid for sid in ids if ids.count(sid) > 1})
            raise SchedulerError(f"存在重复的 subtask_id：{duplicated}")
        id_set = set(ids)
        known_external = set(external_deps or ())

        indegree: dict[str, int] = {sid: 0 for sid in ids}
        dependents: dict[str, list[str]] = {sid: [] for sid in ids}
        for spec in subtasks:
            for dep in spec.dependencies:
                if dep == spec.subtask_id:
                    raise SchedulerError(f"子任务 {spec.subtask_id} 依赖自身，非法")
                if dep not in id_set:
                    if dep in known_external:
                        continue
                    raise SchedulerError(
                        f"子任务 {spec.subtask_id} 依赖不存在的子任务 {dep!r}"
                    )
                indegree[spec.subtask_id] += 1
                dependents[dep].append(spec.subtask_id)

        queue = [sid for sid in ids if indegree[sid] == 0]
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for nxt in dependents[current]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(ids):
            remaining = sorted(set(ids) - set(order))
            raise SchedulerError(f"子任务依赖图存在循环依赖：{remaining}")
        return order

    def _validate_dag(
        self,
        subtasks: list[SubtaskSpec],
        *,
        external_deps: set[str] | None = None,
    ) -> None:
        """调度前完成依赖图全量校验（委托拓扑序计算）。"""
        self.topological_order(subtasks, external_deps=external_deps)

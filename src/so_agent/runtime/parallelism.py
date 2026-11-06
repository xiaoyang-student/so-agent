"""确定性并行度决策内核（多 Agent 效率治理）。

设计约束：并行与否由代码决定，模型只能给出依赖关系与工具声明。

- 计划形状（层宽 / 深度 / 写冲突）由纯函数编译，不调用模型；
- G1 形状资格 / G2 独立性证明 / G3 收益预检 在规划阶段一次判定；
- G4 运行时熔断由 :class:`RuntimeBreaker` 在调度期生效：provider 类
  故障或连续失败达到阈值后，剩余子任务收敛为串行，避免并发压力
  触发级联超时与重试放大；
- G5 事后校准需要真实运行样本，本模块只保留字段，不臆造结论。

本模块不依赖任何业务状态，可独立单元测试。
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# 只读工具白名单：仅允许明确只读的工具参与并行；未知工具一律按可写
# 处理（fail-closed），防止两个子任务并发改写同一文件造成返工。
READ_ONLY_TOOLS: frozenset[str] = frozenset({"read_file"})

# 判定阈值（与 docs/生产化重构设计.md 的五道闸门一致）
MIN_FAN_OUT_WIDTH: int = 3
NET_GAIN_THRESHOLD: float = 1.3
COORDINATION_COST: float = 0.25
MERGE_COST: float = 0.10
PAIR_CONCURRENCY: int = 2
LIMITED_CONCURRENCY: int = 2

# 效率目标（任务 103）：时间优先时接受更低的净收益门槛，但要求壁钟
# 加速比越过最小阈值——并行带来的额外 token 必须换来可感知的时间
# 收益，避免“巨大 token 换微小时间”。两个阈值对 token 成本本身不做
# 臆测建模，绝对消耗仍由任务级 token 预算熔断兜底。
OBJECTIVE_TIME_FIRST: str = "time_first"
OBJECTIVE_TOKEN_FIRST: str = "token_first"
TIME_FIRST_NET_GAIN_THRESHOLD: float = 1.10
TIME_FIRST_MIN_SPEEDUP: float = 1.10

MODE_SERIAL: str = "serial"
MODE_PAIR: str = "pair"
MODE_FAN_OUT: str = "fan_out"
MODE_LIMITED: str = "limited"

# 运行时熔断：这些失败类型说明并发压力或服务端容量已不可持续
PROVIDER_FAILURE_TYPES: frozenset[str] = frozenset(
    {"network", "rate_limit", "timeout", "provider"}
)


@dataclass(frozen=True)
class PlanShape:
    """计划的结构化形状（由依赖图确定性编译）。"""

    subtask_count: int
    width: int
    depth: int
    level_sizes: tuple[int, ...]
    parallel_fraction: float
    max_branch_ratio: float
    critical_path_ratio: float
    write_conflicts: tuple[str, ...] = ()
    conflict_subtask_ids: tuple[str, ...] = ()
    valid: bool = True
    issues: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典（事件留痕用）。"""
        return {
            "subtask_count": self.subtask_count,
            "width": self.width,
            "depth": self.depth,
            "level_sizes": list(self.level_sizes),
            "parallel_fraction": round(self.parallel_fraction, 4),
            "max_branch_ratio": round(self.max_branch_ratio, 4),
            "critical_path_ratio": round(self.critical_path_ratio, 4),
            "write_conflicts": list(self.write_conflicts),
            "conflict_subtask_ids": list(self.conflict_subtask_ids),
            "valid": self.valid,
            "issues": list(self.issues),
        }


def _subtask_id(spec: Any) -> str:
    return str(getattr(spec, "subtask_id", "") or "").strip()


def _dependencies(spec: Any) -> list[str]:
    raw = getattr(spec, "dependencies", None) or []
    return [str(dep).strip() for dep in raw if str(dep).strip()]


def _allowed_tools(spec: Any) -> list[str]:
    raw = getattr(spec, "allowed_tools", None) or []
    return [str(tool).strip() for tool in raw if str(tool).strip()]


def _normalize_scope_path(value: Any) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _write_scope(spec: Any) -> list[str]:
    raw = getattr(spec, "write_scope", None) or []
    scope: list[str] = []
    for item in raw:
        path = _normalize_scope_path(item)
        if path and path not in scope:
            scope.append(path)
    return scope


def is_write_capable(spec: Any) -> bool:
    """子任务是否可能与文件系统写操作相关（保守判定）。

    显式声明了 ``write_scope`` 的一律视为可写；无工具或仅持有只读
    白名单工具时视为只读；其余一律视为可写，因为未知工具的实际
    副作用无法在规划期证明。
    """
    if _write_scope(spec):
        return True
    tools = _allowed_tools(spec)
    if not tools:
        return False
    return any(tool not in READ_ONLY_TOOLS for tool in tools)


def _paths_overlap(left: str, right: str) -> bool:
    """两个沙箱相对路径是否存在写重叠（相等 / 前缀目录 / 通配匹配）。"""
    if not left or not right:
        return True  # 空路径视为沙箱根，与一切写入冲突
    if left == right:
        return True
    if left.startswith(right + "/") or right.startswith(left + "/"):
        return True
    if any(ch in left or ch in right for ch in "*?["):
        return fnmatch.fnmatch(left, right) or fnmatch.fnmatch(right, left)
    return False


def _scopes_conflict(left: list[str], right: list[str]) -> bool:
    """写范围是否冲突：任一侧未声明即按未知处理（fail-closed）。"""
    if not left or not right:
        return True
    return any(_paths_overlap(a, b) for a in left for b in right)



def _build_layers(
    specs: Sequence[Any],
) -> tuple[list[list[str]], list[str], list[str]]:
    """Kahn 分层：返回 (层列表, 问题清单, 环中残留节点)。"""
    ids: list[str] = []
    issues: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for spec in specs:
        sid = _subtask_id(spec)
        if not sid:
            issues.append("存在 subtask_id 为空的子任务")
            continue
        if sid in seen:
            duplicates.add(sid)
        seen.add(sid)
        ids.append(sid)
    if duplicates:
        issues.append(f"存在重复的 subtask_id：{sorted(duplicates)}")

    id_set = set(ids)
    deps: dict[str, list[str]] = {}
    missing: set[str] = set()
    for spec in specs:
        sid = _subtask_id(spec)
        if not sid or sid not in id_set:
            continue
        kept: list[str] = []
        for dep in _dependencies(spec):
            if dep == sid:
                continue
            if dep not in id_set:
                missing.add(dep)
                continue
            if dep not in kept:
                kept.append(dep)
        deps[sid] = kept
    if missing:
        issues.append(f"依赖指向不存在的子任务：{sorted(missing)}")

    layers: list[list[str]] = []
    remaining = set(deps)
    processed: set[str] = set()
    while remaining:
        layer = sorted(
            sid
            for sid in remaining
            if all(dep in processed for dep in deps.get(sid, []))
        )
        if not layer:
            break
        layers.append(layer)
        processed.update(layer)
        remaining.difference_update(layer)
    return layers, issues, sorted(remaining)


def compile_plan_shape(specs: Sequence[Any]) -> PlanShape:
    """把子任务清单编译为计划形状（纯函数，不抛异常）。"""
    specs = list(specs)
    n = len(specs)
    layers, issues, residual = _build_layers(specs)
    if residual:
        issues.append(f"依赖图存在环（涉及 {residual}）")
    valid = not issues and n > 0

    level_sizes = tuple(len(layer) for layer in layers)
    width = max(level_sizes) if level_sizes else 0
    depth = len(layers)
    parallel_work = sum(size for size in level_sizes if size > 1)
    parallel_fraction = parallel_work / n if n else 0.0
    max_branch_ratio = width / n if n else 0.0
    critical_path_ratio = depth / n if n else 0.0

    spec_by_id = {_subtask_id(spec): spec for spec in specs if _subtask_id(spec)}
    conflicts: list[str] = []
    conflict_ids: list[str] = []
    for index, layer in enumerate(layers, start=1):
        writers = [
            sid
            for sid in layer
            if sid in spec_by_id and is_write_capable(spec_by_id[sid])
        ]
        if len(writers) <= 1:
            continue
        # 写范围按声明判定：双方都声明且互不重叠才允许并行；任一方
        # 未声明即视为未知副作用（fail-closed），保守判为冲突。
        pairs: list[tuple[str, str]] = []
        for pos, left in enumerate(writers):
            left_scope = _write_scope(spec_by_id[left])
            for right in writers[pos + 1 :]:
                if _scopes_conflict(left_scope, _write_scope(spec_by_id[right])):
                    pairs.append((left, right))
        if pairs:
            detail = ", ".join(f"{left}↔{right}" for left, right in pairs)
            conflicts.append(
                f"第 {index} 层写范围冲突或未声明（{detail}）"
            )
            conflict_ids.extend(sid for pair in pairs for sid in pair)

    return PlanShape(
        subtask_count=n,
        width=width,
        depth=depth,
        level_sizes=level_sizes,
        parallel_fraction=parallel_fraction,
        max_branch_ratio=max_branch_ratio,
        critical_path_ratio=critical_path_ratio,
        write_conflicts=tuple(conflicts),
        conflict_subtask_ids=tuple(dict.fromkeys(conflict_ids)),
        valid=valid,
        issues=tuple(issues),
    )


def estimate_speedup(shape: PlanShape, workers: int) -> float:
    """Amdahl 上界：``S = 1 / ((1-p) + p/N)``，N 取实际并发度。"""
    n = max(1, int(workers))
    p = max(0.0, min(1.0, shape.parallel_fraction))
    return 1.0 / ((1.0 - p) + p / n)


def estimate_net_gain(shape: PlanShape, workers: int) -> float:
    """净收益：扣除协调与合并成本后的有效加速比。"""
    speedup = estimate_speedup(shape, workers)
    return speedup * (1.0 - COORDINATION_COST) * (1.0 - MERGE_COST)


@dataclass(frozen=True)
class ParallelismDecision:
    """并行度决策结果（可审计、可回放）。"""

    mode: str
    concurrency: int
    reason: str
    shape: PlanShape
    workers: int
    estimated_speedup: float
    net_gain: float
    objective: str = OBJECTIVE_TIME_FIRST
    net_gain_threshold: float = TIME_FIRST_NET_GAIN_THRESHOLD
    min_speedup: float = TIME_FIRST_MIN_SPEEDUP
    speedup_ok: bool = True
    gates: Mapping[str, bool] = field(default_factory=dict)

    @property
    def is_serial(self) -> bool:
        return self.mode == MODE_SERIAL

    def as_event_payload(self) -> dict[str, Any]:
        """转换为事件载荷（JSON 兼容）。"""
        return {
            "mode": self.mode,
            "concurrency": self.concurrency,
            "reason": self.reason,
            "workers": self.workers,
            "estimated_speedup": round(self.estimated_speedup, 4),
            "net_gain": round(self.net_gain, 4),
            "objective": self.objective,
            "net_gain_threshold": round(self.net_gain_threshold, 4),
            "min_speedup": round(self.min_speedup, 4),
            "speedup_ok": self.speedup_ok,
            "gates": dict(self.gates),
            "shape": self.shape.as_dict(),
        }


def _normalize_objective(objective: Any) -> str:
    """未知目标一律按 time_first 处理（用户默认诉求：先省时间）。"""
    value = str(objective or "").strip().lower()
    return (
        OBJECTIVE_TOKEN_FIRST
        if value == OBJECTIVE_TOKEN_FIRST
        else OBJECTIVE_TIME_FIRST
    )


def _resolve_thresholds(
    objective: str,
    net_gain_threshold: float | None,
    min_speedup: float | None,
) -> tuple[float, float]:
    """解析门槛：显式值 > 0 时覆盖目标默认，否则按目标取默认。"""
    time_first = objective == OBJECTIVE_TIME_FIRST
    default_gain = (
        TIME_FIRST_NET_GAIN_THRESHOLD if time_first else NET_GAIN_THRESHOLD
    )
    default_speedup = TIME_FIRST_MIN_SPEEDUP if time_first else 1.0
    gain = (
        float(net_gain_threshold)
        if net_gain_threshold is not None and float(net_gain_threshold) > 0
        else default_gain
    )
    floor = (
        float(min_speedup)
        if min_speedup is not None and float(min_speedup) > 0
        else default_speedup
    )
    return max(1.0, gain), max(1.0, floor)


def decide_parallelism(
    specs: Sequence[Any],
    *,
    max_concurrency: int = 3,
    objective: str = OBJECTIVE_TIME_FIRST,
    net_gain_threshold: float | None = None,
    min_speedup: float | None = None,
) -> ParallelismDecision:
    """按五道闸门的 G1–G3 做确定性并行度决策。

    - 形状非法 / 线性计划 / 层内写冲突 → 串行；
    - 层宽 ≥ 3 且净收益达标 → fan_out；
    - 层宽 ≥ 3 但净收益不达标、低并发仍能省时间 → limited；
    - 层宽 = 2 且加速比达标 → pair。

    ``objective`` 只影响门槛，不影响判定逻辑：

    - ``time_first``（默认）：净收益门槛降至 1.10，但要求壁钟加速比
      不低于 ``min_speedup``（默认 1.10）；并行多花 token 必须换来
      可感知的时间收益，否则串行；
    - ``token_first``：维持原 1.30 净收益门槛，最小加速比 1.0。

    额外 token 的绝对上限仍由任务级 token 预算熔断负责，本函数不臆造
    token 成本模型。G4（运行时熔断）在调度期由 :class:`RuntimeBreaker`
    生效；G5（事后校准）缺少真实样本，固定为 ``False``。
    """
    shape = compile_plan_shape(specs)
    cap = max(1, int(max_concurrency))
    objective_name = _normalize_objective(objective)
    gain_threshold, speedup_floor = _resolve_thresholds(
        objective_name, net_gain_threshold, min_speedup
    )
    meta = {
        "objective": objective_name,
        "net_gain_threshold": gain_threshold,
        "min_speedup": speedup_floor,
    }

    if not shape.valid:
        reason = "计划形状非法：" + "；".join(shape.issues or ("未知",))
        return ParallelismDecision(
            mode=MODE_SERIAL,
            concurrency=1,
            reason=reason,
            shape=shape,
            workers=1,
            estimated_speedup=1.0,
            net_gain=1.0,
            speedup_ok=False,
            **meta,
            gates={
                "g1_shape": False,
                "g2_independence": False,
                "g3_gain": False,
                "g3_speedup": False,
                "g4_runtime": True,
                "g5_calibration": False,
            },
        )

    if shape.width <= 1:
        return ParallelismDecision(
            mode=MODE_SERIAL,
            concurrency=1,
            reason="线性计划（层宽 ≤ 1），无并行收益",
            shape=shape,
            workers=1,
            estimated_speedup=1.0,
            net_gain=1.0,
            speedup_ok=False,
            **meta,
            gates={
                "g1_shape": False,
                "g2_independence": True,
                "g3_gain": False,
                "g3_speedup": False,
                "g4_runtime": True,
                "g5_calibration": False,
            },
        )

    if shape.write_conflicts:
        return ParallelismDecision(
            mode=MODE_SERIAL,
            concurrency=1,
            reason="层内写范围冲突或未声明：" + "；".join(shape.write_conflicts),
            shape=shape,
            workers=1,
            estimated_speedup=1.0,
            net_gain=1.0,
            speedup_ok=False,
            **meta,
            gates={
                "g1_shape": True,
                "g2_independence": False,
                "g3_gain": False,
                "g3_speedup": False,
                "g4_runtime": True,
                "g5_calibration": False,
            },
        )

    workers = min(shape.width, cap)

    def evaluate(concurrency: int) -> tuple[float, float, bool, bool]:
        concurrency = max(1, int(concurrency))
        speedup = estimate_speedup(shape, concurrency)
        net_gain = estimate_net_gain(shape, concurrency)
        return (
            speedup,
            net_gain,
            speedup >= speedup_floor,
            net_gain >= gain_threshold,
        )

    full_speedup, full_net, full_speedup_ok, full_gain_ok = evaluate(workers)
    gain_ok = full_gain_ok

    if shape.width >= MIN_FAN_OUT_WIDTH and full_gain_ok and full_speedup_ok:
        mode = MODE_FAN_OUT
        concurrency = workers
        speedup, net_gain, speedup_ok = full_speedup, full_net, True
        reason = (
            f"层宽 {shape.width}、加速比 {speedup:.2f}、净收益 "
            f"{net_gain:.2f} ≥ {gain_threshold:.2f}，允许扇出"
        )
    elif shape.width >= MIN_FAN_OUT_WIDTH:
        limited_workers = min(LIMITED_CONCURRENCY, workers)
        speedup, net_gain, speedup_ok, _ = evaluate(limited_workers)
        if speedup_ok:
            mode = MODE_LIMITED
            concurrency = limited_workers
            reason = (
                f"层宽 {shape.width} 但净收益 {full_net:.2f} < "
                f"{gain_threshold:.2f}，收敛到并发 {limited_workers}"
                f"（加速比 {speedup:.2f} ≥ {speedup_floor:.2f}）"
            )
        else:
            mode = MODE_SERIAL
            concurrency = 1
            speedup, net_gain = 1.0, 1.0
            reason = (
                f"并行加速比 {full_speedup:.2f} 低于最小要求 "
                f"{speedup_floor:.2f}，额外 token 换不到可感知的时间收益"
            )
    elif cap >= 2:
        pair_workers = min(PAIR_CONCURRENCY, workers)
        speedup, net_gain, speedup_ok, _ = evaluate(pair_workers)
        if speedup_ok:
            mode = MODE_PAIR
            concurrency = pair_workers
            reason = (
                f"层宽 2 的独立分支，加速比 {speedup:.2f} ≥ "
                f"{speedup_floor:.2f}，保留成对并行"
            )
        else:
            mode = MODE_SERIAL
            concurrency = 1
            speedup, net_gain = 1.0, 1.0
            speedup_ok = False
            reason = (
                f"并行加速比 {full_speedup:.2f} 低于最小要求 "
                f"{speedup_floor:.2f}，额外 token 换不到可感知的时间收益"
            )
    else:
        mode = MODE_SERIAL
        concurrency = 1
        speedup, net_gain, speedup_ok = 1.0, 1.0, False
        reason = "并发上限为 1，无法并行"

    return ParallelismDecision(
        mode=mode,
        concurrency=concurrency,
        reason=reason,
        shape=shape,
        workers=workers,
        estimated_speedup=speedup,
        net_gain=net_gain,
        speedup_ok=speedup_ok,
        **meta,
        gates={
            "g1_shape": shape.width >= 2,
            "g2_independence": True,
            "g3_gain": gain_ok,
            "g3_speedup": speedup_ok,
            "g4_runtime": True,
            "g5_calibration": False,
        },
    )


@dataclass
class RuntimeBreaker:
    """G4 运行时熔断器：失败压力下把剩余子任务收敛为串行。

    触发条件（任一）：
    - provider 类失败（network / rate_limit / timeout / provider）累计
      达到 ``provider_failure_limit``；
    - 连续失败达到 ``consecutive_failure_limit``。

    熔断后不会自动恢复：同一任务余下的子任务全部串行执行，避免并发
    压力继续放大重试与超时。
    """

    provider_failure_limit: int = 2
    consecutive_failure_limit: int = 3
    _provider_failures: int = field(default=0, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _tripped: bool = field(default=False, init=False)
    _trip_reason: str = field(default="", init=False)

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def record(self, *, success: bool, failure_type: str | None = None) -> bool:
        """记录一次子任务结果；返回熔断器当前是否已触发。"""
        if success:
            self._consecutive_failures = 0
            return self._tripped
        self._consecutive_failures += 1
        normalized = (failure_type or "").strip().lower()
        if normalized in PROVIDER_FAILURE_TYPES:
            self._provider_failures += 1
        if self._tripped:
            return True
        if self._provider_failures >= max(1, int(self.provider_failure_limit)):
            self._trip(
                f"provider 类失败累计 {self._provider_failures} 次"
            )
        elif self._consecutive_failures >= max(
            1, int(self.consecutive_failure_limit)
        ):
            self._trip(f"连续失败 {self._consecutive_failures} 次")
        return self._tripped

    def _trip(self, reason: str) -> None:
        self._tripped = True
        self._trip_reason = reason

    def reset(self) -> None:
        self._provider_failures = 0
        self._consecutive_failures = 0
        self._tripped = False
        self._trip_reason = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "tripped": self._tripped,
            "trip_reason": self._trip_reason,
            "provider_failures": self._provider_failures,
            "consecutive_failures": self._consecutive_failures,
        }


def record_breaker_result(
    breaker: RuntimeBreaker | None,
    result: Any,
) -> bool:
    """从 ExecutionResult 记录熔断样本；返回是否在本次调用中触发。"""
    if breaker is None:
        return False
    was_tripped = breaker.tripped
    metadata = getattr(result, "metadata", None) or {}
    failure_type = None
    if isinstance(metadata, Mapping):
        failure_type = metadata.get("failure_type")
    breaker.record(
        success=bool(getattr(result, "success", False)),
        failure_type=str(failure_type) if failure_type else None,
    )
    return breaker.tripped and not was_tripped


def collapse_ready(
    ready: Iterable[str],
    running: Mapping[Any, str],
    breaker: RuntimeBreaker | None,
) -> list[str]:
    """熔断时的派发收敛：已有任务运行时不追加，空闲时只放行一个。"""
    candidates = list(ready)
    if breaker is None or not breaker.tripped:
        return candidates
    if running:
        return []
    return candidates[:1]

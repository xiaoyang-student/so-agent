"""确定性失败恢复内核（任务 46）：分类 / 指纹 / 抑制 / 恢复策略。

本模块提供与 LLM 无关的纯确定性判定，供调度器（SubtaskDAGScheduler）
在真实失败时点调用：

- ``FailureType``：失败分类值域（字符串枚举，可平滑序列化与持久化）；
- ``FailureClassifier``：从失败结果 / 异常对象 / 文本做确定性分类
  （有序规则匹配；判定输入仅为结构化失败证据，LLM 输出无法改变结果）；
- ``normalize_failure_text`` / ``failure_fingerprint``：稳定失败指纹，
  消除 UUID / 时间戳 / 路径 / 长十六进制 / 数字等易变内容，使同一根因
  的重复失败收敛到同一指纹（跨进程、跨运行稳定）；
- ``FailureDeduplicator``：按作用域（task / subtask / agent）对指纹做
  连续计数，同因连续失败达到阈值后抑制重试（确定性去重，防无限重试）；
- ``RecoveryPolicy`` / ``RecoveryDecision`` / ``RecoveryAction``：纯数据
  恢复策略矩阵，决定重试 / 契约修复 / 替换 / 重规划 / 停止。仅语义能力
  与实现类故障（quality / tool / sandbox）允许替换；provider 类故障
  （network / rate_limit / timeout）可重试但禁止替换；format 类输出
  契约失败禁止替换（任务 69）：首次仅允许受限契约修复（修复轮不占
  业务尝试额度，不重跑完整业务任务），同一格式失败再现或修复额度
  耗尽即停止重试并转重规划（REPLAN），绝不发起 Replacement Agent。

安全不变量：
- 判定输入仅为结构化失败证据（错误文本、异常类型、结果状态），
  模型输出无法修改分类、指纹或策略矩阵；
- 来源信任边界（任务 69）：系统级失败类型（network / rate_limit /
  timeout / permission / invalid_input / budget / internal）只能由
  控制层证据决定——调度器捕获的异常对象（异常类名、HTTP 状态码）
  或工具 / 沙箱抛出的领域异常；Agent 返回的自由文本（执行结果的
  error / output）仅可判定低信任类别（quality / format / tool /
  sandbox，见 ``AGENT_TRUSTED_TEXT_TYPES``），不得凭关键词伪装
  provider / permission / budget / internal 等可信系统失败来改变
  恢复方向；
- 全部判定为纯函数 / 显式状态映射；退避抖动经构造期注入的 jitter
  函数显式引入（便于测试以确定性替身替换）。

范围说明（任务 46 修订）：预算与上下文治理已从本任务范围中排除——
本模块不包含任何预算计费 / token-cost 结算逻辑；pipeline deadline
由既有 PipelineGate 独家承担，这里不引入第二个预算状态源。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

from so_agent.models import ExecutionResult

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查（避免运行期依赖方向倒置）
    from so_agent.config import Settings

__all__ = [
    "FailureType",
    "PROVIDER_FAILURE_TYPES",
    "REPLACEABLE_FAILURE_TYPES",
    "CONTRACT_FAILURE_TYPES",
    "NON_RETRYABLE_FAILURE_TYPES",
    "BACKOFF_FAILURE_TYPES",
    "AGENT_TRUSTED_TEXT_TYPES",
    "normalize_failure_text",
    "failure_fingerprint",
    "FailureClassifier",
    "FailureDeduplicator",
    "RecoveryAction",
    "RecoveryDecision",
    "RecoveryPolicy",
]


class FailureType(str, Enum):
    """失败分类值域（确定性枚举，序列化为字符串）。

    分类器可能返回的值：network / rate_limit / timeout / permission /
    invalid_input / budget / quality / format / tool / sandbox /
    internal。``REPEATED`` 为抑制语义的保留标记（同因重复），分类器
    不会返回该值。
    """

    NETWORK = "network"            # 网络 / 网关 / 上游连接故障（provider 类）
    RATE_LIMIT = "rate_limit"      # 上游限流（provider 类）
    TIMEOUT = "timeout"            # 超时（provider 类）
    PERMISSION = "permission"      # 权限 / 授权缺失（不可重试）
    INVALID_INPUT = "invalid_input"  # 输入 / 参数非法（不可重试）
    BUDGET = "budget"              # 配额 / 计费耗尽（不可重试）
    QUALITY = "quality"            # 语义质量失败（可重试、可替换）
    FORMAT = "format"              # 输出契约 / 格式失败（受限契约修复；禁止替换，任务 69）
    TOOL = "tool"                  # 工具实现失败（可重试、可替换）
    SANDBOX = "sandbox"            # 沙箱 / 运行环境实现类失败（可重试、可替换）
    INTERNAL = "internal"          # 内部未知异常（保守不可重试）
    REPEATED = "repeated"          # 保留标记：同因重复（分类器不产生）


#: provider（模型 / 上游服务）类故障：可重试（network / rate_limit 带
#: 退避），但禁止 replacement——故障不源于子 Agent 的语义能力或实现。
PROVIDER_FAILURE_TYPES: frozenset[FailureType] = frozenset(
    {FailureType.NETWORK, FailureType.RATE_LIMIT, FailureType.TIMEOUT}
)

#: 语义能力 / 实现类故障：耗尽重试后允许 replacement。format 不在
#: 此集合内（任务 69）：输出契约失败指向规格 / 契约描述问题，换一个
#: Agent 并不会让契约更清楚——仅允许受限契约修复，同因再现或修复
#: 额度耗尽即转重规划，绝不发起替换。
REPLACEABLE_FAILURE_TYPES: frozenset[FailureType] = frozenset(
    {
        FailureType.QUALITY,
        FailureType.TOOL,
        FailureType.SANDBOX,
    }
)

#: 契约修复类故障（任务 69）：首次仅允许受限契约修复动作
#: （REPAIR_CONTRACT，修复轮不占业务尝试额度）；同一格式失败再现
#: （同因抑制）或修复额度耗尽后必须停止重试并转重规划（REPLAN），
#: 绝不创建 Replacement Agent，也不得重跑完整业务任务冒充契约修复。
CONTRACT_FAILURE_TYPES: frozenset[FailureType] = frozenset(
    {FailureType.FORMAT}
)

#: 不可重试故障：首次判定即终止本 Agent 的恢复路径（停止重试）。
NON_RETRYABLE_FAILURE_TYPES: frozenset[FailureType] = frozenset(
    {
        FailureType.PERMISSION,
        FailureType.INVALID_INPUT,
        FailureType.BUDGET,
        FailureType.INTERNAL,
    }
)

#: 需要指数退避的重试类型（provider 的瞬时过载 / 网络类故障）。
BACKOFF_FAILURE_TYPES: frozenset[FailureType] = frozenset(
    {FailureType.NETWORK, FailureType.RATE_LIMIT}
)

# ---------------------------------------------------------------------------
# 稳定失败指纹
# ---------------------------------------------------------------------------
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
_ISO_TIMESTAMP_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[t ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:z|[+-]\d{2}:?\d{2})?)?"
)
_WINDOWS_PATH_RE = re.compile(r"[a-z]:\\[^\s\"']+")
_POSIX_PATH_RE = re.compile(r"/(?:[^/\s\"']+/)+[^/\s\"']*")
_LONG_HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b")
_NUMBER_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_failure_text(text: str) -> str:
    """规范化失败文本：消除 UUID / 时间戳 / 文件路径 / 长十六进制串 /
    数字 / 空白等易变内容，使同一根因的失败文本收敛到同一形式。

    规范化是确定性的（仅依赖正则替换规则），同一输入必然得到同一输出。
    """
    normalized = (text or "").casefold()
    normalized = _UUID_RE.sub("<uuid>", normalized)
    normalized = _ISO_TIMESTAMP_RE.sub("<time>", normalized)
    normalized = _WINDOWS_PATH_RE.sub("<path>", normalized)
    normalized = _POSIX_PATH_RE.sub("<path>", normalized)
    normalized = _LONG_HEX_RE.sub("<hex>", normalized)
    normalized = _NUMBER_RE.sub("<num>", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return normalized


def failure_fingerprint(failure_type: FailureType, text: str) -> str:
    """稳定失败指纹：``sha256(类型 :: 规范化文本)`` 的前 16 位十六进制。

    指纹按失败类型隔离（同文本不同分类得到不同指纹），仅依赖确定性
    规范化规则，不含随机数、时间或地址等易变因子。
    """
    payload = f"{failure_type.value}::{normalize_failure_text(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------
# 有序文本规则表：先命中先返回（顺序即优先级，确定性）。
_TEXT_RULES: tuple[tuple[FailureType, tuple[str, ...]], ...] = (
    (
        FailureType.RATE_LIMIT,
        ("rate limit", "rate_limit", "ratelimit", "too many requests", "429"),
    ),
    (
        FailureType.PERMISSION,
        (
            "permission denied",
            "permission",
            "forbidden",
            "unauthorized",
            "not authorized",
            "未授权",
            "无权限",
            "权限不足",
            "401",
            "403",
        ),
    ),
    (
        FailureType.BUDGET,
        (
            "insufficient quota",
            "quota exceeded",
            "exceeded your current quota",
            "billing",
            "budget",
            "402",
        ),
    ),
    (
        FailureType.NETWORK,
        (
            "internal server error",
            "bad gateway",
            "gateway timeout",
            "service unavailable",
            "connection",
            "network",
            "socket",
            "dns",
            "could not connect",
            "broken pipe",
            "remote end closed",
            "502",
            "503",
            "504",
        ),
    ),
    (
        FailureType.TIMEOUT,
        ("timeout", "timed out", "time out", "deadline exceeded", "超时"),
    ),
    (
        FailureType.FORMAT,
        (
            "format",
            "json",
            "schema",
            "parse",
            "deserialize",
            "契约",
            "格式",
            "解析",
        ),
    ),
    (
        FailureType.INVALID_INPUT,
        ("invalid", "validation", "bad request", "参数", "400"),
    ),
    (
        FailureType.SANDBOX,
        ("sandbox", "沙箱", "container", "docker", "cgroup"),
    ),
    (
        FailureType.TOOL,
        ("tool error", "tool_error", "tool failed", "tool invocation", "mcp", "工具"),
    ),
    (
        FailureType.INTERNAL,
        ("internal", "内部错误", "unexpected error"),
    ),
)

# Agent 自由文本可判定的低信任失败类别（任务 69 来源信任边界）：
#: 执行结果（ExecutionResult）的 error / output 是子 Agent 或其执行
#: 链路可自由书写的文本，仅允许命中实现类低信任规则（FORMAT / TOOL /
#: SANDBOX，QUALITY 为未命中默认值）。provider 类（network /
#: rate_limit / timeout）与不可重试类（permission / invalid_input /
#: budget / internal）属于系统级失败，只能由控制层结构化证据决定：
#: 调度器捕获的异常对象（类名 / HTTP 状态码）、工具或沙箱抛出的
#: 领域异常。子 Agent 在 error 里写 “429” / “permission denied” /
#: “budget exhausted” 等关键词不得伪装成可信系统失败。
AGENT_TRUSTED_TEXT_TYPES: frozenset[FailureType] = frozenset(
    {
        FailureType.QUALITY,
        FailureType.FORMAT,
        FailureType.TOOL,
        FailureType.SANDBOX,
    }
)

# 低信任文本规则子表：从完整规则表按可判类别过滤派生（顺序保持）。
# 完整规则表仅供高信任入口使用（classify_exception 的异常文本回退）。
_AGENT_TEXT_RULES: tuple[tuple[FailureType, tuple[str, ...]], ...] = tuple(
    (failure_type, keywords)
    for failure_type, keywords in _TEXT_RULES
    if failure_type in AGENT_TRUSTED_TEXT_TYPES
)

# 异常类名规则（大小写折叠后的类名子串匹配，先命中先返回）
_EXCEPTION_NAME_RULES: tuple[tuple[FailureType, tuple[str, ...]], ...] = (
    (FailureType.TIMEOUT, ("timeout",)),
    (FailureType.RATE_LIMIT, ("ratelimit",)),
    (FailureType.NETWORK, ("connection", "network", "internalservererror")),
    (FailureType.PERMISSION, ("authentication", "permission", "authorization")),
    (FailureType.INVALID_INPUT, ("badrequest",)),
    (FailureType.BUDGET, ("quota", "billing")),
)

# HTTP 状态码映射（异常对象携带 status_code 时作为次要依据）
_STATUS_CODE_MAP: dict[int, FailureType] = {
    400: FailureType.INVALID_INPUT,
    401: FailureType.PERMISSION,
    402: FailureType.BUDGET,
    403: FailureType.PERMISSION,
    408: FailureType.TIMEOUT,
    429: FailureType.RATE_LIMIT,
    502: FailureType.NETWORK,
    503: FailureType.NETWORK,
    504: FailureType.NETWORK,
}


def _match_rules(
    rules: tuple[tuple[FailureType, tuple[str, ...]], ...],
    text: str,
    *,
    default: FailureType,
) -> FailureType:
    """按有序规则表做子串匹配（先命中先返回；空文本返回默认值）。"""
    lowered = (text or "").casefold()
    if not lowered:
        return default
    for failure_type, keywords in rules:
        if any(keyword in lowered for keyword in keywords):
            return failure_type
    return default


class FailureClassifier:
    """确定性失败分类器（无状态、纯函数式；任务 69 增设来源信任边界）。

    - ``classify_result``：以失败结果文本（error 优先，回退 output）按
      **低信任规则子表**（FORMAT / TOOL / SANDBOX）分类；无命中时默认
      QUALITY（子 Agent 语义失败）。系统级失败类型（provider 类与
      不可重试类）绝不因该文本的关键词成立——那是 Agent 侧可自由
      书写的文本，不构成可信证据；
    - ``classify_exception``：对**控制层捕获的异常对象**分类（高信任
      证据：类名 → HTTP 状态码 → 异常文本完整规则，最后保守默认
      INTERNAL）。真实系统失败（网络 / 限流 / 权限 / 配额 / 内部
      错误）应经异常对象进入本入口，保留 status_code 等结构化属性；
    - ``classify_text``：按**完整规则表**的通用有序匹配，默认值由
      调用方指定。仅供控制层可信文本（如异常消息回退）使用；
      Agent 自由文本必须走 ``classify_result`` 的低信任路径。

    注意：``asyncio.CancelledError`` 属于取消语义，不得传入本分类器
    （调度器直接向上传播，不进入失败分类路径）。
    """

    def classify_result(self, result: ExecutionResult) -> FailureType:
        """对失败的执行结果做分类（低信任文本路径，任务 69）。

        执行结果的 error / output 是 Agent 侧可自由书写的文本：仅按
        低信任规则子表（FORMAT / TOOL / SANDBOX）匹配，未命中按
        QUALITY（子 Agent 语义失败）处理。子 Agent 自述 “429” /
        “permission denied” / “budget exhausted” / “internal server
        error” 等关键词不得判定为对应的系统级失败类型；真实系统失败
        必须经控制层异常对象（``classify_exception``）等高信任证据。

        成功结果按 QUALITY 返回（调用方不应对成功结果做恢复判定）。
        """
        if result.success:
            return FailureType.QUALITY
        text = result.error or result.output or ""
        return _match_rules(_AGENT_TEXT_RULES, text, default=FailureType.QUALITY)

    def classify_exception(self, exc: BaseException) -> FailureType:
        """按控制层捕获的异常对象分类（高信任：类名 → 状态码 → 文本）。"""
        name = type(exc).__name__.casefold()
        for failure_type, keywords in _EXCEPTION_NAME_RULES:
            if any(keyword in name for keyword in keywords):
                return failure_type
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and 400 <= status < 600:
            if status in _STATUS_CODE_MAP:
                return _STATUS_CODE_MAP[status]
            return FailureType.NETWORK if status >= 500 else FailureType.INVALID_INPUT
        return self.classify_text(f"{name}: {exc}", default=FailureType.INTERNAL)

    def classify_text(
        self, text: str, *, default: FailureType
    ) -> FailureType:
        """按完整规则表对控制层可信文本分类；全部未命中时返回 ``default``。

        仅用于高信任文本（异常消息回退、控制层自构错误文本）；Agent
        自由文本（执行结果 error / output）必须走 ``classify_result``
        的低信任子表，不得调用本方法。
        """
        return _match_rules(_TEXT_RULES, text, default=default)


# ---------------------------------------------------------------------------
# 同因抑制
# ---------------------------------------------------------------------------
@dataclass
class _DedupState:
    fingerprint: str
    count: int


class FailureDeduplicator:
    """同因（指纹）连续失败计数器（作用域粒度，连续语义）。

    作用域为不透明元组（调度器使用 ``(task_id, subtask_id, agent_id)``）：
    同一作用域内同一指纹连续出现时计数递增，指纹变化即从 1 重新计数。
    出现次数达到阈值后 :meth:`suppressed` 返回 True，调用方据此停止重试
    （抑制）。替代 Agent 与新计划版本是新的作用域，天然独立计数重置。
    """

    def __init__(self, threshold: int = 2) -> None:
        self._threshold = max(1, int(threshold))
        self._states: dict[tuple[str, ...], _DedupState] = {}

    @property
    def threshold(self) -> int:
        """抑制阈值（同指纹连续失败达到该次数即抑制重试）。"""
        return self._threshold

    def record(self, scope: tuple[str, ...], fingerprint: str) -> int:
        """记录一次失败，返回该指纹在作用域内的连续出现次数。"""
        state = self._states.get(scope)
        if state is None or state.fingerprint != fingerprint:
            state = _DedupState(fingerprint=fingerprint, count=1)
            self._states[scope] = state
        else:
            state.count += 1
        return state.count

    def suppressed(self, occurrences: int) -> bool:
        """出现次数是否达到抑制阈值（True 表示应停止重试）。"""
        return occurrences >= self._threshold

    def reset(self, scope: tuple[str, ...]) -> None:
        """清除某作用域的计数（作用域生命周期结束时可选调用）。"""
        self._states.pop(scope, None)


# ---------------------------------------------------------------------------
# 恢复策略
# ---------------------------------------------------------------------------
class RecoveryAction(str, Enum):
    """恢复动作值域（字符串枚举，可平滑写入事件与执行记录）。"""

    RETRY = "retry"                      # 原 Agent 重试（可携带退避延迟；消耗业务尝试额度）
    REPAIR_CONTRACT = "repair_contract"  # 受限契约修复轮（任务 69：不占业务
                                          # 尝试额度，仅修输出契约，不重跑业务任务）
    REPLACE = "replace"                  # 停止重试，发起替换
    REPLAN = "replan"                    # 停止重试，上报分解问题触发重规划
    STOP = "stop"                        # 终止恢复路径（不可重试类故障）


@dataclass(frozen=True)
class RecoveryDecision:
    """一次恢复判定结果（纯数据，便于审计事件与执行记录落账）。"""

    action: RecoveryAction
    failure_type: FailureType
    fingerprint: str
    occurrences: int
    suppressed: bool
    reason: str
    backoff_seconds: float = 0.0

    @property
    def is_terminal(self) -> bool:
        """是否终止本 Agent 的尝试循环（交由调度器接手恢复动作）。"""
        return self.action in (
            RecoveryAction.REPLACE,
            RecoveryAction.REPLAN,
            RecoveryAction.STOP,
        )


class RecoveryPolicy:
    """确定性恢复策略矩阵（纯数据映射，LLM 无法修改；任务 69 修订）。

    决策优先级（自上而下，先命中先返回）：

    1. 不可重试类（permission / invalid_input / budget / internal）→ STOP；
    2. 同指纹连续失败达到抑制阈值 → 语义 / 实现类 REPLACE；provider 类
       与契约类（format）REPLAN（均禁止替换，措辞区分来源）；
    3. 输出契约类（format，见 ``CONTRACT_FAILURE_TYPES``）→ 修复额度内
       REPAIR_CONTRACT（受限修复轮，不占业务尝试额度，故先于业务
       尝试上限判定）；额度用尽 → REPLAN（禁止替换，任务 69）；
    4. 尝试次数已达上限 → 语义 / 实现类 REPLACE、provider 类 REPLAN、
       其余禁替换类型 REPLAN；
    5. network / rate_limit → 指数退避（cap 截断，可注入抖动）后 RETRY；
    6. 其余可重试类型 → 立即 RETRY。

    Args:
        repair_contract_max_attempts: 契约修复重试上限（0 表示不修复）。
        backoff_base_seconds: 退避基数（秒）。
        backoff_cap_seconds: 单次退避上限（秒）。
        backoff_jitter_ratio: 抖动比例（0~1；最终延迟乘 1 + ratio * jitter）。
        jitter: 抖动采样函数（默认 ``random.random``；测试注入固定值
            以获得确定性延迟）。
    """

    def __init__(
        self,
        *,
        repair_contract_max_attempts: int = 1,
        backoff_base_seconds: float = 1.0,
        backoff_cap_seconds: float = 30.0,
        backoff_jitter_ratio: float = 0.2,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._repair_max = max(0, int(repair_contract_max_attempts))
        self._backoff_base = max(0.0, float(backoff_base_seconds))
        self._backoff_cap = max(0.0, float(backoff_cap_seconds))
        self._jitter_ratio = min(1.0, max(0.0, float(backoff_jitter_ratio)))
        self._jitter = jitter

    @classmethod
    def from_settings(cls, settings: "Settings") -> "RecoveryPolicy":
        """从全局配置构造策略（参数见 Settings.recovery_* 字段）。"""
        return cls(
            repair_contract_max_attempts=settings.recovery_repair_contract_max_attempts,
            backoff_base_seconds=settings.recovery_backoff_base_seconds,
            backoff_cap_seconds=settings.recovery_backoff_cap_seconds,
            backoff_jitter_ratio=settings.recovery_backoff_jitter_ratio,
        )

    def decide(
        self,
        *,
        failure_type: FailureType,
        fingerprint: str,
        occurrences: int,
        suppressed: bool,
        attempt: int,
        max_attempts: int,
        repair_used: int = 0,
    ) -> RecoveryDecision:
        """给出恢复动作（纯函数式判定：无副作用；抖动仅为显式注入源）。"""
        occ = max(1, int(occurrences))
        attempt = max(1, int(attempt))
        max_attempts = max(1, int(max_attempts))

        def _decision(
            action: RecoveryAction, reason: str, backoff: float = 0.0
        ) -> RecoveryDecision:
            return RecoveryDecision(
                action=action,
                failure_type=failure_type,
                fingerprint=fingerprint,
                occurrences=occ,
                suppressed=suppressed,
                reason=reason,
                backoff_seconds=backoff,
            )

        # 1) 不可重试类：首次判定即终止恢复路径
        if failure_type in NON_RETRYABLE_FAILURE_TYPES:
            return _decision(
                RecoveryAction.STOP,
                f"{failure_type.value} 类故障不可重试，终止恢复路径",
            )

        # 2) 同因连续失败达到抑制阈值：停止重试（确定性去重）。契约类
        #（format）在此终止：同一格式失败再现即转重规划，绝不再修复、
        # 更不替换（任务 69）
        if suppressed:
            if failure_type in REPLACEABLE_FAILURE_TYPES:
                return _decision(
                    RecoveryAction.REPLACE,
                    f"同因失败连续 {occ} 次（指纹 {fingerprint[:8]}），停止重试并替换",
                )
            if failure_type in PROVIDER_FAILURE_TYPES:
                return _decision(
                    RecoveryAction.REPLAN,
                    f"同因失败连续 {occ} 次（provider 类故障禁止替换），转入重规划",
                )
            return _decision(
                RecoveryAction.REPLAN,
                f"同因失败连续 {occ} 次（指纹 {fingerprint[:8]}；"
                f"{failure_type.value} 类故障禁止替换），转入重规划",
            )

        # 3) 输出契约类（任务 69）：修复额度内进入受限契约修复轮
        #（REPAIR_CONTRACT，不占业务尝试额度，因此先于尝试上限判定，
        # 修复机会不被业务上限剥夺）；额度用尽转重规划，绝不替换、
        # 也不得重跑完整业务任务冒充修复
        if failure_type in CONTRACT_FAILURE_TYPES:
            if repair_used < self._repair_max:
                return _decision(
                    RecoveryAction.REPAIR_CONTRACT,
                    f"输出契约失败，进入受限契约修复轮"
                    f"（第 {repair_used + 1} 次修复，不占业务尝试额度）",
                )
            return _decision(
                RecoveryAction.REPLAN,
                "输出契约修复额度已用尽，停止重试并转重规划（format 禁止替换）",
            )

        # 4) 尝试次数已达上限
        if attempt >= max_attempts:
            if failure_type in REPLACEABLE_FAILURE_TYPES:
                return _decision(
                    RecoveryAction.REPLACE,
                    f"尝试次数已达上限（{attempt}/{max_attempts}），发起替换",
                )
            if failure_type in PROVIDER_FAILURE_TYPES:
                return _decision(
                    RecoveryAction.REPLAN,
                    f"尝试次数已达上限（{attempt}/{max_attempts}），"
                    f"provider 类故障禁止替换，转入重规划",
                )
            return _decision(
                RecoveryAction.REPLAN,
                f"尝试次数已达上限（{attempt}/{max_attempts}）；"
                f"{failure_type.value} 类故障禁止替换，转入重规划",
            )

        # 5) 退避重试（network / rate_limit 瞬时故障）
        if failure_type in BACKOFF_FAILURE_TYPES:
            delay = self._next_backoff(occ)
            return _decision(
                RecoveryAction.RETRY,
                f"{failure_type.value} 类瞬时故障，退避 {delay:.2f}s 后重试",
                backoff=delay,
            )

        # 6) 其余可重试类型立即重试
        return _decision(
            RecoveryAction.RETRY, f"{failure_type.value} 类失败，按策略重试"
        )

    def _next_backoff(self, occurrences: int) -> float:
        """指数退避延迟：``min(cap, base * 2^(n-1)) * (1 + jitter_ratio * jitter)``。"""
        if self._backoff_base <= 0.0 or self._backoff_cap <= 0.0:
            return 0.0
        raw = self._backoff_base * (2 ** (max(1, occurrences) - 1))
        capped = min(self._backoff_cap, raw)
        jitter_value = min(1.0, max(0.0, float(self._jitter())))
        return capped * (1.0 + self._jitter_ratio * jitter_value)

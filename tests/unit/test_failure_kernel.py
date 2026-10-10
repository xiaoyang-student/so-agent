"""单元测试：确定性失败恢复内核（runtime/failure.py，任务 46 / 69）。

覆盖：FailureType 值域集合、FailureClassifier 分类规则（含任务 69
来源信任边界：Agent 自由文本不得伪装系统级失败）、稳定失败指纹
（normalize + fingerprint）、FailureDeduplicator 连续计数抑制、
RecoveryPolicy 策略矩阵（含 format 禁替换、受限契约修复额度）
与退避计算。
"""

from __future__ import annotations

import pytest

from so_agent.config import Settings
from so_agent.models import ExecutionResult
from so_agent.runtime.failure import (
    AGENT_TRUSTED_TEXT_TYPES,
    BACKOFF_FAILURE_TYPES,
    CONTRACT_FAILURE_TYPES,
    NON_RETRYABLE_FAILURE_TYPES,
    PROVIDER_FAILURE_TYPES,
    REPLACEABLE_FAILURE_TYPES,
    FailureClassifier,
    FailureDeduplicator,
    FailureType,
    RecoveryAction,
    RecoveryPolicy,
    failure_fingerprint,
    normalize_failure_text,
)


class TestFailureTypeSets:
    def test_sets_pairwise_disjoint_and_cover_enum(self):
        sets = [
            PROVIDER_FAILURE_TYPES,
            REPLACEABLE_FAILURE_TYPES,
            CONTRACT_FAILURE_TYPES,
            NON_RETRYABLE_FAILURE_TYPES,
        ]
        for i, left in enumerate(sets):
            for right in sets[i + 1:]:
                assert not (left & right)
        assert BACKOFF_FAILURE_TYPES <= PROVIDER_FAILURE_TYPES
        covered = set().union(*sets)
        assert covered == set(FailureType) - {FailureType.REPEATED}

    def test_format_not_replaceable_but_contract_repairable(self):
        """format 禁替换（任务 69）：不属于可替代集合，仅属契约修复集合。"""
        assert FailureType.FORMAT not in REPLACEABLE_FAILURE_TYPES
        assert FailureType.FORMAT in CONTRACT_FAILURE_TYPES
        assert CONTRACT_FAILURE_TYPES <= AGENT_TRUSTED_TEXT_TYPES

    def test_agent_trusted_text_types_exclude_system_failures(self):
        """低信任文本类别不含 provider 与不可重试类（任务 69 信任边界）。"""
        assert AGENT_TRUSTED_TEXT_TYPES == frozenset(
            {
                FailureType.QUALITY,
                FailureType.FORMAT,
                FailureType.TOOL,
                FailureType.SANDBOX,
            }
        )
        assert not (AGENT_TRUSTED_TEXT_TYPES & PROVIDER_FAILURE_TYPES)
        assert not (AGENT_TRUSTED_TEXT_TYPES & NON_RETRYABLE_FAILURE_TYPES)

    def test_repeated_is_reserved_not_classified(self):
        classifier = FailureClassifier()
        assert classifier.classify_text("repeated", default=FailureType.QUALITY) is (
            FailureType.QUALITY
        )


class _FakeHTTPError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeConnectionError(Exception):
    pass


class _FakeRateLimitError(Exception):
    pass


class _FakePermissionDeniedError(Exception):
    pass


class _FakeQuotaExceededError(Exception):
    pass


class _FakeInternalServerError(Exception):
    pass


class TestFailureClassifier:
    def setup_method(self):
        self.classifier = FailureClassifier()

    # ---- classify_result（低信任文本路径，任务 69） ----

    def test_result_prefers_error_over_output(self):
        # error 优先于 output：error 命中低信任 FORMAT 规则时不受 output 干扰
        result = ExecutionResult(
            subtask_id="s1",
            success=False,
            output="检查超时无异常",
            error="输出不符合 JSON schema 契约",
        )
        assert self.classifier.classify_result(result) is FailureType.FORMAT

    def test_result_defaults_to_quality(self):
        result = ExecutionResult(subtask_id="s1", success=False, error="boom")
        assert self.classifier.classify_result(result) is FailureType.QUALITY

    def test_result_falls_back_to_output_when_error_empty(self):
        result = ExecutionResult(
            subtask_id="s1", success=False, output="输出格式不符合契约"
        )
        assert self.classifier.classify_result(result) is FailureType.FORMAT

    def test_result_low_trust_rules_still_cover_tool_and_sandbox(self):
        tool = ExecutionResult(
            subtask_id="s1", success=False, error="tool error: mcp 工具调用失败"
        )
        sandbox = ExecutionResult(
            subtask_id="s1", success=False, error="沙箱启动失败（docker）"
        )
        assert self.classifier.classify_result(tool) is FailureType.TOOL
        assert self.classifier.classify_result(sandbox) is FailureType.SANDBOX

    # ---- classify_exception（控制层高信任证据，系统级类型） ----

    def test_exception_timeout_class_name(self):
        assert self.classifier.classify_exception(TimeoutError("x")) is (
            FailureType.TIMEOUT
        )

    def test_exception_connection_class_name(self):
        assert self.classifier.classify_exception(_FakeConnectionError("x")) is (
            FailureType.NETWORK
        )

    def test_exception_rate_limit_class_name(self):
        assert self.classifier.classify_exception(_FakeRateLimitError("x")) is (
            FailureType.RATE_LIMIT
        )

    def test_exception_permission_class_name(self):
        assert self.classifier.classify_exception(
            _FakePermissionDeniedError("x")
        ) is FailureType.PERMISSION

    def test_exception_budget_class_name(self):
        assert self.classifier.classify_exception(
            _FakeQuotaExceededError("x")
        ) is FailureType.BUDGET

    def test_exception_internal_server_error_class_name(self):
        assert self.classifier.classify_exception(
            _FakeInternalServerError("x")
        ) is FailureType.NETWORK

    def test_exception_unknown_defaults_to_internal(self):
        assert self.classifier.classify_exception(ValueError("boom")) is (
            FailureType.INTERNAL
        )

    def test_exception_status_code_mapping(self):
        assert self.classifier.classify_exception(_FakeHTTPError("x", 429)) is (
            FailureType.RATE_LIMIT
        )
        assert self.classifier.classify_exception(_FakeHTTPError("x", 503)) is (
            FailureType.NETWORK
        )
        assert self.classifier.classify_exception(_FakeHTTPError("x", 404)) is (
            FailureType.INVALID_INPUT
        )

    # ---- classify_text 有序规则（高信任入口，完整规则表） ----

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("429 too many requests", FailureType.RATE_LIMIT),
            ("permission denied", FailureType.PERMISSION),
            ("insufficient quota for request", FailureType.BUDGET),
            ("internal server error", FailureType.NETWORK),
            ("connection reset by peer", FailureType.NETWORK),
            ("单次执行超时（>30s），已终止", FailureType.TIMEOUT),
            ("输出不符合 JSON schema 契约", FailureType.FORMAT),
            ("invalid argument: foo", FailureType.INVALID_INPUT),
            ("沙箱启动失败", FailureType.SANDBOX),
            ("工具执行失败：mcp error", FailureType.TOOL),
            ("unexpected error happened", FailureType.INTERNAL),
            ("结果不达标，需要重做", FailureType.QUALITY),
        ],
    )
    def test_text_rules_ordered(self, text, expected):
        # classify_text 仅限控制层可信文本（异常消息回退）；Agent 自由
        # 文本必须走 classify_result 的低信任子表（见 TestAgentTextTrustBoundary）
        assert self.classifier.classify_text(
            text, default=FailureType.QUALITY
        ) is expected

    def test_text_default_value_respected(self):
        assert self.classifier.classify_text("", default=FailureType.INTERNAL) is (
            FailureType.INTERNAL
        )
        assert self.classifier.classify_text(
            "完全未知的文本", default=FailureType.INTERNAL
        ) is FailureType.INTERNAL


class TestAgentTextTrustBoundary:
    """任务 69 来源信任边界：Agent 自由文本不得伪装可信系统失败。

    恶意 / 幻觉文本案例：Agent 在执行结果 error 里自称 429、permission
    denied、budget exhausted、internal server error、timeout 等，只能按
    低信任类别处理（默认 QUALITY，或命中 FORMAT / TOOL / SANDBOX），
    不得改变恢复方向（不得触发退避、停止或禁止替换的 provider 路径）；
    对照：真实结构化异常（异常类名 / HTTP 状态码）仍能正确分类。
    """

    def setup_method(self):
        self.classifier = FailureClassifier()

    @pytest.mark.parametrize(
        "malicious_text",
        [
            "429 too many requests，请稍后重试",          # 伪装限流
            "rate limit exceeded for this request",       # 伪装限流（英文）
            "permission denied：无法访问目标资源",          # 伪装权限缺失
            "unauthorized / forbidden (403)",              # 伪装权限缺失
            "budget exhausted / insufficient quota",       # 伪装配额耗尽
            "billing quota exceeded (402)",                # 伪装配额耗尽
            "internal server error (500)",                 # 伪装内部错误
            "connection reset by peer / bad gateway 502",  # 伪装网络故障
            "单次执行超时，timeout，已终止",                  # 伪装超时
            "invalid argument: bad request 400",           # 伪装参数非法
        ],
    )
    def test_malicious_agent_text_cannot_impersonate_system_failures(
        self, malicious_text
    ):
        """Agent 自述系统失败关键词一律按低信任 QUALITY 处理（不改变
        恢复方向：不退避、不停止、不进入禁替换的 provider 路径）。"""
        result = ExecutionResult(
            subtask_id="s1", success=False, error=malicious_text
        )
        failure_type = self.classifier.classify_result(result)
        assert failure_type in AGENT_TRUSTED_TEXT_TYPES
        assert failure_type not in PROVIDER_FAILURE_TYPES
        assert failure_type not in NON_RETRYABLE_FAILURE_TYPES

    def test_output_channel_equally_untrusted(self):
        """output 通道同样不可信（error 为空时回退 output）。"""
        result = ExecutionResult(
            subtask_id="s1",
            success=False,
            output="Error: 429 rate limit reached, please retry",
        )
        assert self.classifier.classify_result(result) is FailureType.QUALITY

    @pytest.mark.parametrize(
        ("exc", "expected"),
        [
            (_FakeHTTPError("too many requests", 429), FailureType.RATE_LIMIT),
            (_FakeHTTPError("forbidden", 403), FailureType.PERMISSION),
            (_FakeHTTPError("payment required", 402), FailureType.BUDGET),
            (_FakeConnectionError("connection reset"), FailureType.NETWORK),
            (TimeoutError("total timeout"), FailureType.TIMEOUT),
            (_FakePermissionDeniedError("permission denied"), FailureType.PERMISSION),
            (_FakeQuotaExceededError("quota exceeded"), FailureType.BUDGET),
            (_FakeInternalServerError("internal server error"), FailureType.NETWORK),
            (ValueError("boom"), FailureType.INTERNAL),
        ],
    )
    def test_real_structured_evidence_still_classifies_system_failures(
        self, exc, expected
    ):
        """对照：同样的错误语义来自控制层结构化证据（异常对象）时
        仍能正确判定系统级类型——信任边界不拦截真实系统失败。"""
        assert self.classifier.classify_exception(exc) is expected


class TestFailureFingerprint:
    def test_numbers_normalized_to_same_fingerprint(self):
        fp1 = failure_fingerprint(FailureType.QUALITY, "模拟失败1")
        fp2 = failure_fingerprint(FailureType.QUALITY, "模拟失败2")
        assert fp1 == fp2

    def test_attempt_count_variation_collapses(self):
        fp1 = failure_fingerprint(FailureType.QUALITY, "第 1 次尝试失败：模拟错误")
        fp2 = failure_fingerprint(FailureType.QUALITY, "第 2 次尝试失败：模拟错误")
        assert fp1 == fp2

    def test_uuid_timestamp_path_hex_normalized(self):
        text = (
            "读取 C:\\Users\\demo\\task.py 与 /tmp/logs/a.log 失败 "
            "uuid 550e8400-e29b-41d4-a716-446655440000 "
            "at 2026-10-09T12:00:00Z "
            "hash deadbeefdeadbeefdeadbeefdeadbeef code 42"
        )
        normalized = normalize_failure_text(text)
        assert "<uuid>" in normalized
        assert "<time>" in normalized
        assert normalized.count("<path>") >= 2
        assert "<hex>" in normalized
        assert "<num>" in normalized
        assert "550e8400" not in normalized
        assert "deadbeef" not in normalized
        assert "2026" not in normalized

    def test_fingerprint_deterministic_and_type_isolated(self):
        fp1 = failure_fingerprint(FailureType.QUALITY, "same text")
        fp2 = failure_fingerprint(FailureType.QUALITY, "same text")
        fp_other = failure_fingerprint(FailureType.FORMAT, "same text")
        assert fp1 == fp2
        assert fp1 != fp_other
        assert len(fp1) == 16
        assert all(ch in "0123456789abcdef" for ch in fp1)


class TestFailureDeduplicator:
    def test_consecutive_suppression(self):
        dedup = FailureDeduplicator(threshold=2)
        scope = ("t1", "s1", "a1")
        assert dedup.record(scope, "fp-a") == 1
        assert dedup.record(scope, "fp-a") == 2
        assert dedup.suppressed(1) is False
        assert dedup.suppressed(2) is True
        assert dedup.record(scope, "fp-a") == 3

    def test_fingerprint_change_resets_consecutive_count(self):
        dedup = FailureDeduplicator(threshold=2)
        scope = ("t1", "s1", "a1")
        assert dedup.record(scope, "fp-a") == 1
        assert dedup.record(scope, "fp-a") == 2
        assert dedup.record(scope, "fp-b") == 1
        assert dedup.record(scope, "fp-a") == 1

    def test_scope_isolation_and_reset(self):
        dedup = FailureDeduplicator(threshold=2)
        scope_a = ("t1", "s1", "a1")
        scope_b = ("t1", "s1", "a2")
        assert dedup.record(scope_a, "fp") == 1
        assert dedup.record(scope_b, "fp") == 1
        dedup.reset(scope_a)
        assert dedup.record(scope_a, "fp") == 1
        assert dedup.record(scope_b, "fp") == 2

    def test_threshold_floor_one(self):
        assert FailureDeduplicator(threshold=0).threshold == 1
        assert FailureDeduplicator(threshold=5).threshold == 5


class TestRecoveryPolicy:
    def _policy(self, **overrides) -> RecoveryPolicy:
        params = dict(
            repair_contract_max_attempts=1,
            backoff_base_seconds=1.0,
            backoff_cap_seconds=30.0,
            backoff_jitter_ratio=0.2,
            jitter=lambda: 0.5,
        )
        params.update(overrides)
        return RecoveryPolicy(**params)

    def _decide(self, policy: RecoveryPolicy, failure_type: FailureType, **kw):
        params = dict(
            failure_type=failure_type,
            fingerprint="a" * 16,
            occurrences=1,
            suppressed=False,
            attempt=1,
            max_attempts=3,
        )
        params.update(kw)
        return policy.decide(**params)

    @pytest.mark.parametrize(
        "failure_type",
        [
            FailureType.PERMISSION,
            FailureType.INVALID_INPUT,
            FailureType.BUDGET,
            FailureType.INTERNAL,
        ],
    )
    def test_non_retryable_stops(self, failure_type):
        decision = self._decide(self._policy(), failure_type)
        assert decision.action is RecoveryAction.STOP
        assert decision.is_terminal is True

    @pytest.mark.parametrize(
        "failure_type",
        [
            FailureType.QUALITY,
            FailureType.TOOL,
            FailureType.SANDBOX,
        ],
    )
    def test_suppressed_semantic_replace(self, failure_type):
        decision = self._decide(
            self._policy(), failure_type, occurrences=2, suppressed=True
        )
        assert decision.action is RecoveryAction.REPLACE
        assert decision.is_terminal is True
        assert "停止重试并替换" in decision.reason

    def test_suppressed_format_replans_forbidden_replace(self):
        """format 同因再现（任务 69）：同一格式失败第二次出现即停止
        重试并转重规划，绝不创建 Replacement Agent、绝不再修复。"""
        decision = self._decide(
            self._policy(), FailureType.FORMAT, occurrences=2, suppressed=True
        )
        assert decision.action is RecoveryAction.REPLAN
        assert decision.is_terminal is True
        assert "format 类故障禁止替换" in decision.reason

    @pytest.mark.parametrize(
        "failure_type",
        [FailureType.NETWORK, FailureType.RATE_LIMIT, FailureType.TIMEOUT],
    )
    def test_suppressed_provider_replans_forbidden_replace(self, failure_type):
        decision = self._decide(
            self._policy(), failure_type, occurrences=2, suppressed=True
        )
        assert decision.action is RecoveryAction.REPLAN
        assert "provider 类故障禁止替换" in decision.reason
        assert decision.is_terminal is True

    def test_attempt_limit_replace_or_replan(self):
        policy = self._policy()
        replaced = self._decide(
            policy, FailureType.QUALITY, attempt=3, max_attempts=3
        )
        assert replaced.action is RecoveryAction.REPLACE
        replanned = self._decide(
            policy, FailureType.NETWORK, attempt=3, max_attempts=3
        )
        assert replanned.action is RecoveryAction.REPLAN
        assert "provider 类故障禁止替换" in replanned.reason

    def test_format_repair_contract_within_quota(self):
        decision = self._decide(self._policy(), FailureType.FORMAT)
        assert decision.action is RecoveryAction.REPAIR_CONTRACT
        assert decision.is_terminal is False
        assert "不占业务尝试额度" in decision.reason

    def test_format_repair_precedes_attempt_limit(self):
        """契约修复判定先于业务尝试上限（任务 69）：即使业务 attempt
        已达上限，修复额度未耗尽时仍允许受限修复轮——修复机会不被
        业务上限剥夺，且修复轮不推进 attempt，不会重跑业务任务。"""
        decision = self._decide(
            self._policy(), FailureType.FORMAT, attempt=3, max_attempts=3
        )
        assert decision.action is RecoveryAction.REPAIR_CONTRACT

    def test_format_repair_exhausted_then_replan(self):
        """修复额度耗尽（format 禁替换，任务 69）：停止重试并转重
        规划，不发起替换、不再修复。"""
        decision = self._decide(
            self._policy(), FailureType.FORMAT, repair_used=1
        )
        assert decision.action is RecoveryAction.REPLAN
        assert decision.is_terminal is True
        assert "修复额度已用尽" in decision.reason

    def test_format_repair_disabled_replans_immediately(self):
        """修复关闭（repair_max=0，任务 69）：首次 format 失败直接转
        重规划，不发起替换。"""
        policy = self._policy(repair_contract_max_attempts=0)
        decision = self._decide(policy, FailureType.FORMAT)
        assert decision.action is RecoveryAction.REPLAN
        assert decision.is_terminal is True

    def test_network_backoff_exponential(self):
        policy = self._policy()
        delays = [
            self._decide(policy, FailureType.NETWORK, occurrences=n).backoff_seconds
            for n in (1, 2, 3, 4)
        ]
        assert delays == pytest.approx([1.1, 2.2, 4.4, 8.8])
        for delay, occ in zip(delays, (1, 2, 3, 4)):
            decision = self._decide(
                policy, FailureType.NETWORK, occurrences=occ
            )
            assert decision.action is RecoveryAction.RETRY
            assert decision.is_terminal is False
            assert decision.backoff_seconds == delay

    def test_backoff_capped(self):
        policy = self._policy(backoff_cap_seconds=3.0)
        decision = self._decide(policy, FailureType.NETWORK, occurrences=5)
        assert decision.backoff_seconds == pytest.approx(3.3)

    def test_rate_limit_uses_backoff_path(self):
        decision = self._decide(self._policy(), FailureType.RATE_LIMIT)
        assert decision.action is RecoveryAction.RETRY
        assert decision.backoff_seconds == pytest.approx(1.1)

    def test_timeout_retry_without_backoff(self):
        decision = self._decide(self._policy(), FailureType.TIMEOUT)
        assert decision.action is RecoveryAction.RETRY
        assert decision.backoff_seconds == 0.0

    def test_quality_default_retry(self):
        decision = self._decide(self._policy(), FailureType.QUALITY)
        assert decision.action is RecoveryAction.RETRY
        assert "按策略重试" in decision.reason

    def test_decision_echoes_context(self):
        decision = self._decide(
            self._policy(), FailureType.QUALITY, occurrences=2
        )
        assert decision.failure_type is FailureType.QUALITY
        assert decision.fingerprint == "a" * 16
        assert decision.occurrences == 2
        assert decision.suppressed is False

    def test_from_settings_wires_fields(self):
        settings = Settings(
            recovery_backoff_base_seconds=2.0,
            recovery_backoff_jitter_ratio=0.0,
            recovery_repair_contract_max_attempts=0,
        )
        policy = RecoveryPolicy.from_settings(settings)
        network = self._decide(policy, FailureType.NETWORK)
        assert network.backoff_seconds == pytest.approx(2.0)
        fmt = self._decide(policy, FailureType.FORMAT)
        assert fmt.action is RecoveryAction.REPLAN  # 修复额度 0：format 直接转重规划

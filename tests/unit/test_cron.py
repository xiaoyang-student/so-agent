"""单元测试：五字段 cron 解析与下次触发时间。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from so_agent.cron import CronExpressionError, parse_cron


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0,
        second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


class TestParse:
    def test_every_minute(self):
        expression = parse_cron("* * * * *")
        assert len(expression.minutes) == 60
        assert len(expression.hours) == 24

    def test_step_range_and_list(self):
        expression = parse_cron("*/15 9-18/3 1,15 * *")
        assert expression.minutes == frozenset({0, 15, 30, 45})
        assert expression.hours == frozenset({9, 12, 15, 18})
        assert expression.days_of_month == frozenset({1, 15})

    def test_month_and_weekday_names(self):
        expression = parse_cron("0 0 * JAN MON")
        assert expression.months == frozenset({1})
        assert expression.days_of_week == frozenset({1})

    def test_sunday_accepts_zero_and_seven(self):
        assert parse_cron("0 0 * * 0").days_of_week == frozenset({0})
        assert parse_cron("0 0 * * 7").days_of_week == frozenset({0})

    @pytest.mark.parametrize(
        "expression",
        [
            "",
            "   ",
            "* * * *",
            "* * * * * *",
            "60 * * * *",
            "* 24 * * *",
            "* * 0 * *",
            "* * * 13 *",
            "* * * * 8",
            "*/0 * * * *",
            "5-1 * * * *",
            "a * * * *",
            "1,,2 * * * *",
        ],
    )
    def test_invalid_expressions(self, expression):
        with pytest.raises(CronExpressionError):
            parse_cron(expression)


class TestNextAfter:
    def test_strictly_after_current_minute(self):
        expression = parse_cron("* * * * *")
        assert expression.next_after(utc(2026, 1, 1, 12, 0, 30)) == utc(
            2026, 1, 1, 12, 1
        )

    def test_step_rounds_up_within_hour(self):
        expression = parse_cron("*/15 * * * *")
        assert expression.next_after(utc(2026, 1, 1, 10, 7, 45)) == utc(
            2026, 1, 1, 10, 15
        )

    def test_daily_rolls_to_next_day(self):
        expression = parse_cron("0 9 * * *")
        assert expression.next_after(utc(2026, 1, 1, 9, 0)) == utc(
            2026, 1, 2, 9, 0
        )
        assert expression.next_after(utc(2026, 1, 1, 8, 59)) == utc(
            2026, 1, 1, 9, 0
        )

    def test_weekday_only(self):
        # 2026-01-03 是周六，下一个周一为 2026-01-05
        expression = parse_cron("0 0 * * 1")
        assert expression.next_after(utc(2026, 1, 3, 12, 0)) == utc(
            2026, 1, 5, 0, 0
        )

    def test_dom_dow_use_or_semantics(self):
        expression = parse_cron("0 0 1 * 1")
        # “每月 1 号或每周一”：2025-12-02 之后最近的周一是 12-08
        assert expression.next_after(utc(2025, 12, 2, 0, 0)) == utc(
            2025, 12, 8, 0, 0
        )
        # 2025-12-31 是周三，下一天 1 号（周四）同样命中
        assert expression.next_after(utc(2025, 12, 31, 0, 0)) == utc(
            2026, 1, 1, 0, 0
        )

    def test_dom_only_uses_and_semantics(self):
        # 仅日字段受限：只按日期匹配（不看星期）
        expression = parse_cron("0 0 15 * *")
        assert expression.next_after(utc(2026, 1, 3, 12, 0)) == utc(
            2026, 1, 15, 0, 0
        )

    def test_leap_day_found_across_years(self):
        expression = parse_cron("0 0 29 2 *")
        assert expression.next_after(utc(2025, 3, 1)) == utc(2028, 2, 29, 0, 0)

    def test_naive_datetime_treated_as_utc(self):
        expression = parse_cron("0 9 * * *")
        result = expression.next_after(datetime(2026, 1, 1, 10, 0))
        assert result == utc(2026, 1, 2, 9, 0)

    def test_non_utc_timestamp_normalized(self):
        from datetime import timedelta

        expression = parse_cron("0 9 * * *")
        tz = timezone(timedelta(hours=8))
        moment = datetime(2026, 1, 1, 8, 0, tzinfo=tz)  # UTC 00:00
        assert expression.next_after(moment) == utc(2026, 1, 1, 9, 0)

    def test_no_match_within_search_window(self):
        # 2 月 30 日不存在，10 年搜索窗口内无解
        with pytest.raises(CronExpressionError):
            parse_cron("0 0 30 2 *").next_after(utc(2026, 1, 1))

"""五字段 cron 解析与下次触发时间计算。

支持标准字段 ``minute hour day-of-month month day-of-week``：

- ``*`` 与 ``?``：任意值；
- ``*/n``：按步长；
- ``a-b``、``a-b/n``：范围与范围步长；
- 逗号列表：``1,15,30``；
- 月份与星期支持三字母英文缩写（JAN-DEC / SUN-SAT）。

语义遵循 Vixie cron：day-of-month 与 day-of-week 同时受限时取“或”，
仅一个受限时取“与”。星期允许 0 或 7 表示周日。

时间统一按 UTC 计算；``next_after`` 严格返回晚于给定时刻的下一分钟。
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

__all__ = [
    "CronExpression",
    "CronExpressionError",
    "parse_cron",
]

_MONTH_NAMES = {
    name: index
    for index, name in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
         "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}
_DOW_NAMES = {
    name: index
    for index, name in enumerate(
        ("SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT")
    )
}

# 最多向后搜索 10 年；可覆盖 2 月 29 日等低频表达式。
_MAX_SEARCH_DAYS = 366 * 10


class CronExpressionError(ValueError):
    """cron 表达式非法（字段数、取值范围、步长或语法错误）。"""


@dataclass(frozen=True)
class CronExpression:
    """解析后的五字段 cron 表达式。"""

    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    dom_restricted: bool
    dow_restricted: bool
    source: str

    def next_after(self, moment: datetime) -> datetime:
        """返回严格晚于 ``moment`` 的下一个匹配时刻（UTC，秒归零）。"""
        if moment.tzinfo is None:
            current = moment.replace(tzinfo=timezone.utc)
        else:
            current = moment.astimezone(timezone.utc)
        start = current.replace(second=0, microsecond=0) + timedelta(minutes=1)
        candidate_date: date = start.date()
        first_day = True

        for _ in range(_MAX_SEARCH_DAYS):
            if candidate_date.month in self.months and self._day_matches(candidate_date):
                for hour in sorted(self.hours):
                    if first_day and hour < start.hour:
                        continue
                    for minute in sorted(self.minutes):
                        if first_day and hour == start.hour and minute < start.minute:
                            continue
                        return datetime(
                            candidate_date.year,
                            candidate_date.month,
                            candidate_date.day,
                            hour,
                            minute,
                            tzinfo=timezone.utc,
                        )
            candidate_date += timedelta(days=1)
            first_day = False

        raise CronExpressionError(
            f"cron 表达式在 {_MAX_SEARCH_DAYS} 天内没有可触发时间：{self.source!r}"
        )

    def _day_matches(self, day: date) -> bool:
        dom_match = day.day in self.days_of_month
        # Python: Monday=0；cron: Sunday=0
        cron_dow = (day.weekday() + 1) % 7
        dow_match = cron_dow in self.days_of_week
        if self.dom_restricted and self.dow_restricted:
            return dom_match or dow_match
        if self.dom_restricted:
            return dom_match
        if self.dow_restricted:
            return dow_match
        return True


def parse_cron(expression: str) -> CronExpression:
    """解析五字段 cron 表达式。

    Raises:
        CronExpressionError: 字段数不为 5，或任一字段语法/范围非法时。
    """
    if not isinstance(expression, str) or not expression.strip():
        raise CronExpressionError("cron 表达式不能为空")
    fields = expression.split()
    if len(fields) != 5:
        raise CronExpressionError(
            f"cron 表达式必须为 5 个字段（分 时 日 月 周），实际 {len(fields)} 个："
            f"{expression!r}"
        )
    minute_text, hour_text, dom_text, month_text, dow_text = fields

    minutes = _parse_field(minute_text, 0, 59, "minute")
    hours = _parse_field(hour_text, 0, 23, "hour")
    days_of_month = _parse_field(dom_text, 1, 31, "day-of-month")
    months = _parse_field(month_text, 1, 12, "month", names=_MONTH_NAMES)
    raw_dow = _parse_field(dow_text, 0, 7, "day-of-week", names=_DOW_NAMES)
    days_of_week = frozenset(0 if value == 7 else value for value in raw_dow)

    return CronExpression(
        minutes=minutes,
        hours=hours,
        days_of_month=days_of_month,
        months=months,
        days_of_week=days_of_week,
        dom_restricted=_is_restricted(dom_text),
        dow_restricted=_is_restricted(dow_text),
        source=expression.strip(),
    )


def _is_restricted(text: str) -> bool:
    return text.strip() not in {"*", "?"}


def _parse_field(
    text: str,
    minimum: int,
    maximum: int,
    label: str,
    *,
    names: dict[str, int] | None = None,
) -> frozenset[int]:
    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            raise CronExpressionError(f"{label} 字段存在空项：{text!r}")
        base, step = _split_step(token, label)
        if base in {"*", "?"}:
            start, end = minimum, maximum
        elif "-" in base:
            left, right = base.split("-", 1)
            start = _parse_value(left, label, names)
            end = _parse_value(right, label, names)
        else:
            start = _parse_value(base, label, names)
            end = maximum if step is not None else start
        if start > end:
            raise CronExpressionError(
                f"{label} 字段范围起点大于终点：{token!r}"
            )
        if start < minimum or end > maximum:
            raise CronExpressionError(
                f"{label} 字段超出范围 {minimum}-{maximum}：{token!r}"
            )
        values.update(range(start, end + 1, step or 1))
    return frozenset(values)


def _split_step(token: str, label: str) -> tuple[str, int | None]:
    if "/" not in token:
        return token, None
    base, step_text = token.split("/", 1)
    try:
        step = int(step_text)
    except ValueError as exc:
        raise CronExpressionError(f"{label} 字段步长非法：{token!r}") from exc
    if step <= 0:
        raise CronExpressionError(f"{label} 字段步长必须为正数：{token!r}")
    return base, step


def _parse_value(
    text: str,
    label: str,
    names: dict[str, int] | None,
) -> int:
    text = text.strip().upper()
    if names and text in names:
        return names[text]
    try:
        return int(text)
    except ValueError as exc:
        raise CronExpressionError(f"{label} 字段取值非法：{text!r}") from exc

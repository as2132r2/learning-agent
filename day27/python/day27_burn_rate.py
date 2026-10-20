"""Day27 第 2 步：滚动时间窗口与错误预算燃烧率。

运行：
    python day27/python/day27_burn_rate.py

本步生成6小时确定性事件，并同时计算多个滚动窗口。
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from day27_slo_basic import (
    DATABASE_PATH,
    RequestEvent,
    SLO,
    create_schema,
    open_database,
    save_events,
)


AlertLevel = Literal["normal", "warning", "page"]


@dataclass(frozen=True, slots=True)
class WindowDefinition:
    name: str
    duration: timedelta


@dataclass(slots=True)
class WindowMeasurement:
    """一个滚动窗口的可用率和燃烧率。"""

    name: str
    minutes: int
    window_start: str
    window_end: str
    total_requests: int
    bad_requests: int
    bad_rate: float | None
    availability: float | None
    allowed_bad_rate: float
    burn_rate: float | None
    budget_lifetime_days_if_sustained: float | None


@dataclass(slots=True)
class MultiWindowDecision:
    """多个时间窗口组合后的告警判断。"""

    level: AlertLevel
    fast_burn_triggered: bool
    slow_burn_triggered: bool
    reason: str
    measurements: dict[str, WindowMeasurement]


WINDOWS = (
    WindowDefinition("5m", timedelta(minutes=5)),
    WindowDefinition("30m", timedelta(minutes=30)),
    WindowDefinition("1h", timedelta(hours=1)),
    WindowDefinition("6h", timedelta(hours=6)),
)

# 教学版阈值：高倍燃烧需要短窗和长窗同时成立，减少瞬时尖峰误报。
FAST_BURN_THRESHOLD = 14.4
SLOW_BURN_THRESHOLD = 6.0
SLO_PERIOD_DAYS = 30.0


def create_burn_schema(connection: sqlite3.Connection) -> None:
    """保存每次计算快照，方便后续画图和审计告警依据。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS slo_window_snapshots (
            snapshot_id       TEXT NOT NULL,
            metric_name       TEXT NOT NULL,
            window_name       TEXT NOT NULL,
            window_start      TEXT NOT NULL,
            window_end        TEXT NOT NULL,
            total_requests    INTEGER NOT NULL,
            bad_requests      INTEGER NOT NULL,
            availability      REAL,
            burn_rate         REAL,
            created_at        TEXT NOT NULL,
            PRIMARY KEY (snapshot_id, window_name)
        );
        """
    )
    connection.commit()


def build_six_hour_events(now: datetime) -> list[RequestEvent]:
    """每分钟5个请求；最后1小时制造20%不可用事故。"""

    events: list[RequestEvent] = []
    sequence = 0
    for minute in range(360):
        occurred = now - timedelta(minutes=359 - minute)
        incident = minute >= 300
        for slot in range(5):
            sequence += 1
            if incident and slot == 0:
                # 最近1小时每分钟5个请求中有1个失败：坏请求率20%。
                status = "error"
                quality_passed = False
                latency_ms = 650
            elif not incident and sequence % 200 == 0:
                # 事故前保持约0.5%的背景错误率。
                status = "error"
                quality_passed = False
                latency_ms = 600
            else:
                status = "success"
                quality_passed = sequence % 50 != 0
                latency_ms = 900 + sequence % 1700

            events.append(
                RequestEvent(
                    request_id=f"req-day27-burn-{sequence:04d}",
                    status=status,  # type: ignore[arg-type]
                    quality_passed=quality_passed,
                    latency_ms=latency_ms,
                    total_tokens=1100 + sequence % 900,
                    occurred_at=occurred.isoformat(timespec="seconds"),
                )
            )
    return events


def measure_window(
    connection: sqlite3.Connection,
    definition: WindowDefinition,
    now: datetime,
) -> WindowMeasurement:
    """每次都相对当前now回看，因此窗口会随时间向前滚动。"""

    start = now - definition.duration
    row = connection.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'success' THEN 0 ELSE 1 END) AS bad
        FROM service_request_events
        WHERE request_id LIKE 'req-day27-burn-%'
          AND occurred_at > ?
          AND occurred_at <= ?
        """,
        (
            start.isoformat(timespec="seconds"),
            now.isoformat(timespec="seconds"),
        ),
    ).fetchone()

    total = int(row["total"] or 0)
    bad = int(row["bad"] or 0)
    bad_rate = bad / total if total else None
    availability = 1 - bad_rate if bad_rate is not None else None
    allowed_bad_rate = 1 - SLO.availability_target
    burn_rate = (
        bad_rate / allowed_bad_rate
        if bad_rate is not None and allowed_bad_rate > 0
        else None
    )
    lifetime = (
        SLO_PERIOD_DAYS / burn_rate
        if burn_rate is not None and burn_rate > 0
        else None
    )

    return WindowMeasurement(
        name=definition.name,
        minutes=round(definition.duration.total_seconds() / 60),
        window_start=start.isoformat(timespec="seconds"),
        window_end=now.isoformat(timespec="seconds"),
        total_requests=total,
        bad_requests=bad,
        bad_rate=bad_rate,
        availability=availability,
        allowed_bad_rate=allowed_bad_rate,
        burn_rate=burn_rate,
        budget_lifetime_days_if_sustained=lifetime,
    )


def evaluate_multi_window(
    measurements: dict[str, WindowMeasurement],
) -> MultiWindowDecision:
    """短窗与长窗必须同时超线，才触发对应级别。"""

    def burn(name: str) -> float:
        return measurements[name].burn_rate or 0.0

    fast = burn("5m") >= FAST_BURN_THRESHOLD and burn("1h") >= FAST_BURN_THRESHOLD
    slow = burn("30m") >= SLOW_BURN_THRESHOLD and burn("6h") >= SLOW_BURN_THRESHOLD

    if fast:
        level: AlertLevel = "page"
        reason = (
            f"5m和1h燃烧率均达到{FAST_BURN_THRESHOLD}倍，"
            "错误预算正在快速耗尽，需要立即处理。"
        )
    elif slow:
        level = "warning"
        reason = (
            f"30m和6h燃烧率均达到{SLOW_BURN_THRESHOLD}倍，"
            "存在持续性退化，需要尽快调查。"
        )
    else:
        level = "normal"
        reason = "没有多窗口组合同时超过阈值。"

    return MultiWindowDecision(
        level=level,
        fast_burn_triggered=fast,
        slow_burn_triggered=slow,
        reason=reason,
        measurements=measurements,
    )


def save_snapshot(
    connection: sqlite3.Connection,
    measurements: dict[str, WindowMeasurement],
) -> str:
    snapshot_id = f"snapshot-{uuid4().hex[:12]}"
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    with connection:
        connection.executemany(
            """
            INSERT INTO slo_window_snapshots (
                snapshot_id, metric_name, window_name,
                window_start, window_end, total_requests,
                bad_requests, availability, burn_rate, created_at
            )
            VALUES (?, 'availability', ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    snapshot_id,
                    item.name,
                    item.window_start,
                    item.window_end,
                    item.total_requests,
                    item.bad_requests,
                    item.availability,
                    item.burn_rate,
                    created_at,
                )
                for item in measurements.values()
            ],
        )
    return snapshot_id


def percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def number(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.2f}{suffix}"


def main() -> None:
    now = datetime.now().astimezone().replace(microsecond=0)
    print("Day27 滚动窗口与错误预算燃烧率")
    print("数据库：", DATABASE_PATH)
    print("可用率SLO：", percent(SLO.availability_target))
    print("允许坏请求比例：", percent(1 - SLO.availability_target))
    print("=" * 112)

    with open_database() as connection:
        create_schema(connection)
        create_burn_schema(connection)
        save_events(connection, build_six_hour_events(now))
        measurements = {
            definition.name: measure_window(connection, definition, now)
            for definition in WINDOWS
        }
        decision = evaluate_multi_window(measurements)
        snapshot_id = save_snapshot(connection, measurements)

    print(
        f"{'窗口':<8}{'请求':>8}{'坏请求':>10}{'坏请求率':>14}"
        f"{'可用率':>14}{'燃烧率':>14}{'预算寿命':>16}"
    )
    print("-" * 112)
    for name in ("5m", "30m", "1h", "6h"):
        item = measurements[name]
        print(
            f"{name:<8}{item.total_requests:>8}{item.bad_requests:>10}"
            f"{percent(item.bad_rate):>14}{percent(item.availability):>14}"
            f"{number(item.burn_rate, 'x'):>14}"
            f"{number(item.budget_lifetime_days_if_sustained, '天'):>16}"
        )

    print("\n" + "=" * 112)
    print("多窗口判断：", decision.level)
    print("快速燃烧：", decision.fast_burn_triggered)
    print("慢速燃烧：", decision.slow_burn_triggered)
    print("原因：", decision.reason)
    print("快燃烧规则：5m和1h都 >=", FAST_BURN_THRESHOLD, "倍")
    print("慢燃烧规则：30m和6h都 >=", SLOW_BURN_THRESHOLD, "倍")
    print("快照编号：", snapshot_id)

    print("\n完整判断对象：")
    print(json.dumps(asdict(decision), ensure_ascii=False, indent=2))

    print("\n本步数据流：")
    print("最近6小时RequestEvent")
    print("→ 5m / 30m / 1h / 6h滚动查询")
    print("→ 坏请求率 ÷ SLO允许坏请求率 = burn_rate")
    print("→ 短窗和长窗组合 → normal / warning / page")
    print("→ slo_window_snapshots审计快照")


if __name__ == "__main__":
    main()

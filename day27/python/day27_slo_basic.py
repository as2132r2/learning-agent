"""Day27 第 1 步：SLI、SLO 与错误预算基础。

运行：
    python day27/python/day27_slo_basic.py

本步使用确定性的模拟请求，不调用大模型。
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal


DAY27_DIR = Path(__file__).resolve().parents[1]
DATABASE_DIR = DAY27_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "day27_operations.db"

RequestStatus = Literal["success", "rejected", "error", "timeout", "cancelled"]


@dataclass(frozen=True, slots=True)
class RequestEvent:
    """生产服务一次请求的客观结果。"""

    request_id: str
    status: RequestStatus
    quality_passed: bool
    latency_ms: int
    total_tokens: int
    occurred_at: str


@dataclass(frozen=True, slots=True)
class SLOConfig:
    """服务目标；比例使用0～1，延迟使用毫秒。"""

    availability_target: float = 0.99
    quality_target: float = 0.95
    latency_p95_target_ms: int = 4000


@dataclass(slots=True)
class ErrorBudget:
    """一个比例SLO对应的错误预算。"""

    total_requests: int
    target: float
    allowed_bad_requests: float
    actual_bad_requests: int
    remaining_bad_requests: float
    consumed_percent: float
    exhausted: bool


@dataclass(slots=True)
class SLOReport:
    """一个观察窗口内的完整SLO报告。"""

    window_start: str
    window_end: str
    total_requests: int
    available_requests: int
    quality_passed_requests: int
    availability: float | None
    quality_pass_rate: float | None
    latency_p95_ms: int | None
    availability_target: float
    quality_target: float
    latency_p95_target_ms: int
    availability_met: bool
    quality_met: bool
    latency_met: bool
    overall_met: bool
    availability_error_budget: ErrorBudget
    quality_error_budget: ErrorBudget


SLO = SLOConfig()


def open_database() -> sqlite3.Connection:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS service_request_events (
            request_id       TEXT PRIMARY KEY,
            status           TEXT NOT NULL
                             CHECK (status IN (
                                'success', 'rejected', 'error',
                                'timeout', 'cancelled'
                             )),
            quality_passed   INTEGER NOT NULL CHECK (quality_passed IN (0, 1)),
            latency_ms       INTEGER NOT NULL CHECK (latency_ms >= 0),
            total_tokens     INTEGER NOT NULL CHECK (total_tokens >= 0),
            occurred_at      TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_request_events_time
        ON service_request_events(occurred_at);

        CREATE INDEX IF NOT EXISTS idx_request_events_status
        ON service_request_events(status);
        """
    )
    connection.commit()


def validate_event(event: RequestEvent) -> None:
    if event.status not in {"success", "rejected", "error", "timeout", "cancelled"}:
        raise ValueError("无效的请求状态。")
    if event.latency_ms < 0 or event.total_tokens < 0:
        raise ValueError("latency_ms和total_tokens不能为负数。")
    if event.quality_passed and event.status != "success":
        raise ValueError("只有success请求才能quality_passed=True。")


def save_events(
    connection: sqlite3.Connection,
    events: list[RequestEvent],
) -> None:
    """使用request_id幂等写入；重复运行不会重复计算请求。"""

    for event in events:
        validate_event(event)
    with connection:
        connection.executemany(
            """
            INSERT INTO service_request_events (
                request_id, status, quality_passed,
                latency_ms, total_tokens, occurred_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
                status = excluded.status,
                quality_passed = excluded.quality_passed,
                latency_ms = excluded.latency_ms,
                total_tokens = excluded.total_tokens,
                occurred_at = excluded.occurred_at
            """,
            [
                (
                    event.request_id,
                    event.status,
                    int(event.quality_passed),
                    event.latency_ms,
                    event.total_tokens,
                    event.occurred_at,
                )
                for event in events
            ],
        )


def build_demonstration_events(now: datetime) -> list[RequestEvent]:
    """生成100条确定性事件：97成功、94质量通过、3个不可用请求。"""

    events: list[RequestEvent] = []
    for index in range(1, 101):
        if index <= 94:
            status: RequestStatus = "success"
            quality_passed = True
        elif index <= 97:
            status = "success"
            quality_passed = False
        elif index <= 99:
            status = "error"
            quality_passed = False
        else:
            status = "timeout"
            quality_passed = False

        # 大部分请求约1～3秒，最后5条成功请求较慢，用于观察P95。
        latency_ms = 900 + (index * 37) % 2100
        if 90 <= index <= 94:
            latency_ms = 3300 + (index - 90) * 120
        if status in {"error", "timeout"}:
            latency_ms = 5000 if status == "timeout" else 600

        occurred_at = (now - timedelta(seconds=(100 - index) * 30)).isoformat(
            timespec="seconds"
        )
        events.append(
            RequestEvent(
                request_id=f"req-day27-{index:03d}",
                status=status,
                quality_passed=quality_passed,
                latency_ms=latency_ms,
                total_tokens=1200 + (index * 29) % 1000,
                occurred_at=occurred_at,
            )
        )
    return events


def nearest_rank_percentile(values: list[int], percentile: float) -> int | None:
    """使用nearest-rank算法计算百分位数。"""

    if not values:
        return None
    if not 0 < percentile <= 1:
        raise ValueError("percentile必须在0～1之间。")
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def build_error_budget(
    total: int,
    good: int,
    target: float,
) -> ErrorBudget:
    """根据SLO允许的失败比例计算错误预算消耗。"""

    allowed_bad = total * (1 - target)
    actual_bad = total - good
    remaining = allowed_bad - actual_bad
    consumed = (
        actual_bad / allowed_bad * 100
        if allowed_bad > 0
        else (0.0 if actual_bad == 0 else math.inf)
    )
    return ErrorBudget(
        total_requests=total,
        target=target,
        allowed_bad_requests=round(allowed_bad, 3),
        actual_bad_requests=actual_bad,
        remaining_bad_requests=round(max(0.0, remaining), 3),
        consumed_percent=round(consumed, 1),
        exhausted=actual_bad > allowed_bad,
    )


def calculate_slo_report(
    connection: sqlite3.Connection,
    window_start: datetime,
    window_end: datetime,
    config: SLOConfig,
) -> SLOReport:
    """读取时间窗口中的原始事件并计算SLI、SLO和错误预算。"""

    rows = connection.execute(
        """
        SELECT status, quality_passed, latency_ms
        FROM service_request_events
        WHERE occurred_at >= ? AND occurred_at <= ?
        ORDER BY occurred_at
        """,
        (
            window_start.isoformat(timespec="seconds"),
            window_end.isoformat(timespec="seconds"),
        ),
    ).fetchall()

    total = len(rows)
    available = sum(row["status"] == "success" for row in rows)
    quality_passed = sum(bool(row["quality_passed"]) for row in rows)
    # 延迟SLI只统计成功返回的请求；错误和超时由可用率单独衡量。
    successful_latencies = [
        int(row["latency_ms"])
        for row in rows
        if row["status"] == "success"
    ]

    availability = available / total if total else None
    quality_rate = quality_passed / total if total else None
    latency_p95 = nearest_rank_percentile(successful_latencies, 0.95)

    availability_met = (
        availability is not None and availability >= config.availability_target
    )
    quality_met = quality_rate is not None and quality_rate >= config.quality_target
    latency_met = (
        latency_p95 is not None and latency_p95 <= config.latency_p95_target_ms
    )

    return SLOReport(
        window_start=window_start.isoformat(timespec="seconds"),
        window_end=window_end.isoformat(timespec="seconds"),
        total_requests=total,
        available_requests=available,
        quality_passed_requests=quality_passed,
        availability=availability,
        quality_pass_rate=quality_rate,
        latency_p95_ms=latency_p95,
        availability_target=config.availability_target,
        quality_target=config.quality_target,
        latency_p95_target_ms=config.latency_p95_target_ms,
        availability_met=availability_met,
        quality_met=quality_met,
        latency_met=latency_met,
        overall_met=availability_met and quality_met and latency_met,
        availability_error_budget=build_error_budget(
            total, available, config.availability_target
        ),
        quality_error_budget=build_error_budget(
            total, quality_passed, config.quality_target
        ),
    )


def percent(value: float | None) -> str:
    return "无数据" if value is None else f"{value:.2%}"


def show_budget(name: str, budget: ErrorBudget) -> None:
    print(f"\n{name}错误预算：")
    print("允许坏请求：", budget.allowed_bad_requests)
    print("实际坏请求：", budget.actual_bad_requests)
    print("剩余预算：", budget.remaining_bad_requests)
    print("预算消耗：", f"{budget.consumed_percent}%")
    print("是否耗尽：", budget.exhausted)


def main() -> None:
    now = datetime.now().astimezone().replace(microsecond=0)
    window_start = now - timedelta(hours=1)

    print("Day27 SLI、SLO 与错误预算基础")
    print("数据库：", DATABASE_PATH)
    print("观察窗口：最近1小时")
    print("=" * 88)

    events = build_demonstration_events(now)
    with open_database() as connection:
        create_schema(connection)
        save_events(connection, events)
        report = calculate_slo_report(connection, window_start, now, SLO)

    print("请求总数：", report.total_requests)
    print("成功返回：", report.available_requests)
    print("质量通过：", report.quality_passed_requests)
    print("\nSLI与SLO：")
    print(
        "可用率：",
        percent(report.availability),
        "目标：",
        percent(report.availability_target),
        "结果：",
        "通过" if report.availability_met else "失败",
    )
    print(
        "质量通过率：",
        percent(report.quality_pass_rate),
        "目标：",
        percent(report.quality_target),
        "结果：",
        "通过" if report.quality_met else "失败",
    )
    print(
        "成功请求P95延迟：",
        report.latency_p95_ms,
        "ms；目标 <=",
        report.latency_p95_target_ms,
        "ms；结果：",
        "通过" if report.latency_met else "失败",
    )
    print("总体SLO：", "通过" if report.overall_met else "失败")

    show_budget("可用率", report.availability_error_budget)
    show_budget("质量", report.quality_error_budget)

    print("\n" + "=" * 88)
    print("完整报告对象：")
    import json

    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))

    print("\n本步数据流：")
    print("RequestEvent → SQLite幂等保存")
    print("→ 最近1小时事件 → 计算SLI")
    print("→ 与SLO目标比较 → 计算错误预算")
    print("→ SLOReport")
    print("\nSLI是测量值，SLO是目标值；SLA是对外承诺，三者不要混用。")


if __name__ == "__main__":
    main()

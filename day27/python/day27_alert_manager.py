"""Day27 第 3 步：告警去重、维护期抑制与恢复通知。

运行：
    python day27/python/day27_alert_manager.py

通知仅打印到终端，不会发送真实邮件或消息。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from day27_burn_rate import (
    WINDOWS,
    MultiWindowDecision,
    build_six_hour_events,
    create_burn_schema,
    evaluate_multi_window,
    measure_window,
)
from day27_slo_basic import create_schema, open_database, save_events


AlertStatus = Literal["open", "resolved"]
AlertEventType = Literal[
    "opened",
    "deduplicated",
    "severity_changed",
    "suppressed",
    "resolved",
    "no_change",
]

SEVERITY_ORDER = {"normal": 0, "warning": 1, "page": 2}
SERVICE_NAME = "langchain-rag-api"
ALERT_KEY = f"{SERVICE_NAME}:availability_burn_rate"


@dataclass(slots=True)
class AlertAction:
    """一次告警状态处理的结果。"""

    alert_key: str
    event_type: AlertEventType
    severity: str
    should_notify: bool
    notification_kind: str | None
    occurrence_count: int
    message: str


def create_alert_schema(connection: sqlite3.Connection) -> None:
    """创建当前告警状态、事件历史和维护窗口表。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS alert_states (
            alert_key          TEXT PRIMARY KEY,
            service_name      TEXT NOT NULL,
            status            TEXT NOT NULL
                              CHECK (status IN ('open', 'resolved')),
            severity          TEXT NOT NULL
                              CHECK (severity IN ('warning', 'page')),
            occurrence_count  INTEGER NOT NULL CHECK (occurrence_count > 0),
            first_seen_at      TEXT NOT NULL,
            last_seen_at       TEXT NOT NULL,
            last_reason        TEXT NOT NULL,
            resolved_at        TEXT
        );

        CREATE TABLE IF NOT EXISTS alert_events (
            event_id           TEXT PRIMARY KEY,
            alert_key          TEXT NOT NULL,
            event_type         TEXT NOT NULL,
            severity           TEXT NOT NULL,
            should_notify      INTEGER NOT NULL CHECK (should_notify IN (0, 1)),
            message            TEXT NOT NULL,
            occurred_at        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS maintenance_windows (
            maintenance_id     TEXT PRIMARY KEY,
            service_name       TEXT NOT NULL,
            reason             TEXT NOT NULL,
            starts_at          TEXT NOT NULL,
            ends_at            TEXT NOT NULL,
            enabled            INTEGER NOT NULL CHECK (enabled IN (0, 1))
        );

        CREATE INDEX IF NOT EXISTS idx_alert_events_key_time
        ON alert_events(alert_key, occurred_at);
        """
    )
    connection.commit()


def now_text(now: datetime) -> str:
    return now.isoformat(timespec="seconds")


def add_maintenance_window(
    connection: sqlite3.Connection,
    maintenance_id: str,
    service_name: str,
    reason: str,
    starts_at: datetime,
    ends_at: datetime,
) -> None:
    if ends_at <= starts_at:
        raise ValueError("维护结束时间必须晚于开始时间。")
    with connection:
        connection.execute(
            """
            INSERT INTO maintenance_windows (
                maintenance_id, service_name, reason,
                starts_at, ends_at, enabled
            )
            VALUES (?, ?, ?, ?, ?, 1)
            ON CONFLICT(maintenance_id) DO UPDATE SET
                reason = excluded.reason,
                starts_at = excluded.starts_at,
                ends_at = excluded.ends_at,
                enabled = 1
            """,
            (
                maintenance_id,
                service_name,
                reason,
                now_text(starts_at),
                now_text(ends_at),
            ),
        )


def active_maintenance_reason(
    connection: sqlite3.Connection,
    service_name: str,
    now: datetime,
) -> str | None:
    row = connection.execute(
        """
        SELECT reason
        FROM maintenance_windows
        WHERE service_name = ?
          AND enabled = 1
          AND starts_at <= ?
          AND ends_at >= ?
        ORDER BY starts_at DESC
        LIMIT 1
        """,
        (service_name, now_text(now), now_text(now)),
    ).fetchone()
    return None if row is None else str(row["reason"])


def save_alert_event(
    connection: sqlite3.Connection,
    action: AlertAction,
    now: datetime,
) -> None:
    connection.execute(
        """
        INSERT INTO alert_events (
            event_id, alert_key, event_type, severity,
            should_notify, message, occurred_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            f"alert-event-{uuid4().hex[:12]}",
            action.alert_key,
            action.event_type,
            action.severity,
            int(action.should_notify),
            action.message,
            now_text(now),
        ),
    )


def process_alert(
    connection: sqlite3.Connection,
    alert_key: str,
    service_name: str,
    decision: MultiWindowDecision,
    now: datetime,
) -> AlertAction:
    """把无状态的监控判断转换成有状态的告警动作。"""

    state = connection.execute(
        "SELECT * FROM alert_states WHERE alert_key = ?",
        (alert_key,),
    ).fetchone()

    # 恢复路径：只有之前确实存在open告警，才发送恢复通知。
    if decision.level == "normal":
        if state is None or state["status"] == "resolved":
            action = AlertAction(
                alert_key=alert_key,
                event_type="no_change",
                severity="normal",
                should_notify=False,
                notification_kind=None,
                occurrence_count=0 if state is None else state["occurrence_count"],
                message="指标正常，没有需要关闭的活动告警。",
            )
        else:
            action = AlertAction(
                alert_key=alert_key,
                event_type="resolved",
                severity=state["severity"],
                should_notify=True,
                notification_kind="recovery",
                occurrence_count=state["occurrence_count"],
                message=f"服务已恢复：{decision.reason}",
            )
            with connection:
                connection.execute(
                    """
                    UPDATE alert_states
                    SET status = 'resolved', resolved_at = ?,
                        last_seen_at = ?, last_reason = ?
                    WHERE alert_key = ?
                    """,
                    (now_text(now), now_text(now), decision.reason, alert_key),
                )
                save_alert_event(connection, action, now)
            return action

        with connection:
            save_alert_event(connection, action, now)
        return action

    maintenance_reason = active_maintenance_reason(connection, service_name, now)
    if maintenance_reason is not None:
        action = AlertAction(
            alert_key=alert_key,
            event_type="suppressed",
            severity=decision.level,
            should_notify=False,
            notification_kind=None,
            occurrence_count=0,
            message=f"维护期抑制通知：{maintenance_reason}；{decision.reason}",
        )
        with connection:
            save_alert_event(connection, action, now)
        return action

    # 新告警，或者旧告警已经恢复后再次发生。
    if state is None or state["status"] == "resolved":
        action = AlertAction(
            alert_key=alert_key,
            event_type="opened",
            severity=decision.level,
            should_notify=True,
            notification_kind="firing",
            occurrence_count=1,
            message=f"打开{decision.level}告警：{decision.reason}",
        )
        with connection:
            connection.execute(
                """
                INSERT INTO alert_states (
                    alert_key, service_name, status, severity,
                    occurrence_count, first_seen_at, last_seen_at,
                    last_reason, resolved_at
                )
                VALUES (?, ?, 'open', ?, 1, ?, ?, ?, NULL)
                ON CONFLICT(alert_key) DO UPDATE SET
                    status = 'open', severity = excluded.severity,
                    occurrence_count = 1,
                    first_seen_at = excluded.first_seen_at,
                    last_seen_at = excluded.last_seen_at,
                    last_reason = excluded.last_reason,
                    resolved_at = NULL
                """,
                (
                    alert_key,
                    service_name,
                    decision.level,
                    now_text(now),
                    now_text(now),
                    decision.reason,
                ),
            )
            save_alert_event(connection, action, now)
        return action

    new_count = state["occurrence_count"] + 1
    old_level = state["severity"]
    severity_changed = decision.level != old_level
    escalated = SEVERITY_ORDER[decision.level] > SEVERITY_ORDER[old_level]
    if severity_changed:
        action = AlertAction(
            alert_key=alert_key,
            event_type="severity_changed",
            severity=decision.level,
            should_notify=escalated,
            notification_kind="escalation" if escalated else None,
            occurrence_count=new_count,
            message=(
                f"严重度从{old_level}变为{decision.level}：{decision.reason}"
            ),
        )
    else:
        action = AlertAction(
            alert_key=alert_key,
            event_type="deduplicated",
            severity=decision.level,
            should_notify=False,
            notification_kind=None,
            occurrence_count=new_count,
            message=f"相同告警仍在持续，已去重：{decision.reason}",
        )

    with connection:
        connection.execute(
            """
            UPDATE alert_states
            SET severity = ?, occurrence_count = ?,
                last_seen_at = ?, last_reason = ?
            WHERE alert_key = ?
            """,
            (
                decision.level,
                new_count,
                now_text(now),
                decision.reason,
                alert_key,
            ),
        )
        save_alert_event(connection, action, now)
    return action


def normal_decision() -> MultiWindowDecision:
    return MultiWindowDecision(
        level="normal",
        fast_burn_triggered=False,
        slow_burn_triggered=False,
        reason="所有滚动窗口均已回到阈值以下。",
        measurements={},
    )


def show_action(title: str, action: AlertAction) -> None:
    print(f"\n{title}")
    print("事件：", action.event_type)
    print("严重度：", action.severity)
    print("累计出现：", action.occurrence_count)
    print("是否通知：", action.should_notify)
    print("通知类型：", action.notification_kind)
    print("消息：", action.message)
    if action.should_notify:
        print(f"[模拟通知] {action.notification_kind}: {action.message}")


def main() -> None:
    now = datetime.now().astimezone().replace(microsecond=0)
    print("Day27 告警去重、维护期抑制与恢复通知")
    print("服务：", SERVICE_NAME)
    print("告警键：", ALERT_KEY)
    print("=" * 96)

    with open_database() as connection:
        create_schema(connection)
        create_burn_schema(connection)
        create_alert_schema(connection)
        save_events(connection, build_six_hour_events(now))
        measurements = {
            definition.name: measure_window(connection, definition, now)
            for definition in WINDOWS
        }
        incident = evaluate_multi_window(measurements)

        # 使用本次运行独立的key，便于重复练习完整生命周期。
        run_suffix = uuid4().hex[:8]
        availability_key = f"{ALERT_KEY}:{run_suffix}"
        maintenance_key = f"{SERVICE_NAME}:quality_burn_rate:{run_suffix}"

        first = process_alert(
            connection, availability_key, SERVICE_NAME, incident, now
        )
        repeated = process_alert(
            connection,
            availability_key,
            SERVICE_NAME,
            incident,
            now + timedelta(minutes=1),
        )

        add_maintenance_window(
            connection,
            f"maintenance-{run_suffix}",
            SERVICE_NAME,
            "计划内向量数据库维护",
            now - timedelta(minutes=5),
            now + timedelta(minutes=30),
        )
        suppressed = process_alert(
            connection,
            maintenance_key,
            SERVICE_NAME,
            incident,
            now + timedelta(minutes=2),
        )

        # 为了演示恢复，关闭维护窗口后处理正常判断。
        connection.execute(
            "UPDATE maintenance_windows SET enabled = 0 WHERE maintenance_id = ?",
            (f"maintenance-{run_suffix}",),
        )
        connection.commit()
        recovered = process_alert(
            connection,
            availability_key,
            SERVICE_NAME,
            normal_decision(),
            now + timedelta(minutes=10),
        )
        still_normal = process_alert(
            connection,
            availability_key,
            SERVICE_NAME,
            normal_decision(),
            now + timedelta(minutes=11),
        )

        history_rows = connection.execute(
            """
            SELECT event_type, severity, should_notify, message, occurred_at
            FROM alert_events
            WHERE alert_key IN (?, ?)
            ORDER BY occurred_at, rowid
            """,
            (availability_key, maintenance_key),
        ).fetchall()
        history = [dict(row) for row in history_rows]

    show_action("1. 第一次检测到事故", first)
    show_action("2. 一分钟后仍是同一事故", repeated)
    show_action("3. 维护窗口内的另一条告警", suppressed)
    show_action("4. 指标恢复正常", recovered)
    show_action("5. 恢复后继续正常", still_normal)

    print("\n" + "=" * 96)
    print("本次告警审计时间线：")
    print(json.dumps(history, ensure_ascii=False, indent=2))

    print("\n本步数据流：")
    print("MultiWindowDecision + alert_key")
    print("→ 查询alert_states与maintenance_windows")
    print("→ opened / deduplicated / suppressed / severity_changed / resolved")
    print("→ 只有首次、升级和恢复需要通知")
    print("→ alert_events保存完整审计时间线")


if __name__ == "__main__":
    main()

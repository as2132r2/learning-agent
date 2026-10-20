"""Day27 第 4 步：事故状态机、负责人和审计时间线。

运行：
    python day27/python/day27_incident_lifecycle.py

本步只模拟事故处理，不发送外部通知。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from day27_alert_manager import create_alert_schema
from day27_slo_basic import create_schema, open_database


IncidentStatus = Literal[
    "detected",
    "acknowledged",
    "investigating",
    "mitigated",
    "resolved",
]

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "detected": {"acknowledged"},
    "acknowledged": {"investigating"},
    "investigating": {"mitigated"},
    "mitigated": {"resolved", "investigating"},
    "resolved": {"investigating"},
}


class InvalidTransitionError(RuntimeError):
    """状态不允许这样迁移。"""


class ConcurrentUpdateError(RuntimeError):
    """事故已经被别人更新，当前客户端版本过期。"""


@dataclass(slots=True)
class Incident:
    incident_id: str
    alert_key: str
    title: str
    severity: str
    status: str
    owner: str | None
    version: int
    detected_at: str
    updated_at: str
    resolved_at: str | None


def create_incident_schema(connection: sqlite3.Connection) -> None:
    """创建事故当前状态和追加式时间线。"""

    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS incidents (
            incident_id      TEXT PRIMARY KEY,
            alert_key        TEXT NOT NULL,
            title            TEXT NOT NULL,
            severity         TEXT NOT NULL
                             CHECK (severity IN ('warning', 'page')),
            status           TEXT NOT NULL
                             CHECK (status IN (
                                'detected', 'acknowledged', 'investigating',
                                'mitigated', 'resolved'
                             )),
            owner            TEXT,
            version          INTEGER NOT NULL CHECK (version > 0),
            detected_at      TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            resolved_at      TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_incident_per_alert
        ON incidents(alert_key)
        WHERE status != 'resolved';

        CREATE TABLE IF NOT EXISTS incident_timeline (
            timeline_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id      TEXT NOT NULL,
            event_type       TEXT NOT NULL,
            actor            TEXT NOT NULL,
            from_status      TEXT,
            to_status        TEXT,
            note             TEXT NOT NULL,
            incident_version INTEGER NOT NULL,
            occurred_at      TEXT NOT NULL,
            FOREIGN KEY (incident_id)
                REFERENCES incidents(incident_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_incident_timeline_order
        ON incident_timeline(incident_id, timeline_id);
        """
    )
    connection.commit()


def time_text(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def row_to_incident(row: sqlite3.Row) -> Incident:
    return Incident(
        incident_id=row["incident_id"],
        alert_key=row["alert_key"],
        title=row["title"],
        severity=row["severity"],
        status=row["status"],
        owner=row["owner"],
        version=row["version"],
        detected_at=row["detected_at"],
        updated_at=row["updated_at"],
        resolved_at=row["resolved_at"],
    )


def get_incident(
    connection: sqlite3.Connection,
    incident_id: str,
) -> Incident:
    row = connection.execute(
        "SELECT * FROM incidents WHERE incident_id = ?",
        (incident_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"找不到事故：{incident_id}")
    return row_to_incident(row)


def append_timeline(
    connection: sqlite3.Connection,
    incident_id: str,
    event_type: str,
    actor: str,
    note: str,
    version: int,
    occurred_at: datetime,
    from_status: str | None = None,
    to_status: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO incident_timeline (
            incident_id, event_type, actor, from_status,
            to_status, note, incident_version, occurred_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            incident_id,
            event_type,
            actor,
            from_status,
            to_status,
            note,
            version,
            time_text(occurred_at),
        ),
    )


def create_incident(
    connection: sqlite3.Connection,
    alert_key: str,
    title: str,
    severity: str,
    actor: str,
    now: datetime,
) -> tuple[Incident, bool]:
    """相同alert_key存在活动事故时复用，避免重复建事故。"""

    if severity not in {"warning", "page"}:
        raise ValueError("severity必须是warning或page。")
    old = connection.execute(
        """
        SELECT * FROM incidents
        WHERE alert_key = ? AND status != 'resolved'
        """,
        (alert_key,),
    ).fetchone()
    if old is not None:
        return row_to_incident(old), False

    incident_id = f"inc-{uuid4().hex[:12]}"
    timestamp = time_text(now)
    with connection:
        connection.execute(
            """
            INSERT INTO incidents (
                incident_id, alert_key, title, severity,
                status, owner, version, detected_at,
                updated_at, resolved_at
            )
            VALUES (?, ?, ?, ?, 'detected', NULL, 1, ?, ?, NULL)
            """,
            (incident_id, alert_key, title, severity, timestamp, timestamp),
        )
        append_timeline(
            connection,
            incident_id,
            "incident_created",
            actor,
            f"由{severity}告警创建事故：{title}",
            1,
            now,
            to_status="detected",
        )
    return get_incident(connection, incident_id), True


def transition_incident(
    connection: sqlite3.Connection,
    incident_id: str,
    target_status: IncidentStatus,
    actor: str,
    note: str,
    expected_version: int,
    now: datetime,
) -> Incident:
    """校验状态机和版本后，原子更新状态并追加时间线。"""

    current = get_incident(connection, incident_id)
    if current.version != expected_version:
        raise ConcurrentUpdateError(
            f"期望版本{expected_version}，数据库已经是版本{current.version}；"
            "请刷新后重试。"
        )
    allowed = ALLOWED_TRANSITIONS.get(current.status, set())
    if target_status not in allowed:
        raise InvalidTransitionError(
            f"不允许从{current.status}直接变为{target_status}；"
            f"允许的下一状态={sorted(allowed)}。"
        )
    if not actor.strip() or not note.strip():
        raise ValueError("actor和note不能为空。")

    new_version = current.version + 1
    resolved_at = time_text(now) if target_status == "resolved" else None
    with connection:
        cursor = connection.execute(
            """
            UPDATE incidents
            SET status = ?, version = ?, updated_at = ?, resolved_at = ?
            WHERE incident_id = ? AND version = ?
            """,
            (
                target_status,
                new_version,
                time_text(now),
                resolved_at,
                incident_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise ConcurrentUpdateError("事故被并发更新，请刷新后重试。")
        append_timeline(
            connection,
            incident_id,
            "status_changed",
            actor,
            note,
            new_version,
            now,
            from_status=current.status,
            to_status=target_status,
        )
    return get_incident(connection, incident_id)


def assign_owner(
    connection: sqlite3.Connection,
    incident_id: str,
    owner: str,
    actor: str,
    expected_version: int,
    now: datetime,
) -> Incident:
    """分配负责人同样使用乐观锁并写时间线。"""

    current = get_incident(connection, incident_id)
    if current.version != expected_version:
        raise ConcurrentUpdateError(
            f"负责人分配失败：期望版本{expected_version}，"
            f"实际版本{current.version}。"
        )
    if not owner.strip():
        raise ValueError("owner不能为空。")

    new_version = current.version + 1
    with connection:
        cursor = connection.execute(
            """
            UPDATE incidents
            SET owner = ?, version = ?, updated_at = ?
            WHERE incident_id = ? AND version = ?
            """,
            (owner, new_version, time_text(now), incident_id, expected_version),
        )
        if cursor.rowcount != 1:
            raise ConcurrentUpdateError("事故被并发更新，请刷新后重试。")
        append_timeline(
            connection,
            incident_id,
            "owner_changed",
            actor,
            f"负责人设为{owner}",
            new_version,
            now,
            from_status=current.status,
            to_status=current.status,
        )
    return get_incident(connection, incident_id)


def add_note(
    connection: sqlite3.Connection,
    incident_id: str,
    actor: str,
    note: str,
    now: datetime,
) -> None:
    """普通备注不改变状态和版本，但永久追加到时间线。"""

    current = get_incident(connection, incident_id)
    if not note.strip():
        raise ValueError("note不能为空。")
    with connection:
        append_timeline(
            connection,
            incident_id,
            "note_added",
            actor,
            note,
            current.version,
            now,
            from_status=current.status,
            to_status=current.status,
        )


def load_timeline(
    connection: sqlite3.Connection,
    incident_id: str,
) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT timeline_id, event_type, actor,
               from_status, to_status, note,
               incident_version, occurred_at
        FROM incident_timeline
        WHERE incident_id = ?
        ORDER BY timeline_id
        """,
        (incident_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def show_incident(label: str, incident: Incident) -> None:
    print(
        f"{label:<12} status={incident.status:<14} "
        f"owner={str(incident.owner):<12} version={incident.version}"
    )


def main() -> None:
    base_time = datetime.now().astimezone().replace(microsecond=0)
    alert_key = f"langchain-rag-api:availability:{uuid4().hex[:8]}"

    print("Day27 事故状态机与审计时间线")
    print("告警键：", alert_key)
    print("合法主流程：detected → acknowledged → investigating → mitigated → resolved")
    print("=" * 104)

    with open_database() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        create_schema(connection)
        create_alert_schema(connection)
        create_incident_schema(connection)

        incident, created = create_incident(
            connection,
            alert_key,
            "RAG API可用率错误预算快速燃烧",
            "page",
            "monitoring-system",
            base_time,
        )
        show_incident("创建事故", incident)
        print("是否新建：", created)

        print("\n尝试跳过处理步骤，直接从detected变成resolved：")
        try:
            transition_incident(
                connection,
                incident.incident_id,
                "resolved",
                "engineer-leo",
                "尝试直接关闭",
                incident.version,
                base_time + timedelta(minutes=1),
            )
        except InvalidTransitionError as error:
            print("迁移被拒绝：", error)

        incident = transition_incident(
            connection,
            incident.incident_id,
            "acknowledged",
            "engineer-leo",
            "已收到告警，开始响应。",
            incident.version,
            base_time + timedelta(minutes=2),
        )
        show_incident("确认事故", incident)

        stale_version = incident.version
        incident = assign_owner(
            connection,
            incident.incident_id,
            "engineer-leo",
            "incident-bot",
            incident.version,
            base_time + timedelta(minutes=3),
        )
        show_incident("分配负责人", incident)

        print("\n模拟另一个页面仍使用过期version更新：")
        try:
            assign_owner(
                connection,
                incident.incident_id,
                "engineer-xiaolin",
                "stale-browser",
                stale_version,
                base_time + timedelta(minutes=4),
            )
        except ConcurrentUpdateError as error:
            print("并发更新被拒绝：", error)

        incident = transition_incident(
            connection,
            incident.incident_id,
            "investigating",
            "engineer-leo",
            "开始检查模型网关、超时和上游错误率。",
            incident.version,
            base_time + timedelta(minutes=5),
        )
        show_incident("开始调查", incident)

        add_note(
            connection,
            incident.incident_id,
            "engineer-leo",
            "发现候选版本上游连接错误增加，准备回退control。",
            base_time + timedelta(minutes=8),
        )

        incident = transition_incident(
            connection,
            incident.incident_id,
            "mitigated",
            "engineer-leo",
            "已停止候选流量并全部回退control，燃烧率开始下降。",
            incident.version,
            base_time + timedelta(minutes=12),
        )
        show_incident("完成缓解", incident)

        incident = transition_incident(
            connection,
            incident.incident_id,
            "resolved",
            "incident-commander",
            "连续观察10分钟，短窗和长窗均恢复正常。",
            incident.version,
            base_time + timedelta(minutes=22),
        )
        show_incident("解决事故", incident)

        timeline = load_timeline(connection, incident.incident_id)

    print("\n" + "=" * 104)
    print("最终事故对象：")
    print(json.dumps(asdict(incident), ensure_ascii=False, indent=2))
    print("\n不可变审计时间线：")
    print(json.dumps(timeline, ensure_ascii=False, indent=2))

    print("\n本步数据流：")
    print("page告警 → create_incident() → detected")
    print("→ 状态机校验 + expected_version乐观锁")
    print("→ acknowledged → investigating → mitigated → resolved")
    print("→ 每次操作追加incident_timeline")
    print("→ 当前状态用于协作，时间线用于审计和复盘")


if __name__ == "__main__":
    main()

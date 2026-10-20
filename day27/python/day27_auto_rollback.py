"""Day27 第 5 步：自动回滚、冷却期和恢复验证。

运行：
    python day27/python/day27_auto_rollback.py

本步使用模拟版本和健康探测，不会修改Day26实验数据库。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

from day27_slo_basic import create_schema, open_database


ProtectionStatus = Literal["normal", "cooldown", "verifying"]
RecoveryDecision = Literal[
    "cooldown",
    "not_enough_probes",
    "probe_failed",
    "recovered",
]

SERVICE_NAME = "langchain-rag-api"
CONTROL_VERSION = "rag-control-v1"
CANDIDATE_VERSION = "rag-candidate-v2"
COOLDOWN_MINUTES = 10
REQUIRED_CONSECUTIVE_PROBES = 3
MAX_RECOVERY_LATENCY_MS = 4000


@dataclass(slots=True)
class ReleaseState:
    service_name: str
    active_version: str
    last_known_good_version: str
    protection_status: str
    cooldown_until: str | None
    state_version: int
    updated_at: str


@dataclass(slots=True)
class RollbackResult:
    action_id: str
    incident_id: str
    from_version: str
    to_version: str
    executed: bool
    reason: str
    cooldown_until: str


@dataclass(slots=True)
class RecoveryResult:
    decision: RecoveryDecision
    successful_probes: int
    required_probes: int
    message: str
    state: ReleaseState


def create_rollback_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS release_versions (
            version_id         TEXT PRIMARY KEY,
            service_name       TEXT NOT NULL,
            status             TEXT NOT NULL
                               CHECK (status IN (
                                  'active', 'stable', 'quarantined', 'inactive'
                               )),
            config_hash        TEXT NOT NULL,
            created_at         TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS service_release_state (
            service_name             TEXT PRIMARY KEY,
            active_version           TEXT NOT NULL,
            last_known_good_version  TEXT NOT NULL,
            protection_status        TEXT NOT NULL
                                     CHECK (protection_status IN (
                                        'normal', 'cooldown', 'verifying'
                                     )),
            cooldown_until           TEXT,
            state_version            INTEGER NOT NULL CHECK (state_version > 0),
            updated_at               TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rollback_actions (
            action_id          TEXT PRIMARY KEY,
            incident_id        TEXT NOT NULL UNIQUE,
            service_name       TEXT NOT NULL,
            from_version       TEXT NOT NULL,
            to_version         TEXT NOT NULL,
            reason             TEXT NOT NULL,
            executed_at        TEXT NOT NULL,
            cooldown_until     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS recovery_probes (
            probe_id           TEXT PRIMARY KEY,
            action_id          TEXT NOT NULL,
            available          INTEGER NOT NULL CHECK (available IN (0, 1)),
            quality_passed     INTEGER NOT NULL CHECK (quality_passed IN (0, 1)),
            latency_ms         INTEGER NOT NULL CHECK (latency_ms >= 0),
            successful         INTEGER NOT NULL CHECK (successful IN (0, 1)),
            details            TEXT NOT NULL,
            probed_at          TEXT NOT NULL,
            FOREIGN KEY (action_id)
                REFERENCES rollback_actions(action_id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_recovery_probe_action_time
        ON recovery_probes(action_id, probed_at DESC);
        """
    )
    connection.commit()


def time_text(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def initialize_demo_release(
    connection: sqlite3.Connection,
    now: datetime,
) -> None:
    """每次练习开始时模拟candidate当前正在承载流量。"""

    with connection:
        connection.execute(
            """
            INSERT INTO release_versions (
                version_id, service_name, status, config_hash, created_at
            ) VALUES (?, ?, 'stable', 'control-hash-001', ?)
            ON CONFLICT(version_id) DO UPDATE SET status = 'stable'
            """,
            (CONTROL_VERSION, SERVICE_NAME, time_text(now)),
        )
        connection.execute(
            """
            INSERT INTO release_versions (
                version_id, service_name, status, config_hash, created_at
            ) VALUES (?, ?, 'active', 'candidate-hash-002', ?)
            ON CONFLICT(version_id) DO UPDATE SET status = 'active'
            """,
            (CANDIDATE_VERSION, SERVICE_NAME, time_text(now)),
        )
        connection.execute(
            """
            INSERT INTO service_release_state (
                service_name, active_version, last_known_good_version,
                protection_status, cooldown_until, state_version, updated_at
            ) VALUES (?, ?, ?, 'normal', NULL, 1, ?)
            ON CONFLICT(service_name) DO UPDATE SET
                active_version = excluded.active_version,
                last_known_good_version = excluded.last_known_good_version,
                protection_status = 'normal',
                cooldown_until = NULL,
                state_version = service_release_state.state_version + 1,
                updated_at = excluded.updated_at
            """,
            (
                SERVICE_NAME,
                CANDIDATE_VERSION,
                CONTROL_VERSION,
                time_text(now),
            ),
        )


def load_release_state(connection: sqlite3.Connection) -> ReleaseState:
    row = connection.execute(
        "SELECT * FROM service_release_state WHERE service_name = ?",
        (SERVICE_NAME,),
    ).fetchone()
    if row is None:
        raise LookupError("没有找到服务发布状态。")
    return ReleaseState(
        service_name=row["service_name"],
        active_version=row["active_version"],
        last_known_good_version=row["last_known_good_version"],
        protection_status=row["protection_status"],
        cooldown_until=row["cooldown_until"],
        state_version=row["state_version"],
        updated_at=row["updated_at"],
    )


def execute_rollback(
    connection: sqlite3.Connection,
    incident_id: str,
    reason: str,
    now: datetime,
) -> RollbackResult:
    """同一incident_id只执行一次真实回滚。"""

    old_action = connection.execute(
        "SELECT * FROM rollback_actions WHERE incident_id = ?",
        (incident_id,),
    ).fetchone()
    if old_action is not None:
        return RollbackResult(
            action_id=old_action["action_id"],
            incident_id=old_action["incident_id"],
            from_version=old_action["from_version"],
            to_version=old_action["to_version"],
            executed=False,
            reason="同一事故已经执行过回滚，本次幂等返回。",
            cooldown_until=old_action["cooldown_until"],
        )

    state = load_release_state(connection)
    if state.active_version == state.last_known_good_version:
        raise RuntimeError("当前已经运行last-known-good版本，无需再次回滚。")

    action_id = f"rollback-{uuid4().hex[:12]}"
    cooldown_until = now + timedelta(minutes=COOLDOWN_MINUTES)
    new_state_version = state.state_version + 1

    # 隔离candidate、切换active和写审计记录必须在同一个事务中。
    with connection:
        connection.execute(
            "UPDATE release_versions SET status = 'quarantined' WHERE version_id = ?",
            (state.active_version,),
        )
        connection.execute(
            "UPDATE release_versions SET status = 'active' WHERE version_id = ?",
            (state.last_known_good_version,),
        )
        connection.execute(
            """
            UPDATE service_release_state
            SET active_version = last_known_good_version,
                protection_status = 'cooldown',
                cooldown_until = ?,
                state_version = ?,
                updated_at = ?
            WHERE service_name = ? AND state_version = ?
            """,
            (
                time_text(cooldown_until),
                new_state_version,
                time_text(now),
                SERVICE_NAME,
                state.state_version,
            ),
        )
        connection.execute(
            """
            INSERT INTO rollback_actions (
                action_id, incident_id, service_name,
                from_version, to_version, reason,
                executed_at, cooldown_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action_id,
                incident_id,
                SERVICE_NAME,
                state.active_version,
                state.last_known_good_version,
                reason,
                time_text(now),
                time_text(cooldown_until),
            ),
        )

    return RollbackResult(
        action_id=action_id,
        incident_id=incident_id,
        from_version=state.active_version,
        to_version=state.last_known_good_version,
        executed=True,
        reason=reason,
        cooldown_until=time_text(cooldown_until),
    )


def record_probe(
    connection: sqlite3.Connection,
    action_id: str,
    available: bool,
    quality_passed: bool,
    latency_ms: int,
    details: str,
    now: datetime,
) -> bool:
    """恢复探测必须同时满足可用、质量和延迟三项。"""

    if latency_ms < 0:
        raise ValueError("latency_ms不能为负数。")
    successful = (
        available
        and quality_passed
        and latency_ms <= MAX_RECOVERY_LATENCY_MS
    )
    with connection:
        connection.execute(
            """
            INSERT INTO recovery_probes (
                probe_id, action_id, available, quality_passed,
                latency_ms, successful, details, probed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"probe-{uuid4().hex[:12]}",
                action_id,
                int(available),
                int(quality_passed),
                latency_ms,
                int(successful),
                details,
                time_text(now),
            ),
        )
    return successful


def evaluate_recovery(
    connection: sqlite3.Connection,
    action_id: str,
    now: datetime,
) -> RecoveryResult:
    """冷却结束后，要求最近连续N次探测全部成功。"""

    state = load_release_state(connection)
    if state.cooldown_until is None:
        return RecoveryResult(
            decision="recovered",
            successful_probes=REQUIRED_CONSECUTIVE_PROBES,
            required_probes=REQUIRED_CONSECUTIVE_PROBES,
            message="服务不在保护期。",
            state=state,
        )

    cooldown_until = datetime.fromisoformat(state.cooldown_until)
    if now < cooldown_until:
        return RecoveryResult(
            decision="cooldown",
            successful_probes=0,
            required_probes=REQUIRED_CONSECUTIVE_PROBES,
            message=f"仍在冷却期，最早于{state.cooldown_until}验证恢复。",
            state=state,
        )

    probes = connection.execute(
        """
        SELECT successful, latency_ms, probed_at
        FROM recovery_probes
        WHERE action_id = ?
        ORDER BY probed_at DESC, rowid DESC
        LIMIT ?
        """,
        (action_id, REQUIRED_CONSECUTIVE_PROBES),
    ).fetchall()

    if len(probes) < REQUIRED_CONSECUTIVE_PROBES:
        return RecoveryResult(
            decision="not_enough_probes",
            successful_probes=sum(bool(row["successful"]) for row in probes),
            required_probes=REQUIRED_CONSECUTIVE_PROBES,
            message="冷却已结束，但健康探测数量不足。",
            state=state,
        )

    successful_count = sum(bool(row["successful"]) for row in probes)
    if successful_count != REQUIRED_CONSECUTIVE_PROBES:
        with connection:
            connection.execute(
                """
                UPDATE service_release_state
                SET protection_status = 'verifying', updated_at = ?
                WHERE service_name = ?
                """,
                (time_text(now), SERVICE_NAME),
            )
        return RecoveryResult(
            decision="probe_failed",
            successful_probes=successful_count,
            required_probes=REQUIRED_CONSECUTIVE_PROBES,
            message="最近连续探测中仍有失败，继续保持保护状态。",
            state=load_release_state(connection),
        )

    with connection:
        connection.execute(
            """
            UPDATE service_release_state
            SET protection_status = 'normal', cooldown_until = NULL,
                state_version = state_version + 1, updated_at = ?
            WHERE service_name = ?
            """,
            (time_text(now), SERVICE_NAME),
        )
    return RecoveryResult(
        decision="recovered",
        successful_probes=successful_count,
        required_probes=REQUIRED_CONSECUTIVE_PROBES,
        message=(
            "冷却结束且连续健康探测通过，control服务恢复稳定；"
            "candidate仍保持quarantined。"
        ),
        state=load_release_state(connection),
    )


def load_versions(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT version_id, status, config_hash, created_at
        FROM release_versions
        WHERE service_name = ?
        ORDER BY version_id
        """,
        (SERVICE_NAME,),
    ).fetchall()
    return [dict(row) for row in rows]


def main() -> None:
    now = datetime.now().astimezone().replace(microsecond=0)
    incident_id = f"inc-auto-{uuid4().hex[:10]}"
    print("Day27 自动回滚、冷却期与恢复验证")
    print("服务：", SERVICE_NAME)
    print("事故：", incident_id)
    print("=" * 100)

    with open_database() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        create_schema(connection)
        create_rollback_schema(connection)
        initialize_demo_release(connection, now)

        before = load_release_state(connection)
        print("回滚前状态：")
        print(json.dumps(asdict(before), ensure_ascii=False, indent=2))

        rollback = execute_rollback(
            connection,
            incident_id,
            "page告警确认candidate导致可用率错误预算快速燃烧。",
            now,
        )
        repeated = execute_rollback(
            connection,
            incident_id,
            "重复执行不应再次切换版本。",
            now + timedelta(seconds=5),
        )
        after = load_release_state(connection)

        print("\n回滚结果：")
        print(json.dumps(asdict(rollback), ensure_ascii=False, indent=2))
        print("重复回滚executed：", repeated.executed)
        print("当前活动版本：", after.active_version)
        print("保护状态：", after.protection_status)

        early = evaluate_recovery(
            connection, rollback.action_id, now + timedelta(minutes=5)
        )
        print("\n冷却期内检查：", early.decision, "-", early.message)

        # 冷却期内和刚结束时进行三次成功探测。
        for index, minutes in enumerate((6, 8, 10), start=1):
            success = record_probe(
                connection,
                rollback.action_id,
                available=True,
                quality_passed=True,
                latency_ms=1700 + index * 100,
                details=f"control健康探测{index}",
                now=now + timedelta(minutes=minutes),
            )
            print(f"探测{index}：", "成功" if success else "失败")

        recovered = evaluate_recovery(
            connection, rollback.action_id, now + timedelta(minutes=11)
        )
        versions = load_versions(connection)

    print("\n恢复判断：")
    print(json.dumps(asdict(recovered), ensure_ascii=False, indent=2))
    print("\n版本状态：")
    print(json.dumps(versions, ensure_ascii=False, indent=2))

    print("\n本步数据流：")
    print("page事故 → execute_rollback(incident_id)")
    print("→ candidate=quarantined + control=active + cooldown")
    print("→ 重复incident_id幂等返回")
    print("→ 冷却期结束 + 最近连续3次探测成功")
    print("→ protection_status=normal")
    print("→ control确认稳定，但candidate不会自动重新启用")


if __name__ == "__main__":
    main()

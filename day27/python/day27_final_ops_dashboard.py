"""Day27 第 6 步：SLO、告警、事故与回滚生产运维控制台。

运行：
    python day27/python/day27_final_ops_dashboard.py

浏览器：
    http://127.0.0.1:8036
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from day27_alert_manager import (
    ALERT_KEY,
    SERVICE_NAME,
    create_alert_schema,
    process_alert,
)
from day27_auto_rollback import (
    create_rollback_schema,
    execute_rollback,
    initialize_demo_release,
    load_release_state,
    load_versions,
)
from day27_burn_rate import (
    WINDOWS,
    build_six_hour_events,
    create_burn_schema,
    evaluate_multi_window,
    measure_window,
    save_snapshot,
)
from day27_incident_lifecycle import (
    create_incident,
    create_incident_schema,
)
from day27_slo_basic import DATABASE_PATH, create_schema, open_database, save_events


HOST, PORT = "127.0.0.1", 8036
DAY27_DIR = Path(__file__).resolve().parents[1]
HTML = DAY27_DIR / "html" / "day27_final.html"
JS = DAY27_DIR / "javascript" / "day27_final.js"
CSS = DAY27_DIR / "css" / "day27_final.css"
MONITOR_INTERVAL_SECONDS = 15
DASHBOARD_ALERT_KEY = f"{ALERT_KEY}:dashboard"

monitor_lock = asyncio.Lock()


def initialize_all() -> None:
    """建表并确保至少存在一份发布状态；不会重置已有回滚结果。"""

    now = datetime.now().astimezone().replace(microsecond=0)
    with open_database() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        create_schema(connection)
        create_burn_schema(connection)
        create_alert_schema(connection)
        create_incident_schema(connection)
        create_rollback_schema(connection)
        state = connection.execute(
            "SELECT 1 FROM service_release_state WHERE service_name = ?",
            (SERVICE_NAME,),
        ).fetchone()
        if state is None:
            initialize_demo_release(connection, now)


def latest_active_incident(connection: sqlite3.Connection) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT * FROM incidents
        WHERE alert_key = ? AND status != 'resolved'
        ORDER BY detected_at DESC
        LIMIT 1
        """,
        (DASHBOARD_ALERT_KEY,),
    ).fetchone()
    return None if row is None else dict(row)


def monitor_cycle_sync() -> dict[str, Any]:
    """一次完整监控循环；同步数据库操作在线程中执行。"""

    now = datetime.now().astimezone().replace(microsecond=0)
    with open_database() as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        # 更新时间戳，让教学事故持续出现在当前滚动窗口。
        save_events(connection, build_six_hour_events(now))
        measurements = {
            definition.name: measure_window(connection, definition, now)
            for definition in WINDOWS
        }
        decision = evaluate_multi_window(measurements)
        snapshot_id = save_snapshot(connection, measurements)
        alert_action = process_alert(
            connection,
            DASHBOARD_ALERT_KEY,
            SERVICE_NAME,
            decision,
            now,
        )

        incident_id = None
        rollback_summary: dict[str, Any] | None = None
        if decision.level == "page":
            incident, _ = create_incident(
                connection,
                DASHBOARD_ALERT_KEY,
                "RAG API可用率错误预算快速燃烧",
                "page",
                "ops-monitor",
                now,
            )
            incident_id = incident.incident_id
            release = load_release_state(connection)
            if release.active_version != release.last_known_good_version:
                rollback = execute_rollback(
                    connection,
                    incident.incident_id,
                    "多窗口快速燃烧告警触发自动回滚。",
                    now,
                )
                rollback_summary = asdict(rollback)
            else:
                rollback_summary = {
                    "executed": False,
                    "reason": "服务已经运行last-known-good版本，无需重复回滚。",
                }

        return {
            "ran_at": now.isoformat(timespec="seconds"),
            "snapshot_id": snapshot_id,
            "decision": {
                "level": decision.level,
                "reason": decision.reason,
                "fast_burn_triggered": decision.fast_burn_triggered,
                "slow_burn_triggered": decision.slow_burn_triggered,
            },
            "alert_action": asdict(alert_action),
            "incident_id": incident_id,
            "rollback": rollback_summary,
        }


async def run_monitor_cycle() -> dict[str, Any]:
    async with monitor_lock:
        return await asyncio.to_thread(monitor_cycle_sync)


def read_overview_sync() -> dict[str, Any]:
    """读取当前状态和最近审计记录，避免前端自行拼接事实。"""

    now = datetime.now().astimezone().replace(microsecond=0)
    with open_database() as connection:
        measurements = {
            definition.name: measure_window(connection, definition, now)
            for definition in WINDOWS
        }
        decision = evaluate_multi_window(measurements)
        alert = connection.execute(
            "SELECT * FROM alert_states WHERE alert_key = ?",
            (DASHBOARD_ALERT_KEY,),
        ).fetchone()
        incident = latest_active_incident(connection)
        release = load_release_state(connection)
        versions = load_versions(connection)
        alert_events = connection.execute(
            """
            SELECT event_type, severity, should_notify, message, occurred_at
            FROM alert_events
            WHERE alert_key = ?
            ORDER BY occurred_at DESC, rowid DESC
            LIMIT 12
            """,
            (DASHBOARD_ALERT_KEY,),
        ).fetchall()
        rollback_rows = connection.execute(
            """
            SELECT action_id, incident_id, from_version,
                   to_version, reason, executed_at, cooldown_until
            FROM rollback_actions
            ORDER BY executed_at DESC
            LIMIT 5
            """
        ).fetchall()
        timeline_rows: list[sqlite3.Row] = []
        if incident is not None:
            timeline_rows = connection.execute(
                """
                SELECT event_type, actor, from_status,
                       to_status, note, occurred_at
                FROM incident_timeline
                WHERE incident_id = ?
                ORDER BY timeline_id DESC
                LIMIT 12
                """,
                (incident["incident_id"],),
            ).fetchall()

    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "service": SERVICE_NAME,
        "monitor_interval_seconds": MONITOR_INTERVAL_SECONDS,
        "decision": {
            "level": decision.level,
            "reason": decision.reason,
            "fast_burn_triggered": decision.fast_burn_triggered,
            "slow_burn_triggered": decision.slow_burn_triggered,
        },
        "windows": {name: asdict(item) for name, item in measurements.items()},
        "alert": None if alert is None else dict(alert),
        "incident": incident,
        "incident_timeline": [dict(row) for row in timeline_rows],
        "release": asdict(release),
        "versions": versions,
        "alert_events": [dict(row) for row in alert_events],
        "rollbacks": [dict(row) for row in rollback_rows],
    }


async def overview_handler(request: web.Request) -> web.Response:
    del request
    return web.json_response(await asyncio.to_thread(read_overview_sync))


async def run_monitor_handler(request: web.Request) -> web.Response:
    del request
    result = await run_monitor_cycle()
    return web.json_response({"ok": True, "cycle": result})


async def health_handler(request: web.Request) -> web.Response:
    del request
    return web.json_response(
        {
            "status": "ok",
            "service": "day27-ops-dashboard",
            "database": str(DATABASE_PATH),
        }
    )


async def file_handler(request: web.Request) -> web.FileResponse:
    return web.FileResponse(
        {"/": HTML, "/app.js": JS, "/app.css": CSS}[request.path]
    )


@web.middleware
async def errors(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException as error:
        return web.json_response(
            {"ok": False, "error": error.text or error.reason},
            status=error.status,
        )
    except Exception as error:
        print(f"[控制台错误] {type(error).__name__}: {error}")
        return web.json_response(
            {"ok": False, "error": "服务器内部错误。"}, status=500
        )


async def monitor_worker(app: web.Application) -> None:
    """后台周期任务；异常被记录，但不会让整个HTTP服务退出。"""

    del app
    while True:
        try:
            result = await run_monitor_cycle()
            print(
                f"[监控循环] {result['ran_at']} "
                f"level={result['decision']['level']} "
                f"alert={result['alert_action']['event_type']}"
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            print(f"[监控循环失败] {type(error).__name__}: {error}")
        await asyncio.sleep(MONITOR_INTERVAL_SECONDS)


async def start_background_monitor(app: web.Application) -> None:
    app["monitor_task"] = asyncio.create_task(
        monitor_worker(app), name="day27-slo-monitor"
    )


async def stop_background_monitor(app: web.Application) -> None:
    task: asyncio.Task[None] | None = app.get("monitor_task")
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def create_app() -> web.Application:
    app = web.Application(middlewares=[errors])
    app.router.add_get("/", file_handler)
    app.router.add_get("/app.js", file_handler)
    app.router.add_get("/app.css", file_handler)
    app.router.add_get("/health", health_handler)
    app.router.add_get("/api/overview", overview_handler)
    app.router.add_post("/api/monitor/run", run_monitor_handler)
    app.on_startup.append(start_background_monitor)
    app.on_cleanup.append(stop_background_monitor)
    return app


def main() -> None:
    initialize_all()
    print(f"Day27 生产运维控制台：http://{HOST}:{PORT}")
    print("数据库：", DATABASE_PATH)
    print("后台监控间隔：", MONITOR_INTERVAL_SECONDS, "秒")
    print("当前状态预览：")
    print(json.dumps(read_overview_sync()["decision"], ensure_ascii=False, indent=2))
    web.run_app(create_app(), host=HOST, port=PORT, print=None)


if __name__ == "__main__":
    main()

"""Day15 第十三步：记录不包含凭证的认证安全审计日志。"""

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

import day15_refresh_api as refresh_api
from day15_login_lockout_api import protected_login


AUDIT_DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "day15_auth_audit.db"
)


def write_audit(
    event_type: str,
    user_id: str,
    client_ip: str,
    outcome: str,
    detail: str
) -> None:
    """只记录安全事件元数据，不记录密码和令牌。"""

    connection = sqlite3.connect(AUDIT_DATABASE_PATH)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_audit(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                client_ip TEXT NOT NULL,
                outcome TEXT NOT NULL,
                detail TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO auth_audit(
                occurred_at, event_type, user_id,
                client_ip, outcome, detail
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                event_type,
                user_id,
                client_ip,
                outcome,
                detail
            )
        )
        connection.commit()
    finally:
        connection.close()


async def login_handler(request: web.Request) -> web.Response:
    data = await refresh_api.read_json(request)
    user_id = data.get("user_id")
    password = data.get("password")

    if not isinstance(user_id, str) or not isinstance(password, str):
        return web.json_response({"error": "invalid_input"}, status=422)

    user_id = user_id.strip()
    client_ip = request.remote or "unknown"
    status, tokens, retry_after = await asyncio.to_thread(
        protected_login, user_id, password
    )

    if status == "locked":
        await asyncio.to_thread(
            write_audit,
            "login_blocked",
            user_id,
            client_ip,
            "denied",
            f"account_locked retry_after={retry_after}"
        )
        return web.json_response(
            {"error": "account_temporarily_locked", "retry_after": retry_after},
            status=429,
            headers={"Retry-After": str(retry_after)}
        )

    if status == "invalid":
        await asyncio.to_thread(
            write_audit,
            "login_failed",
            user_id,
            client_ip,
            "denied",
            "invalid_credentials"
        )
        return web.json_response(
            {"error": "invalid_credentials", "message": "用户名或密码错误。"},
            status=401
        )

    await asyncio.to_thread(
        write_audit,
        "login_succeeded",
        user_id,
        client_ip,
        "allowed",
        "password_verified"
    )
    access_token, refresh_token = tokens
    return web.json_response({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer"
    })


app = web.Application(client_max_size=10_000)
app.router.add_post("/auth/login", login_handler)
app.router.add_post("/auth/refresh", refresh_api.refresh_handler)
app.router.add_post("/auth/logout", refresh_api.logout_handler)


if __name__ == "__main__":
    print("Day15认证审计API已启动。")
    print("审计数据库：", AUDIT_DATABASE_PATH)
    web.run_app(app, host="127.0.0.1", port=8000)

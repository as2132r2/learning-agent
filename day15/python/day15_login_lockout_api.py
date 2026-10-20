"""Day15 第十二步：登录失败计数与账号临时锁定。"""

import asyncio
import sqlite3
import time
from pathlib import Path

from aiohttp import web

import day15_refresh_api as refresh_api
from day15_real_login_api import authenticate_and_issue


MAX_FAILURES = 5
LOCK_SECONDS = 60
SECURITY_DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "day15_login_security.db"
)


def initialize_security_database(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS login_security(
            user_id TEXT PRIMARY KEY,
            failure_count INTEGER NOT NULL DEFAULT 0,
            locked_until INTEGER,
            last_failure_at INTEGER
        )
        """
    )
    connection.commit()


def protected_login(
    user_id: str,
    password: str
) -> tuple[str, tuple[str, str] | None, int | None]:
    """返回(status, tokens, retry_after)。"""

    connection = sqlite3.connect(SECURITY_DATABASE_PATH)
    initialize_security_database(connection)
    now = int(time.time())

    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT failure_count, locked_until
            FROM login_security WHERE user_id = ?
            """,
            (user_id,)
        ).fetchone()

        if row is not None and row[1] is not None and now < row[1]:
            connection.rollback()
            return "locked", None, row[1] - now

        tokens = authenticate_and_issue(user_id, password)

        if tokens is not None:
            connection.execute(
                """
                INSERT INTO login_security(user_id, failure_count, locked_until)
                VALUES (?, 0, NULL)
                ON CONFLICT(user_id) DO UPDATE SET
                    failure_count = 0,
                    locked_until = NULL
                """,
                (user_id,)
            )
            connection.commit()
            return "success", tokens, None

        old_count = row[0] if row is not None else 0
        new_count = old_count + 1
        locked_until = now + LOCK_SECONDS if new_count >= MAX_FAILURES else None

        connection.execute(
            """
            INSERT INTO login_security(
                user_id, failure_count, locked_until, last_failure_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                failure_count = excluded.failure_count,
                locked_until = excluded.locked_until,
                last_failure_at = excluded.last_failure_at
            """,
            (user_id, new_count, locked_until, now)
        )
        connection.commit()

        if locked_until is not None:
            return "locked", None, LOCK_SECONDS
        return "invalid", None, None

    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


async def login_handler(request: web.Request) -> web.Response:
    data = await refresh_api.read_json(request)
    user_id = data.get("user_id")
    password = data.get("password")

    if not isinstance(user_id, str) or not isinstance(password, str):
        return web.json_response({"error": "invalid_input"}, status=422)

    status, tokens, retry_after = await asyncio.to_thread(
        protected_login,
        user_id.strip(),
        password
    )

    if status == "locked":
        return web.json_response(
            {
                "error": "account_temporarily_locked",
                "message": "登录失败次数过多，请稍后重试。",
                "retry_after": retry_after
            },
            status=429,
            headers={"Retry-After": str(retry_after)}
        )

    if status == "invalid":
        return web.json_response(
            {"error": "invalid_credentials", "message": "用户名或密码错误。"},
            status=401
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
    print("Day15登录锁定API已启动。")
    print(f"连续失败{MAX_FAILURES}次，锁定{LOCK_SECONDS}秒。")
    web.run_app(app, host="127.0.0.1", port=8000)

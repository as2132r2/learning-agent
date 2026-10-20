"""Day15 第九步：Refresh Token的HTTP登录、刷新和退出流程。"""

import asyncio
import sqlite3

from aiohttp import web

from day15_refresh_token_basic import (
    DATABASE_PATH,
    RefreshTokenError,
    hash_refresh_token,
    initialize_database,
    issue_token_pair,
    rotate_refresh_token,
)


def with_connection(operation, *args):
    """每次操作使用独立SQLite连接。"""

    connection = sqlite3.connect(DATABASE_PATH)
    try:
        initialize_database(connection)
        return operation(connection, *args)
    finally:
        connection.close()


def revoke_refresh_token(connection: sqlite3.Connection, raw_token: str) -> None:
    token_hash = hash_refresh_token(raw_token)
    cursor = connection.execute(
        """
        UPDATE refresh_tokens
        SET used_at = COALESCE(used_at, unixepoch())
        WHERE token_hash = ?
        """,
        (token_hash,)
    )
    connection.commit()

    if cursor.rowcount == 0:
        raise RefreshTokenError("Refresh Token不存在。")


async def read_json(request: web.Request) -> dict:
    try:
        data = await request.json()
    except Exception as error:
        raise web.HTTPBadRequest(
            text='{"error":"invalid_json"}',
            content_type="application/json"
        ) from error

    if not isinstance(data, dict):
        raise web.HTTPUnprocessableEntity(
            text='{"error":"JSON最外层必须是对象"}',
            content_type="application/json"
        )
    return data


async def demo_login_handler(request: web.Request) -> web.Response:
    """教学模拟登录：下一步会替换成密码验证。"""

    data = await read_json(request)
    user_id = data.get("user_id")

    if not isinstance(user_id, str) or not user_id.strip():
        return web.json_response({"error": "user_id不能为空"}, status=422)

    access, refresh = await asyncio.to_thread(
        with_connection,
        issue_token_pair,
        user_id.strip()
    )
    return web.json_response({
        "access_token": access,
        "refresh_token": refresh,
        "token_type": "Bearer"
    })


async def refresh_handler(request: web.Request) -> web.Response:
    data = await read_json(request)
    refresh_token = data.get("refresh_token")

    if not isinstance(refresh_token, str) or not refresh_token.strip():
        return web.json_response(
            {"error": "refresh_token不能为空"}, status=422
        )

    try:
        access, new_refresh = await asyncio.to_thread(
            with_connection,
            rotate_refresh_token,
            refresh_token.strip()
        )
    except RefreshTokenError as error:
        return web.json_response(
            {"error": "invalid_refresh_token", "message": str(error)},
            status=401
        )

    return web.json_response({
        "access_token": access,
        "refresh_token": new_refresh,
        "token_type": "Bearer"
    })


async def logout_handler(request: web.Request) -> web.Response:
    data = await read_json(request)
    refresh_token = data.get("refresh_token")

    if not isinstance(refresh_token, str) or not refresh_token.strip():
        return web.json_response({"error": "refresh_token不能为空"}, status=422)

    try:
        await asyncio.to_thread(
            with_connection,
            revoke_refresh_token,
            refresh_token.strip()
        )
    except RefreshTokenError as error:
        return web.json_response(
            {"error": "invalid_refresh_token", "message": str(error)},
            status=401
        )

    return web.json_response({"logged_out": True})


app = web.Application(client_max_size=10_000)
app.router.add_post("/auth/demo-login", demo_login_handler)
app.router.add_post("/auth/refresh", refresh_handler)
app.router.add_post("/auth/logout", logout_handler)


if __name__ == "__main__":
    print("Day15 Refresh Token HTTP API已启动。")
    print("注意：demo-login仅用于教学，不是真实身份验证。")
    web.run_app(app, host="127.0.0.1", port=8000)

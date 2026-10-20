"""Day15 第十一步：密码验证成功后签发Access和Refresh Token。"""

import asyncio
import sqlite3

from aiohttp import web

import day15_refresh_api as refresh_api
from day15_password_auth_basic import (
    DATABASE_PATH as USER_DATABASE_PATH,
    initialize_database as initialize_user_database,
    verify_user,
)
from day15_refresh_token_basic import issue_token_pair


def authenticate_and_issue(user_id: str, password: str) -> tuple[str, str] | None:
    """在线程中验证密码，成功后签发令牌对。"""

    user_connection = sqlite3.connect(USER_DATABASE_PATH)
    try:
        initialize_user_database(user_connection)
        if not verify_user(user_connection, user_id, password):
            return None
    finally:
        user_connection.close()

    return refresh_api.with_connection(issue_token_pair, user_id)


async def login_handler(request: web.Request) -> web.Response:
    data = await refresh_api.read_json(request)
    user_id = data.get("user_id")
    password = data.get("password")

    if not isinstance(user_id, str) or not isinstance(password, str):
        return web.json_response(
            {"error": "user_id和password必须是字符串"},
            status=422
        )

    tokens = await asyncio.to_thread(
        authenticate_and_issue,
        user_id.strip(),
        password
    )

    if tokens is None:
        # 不区分用户不存在和密码错误，减少账号枚举信息。
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
    print("Day15真实密码登录API已启动。")
    print("POST /auth/login、/auth/refresh、/auth/logout")
    web.run_app(app, host="127.0.0.1", port=8000)

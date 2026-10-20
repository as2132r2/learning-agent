"""Day15 第七步：使用jti撤销尚未过期的访问令牌。"""

from datetime import datetime

from aiohttp import web

import day15_scope_api as scope_api
import day15_user_task_api as base


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@web.middleware
async def revocation_middleware(
    request: web.Request,
    handler
) -> web.StreamResponse:
    """认证通过后，检查当前令牌jti是否在撤销表中。"""

    if request.path == "/health":
        return await handler(request)

    identity = request.get("identity")
    if identity is None:
        # 正常情况下无效令牌已被前一层认证中间件拒绝。
        return await handler(request)

    token_id = identity["jti"]
    cursor = await base.db().execute(
        "SELECT 1 FROM revoked_tokens WHERE jti = ?",
        (token_id,)
    )
    revoked = await cursor.fetchone()
    await cursor.close()

    if revoked is not None:
        return web.json_response(
            {
                "error": "token_revoked",
                "message": "该访问令牌已经被撤销。"
            },
            status=401,
            headers={"WWW-Authenticate": "Bearer"}
        )

    return await handler(request)


async def revoke_current_token_handler(request: web.Request) -> web.Response:
    """用户撤销当前正在使用的令牌，相当于服务端退出登录。"""

    identity = request["identity"]

    await base.db().execute(
        """
        INSERT OR IGNORE INTO revoked_tokens(
            jti, user_id, expires_at, revoked_at
        ) VALUES (?, ?, ?, ?)
        """,
        (
            identity["jti"],
            identity["sub"],
            identity["exp"],
            now_text()
        )
    )
    await base.db().commit()

    return web.json_response({
        "revoked": True,
        "token_id": identity["jti"],
        "user_id": identity["sub"],
        "message": "当前令牌已撤销，后续请求将被拒绝。"
    })


async def create_revocation_table(_app: web.Application) -> None:
    await base.db().execute(
        """
        CREATE TABLE IF NOT EXISTS revoked_tokens(
            jti TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            revoked_at TEXT NOT NULL
        )
        """
    )
    await base.db().commit()


# authentication_middleware已经存在；追加后执行顺序为认证→撤销检查→Handler。
base.app.middlewares.append(revocation_middleware)
base.app.router.add_post("/auth/revoke", revoke_current_token_handler)
base.app.on_startup.append(create_revocation_table)


if __name__ == "__main__":
    print("Day15令牌撤销API已启动。")
    print("撤销当前令牌：POST /auth/revoke")
    web.run_app(base.app, host="127.0.0.1", port=8000)


# 签名验证通过 → 查询revoked_tokens(jti)
# 不存在 → 继续请求
# 已存在 → 401 token_revoked
# POST /auth/revoke → 当前jti写入撤销表

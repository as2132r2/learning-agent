"""Day15 第三步：将签名令牌接入aiohttp认证中间件。"""

import os

from aiohttp import web
from dotenv import load_dotenv

from day15_signed_token_basic import TokenError, verify_token


load_dotenv()


def load_secret() -> bytes:
    secret_text = os.getenv("DAY15_TOKEN_SECRET", "").strip()
    if len(secret_text) < 32:
        raise RuntimeError("DAY15_TOKEN_SECRET至少需要32个字符。")
    return secret_text.encode("utf-8")


TOKEN_SECRET = load_secret()
PUBLIC_PATHS = {"/health"}


def extract_bearer_token(request: web.Request) -> str:
    """从Authorization请求头提取Bearer Token。"""

    authorization = request.headers.get("Authorization", "")
    scheme, separator, token = authorization.partition(" ")

    if separator != " " or scheme.lower() != "bearer" or not token.strip():
        raise ValueError("缺少Authorization: Bearer <token>。")

    return token.strip()


@web.middleware
async def authentication_middleware(
    request: web.Request,
    handler
) -> web.StreamResponse:
    """公开接口直接通过；其他接口必须先验证身份。"""

    if request.path in PUBLIC_PATHS:
        return await handler(request)

    try:
        token = extract_bearer_token(request)
        identity = verify_token(token, TOKEN_SECRET)
    except (ValueError, TokenError) as error:
        return web.json_response(
            {
                "error": "unauthorized",
                "message": str(error)
            },
            status=401,
            headers={
                "WWW-Authenticate": "Bearer"
            }
        )

    # Handler不再读取客户端传来的user_id，而是使用已验证载荷中的sub。
    request["identity"] = identity
    return await handler(request)


async def health_handler(_request: web.Request) -> web.Response:
    """公开健康检查。"""

    return web.json_response({
        "status": "ok",
        "service": "day15-auth-api"
    })


async def me_handler(request: web.Request) -> web.Response:
    """返回认证中间件确认的当前身份。"""

    identity = request["identity"]

    return web.json_response({
        "authenticated": True,
        "user_id": identity["sub"],
        "token_id": identity["jti"],
        "issued_at": identity["iat"],
        "expires_at": identity["exp"]
    })


app = web.Application(middlewares=[authentication_middleware])
app.router.add_get("/health", health_handler)
app.router.add_get("/me", me_handler)


if __name__ == "__main__":
    print("Day15认证API已启动。")
    print("公开接口：GET /health")
    print("认证接口：GET /me")
    web.run_app(app, host="127.0.0.1", port=8000)


# HTTP请求
#   ↓ authentication_middleware
# Authorization Header → Bearer Token → verify_token()
#   ↓ 验证通过
# request["identity"] = payload
#   ↓
# /me Handler读取可信sub，而不是客户端提交的user_id

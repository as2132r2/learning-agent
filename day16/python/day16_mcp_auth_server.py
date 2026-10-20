"""Day16 第十五步（服务端）：需要 Bearer Token 的 HTTP MCP Server。

运行方式：
    python day16/python/day16_mcp_auth_server.py

教学令牌：day16-demo-token
MCP 地址：http://127.0.0.1:8002/mcp

注意：硬编码令牌只用于学习。生产环境应验证 JWT 签名或调用 OAuth 2.0
Token Introspection endpoint，不能把真实密钥写进源代码。
"""

from __future__ import annotations

from pydantic import AnyHttpUrl

from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings


RESOURCE_URL = "http://127.0.0.1:8002/mcp"
DEMO_TOKEN = "day16-demo-token"
REQUIRED_SCOPE = "tools:use"


class StaticTokenVerifier(TokenVerifier):
    """把收到的 Bearer Token 与教学令牌进行比较。"""

    async def verify_token(self, token: str) -> AccessToken | None:
        if token != DEMO_TOKEN:
            return None

        return AccessToken(
            token=token,
            client_id="day16-client",
            subject="user-xiaolin",
            scopes=[REQUIRED_SCOPE],
            resource=RESOURCE_URL,
        )


mcp_server = MCPServer(
    name="day16-auth-server",
    title="Day16 认证 MCP 服务",
    description="只有携带有效 Bearer Token 的客户端才能使用。",
    version="1.0.0",
    token_verifier=StaticTokenVerifier(),
    auth=AuthSettings(
        # 演示中不真正访问这个地址；它表示令牌签发方。
        issuer_url=AnyHttpUrl("https://auth.example.com"),
        resource_server_url=AnyHttpUrl(RESOURCE_URL),
        required_scopes=[REQUIRED_SCOPE],
        validate_token_resource=True,
    ),
)


@mcp_server.tool()
async def whoami() -> dict[str, object]:
    """返回当前已认证客户端的身份与权限。"""

    access_token = get_access_token()
    if access_token is None:
        # 正常情况下不会发生，因为未认证请求已被 HTTP 中间件拒绝。
        return {"authenticated": False}

    return {
        "authenticated": True,
        "client_id": access_token.client_id,
        "subject": access_token.subject,
        "scopes": access_token.scopes,
        "resource": access_token.resource,
    }


@mcp_server.tool()
async def secure_add(a: float, b: float) -> float:
    """只有通过认证的客户端才能调用的加法工具。"""

    return a + b


def main() -> None:
    """在 8002 端口启动受保护的 MCP Resource Server。"""

    print("Day16 认证 MCP Server 正在启动……")
    print("MCP 地址：", RESOURCE_URL)
    print("教学令牌：", DEMO_TOKEN)
    print("要求权限：", REQUIRED_SCOPE)
    print("按 Ctrl+C 停止服务。")

    try:
        mcp_server.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=8002,
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=True,
        )
    except KeyboardInterrupt:
        print("\n收到停止信号，认证 MCP Server 已关闭。")


if __name__ == "__main__":
    main()

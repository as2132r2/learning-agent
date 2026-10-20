"""Day16 第十六步（服务端）：根据 AccessToken scopes 控制 MCP 工具权限。

运行方式：
    python day16/python/day16_mcp_scope_server.py

MCP 地址：http://127.0.0.1:8003/mcp

教学令牌：
- day16-reader-token   → account:read
- day16-operator-token → account:read + account:write
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AnyHttpUrl

from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver.exceptions import ToolError


RESOURCE_URL = "http://127.0.0.1:8003/mcp"

KNOWN_TOKENS: dict[str, AccessToken] = {
    "day16-reader-token": AccessToken(
        token="day16-reader-token",
        client_id="reader-client",
        subject="user-reader",
        scopes=["account:read"],
        resource=RESOURCE_URL,
    ),
    "day16-operator-token": AccessToken(
        token="day16-operator-token",
        client_id="operator-client",
        subject="user-operator",
        scopes=["account:read", "account:write"],
        resource=RESOURCE_URL,
    ),
}


class ScopeTokenVerifier(TokenVerifier):
    """验证教学令牌，并返回其中携带的身份与 scopes。"""

    async def verify_token(self, token: str) -> AccessToken | None:
        return KNOWN_TOKENS.get(token)


mcp_server = MCPServer(
    name="day16-scope-server",
    title="Day16 Scope 授权服务",
    version="1.0.0",
    token_verifier=ScopeTokenVerifier(),
    auth=AuthSettings(
        issuer_url=AnyHttpUrl("https://auth.example.com"),
        resource_server_url=AnyHttpUrl(RESOURCE_URL),
        # 这里只要求成功认证；每个工具再检查自己的具体 scope。
        required_scopes=[],
        validate_token_resource=True,
    ),
)


ACCOUNTS: dict[str, float] = {
    "account-001": 1000.0,
    "account-002": 200.0,
}


def require_scope(required_scope: str) -> AccessToken:
    """取得当前身份，并确认它拥有指定 scope。"""

    access_token = get_access_token()
    if access_token is None:
        raise ToolError("当前请求没有认证身份。")

    if required_scope not in access_token.scopes:
        raise ToolError(
            f"用户 {access_token.subject!r} 缺少权限 {required_scope!r}。"
        )

    return access_token


@mcp_server.tool()
async def check_balance(
    account: Literal["account-001", "account-002"],
) -> dict[str, Any]:
    """查询账户余额；要求 account:read。"""

    access_token = require_scope("account:read")
    return {
        "requested_by": access_token.subject,
        "required_scope": "account:read",
        "account": account,
        "balance": ACCOUNTS[account],
    }


@mcp_server.tool()
async def transfer_money(
    from_account: Literal["account-001", "account-002"],
    to_account: Literal["account-001", "account-002"],
    amount: float,
) -> dict[str, Any]:
    """模拟转账；要求 account:write。"""

    access_token = require_scope("account:write")

    if from_account == to_account:
        raise ToolError("付款账户和收款账户不能相同。")
    if amount <= 0:
        raise ToolError("转账金额必须大于 0。")
    if ACCOUNTS[from_account] < amount:
        raise ToolError("余额不足。")

    ACCOUNTS[from_account] -= amount
    ACCOUNTS[to_account] += amount

    return {
        "requested_by": access_token.subject,
        "required_scope": "account:write",
        "status": "success",
        "amount": amount,
        "balances": dict(ACCOUNTS),
    }


def main() -> None:
    """在 8003 端口启动带工具级 scope 授权的 MCP Server。"""

    print("Day16 Scope MCP Server 正在启动……")
    print("MCP 地址：", RESOURCE_URL)
    print("只读令牌：day16-reader-token")
    print("操作员令牌：day16-operator-token")
    print("按 Ctrl+C 停止服务。")

    try:
        mcp_server.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=8003,
            streamable_http_path="/mcp",
            stateless_http=True,
            json_response=True,
        )
    except KeyboardInterrupt:
        print("\n收到停止信号，Scope MCP Server 已关闭。")


if __name__ == "__main__":
    main()

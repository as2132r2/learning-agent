"""Day16 第十六步（客户端）：用不同 scope 的令牌调用相同 MCP 工具。

先启动服务端：
    python day16/python/day16_mcp_scope_server.py

只读用户：
    python day16/python/day16_mcp_scope_client.py reader

操作员：
    python day16/python/day16_mcp_scope_client.py operator
"""

from __future__ import annotations

import asyncio
import sys

import httpx2
import mcp_types as types
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


SERVER_URL = "http://127.0.0.1:8003/mcp"

TOKENS = {
    "reader": "day16-reader-token",
    "operator": "day16-operator-token",
}


def show_result(title: str, result: types.CallToolResult) -> None:
    """显示工具成功或授权失败的结果。"""

    print("\n" + "=" * 70)
    print(title)
    print("is_error：", result.is_error)
    print("structured_content：", result.structured_content)

    for content in result.content:
        if isinstance(content, types.TextContent):
            print("text：", content.text)


async def main() -> None:
    """用指定角色令牌依次调用读取和写入工具。"""

    role = sys.argv[1].lower() if len(sys.argv) == 2 else "reader"
    token = TOKENS.get(role)

    if token is None:
        print("角色只能是 reader 或 operator。")
        return

    print("Day16 Scope MCP Client 已启动。")
    print("当前角色：", role)
    print("服务地址：", SERVER_URL)

    headers = {"Authorization": f"Bearer {token}"}

    async with httpx2.AsyncClient(headers=headers) as http_client:
        transport = streamable_http_client(
            SERVER_URL,
            http_client=http_client,
        )

        try:
            async with Client(transport) as client:
                balance_result = await client.call_tool(
                    "check_balance",
                    arguments={"account": "account-001"},
                )
                show_result("调用 check_balance", balance_result)

                transfer_result = await client.call_tool(
                    "transfer_money",
                    arguments={
                        "from_account": "account-001",
                        "to_account": "account-002",
                        "amount": 100,
                    },
                )
                show_result("调用 transfer_money", transfer_result)
        except Exception as error:
            print("\n认证或连接失败：", type(error).__name__, error)
            return

    print("\n" + "=" * 70)
    print("授权数据流：")
    print("Bearer Token → TokenVerifier → AccessToken.scopes")
    print("→ check_balance 检查 account:read")
    print("→ transfer_money 检查 account:write")
    print("→ scope 满足才执行工具业务逻辑")


if __name__ == "__main__":
    asyncio.run(main())

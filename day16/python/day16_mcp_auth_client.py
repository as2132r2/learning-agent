"""Day16 第十五步（客户端）：携带 Bearer Token 访问 MCP Server。

先启动服务端：
    python day16/python/day16_mcp_auth_server.py

有效令牌测试：
    python day16/python/day16_mcp_auth_client.py valid

错误令牌测试：
    python day16/python/day16_mcp_auth_client.py wrong

无令牌测试：
    python day16/python/day16_mcp_auth_client.py none
"""

from __future__ import annotations

import asyncio
import sys

import httpx2
import mcp_types as types
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


SERVER_URL = "http://127.0.0.1:8002/mcp"
DEMO_TOKEN = "day16-demo-token"


def select_token(mode: str) -> str | None:
    """根据命令行模式返回教学令牌、错误令牌或不返回令牌。"""

    if mode == "valid":
        return DEMO_TOKEN
    if mode == "wrong":
        return "wrong-token"
    if mode == "none":
        return None
    raise ValueError("模式只能是 valid、wrong 或 none。")


async def main() -> None:
    """创建带认证头的 HTTP transport，并调用受保护工具。"""

    mode = sys.argv[1].lower() if len(sys.argv) == 2 else "valid"

    try:
        token = select_token(mode)
    except ValueError as error:
        print(error)
        return

    headers: dict[str, str] = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    print("Day16 认证 MCP Client 已启动。")
    print("服务地址：", SERVER_URL)
    print("测试模式：", mode)
    print("是否携带 Authorization：", "Authorization" in headers)

    # MCP Client 暂时不直接接收 headers；把带认证头的 AsyncClient
    # 交给 Streamable HTTP transport。
    async with httpx2.AsyncClient(headers=headers) as http_client:
        transport = streamable_http_client(
            SERVER_URL,
            http_client=http_client,
        )

        try:
            async with Client(transport) as client:
                print("\n认证成功，MCP 连接已经建立。")
                print("协议版本：", client.protocol_version)

                tools_result = await client.list_tools()
                print("可用工具：", [tool.name for tool in tools_result.tools])

                identity_result = await client.call_tool("whoami")
                print("\nwhoami 返回：")
                print(identity_result.structured_content)

                add_result = await client.call_tool(
                    "secure_add",
                    arguments={"a": 20, "b": 4},
                )
                print("\nsecure_add(20, 4) 返回：")
                print(add_result.structured_content)
        except Exception as error:
            print("\n认证或连接失败。")
            print("错误类型：", type(error).__name__)
            print("错误信息：", error)
            print("服务端没有执行任何受保护工具。")
            return

    print("\n" + "=" * 70)
    print("认证数据流：")
    print("Bearer Token")
    print("→ Authorization 请求头")
    print("→ Streamable HTTP")
    print("→ MCP BearerAuthBackend")
    print("→ StaticTokenVerifier.verify_token()")
    print("→ 检查 scope 和 resource")
    print("→ 通过后才进入 MCP tools/list / tools/call")


if __name__ == "__main__":
    asyncio.run(main())

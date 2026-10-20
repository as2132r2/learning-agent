"""Day16 第九步（客户端）：连接独立的 HTTP MCP Server。

先在终端一运行：
    python day16/python/day16_mcp_http_server.py

再在终端二运行：
    python day16/python/day16_mcp_http_client.py
"""

from __future__ import annotations

import asyncio
import json

import mcp_types as types
from mcp import Client


SERVER_URL = "http://127.0.0.1:8000/mcp"


def show_tools(result: types.ListToolsResult) -> None:
    """显示远程 MCP Server 提供的工具。"""

    print("\n" + "=" * 70)
    print("远程工具数量：", len(result.tools))

    for index, tool in enumerate(result.tools, start=1):
        print(f"\n工具 {index}")
        print("名称：", tool.name)
        print("说明：", tool.description)
        print("输入 Schema：")
        print(json.dumps(tool.input_schema, ensure_ascii=False, indent=2))


def show_result(title: str, result: types.CallToolResult) -> None:
    """显示一次远程工具调用的返回结果。"""

    print("\n" + "=" * 70)
    print(title)
    print("返回类型：", type(result))
    print("是否错误：", result.is_error)
    print("structured_content：", result.structured_content)

    for content in result.content:
        if isinstance(content, types.TextContent):
            print("文本结果：", content.text)


async def main() -> None:
    """通过 HTTP 发现工具，并完成 (20+4)*3。"""

    client_info = types.Implementation(
        name="day16-http-client",
        version="1.0.0",
    )

    print("Day16 HTTP MCP Client 已启动。")
    print("准备连接：", SERVER_URL)

    try:
        # 向 Client 传入 HTTP URL，它会自动使用 Streamable HTTP transport。
        async with Client(
            SERVER_URL,
            client_info=client_info,
        ) as client:
            print("连接成功。")
            print("协议版本：", client.protocol_version)
            print("服务端身份：", client.server_info)

            tools_result = await client.list_tools()
            show_tools(tools_result)

            add_result = await client.call_tool(
                "add",
                arguments={"a": 20, "b": 4},
            )
            show_result("远程调用 add(20, 4)", add_result)

            if not isinstance(add_result.structured_content, dict):
                raise RuntimeError("add 没有返回结构化结果。")
            first_result = add_result.structured_content["result"]

            multiply_result = await client.call_tool(
                "multiply",
                arguments={"a": first_result, "b": 3},
            )
            show_result(
                f"远程调用 multiply({first_result}, 3)",
                multiply_result,
            )
    except OSError as error:
        print("\n连接失败：", error)
        print("请确认服务端已在另一个终端启动。")
        return

    print("\n" + "=" * 70)
    print("最终结果：", multiply_result.structured_content)
    print("\n本步数据流：")
    print("HTTP MCP Client")
    print("→ POST http://127.0.0.1:8000/mcp")
    print("→ Streamable HTTP transport")
    print("→ 独立 MCP Server 进程")
    print("→ add / multiply")
    print("→ HTTP MCP 响应")
    print("→ CallToolResult")


if __name__ == "__main__":
    asyncio.run(main())

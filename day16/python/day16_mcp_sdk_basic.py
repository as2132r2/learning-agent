"""Day16 第三步：使用正式 MCP Python SDK 创建工具服务和客户端。

运行方式：
    python day16/python/day16_mcp_sdk_basic.py

这一步使用 SDK 提供的进程内连接，先专注理解工具注册、自动生成 Schema、
工具发现和工具调用。消息解析和响应创建都由正式 SDK 完成，不再由我们
手写 JSON-RPC 分派代码。下一步再单独学习正式 SDK 的网络传输。

当前代码针对 mcp 2.x；mcp 1.x 旧教程中的 FastMCP 已经更名为 MCPServer。
"""

from __future__ import annotations

import asyncio
import json
import sys

import mcp_types as types
from mcp import Client
from mcp.server import MCPServer


# ---------------------------------------------------------------------------
# 服务端：创建 MCP Server 并注册工具
# ---------------------------------------------------------------------------

mcp_server = MCPServer(
    name="day16-calculator",
    title="Day16 计算器工具服务",
    description="用于学习正式 MCP SDK 的加法和乘法服务。",
    version="1.0.0",
)


@mcp_server.tool()
async def add(a: float, b: float) -> float:
    """计算两个数字的和。"""

    # 继续把服务日志写入 stderr，便于以后切换到 stdio 传输。
    print(f"[MCP Server 执行 add] {a} + {b}", file=sys.stderr, flush=True)
    return a + b


@mcp_server.tool()
async def multiply(a: float, b: float) -> float:
    """计算两个数字的乘积。"""

    print(
        f"[MCP Server 执行 multiply] {a} × {b}",
        file=sys.stderr,
        flush=True,
    )
    return a * b


# ---------------------------------------------------------------------------
# 客户端：连接服务端、发现工具、调用工具
# ---------------------------------------------------------------------------

def print_tools(tools_result: types.ListToolsResult) -> None:
    """显示 SDK 返回的工具对象及其输入 Schema。"""

    print("\n" + "=" * 70)
    print("服务端公开的工具：")

    for index, tool in enumerate(tools_result.tools, start=1):
        print(f"\n工具 {index}")
        print("名称：", tool.name)
        print("说明：", tool.description)
        print("输入 Schema：")
        print(
            json.dumps(
                tool.input_schema,
                ensure_ascii=False,
                indent=2,
            )
        )


def print_call_result(title: str, result: types.CallToolResult) -> None:
    """显示一次 MCP 工具调用的完整返回数据。"""

    print("\n" + "=" * 70)
    print(title)
    print("返回对象类型：", type(result))
    print("是否为工具错误：", result.is_error)

    print("content：")
    for item in result.content:
        print("- 元素类型：", type(item))
        if isinstance(item, types.TextContent):
            print("  文本：", item.text)
        else:
            print("  内容：", item)

    print("structuredContent：", result.structured_content)


async def run_client() -> None:
    """在进程内连接 MCP Server，依次发现和调用工具。"""

    client_info = types.Implementation(
        name="day16-sdk-client",
        version="1.0.0",
    )

    print("Day16 正式 MCP SDK 客户端已启动。")
    print("连接方式：进程内连接（不启动端口，不创建子进程）")

    # 直接把 MCPServer 对象交给 Client，会建立进程内连接。
    # Client 仍然会完成正式的服务发现和请求分派。
    async with Client(
        mcp_server,
        client_info=client_info,
    ) as client:
        print("\n" + "=" * 70)
        print("MCP 服务发现完成。")
        print("客户端协商使用的协议版本：", client.protocol_version)
        print("服务端身份：", client.server_info)
        print("服务端能力：")
        print(
            json.dumps(
                client.server_capabilities.model_dump(
                    by_alias=True,
                    exclude_none=True,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )

        tools_result = await client.list_tools()
        print_tools(tools_result)

        add_result = await client.call_tool(
            "add",
            arguments={"a": 20, "b": 4},
        )
        print_call_result("调用 add(a=20, b=4)", add_result)

        # 从正式 MCP 返回对象中读取结构化数值。
        if not isinstance(add_result.structured_content, dict):
            raise RuntimeError("add 没有返回 structuredContent。")
        first_result = add_result.structured_content["result"]

        multiply_result = await client.call_tool(
            "multiply",
            arguments={"a": first_result, "b": 3},
        )
        print_call_result(
            f"调用 multiply(a={first_result}, b=3)",
            multiply_result,
        )

    print("\n" + "=" * 70)
    print("离开 async with 后，客户端连接已经关闭。")
    print("本步数据流：")
    print("Python 函数 + 类型注解 + docstring")
    print("→ @mcp_server.tool() 自动生成工具定义和 JSON Schema")
    print("→ Client.list_tools() 发现工具")
    print("→ Client.call_tool() 调用工具")
    print("→ SDK 解析为 CallToolResult 对象")


def main() -> None:
    """启动异步 MCP 客户端练习。"""

    asyncio.run(run_client())


if __name__ == "__main__":
    main()

"""Day16 第十二步：区分 MCP 工具错误与协议错误。

运行方式：
    python day16/python/day16_mcp_error_handling.py

本步骤演示三种结果：
1. 工具正常执行：返回 CallToolResult，is_error=False；
2. 工具内部失败：仍返回 CallToolResult，is_error=True；
3. 请求不存在的工具：高层服务同样返回 is_error=True；
4. 读取不存在的资源：抛出 MCPError，属于协议/请求错误。
"""

from __future__ import annotations

import asyncio

import mcp_types as types
from mcp import Client
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.shared.exceptions import MCPError


mcp_server = MCPServer(
    name="day16-error-demo",
    title="Day16 MCP 错误处理服务",
    description="演示正常结果、工具错误和协议错误。",
    version="1.0.0",
)


@mcp_server.tool()
async def divide(a: float, b: float) -> float:
    """计算 a 除以 b；除数不能为零。"""

    print(f"[执行 divide] {a} ÷ {b}")

    if b == 0:
        # 可预期的业务失败使用 ToolError，服务端不会打印意外异常堆栈。
        raise ToolError("除数 b 不能为 0。")

    return a / b


def show_call_result(title: str, result: types.CallToolResult) -> None:
    """显示正常结果或工具错误结果。"""

    print("\n" + "=" * 70)
    print(title)
    print("返回对象类型：", type(result))
    print("is_error：", result.is_error)
    print("structured_content：", result.structured_content)
    print("content：")

    for content in result.content:
        if isinstance(content, types.TextContent):
            print("-", content.text)
        else:
            print("-", content)

    if result.is_error:
        print("结论：MCP 请求成功到达服务端，但工具执行失败。")
    else:
        print("结论：MCP 请求和工具执行都成功。")


async def main() -> None:
    """依次触发成功、工具失败和协议失败。"""

    print("Day16 MCP 错误处理练习已启动。")

    async with Client(mcp_server) as client:
        # 场景一：请求合法，工具执行成功。
        success_result = await client.call_tool(
            "divide",
            arguments={"a": 20, "b": 4},
        )
        show_call_result("场景一：divide(20, 4)", success_result)

        # 场景二：工具存在、参数类型正确，但业务规则不允许除以零。
        tool_error_result = await client.call_tool(
            "divide",
            arguments={"a": 20, "b": 0},
        )
        show_call_result("场景二：divide(20, 0)", tool_error_result)

        # 场景三：未知工具也由高层 MCPServer 包装成工具错误结果。
        print("\n" + "=" * 70)
        print("场景三：调用不存在的工具 subtract")
        unknown_tool_result = await client.call_tool(
            "subtract",
            arguments={"a": 20, "b": 4},
        )
        show_call_result("未知工具的返回结果", unknown_tool_result)

        # 场景四：读取一个不存在的 Resource 会产生 MCPError。
        print("\n" + "=" * 70)
        print("场景四：读取不存在的资源 course://missing")

        try:
            await client.read_resource("course://missing")
        except MCPError as error:
            print("捕获到异常类型：", type(error))
            print("错误码：", error.code)
            print("错误信息：", error.message)
            print("结论：请求没有获得目标资源，SDK 抛出 MCPError。")

        # 上面的错误不会破坏连接；再次调用合法工具仍然能够成功。
        recovery_result = await client.call_tool(
            "divide",
            arguments={"a": 9, "b": 3},
        )
        show_call_result(
            "场景五：发生错误后再次调用 divide(9, 3)",
            recovery_result,
        )

    print("\n" + "=" * 70)
    print("错误分层：")
    print("工具成功     → CallToolResult(is_error=False)")
    print("工具执行失败 → CallToolResult(is_error=True)")
    print("未知工具     → CallToolResult(is_error=True)")
    print("资源不存在   → raise MCPError")
    print("连接失败     → HTTP/网络/传输异常")


if __name__ == "__main__":
    asyncio.run(main())

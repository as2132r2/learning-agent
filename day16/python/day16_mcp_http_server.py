"""Day16 第九步（服务端）：通过 Streamable HTTP 发布 MCP 工具。

运行方式：
    python day16/python/day16_mcp_http_server.py

服务地址：
    http://127.0.0.1:8000/mcp

健康检查：
    http://127.0.0.1:8000/health
"""

from __future__ import annotations

import sys

from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse


mcp_server = MCPServer(
    name="day16-http-calculator",
    title="Day16 HTTP MCP 计算器",
    description="通过 Streamable HTTP 提供加法和乘法工具。",
    version="1.0.0",
)


@mcp_server.tool()
async def add(a: float, b: float) -> float:
    """计算两个数字的和。"""

    print(f"[HTTP MCP Tool 执行 add] {a} + {b}", file=sys.stderr, flush=True)
    return a + b


@mcp_server.tool()
async def multiply(a: float, b: float) -> float:
    """计算两个数字的乘积。"""

    print(
        f"[HTTP MCP Tool 执行 multiply] {a} × {b}",
        file=sys.stderr,
        flush=True,
    )
    return a * b


@mcp_server.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """普通 HTTP 健康检查，不属于 MCP 协议方法。"""

    return JSONResponse(
        {
            "status": "ok",
            "service": "day16-http-calculator",
            "mcp_endpoint": "/mcp",
        }
    )


def main() -> None:
    """启动 Streamable HTTP MCP Server。"""

    print("Day16 HTTP MCP Server 正在启动……")
    print("MCP 地址：http://127.0.0.1:8000/mcp")
    print("健康检查：http://127.0.0.1:8000/health")
    print("按 Ctrl+C 停止服务。")

    try:
        mcp_server.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=8000,
            streamable_http_path="/mcp",
            # 当前协议支持无状态 HTTP；请求可由不同服务实例处理。
            stateless_http=True,
            # 客户端得到普通 JSON 响应，便于当前练习观察。
            json_response=True,
        )
    except KeyboardInterrupt:
        print("\n收到停止信号，MCP Server 已关闭。")


if __name__ == "__main__":
    main()
